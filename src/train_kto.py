"""Train M7: MACA's MV-KTO on individually labelled debate answers (ADR-0009).

KTO (Kahneman-Tversky Optimisation) needs no pairs. Each answer carries its own
binary label -- desirable if the persona's verdict matched the debate majority,
undesirable if it dissented -- so **unanimous items still train**, which is the
whole reason this arm exists: MV-DPO could only learn from the 15% of items
where the personas disagreed after debate.

For one answer ``y`` to prompt ``x``, with ``r = log pi(y|x) - log ref(y|x)``::

    desirable:   loss = w_D * (1 - sigmoid( beta * (r - z_ref) ))
    undesirable: loss = w_U * (1 - sigmoid( beta * (z_ref - r) ))

``z_ref`` is a baseline: the model's average KL from the reference, estimated on
**mismatched** prompt/answer pairs so it measures drift rather than content.
Following the KTO paper it is clamped at zero and not back-propagated through.

Two documented deviations, both recorded in ADR-0009:

- The classes are imbalanced (4.8:1 desirable:undesirable with round-1 text), so
  ``w_U`` lifts the rare class back into KTO's recommended working range.
- Training conditions on the baseline prompt alone, never the debate context,
  because a deployed trusted monitor has no peers to read.

Everything not specific to KTO is shared with M4 and M5: the 4-bit base model
and LoRA adapter, the prompt encoding, the optimiser, seed and adapter saving.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import config
from src.data import SplitName, load_split
from src.debate import load_consensus_responses
from src.train_dpo import sequence_logps
from src.train_sft import (
    encode_prompt,
    encode_response,
    load_student,
    load_tokenizer,
    save_run,
    strip_label,
)
from src.utils import format_duration, get_logger, run_dir, set_seed

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

SMOKE_EXAMPLES = 40
SMOKE_EPOCHS = 1
SMOKE_GRAD_ACCUM = 2

#: KTO's recommended desirable:undesirable working range for the weighted
#: counts, from the paper. ``balanced_undesirable_weight`` targets its middle.
KTO_RATIO_RANGE = (1.0, 4.0 / 3.0)


@dataclass
class KTOExample:
    """One tokenised answer with its own binary label.

    Attributes:
        item_id: The source item.
        input_ids: Prompt followed by the answer.
        labels: ``-100`` on the prompt, the answer's ids after it.
        desirable: Whether this answer agreed with the debate majority.
    """

    item_id: str
    input_ids: list[int]
    labels: list[int]
    desirable: bool


def balanced_undesirable_weight(
    n_desirable: int, n_undesirable: int, desirable_weight: float = 1.0
) -> float:
    """Weight that pulls an imbalanced split into KTO's recommended range.

    KTO asks that ``n_D * w_D / (n_U * w_U)`` sit in [1, 4/3]. Our debate is
    far outside it, so this returns the ``w_U`` putting the ratio at the middle
    of that range.

    Args:
        n_desirable: Count of agreeing answers.
        n_undesirable: Count of dissenting answers.
        desirable_weight: Weight on the desirable class.

    Returns:
        The undesirable weight, or ``desirable_weight`` if either class is
        empty (the ratio is undefined and nothing can be balanced).
    """
    if n_desirable == 0 or n_undesirable == 0:
        return desirable_weight
    target = sum(KTO_RATIO_RANGE) / 2
    return (n_desirable * desirable_weight) / (n_undesirable * target)


def build_kto_examples(
    items: list[Any],
    consensus: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[KTOExample], int]:
    """Tokenise every agreeing and dissenting answer as its own example.

    Args:
        items: Training items, labels blanked by the caller.
        consensus: Output of ``debate.load_consensus_responses``.
        tokenizer: The base model's tokenizer.

    Returns:
        ``(examples, dropped)``. Examples longer than
        ``config.TRAINING.max_seq_length`` are dropped, never truncated: a
        truncated answer would lose the verdict it is labelled for.
    """
    max_length = config.TRAINING.max_seq_length
    examples: list[KTOExample] = []
    dropped = 0
    for item in items:
        entry = consensus.get(item.item_id)
        if not entry:
            continue
        prompt_ids = encode_prompt(strip_label(item), tokenizer)
        for desirable, key in ((True, "agree"), (False, "dissent")):
            for answer in entry[key]:
                response_ids = encode_response(answer["response"], tokenizer)
                input_ids = prompt_ids + response_ids
                if len(input_ids) > max_length:
                    dropped += 1
                    continue
                examples.append(KTOExample(
                    item_id=item.item_id,
                    input_ids=input_ids,
                    labels=[-100] * len(prompt_ids) + response_ids,
                    desirable=desirable,
                ))
    return examples, dropped


def kl_baseline(
    policy_logps: torch.Tensor, reference_logps_batch: torch.Tensor
) -> torch.Tensor:
    """Estimate KTO's ``z_ref``: mean policy-to-reference drift, clamped at zero.

    Detached, so no gradient flows through the baseline -- it shifts where the
    sigmoid sits, it is not itself optimised.

    Args:
        policy_logps: ``(B,)`` policy log-probs on mismatched pairs.
        reference_logps_batch: ``(B,)`` reference log-probs on the same pairs.

    Returns:
        A scalar tensor, ``>= 0``.
    """
    import torch

    drift = (policy_logps - reference_logps_batch).mean().detach()
    return torch.clamp(drift, min=0.0)


def kto_loss(
    policy_logps: torch.Tensor,
    reference_logps_batch: torch.Tensor,
    desirable: torch.Tensor,
    z_ref: torch.Tensor,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the KTO loss and its diagnostics.

    Desirable answers are pushed above the baseline and dissenting ones below
    it, each weighted by its class weight.

    Args:
        policy_logps: ``(B,)`` policy log-probs.
        reference_logps_batch: ``(B,)`` reference log-probs.
        desirable: ``(B,)`` boolean, true where the answer agreed.
        z_ref: The KL baseline.
        beta: KTO temperature.
        desirable_weight: Weight on agreeing answers.
        undesirable_weight: Weight on dissenting answers.

    Returns:
        ``(loss, parts)``. ``parts`` reports the mean implicit reward of each
        class and the share of answers already on the correct side of the
        baseline.
    """
    import torch

    rewards = beta * (policy_logps - reference_logps_batch)
    desirable_value = torch.sigmoid(rewards - beta * z_ref)
    undesirable_value = torch.sigmoid(beta * z_ref - rewards)

    weights = torch.where(
        desirable,
        torch.full_like(rewards, desirable_weight),
        torch.full_like(rewards, undesirable_weight),
    )
    values = torch.where(desirable, desirable_value, undesirable_value)
    loss = (weights * (1 - values)).sum() / weights.sum()

    correct = torch.where(desirable, rewards > beta * z_ref, rewards < beta * z_ref)
    parts = {
        "reward_desirable": _masked_mean(rewards, desirable),
        "reward_undesirable": _masked_mean(rewards, ~desirable),
        "accuracy": correct.float().mean().item(),
        "z_ref": z_ref.item(),
    }
    return loss, parts


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of ``values`` where ``mask`` is true, or 0.0 if it never is."""
    if not bool(mask.any()):
        return 0.0
    return values[mask].detach().mean().item()


def reference_logps_for(
    model: Any, examples: list[KTOExample], batch_size: int, pad_id: int
) -> list[float]:
    """Score every example under the reference: the adapter switched off.

    Args:
        model: The PEFT model.
        examples: Tokenised examples.
        batch_size: Examples per forward pass.
        pad_id: Padding token id.

    Returns:
        One reference log-probability per example, in order.
    """
    import torch

    model.eval()
    scores: list[float] = []
    with torch.no_grad(), model.disable_adapter():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            logps = sequence_logps(
                model, [e.input_ids for e in batch], [e.labels for e in batch], pad_id
            )
            scores.extend(logps.tolist())
    return scores


def mismatched_batch(
    examples: list[KTOExample], indices: list[int], tokenizer: Any
) -> tuple[list[list[int]], list[list[int]]]:
    """Pair each example's prompt with a different example's answer.

    KTO's baseline term measures how far the policy has drifted from the
    reference in general, so it is estimated on pairings the data never
    contains: prompt ``i`` with answer ``i + 1``.

    Args:
        examples: The batch's source examples.
        indices: Which examples to use.
        tokenizer: Unused, kept so callers need not special-case this.

    Returns:
        ``(sequences, labels)`` for the mismatched pairings.
    """
    sequences, labels = [], []
    for position, index in enumerate(indices):
        other = examples[indices[(position + 1) % len(indices)]]
        current = examples[index]
        prompt_length = sum(1 for label in current.labels if label == -100)
        answer_ids = [
            token for token, label in zip(other.input_ids, other.labels, strict=True)
            if label != -100
        ]
        prompt_ids = current.input_ids[:prompt_length]
        sequences.append(prompt_ids + answer_ids)
        labels.append([-100] * prompt_length + answer_ids)
    return sequences, labels


def train_kto_epochs(
    model: Any,
    examples: list[KTOExample],
    references: list[float],
    epochs: int,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    kl_batch_size: int,
    seed: int,
    pad_id: int,
    tokenizer: Any = None,
) -> list[dict[str, float]]:
    """Train the adapter with KTO.

    Uses the same optimiser, warmup, decay, clipping and seeded shuffling as M4
    and M5, so the arms differ in objective only.

    Args:
        model: The PEFT model.
        examples: Tokenised examples.
        references: Reference log-probs, aligned with ``examples``.
        epochs: Passes over the data.
        beta: KTO temperature.
        desirable_weight: Weight on agreeing answers.
        undesirable_weight: Weight on dissenting answers.
        learning_rate: Peak learning rate.
        batch_size: Examples per micro-batch.
        grad_accum: Micro-batches per optimiser step.
        kl_batch_size: Mismatched pairs used for the baseline each step.
        seed: Shuffle seed.
        pad_id: Padding token id.
        tokenizer: Passed through to ``mismatched_batch``.

    Returns:
        One dict per optimiser step, with the loss and ``kto_loss``'s parts.
    """
    import math

    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)

    batches_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = epochs * math.ceil(batches_per_epoch / grad_accum)
    warmup_steps = max(1, round(0.05 * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    model.train()
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(examples), generator=generator).tolist()
        batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        micro = 0

        for batch_number, indices in enumerate(batches, start=1):
            batch = [examples[i] for i in indices]
            policy = sequence_logps(
                model, [e.input_ids for e in batch], [e.labels for e in batch], pad_id
            )
            reference = torch.tensor(
                [references[i] for i in indices], device=policy.device
            )
            desirable = torch.tensor(
                [e.desirable for e in batch], device=policy.device, dtype=torch.bool
            )

            kl_indices = indices[:kl_batch_size]
            if len(kl_indices) > 1:
                sequences, labels = mismatched_batch(examples, kl_indices, tokenizer)
                with torch.no_grad():
                    kl_policy = sequence_logps(model, sequences, labels, pad_id)
                    with model.disable_adapter():
                        kl_reference = sequence_logps(model, sequences, labels, pad_id)
                z_ref = kl_baseline(kl_policy, kl_reference)
            else:
                z_ref = torch.zeros((), device=policy.device)

            loss, parts = kto_loss(
                policy, reference, desirable, z_ref, beta,
                desirable_weight, undesirable_weight,
            )
            (loss / grad_accum).backward()

            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
            micro += 1

            if micro == grad_accum or batch_number == len(batches):
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                record = {name: value / micro for name, value in totals.items()}
                record.update(
                    step=step + 1, epoch=epoch, lr=scheduler.get_last_lr()[0]
                )
                history.append(record)
                step += 1
                if step == 1 or step % 10 == 0:
                    logger.info(
                        "step %d/%d · epoch %d · loss %.4f · acc %.2f · "
                        "reward D %.2f / U %.2f · z_ref %.2f",
                        step, total_steps, epoch, record["loss"], record["accuracy"],
                        record["reward_desirable"], record["reward_undesirable"],
                        record["z_ref"],
                    )
                totals = {}
                micro = 0

    return history


def summarise_kto_history(history: list[dict[str, float]]) -> str:
    """One line on whether KTO learned anything.

    Args:
        history: Output of ``train_kto_epochs``.

    Returns:
        A sentence for the log and the run metadata.
    """
    if not history:
        return "no optimizer steps ran"
    window = max(1, len(history) // 5)
    first = sum(h["loss"] for h in history[:window]) / window
    last = sum(h["loss"] for h in history[-window:]) / window
    direction = "decreased" if last < first else "did NOT decrease"
    return (
        f"loss {direction} over {len(history)} steps · "
        f"loss {first:.4f} -> {last:.4f} · "
        f"accuracy {history[0]['accuracy']:.2f} -> {history[-1]['accuracy']:.2f}"
    )


def train(
    smoke: bool = False,
    split: SplitName = "train",
    text_source: str = "round1",
    learning_rate: float | None = None,
    undesirable_weight: float | None = None,
    max_examples: int | None = None,
    epochs: float | None = None,
    tag: str = "",
) -> Path:
    """Train M7 end to end. The entry point ``main.py train-kto`` calls.

    The override arguments exist for the diagnostic in ADR-0010: M7's first run
    collapsed to answering "yes" almost always, and the overrides let a short
    low-learning-rate run test whether that collapse is an optimisation
    artefact or the objective faithfully reproducing a near-uninformative
    target. Any override is recorded in the run metadata.

    Args:
        smoke: Train briefly on a handful of examples to prove the loop runs.
        split: The split to train on. Only train is valid for a real run.
        text_source: Which debate text the majority vote selects; see
            ``debate.load_consensus_responses``.
        learning_rate: Override ``config.TRAINING.learning_rate``.
        undesirable_weight: Override the weight computed from the class counts.
        max_examples: Train on this many examples instead of all of them.
        epochs: Override ``config.TRAINING.epochs``.
        tag: Suffix for the run directory, so diagnostics do not overwrite the
            reported run.

    Returns:
        The saved adapter directory.
    """
    started = time.time()
    t = config.TRAINING
    kto = config.KTO
    seed = config.SPLITS.seed
    set_seed(seed)

    items = load_split(split)
    consensus = load_consensus_responses(split, text_source=text_source)
    tokenizer = load_tokenizer()
    examples, dropped = build_kto_examples(items, consensus, tokenizer)
    if smoke:
        examples = examples[:SMOKE_EXAMPLES]
    elif max_examples is not None:
        # Seeded subsample, so a diagnostic still sees both classes.
        import random as _random

        _random.Random(seed).shuffle(examples)
        examples = examples[:max_examples]

    n_desirable = sum(e.desirable for e in examples)
    n_undesirable = len(examples) - n_desirable
    if not n_desirable or not n_undesirable:
        raise ValueError(
            f"KTO needs both classes; got {n_desirable} desirable and "
            f"{n_undesirable} undesirable"
        )
    if undesirable_weight is None:
        undesirable_weight = balanced_undesirable_weight(
            n_desirable, n_undesirable, kto.desirable_weight
        )
    logger.info(
        "built %d examples (%d desirable, %d dissenting, %.1f:1), dropped %d · "
        "undesirable weight %.2f",
        len(examples), n_desirable, n_undesirable,
        n_desirable / max(1, n_undesirable), dropped, undesirable_weight,
    )

    import torch

    torch.manual_seed(seed)
    model = load_student()
    references = reference_logps_for(
        model, examples, t.per_device_batch_size, tokenizer.pad_token_id
    )
    logger.info("reference log-probs computed for %d examples", len(examples))

    passes = t.epochs if epochs is None else epochs
    epochs = SMOKE_EPOCHS if smoke else int(round(passes))
    rate = t.learning_rate if learning_rate is None else learning_rate
    grad_accum = SMOKE_GRAD_ACCUM if smoke else t.gradient_accumulation_steps
    history = train_kto_epochs(
        model,
        examples,
        references,
        epochs=epochs,
        beta=kto.beta,
        desirable_weight=kto.desirable_weight,
        undesirable_weight=undesirable_weight,
        learning_rate=t.learning_rate,
        batch_size=t.per_device_batch_size,
        grad_accum=grad_accum,
        kl_batch_size=kto.kl_batch_size,
        seed=seed,
        pad_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
    )

    directory = run_dir("m7-maca-kto-smoke" if smoke else "m7-maca-kto")
    adapter = save_run(model, history, directory, {
        "arm": "m7",
        "objective": "mv-kto",
        "smoke": smoke,
        "text_source": text_source,
        "beta": kto.beta,
        "learning_rate": rate,
        "max_examples": max_examples,
        "tag": tag,
        "desirable_weight": kto.desirable_weight,
        "undesirable_weight": round(undesirable_weight, 4),
        "epochs": epochs,
        "grad_accum": grad_accum,
        "n_examples": len(examples),
        "n_desirable": n_desirable,
        "n_undesirable": n_undesirable,
        "n_dropped": dropped,
        "seed": seed,
        "seconds": round(time.time() - started, 1),
    })
    logger.info(summarise_kto_history(history))
    logger.info("trained in %s -> %s", format_duration(time.time() - started), adapter)
    return adapter
