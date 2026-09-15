"""Train M5: MACA's MV-DPO on persona-debate preference pairs (ADR-0003).

Each pair is two round-2 debate answers to the same item: one agreeing with
the personas' majority verdict (chosen), one dissenting (rejected). DPO raises
the model's probability of the chosen answer relative to the rejected one,
measured against a frozen reference:

    loss = -log sigmoid( beta * [ (log pi(chosen) - log ref(chosen))
                                 - (log pi(rejected) - log ref(rejected)) ] )

The reference is the base model with the LoRA adapter switched off, so no
second copy of the model is loaded. Its log-probabilities never change, so
they are computed once before training.

Everything that is not specific to DPO is shared with M4: the 4-bit base model
and LoRA adapter, the baseline prompt encoding, the hyperparameters and seed,
and the way the adapter is saved and served. Training conditions on the
baseline prompt alone, not the debate context, to match deployment.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import config
from src.data import SplitName, load_split
from src.debate import load_pairs
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

SMOKE_PAIRS = 20
SMOKE_EPOCHS = 1
SMOKE_GRAD_ACCUM = 2


@dataclass
class PairExample:
    """One tokenised preference pair.

    Attributes:
        item_id: The source item.
        chosen_ids: Prompt followed by the agreeing answer.
        chosen_labels: ``-100`` on the prompt, the answer's ids after it.
        rejected_ids: Prompt followed by the dissenting answer.
        rejected_labels: ``-100`` on the prompt, the answer's ids after it.
    """

    item_id: str
    chosen_ids: list[int]
    chosen_labels: list[int]
    rejected_ids: list[int]
    rejected_labels: list[int]


def tokenise_pair(
    prompt_ids: list[int], chosen: str, rejected: str, tokenizer: Any, pair_id: str
) -> PairExample | None:
    """Tokenise one pair, sharing the prompt between both answers.

    Args:
        prompt_ids: The encoded baseline prompt.
        chosen: The agreeing answer.
        rejected: The dissenting answer.
        tokenizer: The base model's tokenizer.
        pair_id: Identifier recorded on the example.

    Returns:
        The example, or ``None`` if either sequence exceeds the length limit.
    """
    chosen_response = encode_response(chosen, tokenizer)
    rejected_response = encode_response(rejected, tokenizer)
    limit = config.TRAINING.max_seq_length
    if len(prompt_ids) + max(len(chosen_response), len(rejected_response)) > limit:
        return None
    mask = [-100] * len(prompt_ids)
    return PairExample(
        item_id=pair_id,
        chosen_ids=prompt_ids + chosen_response,
        chosen_labels=mask + chosen_response,
        rejected_ids=prompt_ids + rejected_response,
        rejected_labels=mask + rejected_response,
    )


def build_pair_examples(
    pairs: list[dict[str, Any]], split: SplitName, tokenizer: Any
) -> tuple[list[PairExample], int]:
    """Tokenise every pair under the baseline prompt.

    Args:
        pairs: Records from ``debate.load_pairs``.
        split: The split the pairs came from, for looking up each item.
        tokenizer: The base model's tokenizer.

    Returns:
        ``(examples, dropped)``.
    """
    items = {item.item_id: strip_label(item) for item in load_split(split)}
    prompts: dict[str, list[int]] = {}
    examples, dropped = [], 0
    for index, pair in enumerate(pairs):
        item_id = pair["item_id"]
        if item_id not in prompts:
            prompts[item_id] = encode_prompt(items[item_id], tokenizer)
        example = tokenise_pair(
            prompts[item_id], pair["chosen"], pair["rejected"], tokenizer,
            f"{item_id}#{index}",
        )
        if example is None:
            dropped += 1
        else:
            examples.append(example)
    return examples, dropped


def pad_batch(
    sequences: list[list[int]], labels: list[list[int]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Right-pad sequences, masking padding in attention and labels.

    Args:
        sequences: Token ids per sequence.
        labels: Labels per sequence, ``-100`` where masked.
        pad_id: The tokenizer's pad token id.

    Returns:
        ``input_ids``, ``attention_mask`` and ``labels``, each ``(B, T)``.
    """
    import torch

    width = max(len(ids) for ids in sequences)
    return {
        "input_ids": torch.tensor(
            [ids + [pad_id] * (width - len(ids)) for ids in sequences]
        ),
        "attention_mask": torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]
        ),
        "labels": torch.tensor(
            [lab + [-100] * (width - len(lab)) for lab in labels]
        ),
    }


def logps_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum each sequence's log-probability over its supervised tokens.

    Logits at position ``t`` predict the token at ``t + 1``. Computed with
    per-token cross-entropy in float32, which avoids materialising a full
    log-softmax over Qwen's ~152k vocabulary.

    Args:
        logits: ``(B, T, V)``.
        labels: ``(B, T)``, ``-100`` where masked.

    Returns:
        ``(B,)`` summed log-probabilities of the answer tokens.
    """
    from torch.nn import functional

    shifted_logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    token_nll = functional.cross_entropy(
        shifted_logits.transpose(1, 2), targets, ignore_index=-100, reduction="none"
    )
    return -(token_nll * (targets != -100)).sum(dim=-1)


def sequence_logps(
    model: Any, sequences: list[list[int]], labels: list[list[int]], pad_id: int
) -> torch.Tensor:
    """Run the model and return each sequence's answer log-probability.

    Args:
        model: The policy (or reference) model.
        sequences: Token ids per sequence.
        labels: Labels per sequence.
        pad_id: Padding token id.

    Returns:
        ``(B,)`` log-probabilities.
    """
    batch = pad_batch(sequences, labels, pad_id)
    batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits
    return logps_from_logits(logits, batch["labels"])


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the DPO loss and its diagnostics.

    Args:
        policy_chosen: ``(B,)`` policy log-probs of chosen answers.
        policy_rejected: ``(B,)`` policy log-probs of rejected answers.
        reference_chosen: ``(B,)`` reference log-probs of chosen answers.
        reference_rejected: ``(B,)`` reference log-probs of rejected answers.
        beta: DPO temperature.

    Returns:
        ``(loss, parts)``. ``parts`` holds ``reward_accuracy`` (share of pairs
        where the chosen answer's implicit reward is higher) and
        ``reward_margin`` (mean chosen-minus-rejected reward).
    """
    from torch.nn import functional

    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    margin = chosen_reward - rejected_reward
    loss = -functional.logsigmoid(margin).mean()
    parts = {
        "reward_accuracy": (margin > 0).float().mean().item(),
        "reward_margin": margin.detach().mean().item(),
    }
    return loss, parts


def reference_logps(
    model: Any, examples: list[PairExample], batch_size: int, pad_id: int
) -> list[tuple[float, float]]:
    """Score every pair under the frozen reference: the adapter switched off.

    Args:
        model: The PEFT model.
        examples: Tokenised pairs.
        batch_size: Pairs per forward pass.
        pad_id: Padding token id.

    Returns:
        ``(chosen_logp, rejected_logp)`` per example, in order.
    """
    import torch

    model.eval()
    scores: list[tuple[float, float]] = []
    with torch.no_grad(), model.disable_adapter():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            chosen = sequence_logps(
                model, [e.chosen_ids for e in batch], [e.chosen_labels for e in batch],
                pad_id,
            )
            rejected = sequence_logps(
                model, [e.rejected_ids for e in batch],
                [e.rejected_labels for e in batch], pad_id,
            )
            scores.extend(zip(chosen.tolist(), rejected.tolist(), strict=True))
    return scores


def train_dpo_epochs(
    model: Any,
    examples: list[PairExample],
    references: list[tuple[float, float]],
    epochs: int,
    beta: float,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    seed: int,
    pad_id: int,
) -> list[dict[str, float]]:
    """Train the adapter with DPO.

    Uses the same optimiser, warmup, decay, clipping and seeded shuffling as
    M4's training loop, so the two trained arms differ in objective only.

    Args:
        model: The PEFT model.
        examples: Tokenised pairs.
        references: Reference log-probs, aligned with ``examples``.
        epochs: Passes over the pairs.
        beta: DPO temperature.
        learning_rate: Peak learning rate.
        batch_size: Pairs per micro-batch.
        grad_accum: Micro-batches per optimiser step.
        seed: Shuffle seed.
        pad_id: Padding token id.

    Returns:
        One row per optimiser step: ``step``, ``epoch``, ``loss``,
        ``reward_accuracy``, ``reward_margin``, ``lr``.
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    batches_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = epochs * math.ceil(batches_per_epoch / grad_accum)
    warmup_steps = max(1, round(0.05 * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    model.train()
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(examples), generator=generator).tolist()
        batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
        micro = 0

        for index, indices in enumerate(batches):
            batch = [examples[i] for i in indices]
            policy_chosen = sequence_logps(
                model, [e.chosen_ids for e in batch], [e.chosen_labels for e in batch],
                pad_id,
            )
            policy_rejected = sequence_logps(
                model, [e.rejected_ids for e in batch],
                [e.rejected_labels for e in batch], pad_id,
            )
            device = policy_chosen.device
            reference_chosen = torch.tensor(
                [references[i][0] for i in indices], device=device
            )
            reference_rejected = torch.tensor(
                [references[i][1] for i in indices], device=device
            )
            loss, parts = dpo_loss(
                policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                beta,
            )
            (loss / grad_accum).backward()

            totals["loss"] += loss.detach().item()
            totals["reward_accuracy"] += parts["reward_accuracy"]
            totals["reward_margin"] += parts["reward_margin"]
            micro += 1

            if micro == grad_accum or index == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                row = {name: value / micro for name, value in totals.items()}
                row.update(step=step, epoch=epoch, lr=scheduler.get_last_lr()[0])
                history.append(row)
                if step == 1 or step % 10 == 0 or step == total_steps:
                    logger.info(
                        "step %d/%d · epoch %d · loss %.4f · reward acc %.2f · "
                        "margin %.3f",
                        step, total_steps, epoch, row["loss"],
                        row["reward_accuracy"], row["reward_margin"],
                    )
                totals = dict.fromkeys(totals, 0.0)
                micro = 0
    return history


def summarise_dpo_history(history: list[dict[str, float]]) -> str:
    """Describe whether DPO learned to prefer the consensus answers.

    Args:
        history: Rows from ``train_dpo_epochs``.

    Returns:
        A one-line summary comparing the first and last quarter of steps.
    """
    if len(history) < 2:
        return f"only {len(history)} optimizer step(s); too few to judge"
    window = max(1, len(history) // 4)

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    first, last = history[:window], history[-window:]
    went_down = mean(last, "loss") < mean(first, "loss")
    return (
        f"loss {'decreased' if went_down else 'DID NOT DECREASE'} over "
        f"{len(history)} steps · loss {mean(first, 'loss'):.4f} -> "
        f"{mean(last, 'loss'):.4f} · reward accuracy "
        f"{mean(first, 'reward_accuracy'):.2f} -> {mean(last, 'reward_accuracy'):.2f}"
    )


def train(smoke: bool = False, split: SplitName = "train") -> Path:
    """Train M5 end to end. The entry point ``main.py train-dpo`` calls.

    Args:
        smoke: Train on the first ``SMOKE_PAIRS`` pairs for one short epoch,
            to prove the loop runs before the full run.
        split: The split whose pairs to train on.

    Returns:
        The saved adapter directory.
    """
    started = time.time()
    t = config.TRAINING
    seed = config.SPLITS.seed
    set_seed(seed)

    pairs = load_pairs(split)
    if smoke:
        pairs = pairs[:SMOKE_PAIRS]
    tokenizer = load_tokenizer()
    examples, dropped = build_pair_examples(pairs, split, tokenizer)
    if not examples:
        raise ValueError("no preference pairs could be tokenised")
    logger.info("built %d pair examples, dropped %d", len(examples), dropped)

    import torch

    torch.manual_seed(seed)
    model = load_student()
    references = reference_logps(model, examples, t.per_device_batch_size,
                                 tokenizer.pad_token_id)
    logger.info("reference log-probs computed for %d pairs", len(references))

    epochs = SMOKE_EPOCHS if smoke else int(round(t.epochs))
    grad_accum = SMOKE_GRAD_ACCUM if smoke else t.gradient_accumulation_steps
    history = train_dpo_epochs(
        model, examples, references,
        epochs=epochs,
        beta=config.DPO.beta,
        learning_rate=t.learning_rate,
        batch_size=t.per_device_batch_size,
        grad_accum=grad_accum,
        seed=seed,
        pad_id=tokenizer.pad_token_id,
    )

    directory = run_dir("m5-maca-dpo-smoke" if smoke else "m5-maca-dpo")
    adapter = save_run(model, history, directory, {
        "arm": "m5",
        "objective": "MV-DPO",
        "smoke": smoke,
        "beta": config.DPO.beta,
        "epochs": epochs,
        "grad_accum": grad_accum,
        "n_pairs": len(pairs),
        "n_examples": len(examples),
        "n_dropped": dropped,
        "seed": seed,
        "training_context": "baseline prompt only (no debate context)",
        "seconds": round(time.time() - started, 1),
    })
    logger.info(summarise_dpo_history(history))
    logger.info("trained in %s -> %s", format_duration(time.time() - started), adapter)
    return adapter
