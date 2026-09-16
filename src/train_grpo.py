"""MACA's MV-GRPO against the frozen majority verdict.

**Implemented and unit-tested, but never run, so it has no arm number**
(ADR-0010). Running it needs 4-8 GPU hours of rollouts.

The only online arm. For each item the policy samples a group of ``G`` answers,
each is rewarded 1 when its written verdict matches the debate's majority and 0
otherwise, and advantages are normalised **within the group**, so a prompt where
every sample agrees contributes no gradient at all::

    A_i = (r_i - mean(r)) / (std(r) + eps)

    loss = -mean_t[ min(rho_t * A, clip(rho_t, 1-eps, 1+eps) * A) ]
           + kl_weight * mean_t[ KL(pi || ref) ]

``rho_t`` is the per-token probability ratio against the policy that generated
the sample, and the KL term uses the k3 estimator against the adapter-disabled
reference, as in the GRPO paper.

The reward is **frozen**: the majority verdict comes from the debate already
run, so no debate happens inside the training loop. That would keep it comparable to
M5, M6 and M7, which all learn from the same vote, and it is the only reason a
rollout loop fits the schedule at all.

Nothing here reads a ground-truth label. The reward is agreement with the
ensemble's own consensus, exactly as in MACA.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import config
from src.data import Item, SplitName, load_split
from src.debate import load_consensus_responses, parse_verdict
from src.train_dpo import pad_batch
from src.train_sft import (
    encode_prompt,
    load_student,
    load_tokenizer,
    save_run,
    strip_label,
)
from src.utils import format_duration, get_logger, run_dir, set_seed

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

SMOKE_PROMPTS = 8
SMOKE_EPOCHS = 1
SMOKE_GRAD_ACCUM = 2

#: Guards the group-advantage denominator when rewards are nearly identical.
ADVANTAGE_EPSILON = 1e-4


@dataclass
class GRPOPrompt:
    """One item the policy will be rolled out on.

    Attributes:
        item_id: The source item.
        prompt_ids: The chat-templated baseline prompt.
        majority: The debate's majority verdict, ``"yes"`` or ``"no"``. This is
            the reward signal and the only supervision.
    """

    item_id: str
    prompt_ids: list[int]
    majority: str


def build_grpo_prompts(
    items: list[Item], consensus: dict[str, dict[str, Any]], tokenizer: Any
) -> tuple[list[GRPOPrompt], int]:
    """Encode every item whose debate reached a majority.

    Args:
        items: Training items.
        consensus: Output of ``debate.load_consensus_responses``.
        tokenizer: The base model's tokenizer.

    Returns:
        ``(prompts, dropped)``. A prompt is dropped when it leaves no room for
        a full answer inside ``config.TRAINING.max_seq_length``.
    """
    budget = config.TRAINING.max_seq_length - config.GRPO.max_new_tokens
    prompts: list[GRPOPrompt] = []
    dropped = 0
    for item in items:
        entry = consensus.get(item.item_id)
        if not entry:
            continue
        prompt_ids = encode_prompt(strip_label(item), tokenizer)
        if len(prompt_ids) > budget:
            dropped += 1
            continue
        prompts.append(GRPOPrompt(item.item_id, prompt_ids, entry["majority"]))
    return prompts, dropped


def verdict_reward(text: str, majority: str) -> float:
    """Reward one sampled answer: 1 when its verdict matches the consensus.

    An answer whose verdict cannot be parsed scores 0, so the policy is also
    pushed toward the output contract it is served under.

    Args:
        text: The sampled answer.
        majority: The debate's majority verdict.

    Returns:
        ``1.0`` or ``0.0``.
    """
    return 1.0 if parse_verdict(text) == majority else 0.0


def group_advantages(rewards: list[float]) -> list[float]:
    """Normalise rewards within one prompt's group.

    A group where every sample earned the same reward yields all-zero
    advantages and therefore no gradient -- the intended GRPO behaviour, and
    the reason an easy prompt costs a rollout but teaches nothing.

    Args:
        rewards: One group's rewards.

    Returns:
        Advantages, same length and order.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    spread = math.sqrt(variance)
    if spread < ADVANTAGE_EPSILON:
        return [0.0] * len(rewards)
    return [(r - mean) / spread for r in rewards]


def sample_group(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    group_size: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[int]]:
    """Sample ``group_size`` answers to one prompt from the current policy.

    Args:
        model: The PEFT model, adapter enabled.
        tokenizer: The base model's tokenizer.
        prompt_ids: The prompt's token ids.
        group_size: Samples per prompt.
        temperature: Sampling temperature.
        max_new_tokens: Cap per answer.

    Returns:
        One list of generated token ids per sample, prompt excluded, with any
        trailing padding removed.
    """
    import torch

    was_training = model.training
    model.eval()
    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    if was_training:
        model.train()

    completions = []
    for row in generated:
        answer = row[len(prompt_ids):].tolist()
        while answer and answer[-1] == tokenizer.pad_token_id:
            answer.pop()
        completions.append(answer)
    return completions


def token_logps(
    model: Any, sequences: list[list[int]], labels: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of the answer tokens, and their mask.

    GRPO's ratio and KL are per token, so unlike DPO and KTO this keeps the
    token axis instead of summing over it.

    Args:
        model: The policy (or reference) model.
        sequences: Token ids per sequence.
        labels: Labels per sequence, ``-100`` on the prompt.
        pad_id: Padding token id.

    Returns:
        ``(logps, mask)``, both ``(B, T-1)``. ``mask`` is 1 on answer tokens.
    """
    from torch.nn import functional

    batch = pad_batch(sequences, labels, pad_id)
    batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits
    shifted = logits[:, :-1].float()
    targets = batch["labels"][:, 1:]
    mask = (targets != -100).float()
    safe_targets = targets.masked_fill(targets == -100, 0)
    nll = functional.cross_entropy(
        shifted.transpose(1, 2), safe_targets, reduction="none"
    )
    return -nll * mask, mask


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy-gradient loss with a KL penalty.

    Args:
        policy_logps: ``(B, T)`` per-token log-probs under the current policy.
        old_logps: ``(B, T)`` log-probs under the policy that sampled the answer.
        reference_logps: ``(B, T)`` log-probs under the frozen reference.
        advantages: ``(B,)`` group-normalised advantages.
        mask: ``(B, T)``, 1 on answer tokens.
        clip_epsilon: PPO-style ratio clip.
        kl_weight: Weight on the KL penalty.

    Returns:
        ``(loss, parts)`` with the policy and KL terms and the clipped fraction.
    """
    import torch

    ratio = torch.exp(policy_logps - old_logps)
    expanded = advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    per_token = -torch.min(ratio * expanded, clipped * expanded)

    # k3 estimator: non-negative, lower variance than the plain log-ratio.
    log_ratio = reference_logps - policy_logps
    kl = torch.exp(log_ratio) - log_ratio - 1

    tokens = mask.sum().clamp(min=1.0)
    policy_term = (per_token * mask).sum() / tokens
    kl_term = (kl * mask).sum() / tokens
    loss = policy_term + kl_weight * kl_term

    was_clipped = ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).float()
    parts = {
        "policy_loss": policy_term.detach().item(),
        "kl": kl_term.detach().item(),
        "clipped_fraction": ((was_clipped * mask).sum() / tokens).item(),
    }
    return loss, parts


def rollout(
    model: Any,
    tokenizer: Any,
    prompts: list[GRPOPrompt],
    group_size: int,
    temperature: float,
    max_new_tokens: int,
    pad_id: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Sample a group per prompt, score it, and keep only groups that teach.

    Args:
        model: The PEFT model.
        tokenizer: The base model's tokenizer.
        prompts: Prompts to roll out.
        group_size: Samples per prompt.
        temperature: Sampling temperature.
        max_new_tokens: Cap per answer.
        pad_id: Padding token id.

    Returns:
        ``(samples, stats)``. Each sample carries its sequence, labels,
        advantage and the old per-token log-probs. Groups with zero spread are
        dropped, since their advantages are all zero.
    """
    import torch

    samples: list[dict[str, Any]] = []
    rewarded = total = skipped = 0

    for prompt in prompts:
        completions = sample_group(
            model, tokenizer, prompt.prompt_ids, group_size, temperature,
            max_new_tokens,
        )
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completions]
        rewards = [verdict_reward(text, prompt.majority) for text in texts]
        rewarded += sum(rewards)
        total += len(rewards)

        advantages = group_advantages(rewards)
        if all(advantage == 0.0 for advantage in advantages):
            skipped += 1
            continue

        sequences, labels = [], []
        for completion in completions:
            sequences.append(prompt.prompt_ids + completion)
            labels.append([-100] * len(prompt.prompt_ids) + completion)
        with torch.no_grad():
            old_logps, _ = token_logps(model, sequences, labels, pad_id)
        for index, completion in enumerate(completions):
            if not completion:
                continue
            samples.append({
                "item_id": prompt.item_id,
                "sequence": sequences[index],
                "labels": labels[index],
                "advantage": advantages[index],
                "old_logps": old_logps[index].detach().cpu(),
            })

    stats = {
        "reward_mean": rewarded / max(1, total),
        "groups_skipped": float(skipped),
        "samples": float(len(samples)),
    }
    return samples, stats


def train_grpo_epochs(
    model: Any,
    prompts: list[GRPOPrompt],
    tokenizer: Any,
    epochs: int,
    group_size: int,
    temperature: float,
    max_new_tokens: int,
    clip_epsilon: float,
    kl_weight: float,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    seed: int,
    pad_id: int,
    rollout_prompts: int = 8,
) -> list[dict[str, float]]:
    """Train the adapter with GRPO, alternating rollout and update.

    Prompts are consumed in seeded-shuffled blocks of ``rollout_prompts``. Each
    block is rolled out with the *current* policy, then trained on once, which
    keeps the samples near-on-policy without holding a replay buffer.

    Args:
        model: The PEFT model.
        prompts: Prompts to train on.
        tokenizer: The base model's tokenizer.
        epochs: Passes over the prompt set.
        group_size: Samples per prompt.
        temperature: Rollout temperature.
        max_new_tokens: Cap per answer.
        clip_epsilon: PPO-style ratio clip.
        kl_weight: Weight on the KL penalty.
        learning_rate: Peak learning rate.
        batch_size: Samples per micro-batch.
        grad_accum: Micro-batches per optimiser step.
        seed: Shuffle seed.
        pad_id: Padding token id.
        rollout_prompts: Prompts per rollout block.

    Returns:
        One dict per optimiser step.
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)

    blocks_per_epoch = math.ceil(len(prompts) / rollout_prompts)
    total_steps = max(1, epochs * blocks_per_epoch)
    warmup_steps = max(1, round(0.05 * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(prompts), generator=generator).tolist()

        for start in range(0, len(order), rollout_prompts):
            block = [prompts[i] for i in order[start : start + rollout_prompts]]
            model.eval()
            samples, stats = rollout(
                model, tokenizer, block, group_size, temperature, max_new_tokens,
                pad_id,
            )
            if not samples:
                logger.info(
                    "step %d: every group agreed with itself, nothing to learn from",
                    step + 1,
                )
                continue

            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            record: dict[str, float] = {}
            micro = 0
            batches = [
                samples[i : i + batch_size]
                for i in range(0, len(samples), batch_size)
            ]
            for batch_number, batch in enumerate(batches, start=1):
                sequences = [s["sequence"] for s in batch]
                labels = [s["labels"] for s in batch]
                policy, mask = token_logps(model, sequences, labels, pad_id)
                with torch.no_grad(), model.disable_adapter():
                    reference, _ = token_logps(model, sequences, labels, pad_id)

                width = policy.shape[1]
                old = torch.stack([
                    torch.nn.functional.pad(
                        s["old_logps"], (0, width - s["old_logps"].shape[0])
                    )[:width]
                    for s in batch
                ]).to(policy.device)
                advantages = torch.tensor(
                    [s["advantage"] for s in batch], device=policy.device
                )

                loss, parts = grpo_loss(
                    policy, old, reference, advantages, mask, clip_epsilon, kl_weight,
                )
                (loss / grad_accum).backward()

                totals["loss"] = totals.get("loss", 0.0) + loss.item()
                for name, value in parts.items():
                    totals[name] = totals.get(name, 0.0) + value
                micro += 1

                if micro == grad_accum or batch_number == len(batches):
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro_count, micro = micro, 0
                    record = {
                        name: value / micro_count for name, value in totals.items()
                    }
                    record.update(stats)
                    totals = {}

            scheduler.step()
            record.update(step=step + 1, epoch=epoch, lr=scheduler.get_last_lr()[0])
            history.append(record)
            step += 1
            if step == 1 or step % 5 == 0:
                logger.info(
                    "step %d/%d · epoch %d · loss %.4f · reward %.2f · kl %.4f · "
                    "samples %d · groups skipped %d",
                    step, total_steps, epoch, record["loss"], record["reward_mean"],
                    record["kl"], int(record["samples"]), int(record["groups_skipped"]),
                )

    return history


def summarise_grpo_history(history: list[dict[str, float]]) -> str:
    """One line on whether the policy's reward moved.

    Args:
        history: Output of ``train_grpo_epochs``.

    Returns:
        A sentence for the log and the run metadata.
    """
    if not history:
        return "no optimizer steps ran"
    window = max(1, len(history) // 5)
    first = sum(h["reward_mean"] for h in history[:window]) / window
    last = sum(h["reward_mean"] for h in history[-window:]) / window
    direction = "rose" if last > first else "did NOT rise"
    return (
        f"consensus agreement {direction} over {len(history)} steps · "
        f"reward {first:.3f} -> {last:.3f} · "
        f"kl {history[0]['kl']:.4f} -> {history[-1]['kl']:.4f}"
    )


def train(
    smoke: bool = False,
    split: SplitName = "train",
    text_source: str = "round1",
) -> Path:
    """Train an MV-GRPO adapter. The entry point ``main.py train-grpo`` calls.

    Args:
        smoke: Roll out a handful of prompts to prove the loop runs.
        split: The split to train on. Only train is valid for a real run.
        text_source: Passed to the consensus loader. GRPO uses only the
            majority verdict, not the text, so this changes which items have a
            majority at all, not what is trained on.

    Returns:
        The saved adapter directory.
    """
    started = time.time()
    t = config.TRAINING
    grpo = config.GRPO
    seed = config.SPLITS.seed
    set_seed(seed)

    items = load_split(split)
    consensus = load_consensus_responses(split, text_source=text_source)
    tokenizer = load_tokenizer()
    prompts, dropped = build_grpo_prompts(items, consensus, tokenizer)
    if smoke:
        prompts = prompts[:SMOKE_PROMPTS]
    if not prompts:
        raise ValueError("no prompts with a majority verdict; run the debate first")
    logger.info(
        "built %d prompts (%d dropped for length) · group size %d",
        len(prompts), dropped, grpo.group_size,
    )

    import torch

    torch.manual_seed(seed)
    model = load_student()
    epochs = SMOKE_EPOCHS if smoke else int(round(t.epochs))
    grad_accum = SMOKE_GRAD_ACCUM if smoke else t.gradient_accumulation_steps
    history = train_grpo_epochs(
        model,
        prompts,
        tokenizer,
        epochs=epochs,
        group_size=grpo.group_size,
        temperature=grpo.temperature,
        max_new_tokens=grpo.max_new_tokens,
        clip_epsilon=grpo.clip_epsilon,
        kl_weight=grpo.kl_weight,
        learning_rate=t.learning_rate,
        batch_size=t.per_device_batch_size,
        grad_accum=grad_accum,
        seed=seed,
        pad_id=tokenizer.pad_token_id,
    )

    directory = run_dir("mv-grpo-smoke" if smoke else "mv-grpo")
    adapter = save_run(model, history, directory, {
        "arm": "mv-grpo",
        "objective": "mv-grpo",
        "smoke": smoke,
        "text_source": text_source,
        "group_size": grpo.group_size,
        "temperature": grpo.temperature,
        "kl_weight": grpo.kl_weight,
        "clip_epsilon": grpo.clip_epsilon,
        "epochs": epochs,
        "grad_accum": grad_accum,
        "n_prompts": len(prompts),
        "n_dropped": dropped,
        "seed": seed,
        "seconds": round(time.time() - started, 1),
    })
    logger.info(summarise_grpo_history(history))
    logger.info("trained in %s -> %s", format_duration(time.time() - started), adapter)
    return adapter
