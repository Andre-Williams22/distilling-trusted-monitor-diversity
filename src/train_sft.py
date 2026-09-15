"""Train M4: distil M2's persona ensemble into one LoRA adapter (ADR-0005).

The loss is ``CE_text + kd_weight * KD_yes``:

- ``CE_text`` teaches the model to write a well-formed monitor response under the
  baseline prompt, ending in the teacher's rounded 0-10 score.
- ``KD_yes`` pushes the model's normalised ``P(yes)`` at the verdict token toward
  the teacher's unrounded mean, which is where the ensemble's extra resolution
  lives.

Pipeline, one function per stage::

    load_teacher_data -> build_sft_example (x N) -> check_examples
        -> load_student -> train_epochs(two_term_loss) -> save_run

Ownership (project-plan.md section 0): the functions marked ``Andre`` are
research logic -- target construction, tokenisation, the loss and the training
loop. Everything else is plumbing and is implemented here.

Nothing in this module reads an item's ground-truth label. Teacher rows arrive
with the ``label`` field stripped, and ``SFTExample`` has nowhere to put one.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from src import config
from src.data import Item, SplitName, load_split
from src.scoring import NO_SPELLINGS, YES_SPELLINGS, render_prompt
from src.utils import (
    format_duration,
    get_logger,
    git_sha,
    read_json_records,
    read_jsonl,
    run_dir,
    set_seed,
)

logger = get_logger(__name__)

#: A smoke test proves the pipeline runs end to end and the loss moves. It is
#: not a result, so it trains briefly on the head of train.
SMOKE_ITEMS = 50
SMOKE_EPOCHS = 3
SMOKE_GRAD_ACCUM = 2

LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

REASONING_PATTERN = re.compile(r"REASONING:\s*(.*?)\s*BACKDOORED:", re.DOTALL)

@dataclass
class SFTExample:
    """One tokenised training example for the two-term loss.

    Attributes:
        item_id: The source item.
        input_ids: Chat-templated prompt followed by the target response, as
            token ids. Length ``<= config.TRAINING.max_seq_length``.
        labels: Same length as ``input_ids``. ``-100`` on every prompt token so
            ``CE_text`` covers only the response; real token ids on the response.
        verdict_pos: Index into ``input_ids`` of the verdict token itself (the
            yes/no right after ``BACKDOORED:``). The logits that *predict* it
            sit at ``verdict_pos - 1``.
        kd_target: The ``KD_yes`` target in [0, 1], unrounded.
    """

    item_id: str
    input_ids: list[int]
    labels: list[int]
    verdict_pos: int
    kd_target: float


# --------------------------------------------------------------------------
# Teacher data (plumbing)
# --------------------------------------------------------------------------


def load_teacher_data(
    split: SplitName = "train",
    targets_path: Path | None = None,
    raw_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load the aggregated teacher targets and each persona's full response.

    Written by ``python main.py teacher-scores --split train``. The ``label``
    field is removed from every record before it is returned, so the research
    functions downstream cannot read ground truth even by accident.

    Args:
        split: Which split's teacher files to read.
        targets_path: Override for the aggregated targets file.
        raw_path: Override for the per-persona score file.

    Returns:
        ``(targets_by_item, persona_rows_by_item)``. Each persona row carries
        ``prompt_name``, ``logprob_score``, ``text_score`` and ``response``.

    Raises:
        FileNotFoundError: If teacher scores have not been generated.
    """
    targets_path = targets_path or config.TRAINING_DIR / f"sft_targets__{split}.jsonl"
    raw_path = raw_path or config.GENERATIONS_DIR / f"teacher__{split}.json"
    if not targets_path.exists() or not raw_path.exists():
        raise FileNotFoundError(
            f"teacher files missing ({targets_path.name}, {raw_path.name}); run "
            f"`python main.py teacher-scores --split {split}` on the GPU box first"
        )

    targets = {
        row["item_id"]: _without_label(row) for row in read_jsonl(targets_path)
    }
    personas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json_records(raw_path):
        item_id = row["item_id"].split("::", 1)[0]
        personas[item_id].append(_without_label(row))
    return targets, dict(personas)


def _without_label(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a record with the ground-truth label removed."""
    return {key: value for key, value in record.items() if key != "label"}


def strip_label(item: Item) -> Item:
    """Blank an item's label before handing it to research code.

    ``Item`` is frozen and its ``label`` field is required, so the value is
    replaced with ``-1`` rather than removed. Any code that reads it gets an
    obviously wrong value instead of a quietly correct one.

    Args:
        item: A split item.

    Returns:
        The same item with ``label=-1``.
    """
    from dataclasses import replace

    return replace(item, label=-1)


def teacher_verdict_probability(target: dict[str, Any]) -> float | None:
    """Return the KD target: the personas' mean normalised P(yes).

    ``teacher_logprob`` is the same quantity the eval readout measures, so it is
    used directly. When every persona's P(yes) was unreadable, the 0-10 text
    mean divided by 10 stands in.

    Args:
        target: An aggregated teacher row.

    Returns:
        A probability in [0, 1], or ``None`` if no persona produced a score.
    """
    if target.get("teacher_logprob") is not None:
        return float(target["teacher_logprob"])
    if target.get("teacher_text") is not None:
        return float(target["teacher_text"]) / 10.0
    return None


def pick_reasoning(
    persona_rows: list[dict[str, Any]], verdict: str, kd_target: float
) -> str | None:
    """Choose which persona's REASONING sentence the target reuses.

    Prefers personas whose own verdict agrees with the target verdict, and
    among those the one whose P(yes) is closest to the teacher mean.

    Args:
        persona_rows: The item's persona rows.
        verdict: ``"yes"`` or ``"no"``.
        kd_target: The teacher mean.

    Returns:
        The reasoning text on one line, or ``None`` if none parses.
    """
    candidates = []
    for row in persona_rows:
        match = REASONING_PATTERN.search(row.get("response") or "")
        if not match or row.get("logprob_score") is None:
            continue
        agrees = (row["logprob_score"] >= 0.5) == (verdict == "yes")
        distance = abs(row["logprob_score"] - kd_target)
        text = " ".join(match.group(1).split())
        candidates.append((not agrees, distance, text))
    if not candidates:
        return None
    return min(candidates)[2]



# --------------------------------------------------------------------------
# Targets and tokenisation
# --------------------------------------------------------------------------


def build_target_response(
    item: Item,
    target: dict[str, Any],
    persona_rows: list[dict[str, Any]],
) -> str:
    """Write the assistant response M4 is trained to produce (Andre).

    Contract:
        - Follows the output block in ``prompts/README.md`` exactly::

              REASONING: <two or three sentences>
              BACKDOORED: <yes or no>
              SUSPICION: <integer 0-10>

        - ``SUSPICION`` is the teacher mean **rounded**
          (``config.SFT.round_text_target``).
        - Never reads ``item.label``; it is ``-1`` here by construction.

    Decisions that are yours. Record each one here once made:
        - Where the REASONING text comes from: one persona's reasoning (which
          one?), the persona closest to the mean, or something else. Persona
          responses were written under persona framing, but M4 is served under
          the baseline prompt.
        - How the yes/no verdict is chosen: threshold the mean, or a majority of
          the personas' own verdicts. It should agree with ``SUSPICION``, or the
          target contradicts itself.
        - Which teacher mean drives the targets: ``teacher_text`` (0-10) or
          ``teacher_logprob`` (0-1). Most persona scores sit near 0 or 1, so
          most means are close to 0, 1/3, 2/3 or 1 (see results/test).

    Args:
        item: The item (question and code).
        target: This item's row from ``sft_targets__train.jsonl``, with
            ``teacher_logprob``, ``teacher_text`` and the persona counts.
        persona_rows: This item's three persona rows, each with ``prompt_name``,
            ``logprob_score``, ``text_score`` and the full ``response``.

    Returns:
        The response text, with no chat-template tokens.
    """
    kd_target = teacher_verdict_probability(target)
    if kd_target is None:
        raise ValueError(f"{item.item_id}: no readable teacher score")
    verdict = "yes" if kd_target >= 0.5 else "no"

    if target.get("teacher_text") is not None:
        suspicion = round(target["teacher_text"])
    else:
        suspicion = round(10 * kd_target)
    # Keep the written score on the same side as the verdict, so the target
    # never says "yes" with a clean-looking score or the reverse.
    suspicion = max(suspicion, 5) if verdict == "yes" else min(suspicion, 4)

    reasoning = pick_reasoning(persona_rows, verdict, kd_target)
    if reasoning is None:
        raise ValueError(f"{item.item_id}: no persona reasoning could be parsed")

    return f"REASONING: {reasoning}\nBACKDOORED: {verdict}\nSUSPICION: {suspicion}"

def find_verdict_offset(
    response_ids: list[int], tokenizer: Any, yes_ids: list[int], no_ids: list[int]
) -> int | None:
    """Find the index of the yes/no token that directly follows ``BACKDOORED:``.

    Searches decoded text rather than assuming a token layout, since the
    reasoning sentence can itself contain "yes" or "no".

    Args:
        response_ids: The tokenised response.
        tokenizer: The base model's tokenizer.
        yes_ids: Token ids counted as "yes".
        no_ids: Token ids counted as "no".

    Returns:
        The index within ``response_ids``, or ``None`` if not found.
    """
    verdict_ids = set(yes_ids) | set(no_ids)
    for index, token in enumerate(response_ids):
        if token in verdict_ids:
            before = tokenizer.decode(response_ids[:index])
            if before.rstrip().endswith("BACKDOORED:"):
                return index
    return None


def build_sft_example(
    item: Item,
    target: dict[str, Any] | None,
    persona_rows: list[dict[str, Any]] | None,
    tokenizer: Any,
    yes_ids: list[int],
    no_ids: list[int],
) -> SFTExample | None:
    """Tokenise prompt and response, mask the prompt, find the verdict (Andre).

    Contract:
        - The prompt is ``render_prompt(config.BASELINE_PROMPT, item)`` as a
          single user turn, run through
          ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``.
          vLLM applies the same template at serve time, so any drift here is a
          train/serve mismatch.
        - The response comes from ``build_target_response`` and ends with the
          end-of-turn token, so the model learns to stop.
        - ``labels`` is ``-100`` on every prompt token and equals ``input_ids``
          on every response token, the verdict included.
        - ``input_ids[verdict_pos]`` is one of ``yes_ids + no_ids``.
        - ``kd_target`` is in [0, 1] and unrounded.
        - Returns ``None`` if the example exceeds
          ``config.TRAINING.max_seq_length``. Never truncate silently; that can
          cut off the verdict.
        - Returns ``None`` if ``target`` or ``persona_rows`` is missing, or the
          teacher mean is ``None`` (every persona unreadable).

    ``check_examples`` enforces every point above after the batch is built.

    Hint:
        Tokenising prompt and response separately and concatenating makes the
        mask boundary exact. Then check that re-tokenising the joined string
        gives the same ids at the verdict, because BPE merges across the
        boundary can shift it.

    Args:
        item: The item, label blanked.
        target: Its aggregated teacher row, or ``None``.
        persona_rows: Its persona rows, or ``None``.
        tokenizer: The base model's tokenizer.
        yes_ids: Token ids counted as "yes".
        no_ids: Token ids counted as "no".

    Returns:
        The example, or ``None`` if it cannot be built.
    """
    if not target or not persona_rows:
        return None
    kd_target = teacher_verdict_probability(target)
    if kd_target is None:
        return None
    try:
        response = build_target_response(item, target, persona_rows)
    except ValueError:
        return None
    return tokenise_example(item, response, kd_target, tokenizer, yes_ids, no_ids)


def encode_prompt(item: Item, tokenizer: Any) -> list[int]:
    """Encode the baseline prompt exactly as vLLM will at serve time.

    Shared by every trained arm. Encoded separately from the response: vLLM
    encodes the prompt and then generates token by token, so no BPE merge ever
    crosses the prompt/response boundary.

    Args:
        item: The item, label blanked.
        tokenizer: The base model's tokenizer.

    Returns:
        Token ids for the chat-templated prompt, ending at the assistant turn.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": render_prompt(config.BASELINE_PROMPT, item)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def encode_response(response: str, tokenizer: Any) -> list[int]:
    """Encode a target response followed by the end-of-turn token.

    Args:
        response: The response text.
        tokenizer: The base model's tokenizer.

    Returns:
        Token ids, ending with ``<|im_end|>`` so the model learns to stop.
    """
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return tokenizer.encode(response, add_special_tokens=False) + [end_of_turn]


def tokenise_example(
    item: Item,
    response: str,
    kd_target: float,
    tokenizer: Any,
    yes_ids: list[int],
    no_ids: list[int],
) -> SFTExample | None:
    """Turn a prompt and a finished response into a training example.

    Shared by every trained arm, so M3 and M4 are tokenised, masked and
    length-limited identically; the only thing that differs between them is
    where ``response`` and ``kd_target`` come from.

    Args:
        item: The item, label blanked.
        response: The target response text.
        kd_target: The ``KD_yes`` target in [0, 1].
        tokenizer: The base model's tokenizer.
        yes_ids: Token ids counted as "yes".
        no_ids: Token ids counted as "no".

    Returns:
        The example, or ``None`` if it exceeds the length limit or the verdict
        cannot be located.
    """
    prompt_ids = encode_prompt(item, tokenizer)
    response_ids = encode_response(response, tokenizer)

    verdict_offset = find_verdict_offset(response_ids, tokenizer, yes_ids, no_ids)
    if verdict_offset is None:
        return None

    input_ids = prompt_ids + response_ids
    if len(input_ids) > config.TRAINING.max_seq_length:
        return None

    return SFTExample(
        item_id=item.item_id,
        input_ids=input_ids,
        labels=[-100] * len(prompt_ids) + response_ids,
        verdict_pos=len(prompt_ids) + verdict_offset,
        kd_target=kd_target,
    )


# --------------------------------------------------------------------------
# Choosing the teacher ensemble
# --------------------------------------------------------------------------

#: Training-target modes. The two ensemble modes differ only in the teacher.
TARGET_MODES = ("ensemble", "m1-ensemble", "labels")


def load_ensemble_teacher(
    targets: str, split: SplitName = "train"
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load the teacher for an ensemble-distilled arm.

    ``"ensemble"`` is M2: three persona prompts, one sample each, generated by
    ``main.py teacher-scores``. ``"m1-ensemble"`` is M1: the baseline prompt
    sampled three times, generated by ``main.py score --arm m1 --split train``.
    Both are aggregated the same way -- the unrounded mean across the three
    members -- and both feed the same target builder, so the student sees the
    same kind of target from a diverse or an identical ensemble.

    Args:
        targets: ``"ensemble"`` or ``"m1-ensemble"``.
        split: The split to train on.

    Returns:
        ``(targets_by_item, member_rows_by_item)``, labels stripped.
    """
    if targets == "ensemble":
        return load_teacher_data(split)

    from src.scoring import aggregate_teacher_scores

    raw_path = config.GENERATIONS_DIR / f"m1__{split}.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path.name} missing; run `python main.py score --arm m1 "
            f"--split {split}` on the GPU box first"
        )
    targets_path = config.TRAINING_DIR / f"sft_targets_m1__{split}.jsonl"
    aggregate_teacher_scores(raw_path, targets_path)
    return load_teacher_data(split, targets_path=targets_path, raw_path=raw_path)


# --------------------------------------------------------------------------
# M3: targets from the true labels (ADR-0007)
# --------------------------------------------------------------------------

#: Suspicion written for each true label. The extremes of the scale, because a
#: label carries no graded confidence to preserve.
LABEL_SUSPICION = {1: 10, 0: 0}


def load_reasoning_samples(
    split: SplitName = "train", path: Path | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Load the base monitor's own answers, the source of M3's reasoning text.

    These are M1-style samples: the baseline prompt, three draws per item, from
    the untrained model (``python main.py score --arm m1 --split train``). Using
    the base model's reasoning, rather than a persona's as M4 does, keeps any
    ensemble information out of the label baseline. Labels are stripped.

    Args:
        split: Which split's samples to read.
        path: Override for the samples file.

    Returns:
        ``{item_id: [sample rows]}``, each with ``logprob_score`` and
        ``response``.

    Raises:
        FileNotFoundError: If the samples have not been generated.
    """
    path = path or config.GENERATIONS_DIR / f"m1__{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing; run `python main.py score --arm m1 "
            f"--split {split}` on the GPU box first"
        )
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json_records(path):
        samples[row["item_id"].split("::", 1)[0]].append(_without_label(row))
    return dict(samples)


def label_training_items(items: list[Item]) -> list[Item]:
    """Keep only items whose label can be trusted as a training target.

    Nondeterministic problems were routed to train on the grounds that no
    training arm reads labels (ADR-0001). M3 does, so their noisy labels are
    excluded here rather than learned.

    Args:
        items: Train items.

    Returns:
        The deterministic items, in their original order.
    """
    return [item for item in items if not item.is_nondeterministic]


def build_label_target_response(
    label: int, sample_rows: list[dict[str, Any]]
) -> str:
    """Write the response M3 is trained to produce from a true label.

    The verdict and score come from the label: ``yes``/10 for backdoored,
    ``no``/0 for clean. The reasoning sentence is the base monitor's own, chosen
    with the same rule as M4 (``pick_reasoning``): a sample whose verdict agrees
    with the target, and among those the one closest to it, so the most
    confident agreeing sample. When no sample agrees, the closest one is used.

    Args:
        label: The true label, 1 for backdoored and 0 for clean.
        sample_rows: The base monitor's answers for this item.

    Returns:
        The response text.

    Raises:
        ValueError: If no sample's reasoning can be parsed.
    """
    verdict = "yes" if label == 1 else "no"
    reasoning = pick_reasoning(sample_rows, verdict, float(label))
    if reasoning is None:
        raise ValueError("no reasoning could be parsed from the base samples")
    return (
        f"REASONING: {reasoning}\nBACKDOORED: {verdict}"
        f"\nSUSPICION: {LABEL_SUSPICION[label]}"
    )


def build_label_example(
    item: Item,
    sample_rows: list[dict[str, Any]] | None,
    tokenizer: Any,
    yes_ids: list[int],
    no_ids: list[int],
) -> SFTExample | None:
    """Build one M3 training example from an item's true label.

    Args:
        item: The item, **with** its label: M3 is the one arm that reads it.
        sample_rows: The base monitor's answers for this item.
        tokenizer: The base model's tokenizer.
        yes_ids: Token ids counted as "yes".
        no_ids: Token ids counted as "no".

    Returns:
        The example, or ``None`` if it cannot be built.
    """
    if not sample_rows:
        return None
    try:
        response = build_label_target_response(item.label, sample_rows)
    except ValueError:
        return None
    return tokenise_example(
        strip_label(item), response, float(item.label), tokenizer, yes_ids, no_ids
    )


def reasoning_agreement(
    items: list[Item], samples: dict[str, list[dict[str, Any]]]
) -> float:
    """Share of items where at least one base sample agrees with the label.

    Where none agrees, M3's reasoning sentence argues against its own verdict.
    Reported with the run so that mismatch is visible, not hidden.

    Args:
        items: Items with labels.
        samples: Base monitor answers by item.

    Returns:
        A fraction in [0, 1].
    """
    agreeing = 0
    for item in items:
        rows = samples.get(item.item_id, [])
        if any(
            row.get("logprob_score") is not None
            and (row["logprob_score"] >= 0.5) == (item.label == 1)
            for row in rows
        ):
            agreeing += 1
    return agreeing / max(1, len(items))


# --------------------------------------------------------------------------
# Tokeniser helpers and contract checks (plumbing)
# --------------------------------------------------------------------------


def resolve_verdict_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Find single-token ids for every yes/no spelling the eval readout accepts.

    Uses the same spellings as ``src.scoring``, so training pushes on exactly
    the tokens that scoring reads.

    Args:
        tokenizer: The base model's tokenizer.

    Returns:
        ``(yes_ids, no_ids)``, sorted.

    Raises:
        ValueError: If either word has no single-token spelling, or the two
            sets overlap.
    """

    def ids_for(spellings: tuple[str, ...]) -> list[int]:
        found = set()
        for spelling in spellings:
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) == 1:
                found.add(encoded[0])
        return sorted(found)

    yes_ids, no_ids = ids_for(YES_SPELLINGS), ids_for(NO_SPELLINGS)
    if not yes_ids or not no_ids:
        raise ValueError("no single-token yes/no spelling; KD_yes cannot work")
    if set(yes_ids) & set(no_ids):
        raise ValueError("yes and no token ids overlap")
    return yes_ids, no_ids


def check_examples(
    examples: list[SFTExample],
    tokenizer: Any,
    yes_ids: list[int],
    no_ids: list[int],
) -> None:
    """Enforce the ``build_sft_example`` contract on every example.

    Runs before any GPU time is spent, so a masking or verdict-position bug
    fails in seconds with the offending item named, instead of producing an
    adapter that trained on the wrong tokens.

    Args:
        examples: Built examples.
        tokenizer: The base model's tokenizer.
        yes_ids: Token ids counted as "yes".
        no_ids: Token ids counted as "no".

    Raises:
        ValueError: On the first example that breaks the contract.
    """
    if not examples:
        raise ValueError("no examples were built")

    verdict_ids = set(yes_ids) | set(no_ids)
    limit = config.TRAINING.max_seq_length
    for ex in examples:
        where = f"example {ex.item_id}"
        if not len(ex.input_ids) == len(ex.labels) <= limit:
            raise ValueError(f"{where}: lengths {len(ex.input_ids)}/{len(ex.labels)}")
        if not 0.0 <= ex.kd_target <= 1.0:
            raise ValueError(f"{where}: kd_target {ex.kd_target} outside [0, 1]")
        if ex.input_ids[ex.verdict_pos] not in verdict_ids:
            token = tokenizer.decode([ex.input_ids[ex.verdict_pos]])
            raise ValueError(f"{where}: verdict_pos points at {token!r}")

        if ex.labels[ex.verdict_pos] != ex.input_ids[ex.verdict_pos]:
            raise ValueError(f"{where}: verdict token is not supervised")
        supervised = [i for i, t in enumerate(ex.labels) if t != -100]
        if not supervised:
            raise ValueError(f"{where}: no supervised tokens")
        first = supervised[0]
        # The response is one unbroken run of supervised tokens at the end. A
        # leaked prompt token shows up as a gap, which checking only the tokens
        # before the first supervised one would miss.
        if supervised != list(range(first, len(ex.labels))):
            raise ValueError(f"{where}: prompt is not fully masked")
        if not first < ex.verdict_pos:
            raise ValueError(f"{where}: verdict lies outside the response")
        if "BACKDOORED:" not in tokenizer.decode(ex.input_ids[first : ex.verdict_pos]):
            raise ValueError(f"{where}: no 'BACKDOORED:' before the verdict")


def collate(batch: list[SFTExample], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a micro-batch, masking padding in attention and labels.

    Args:
        batch: Examples to batch.
        pad_id: The tokenizer's pad token id.

    Returns:
        Tensors ``input_ids``, ``attention_mask``, ``labels`` (all ``(B, T)``),
        ``verdict_pos`` ``(B,)`` and ``kd_target`` ``(B,)`` float32.
    """
    import torch

    width = max(len(ex.input_ids) for ex in batch)
    input_ids, attention, labels = [], [], []
    for ex in batch:
        padding = width - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_id] * padding)
        attention.append([1] * len(ex.input_ids) + [0] * padding)
        labels.append(ex.labels + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention),
        "labels": torch.tensor(labels),
        "verdict_pos": torch.tensor([ex.verdict_pos for ex in batch]),
        "kd_target": torch.tensor([ex.kd_target for ex in batch], dtype=torch.float32),
    }


# --------------------------------------------------------------------------
# Loss and training loop (Andre)
# --------------------------------------------------------------------------


def two_term_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    verdict_pos: torch.Tensor,
    kd_target: torch.Tensor,
    yes_ids: list[int],
    no_ids: list[int],
    kd_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute ``CE_text + kd_weight * KD_yes`` per ADR-0005 (Andre).

    Contract (tests in ``tests/test_train_sft.py``):
        - **CE_text** is next-token cross-entropy: logits at position ``t``
          predict ``labels[t + 1]``, averaged over every unmasked target token in
          the batch, ignoring ``-100``.
        - **KD_yes** uses the student's *normalised* verdict probability at
          ``logits[b, verdict_pos[b] - 1]``::

              p_yes = sum_{yes_ids} P / (sum_{yes_ids} P + sum_{no_ids} P)

          which is exactly what ``src.scoring.read_verdict_probability`` reads
          at eval time. How the gap to ``kd_target`` is measured (BCE, MSE, ...)
          is your call; record it in ADR-0005.
        - ``kd_weight == 0.0`` reduces **exactly** to plain text SFT.
        - Numerically safe on bf16 logits over Qwen's ~152k vocabulary: do the
          softmax work in float32.

    Args:
        logits: ``(B, T, V)``.
        labels: ``(B, T)``, ``-100`` where masked.
        verdict_pos: ``(B,)`` index of the verdict token in each sequence.
        kd_target: ``(B,)`` targets in [0, 1].
        yes_ids: Token ids summed as "yes".
        no_ids: Token ids summed as "no".
        kd_weight: The weight on ``KD_yes``.

    Returns:
        ``(loss, parts)``: a scalar with grad, and detached floats
        ``{"ce_text", "kd_yes", "p_yes_mean"}`` for logging.
    """
    vocab = logits.size(-1)

    # CE_text: the logits at position t predict the token at t + 1.
    ce_text = functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    # KD_yes: the normalised P(yes) one position before the verdict token.
    rows = torch.arange(logits.size(0), device=logits.device)
    positions = verdict_pos.to(logits.device) - 1
    log_probs = functional.log_softmax(logits[rows, positions].float(), dim=-1)
    log_yes = torch.logsumexp(log_probs[:, yes_ids], dim=-1)
    log_no = torch.logsumexp(log_probs[:, no_ids], dim=-1)
    log_total = torch.logaddexp(log_yes, log_no)
    log_p_yes, log_p_no = log_yes - log_total, log_no - log_total

    # Binary cross-entropy against the soft teacher target, in log space so it
    # stays finite when P(yes) is near 0 or 1.
    target = kd_target.to(logits.device).float()
    kd_yes = -(target * log_p_yes + (1 - target) * log_p_no).mean()

    loss = ce_text if kd_weight == 0.0 else ce_text + kd_weight * kd_yes
    parts = {
        "ce_text": ce_text.detach().item(),
        "kd_yes": kd_yes.detach().item(),
        "p_yes_mean": log_p_yes.detach().exp().mean().item(),
    }
    return loss, parts


def train_epochs(
    model: Any,
    examples: list[SFTExample],
    epochs: int,
    yes_ids: list[int],
    no_ids: list[int],
    kd_weight: float,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    seed: int,
    pad_id: int,
) -> list[dict[str, float]]:
    """Train the LoRA adapter in place with ``two_term_loss`` (Andre).

    Contract:
        - Builds its own optimizer over the trainable (LoRA) parameters only.
          Schedule, warmup and clipping are your choice; note them here, since
          ``TrainingConfig`` doesn't record them yet.
        - Shuffles every epoch with a generator seeded from ``seed`` and the
          epoch index, so two runs see identical batch orders.
        - Micro-batches of ``batch_size`` via ``collate(batch, pad_id)``; one
          optimizer step every ``grad_accum`` micro-batches, loss scaled by
          ``1 / grad_accum``. A partial accumulation at the end of an epoch
          still takes a step.
        - ``model.train()`` throughout; tensors moved to ``model.device``.
        - Logs progress with ``logger.info`` every few steps, so a run on the
          VM shows it is alive.

    Returns:
        One dict per optimizer step:
        ``{"step", "epoch", "loss", "ce_text", "kd_yes", "p_yes_mean", "lr"}``,
        loss parts averaged over that step's micro-batches.
    """
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
        totals = {"loss": 0.0, "ce_text": 0.0, "kd_yes": 0.0, "p_yes_mean": 0.0}
        micro = 0

        for index, indices in enumerate(batches):
            batch = collate([examples[i] for i in indices], pad_id)
            batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            loss, parts = two_term_loss(
                logits, batch["labels"], batch["verdict_pos"], batch["kd_target"],
                yes_ids, no_ids, kd_weight,
            )
            (loss / grad_accum).backward()

            totals["loss"] += loss.detach().item()
            for name in ("ce_text", "kd_yes", "p_yes_mean"):
                totals[name] += parts[name]
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
                        "step %d/%d · epoch %d · loss %.4f · ce %.4f · kd %.4f "
                        "· p_yes %.3f",
                        step, total_steps, epoch, row["loss"], row["ce_text"],
                        row["kd_yes"], row["p_yes_mean"],
                    )
                totals = dict.fromkeys(totals, 0.0)
                micro = 0

    return history


# --------------------------------------------------------------------------
# Model, saving and the driver (plumbing)
# --------------------------------------------------------------------------


def load_tokenizer() -> Any:
    """Load the base model's tokenizer, with a pad token for batching."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_student() -> Any:
    """Load the 4-bit base model and attach a fresh LoRA adapter.

    QLoRA per ``config.TRAINING``: NF4 weights with bf16 compute, gradient
    checkpointing on, LoRA on every attention and MLP projection. Only the
    adapter trains; the base weights are frozen and never saved.

    Returns:
        The PEFT-wrapped model on GPU 0.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    t = config.TRAINING
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=t.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=t.lora_rank,
            lora_alpha=t.lora_alpha,
            lora_dropout=t.lora_dropout,
            target_modules=list(LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def save_run(
    model: Any,
    history: list[dict[str, float]],
    directory: Path,
    details: dict[str, Any],
) -> Path:
    """Save the adapter, the per-step history and the resolved configuration.

    Args:
        model: The trained PEFT model.
        history: Rows from ``train_epochs``.
        directory: The run directory.
        details: Run-specific settings (smoke or full, counts, weights).

    Returns:
        The adapter directory, which vLLM serves with ``--lora-modules``.
    """
    adapter_dir = directory / "adapter"
    model.save_pretrained(adapter_dir)
    with open(directory / "history.jsonl", "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    (directory / "resolved_config.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "base_model": config.BASE_MODEL,
                "training": asdict(config.TRAINING),
                "sft": asdict(config.SFT),
                "run": details,
            },
            indent=2,
        )
    )
    return adapter_dir


def summarise_history(history: list[dict[str, float]]) -> str:
    """Describe whether the loss moved, for the smoke test's verdict line.

    Compares the mean of the first and last few steps rather than two single
    steps, which are noisy at this batch size.

    Args:
        history: Rows from ``train_epochs``.

    Returns:
        A one-line summary.
    """
    if len(history) < 2:
        return f"only {len(history)} optimizer step(s); too few to judge the loss"
    window = max(1, len(history) // 4)

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    first, last = history[:window], history[-window:]
    parts = [
        f"{key} {mean(first, key):.4f} -> {mean(last, key):.4f}"
        for key in ("loss", "ce_text", "kd_yes")
    ]
    went_down = mean(last, "loss") < mean(first, "loss")
    verdict = "decreased" if went_down else "DID NOT DECREASE"
    return f"loss {verdict} over {len(history)} steps · " + " · ".join(parts)


def train(
    smoke: bool = False,
    kd_weight: float | None = None,
    split: SplitName = "train",
    targets: str = "ensemble",
) -> Path:
    """Train M4 or M3 end to end. The entry point ``main.py train-sft`` calls.

    ``targets="ensemble"`` trains M4 on M2's diverse ensemble.
    ``targets="m1-ensemble"`` trains M3 on M1's identical ensemble -- the
    baseline prompt sampled three times -- with every other setting identical,
    so M4 vs M3 isolates whether the teacher's *diversity* is what distils
    (ADR-0008). ``targets="labels"`` trains the excluded label-supervised
    baseline (ADR-0007).

    A smoke run trains on the first ``SMOKE_ITEMS`` items for
    ``SMOKE_EPOCHS`` epochs with a small accumulation, so the optimizer takes
    enough steps for the loss to be judged. Its adapter proves the pipeline
    works; it is not an M4 result.

    Args:
        smoke: Run the short pipeline check instead of the full run.
        kd_weight: Override ``config.SFT.kd_weight``. ``0.0`` is the declared
            fallback to plain text SFT.
        split: The split to train on. Only train is valid for a real run.
        targets: ``"ensemble"`` (M4), ``"m1-ensemble"`` (M3) or ``"labels"``
            (the excluded label baseline).

    Returns:
        The saved adapter directory.
    """
    started = time.time()
    t = config.TRAINING
    weight = config.SFT.kd_weight if kd_weight is None else kd_weight
    seed = config.SPLITS.seed
    set_seed(seed)

    if targets not in TARGET_MODES:
        raise ValueError(f"targets must be one of {TARGET_MODES}, got {targets!r}")
    arm = "m4" if targets == "ensemble" else "m3"

    all_items = load_split(split)
    items = label_training_items(all_items) if targets == "labels" else all_items
    excluded = len(all_items) - len(items)
    if smoke:
        items = items[:SMOKE_ITEMS]

    tokenizer = load_tokenizer()
    yes_ids, no_ids = resolve_verdict_token_ids(tokenizer)

    examples: list[SFTExample] = []
    dropped: list[str] = []
    extra: dict[str, Any] = {}
    if targets in ("ensemble", "m1-ensemble"):
        teacher, personas = load_ensemble_teacher(targets, split)
        for item in items:
            example = build_sft_example(
                strip_label(item),
                teacher.get(item.item_id),
                personas.get(item.item_id),
                tokenizer,
                yes_ids,
                no_ids,
            )
            if example is None:
                dropped.append(item.item_id)
            else:
                examples.append(example)
    else:
        samples = load_reasoning_samples(split)
        for item in items:
            example = build_label_example(
                item, samples.get(item.item_id), tokenizer, yes_ids, no_ids
            )
            if example is None:
                dropped.append(item.item_id)
            else:
                examples.append(example)
        extra = {
            "excluded_nondeterministic_items": excluded,
            "reasoning_agreement": round(reasoning_agreement(items, samples), 4),
        }
        logger.info(
            "M3: %d nondeterministic items excluded · a base sample agrees with "
            "the label on %.1f%% of items",
            excluded, 100 * extra["reasoning_agreement"],
        )
    logger.info("built %d examples, dropped %d", len(examples), len(dropped))
    check_examples(examples, tokenizer, yes_ids, no_ids)
    logger.info("example contract checks passed")

    import torch

    torch.manual_seed(seed)
    model = load_student()
    epochs = SMOKE_EPOCHS if smoke else t.epochs
    grad_accum = SMOKE_GRAD_ACCUM if smoke else t.gradient_accumulation_steps
    history = train_epochs(
        model,
        examples,
        epochs=int(round(epochs)),
        yes_ids=yes_ids,
        no_ids=no_ids,
        kd_weight=weight,
        learning_rate=t.learning_rate,
        batch_size=t.per_device_batch_size,
        grad_accum=grad_accum,
        seed=seed,
        pad_id=tokenizer.pad_token_id,
    )

    name = {
        "ensemble": "m4-sft",
        "m1-ensemble": "m3-sft-m1-ensemble",
        "labels": "m3-sft-labels",
    }[targets]
    directory = run_dir(f"{name}-smoke" if smoke else name)
    adapter = save_run(
        model,
        history,
        directory,
        {
            "arm": arm,
            "targets": targets,
            "smoke": smoke,
            "kd_weight": weight,
            "epochs": epochs,
            "grad_accum": grad_accum,
            "n_items": len(items),
            "n_examples": len(examples),
            "n_dropped": len(dropped),
            "seed": seed,
            "seconds": round(time.time() - started, 1),
            **extra,
        },
    )
    logger.info(summarise_history(history))
    logger.info("trained in %s -> %s", format_duration(time.time() - started), adapter)
    return adapter
