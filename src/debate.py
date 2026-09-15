"""M5 data: a two-round persona debate, then MACA preference pairs (ADR-0003).

Follows MACA's MV-DPO recipe (arXiv:2509.15172) with three documented
adaptations:

1. **Round 1** is each persona scoring the item on its own. These are exactly
   the teacher scores M4 was trained from (``teacher__{split}.json``), reused
   rather than regenerated, so M4 and M5 start from the same persona answers.
2. **Round 2**: each persona sees its own round-1 answer and the other two
   personas' answers, and may revise. MACA uses identical model copies; this
   project uses three different personas, because the teacher's diversity is
   what the project studies (ADR-0002, ADR-0008).
3. **Consensus** is the majority of the round-2 *written verdicts*
   (``BACKDOORED: yes/no``), MACA's vote on a discrete final answer.
   Round-2 responses agreeing with the majority form G+, dissenters G-, and
   every G+ x G- combination becomes a (chosen, rejected) pair.

No ground-truth label is used to build pairs. Training later conditions on the
baseline prompt alone, not the debate context, to match how a monitor is
deployed -- the paper's no-context setting.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src import config
from src.data import Item, SplitName, load_split
from src.scoring import (
    PromptTooLongError,
    get_backend,
    parse_text_score,
    read_verdict_probability,
    render_prompt,
)
from src.utils import (
    get_logger,
    read_json_records,
    read_jsonl,
    write_json_records,
    write_jsonl,
)

logger = get_logger(__name__)

ROUND2_TEMPLATE = config.PROMPTS_DIR / "debate" / "round2.txt"
VERDICT_PATTERN = re.compile(r"BACKDOORED:\s*(yes|no)\b", re.IGNORECASE)
REASONING_PATTERN = re.compile(r"REASONING:\s*(.*?)\s*BACKDOORED:", re.DOTALL)


# --------------------------------------------------------------------------
# Parsing a response
# --------------------------------------------------------------------------


def parse_verdict(text: str) -> str | None:
    """Read the written verdict, the answer MACA votes on.

    Args:
        text: A monitor response.

    Returns:
        ``"yes"`` or ``"no"``, or ``None`` if the format was not followed.
    """
    match = VERDICT_PATTERN.search(text or "")
    return match.group(1).lower() if match else None


def canonical_response(text: str) -> str | None:
    """Rewrite a response into the exact output block, or reject it.

    Chosen and rejected answers are normalised the same way, so DPO cannot
    learn to prefer one answer's whitespace or trailing text over another's --
    only its content.

    Args:
        text: A monitor response.

    Returns:
        ``REASONING / BACKDOORED / SUSPICION`` on three lines, or ``None`` if
        any part is missing.
    """
    reasoning = REASONING_PATTERN.search(text or "")
    verdict = parse_verdict(text)
    score = parse_text_score(text or "")
    if reasoning is None or verdict is None or score is None:
        return None
    sentence = " ".join(reasoning.group(1).split())
    if not sentence:
        return None
    return f"REASONING: {sentence}\nBACKDOORED: {verdict}\nSUSPICION: {score}"


# --------------------------------------------------------------------------
# Round 1 and round 2
# --------------------------------------------------------------------------


def load_round_one(
    split: SplitName = "train", path: Path | None = None
) -> dict[str, dict[str, str]]:
    """Load each persona's independent answer, reused from M4's teacher run.

    Args:
        split: Which split's teacher answers to read.
        path: Override for the teacher file.

    Returns:
        ``{item_id: {persona_name: response}}``, only for items where all three
        personas produced a well-formed answer.

    Raises:
        FileNotFoundError: If teacher scores have not been generated.
    """
    path = path or config.GENERATIONS_DIR / f"teacher__{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing; run `python main.py teacher-scores --split {split}`"
        )
    answers: dict[str, dict[str, str]] = {}
    for row in read_json_records(path):
        item_id = row["item_id"].split("::", 1)[0]
        if canonical_response(row.get("response", "")) is not None:
            answers.setdefault(item_id, {})[row["prompt_name"]] = row["response"]

    personas = {p.stem for p in config.PERSONA_PROMPTS}
    return {i: a for i, a in answers.items() if set(a) == personas}


def round_two_messages(
    item: Item, persona_path: Path, round_one: dict[str, str]
) -> list[dict[str, str]]:
    """Build one persona's round-2 conversation.

    The persona sees its original task, its own round-1 answer as the
    assistant's turn, and then the other two personas' answers.

    Args:
        item: The item under review.
        persona_path: This persona's prompt template.
        round_one: Every persona's round-1 response, by persona name.

    Returns:
        Chat messages ending with the round-2 request.
    """
    own = persona_path.stem
    peers = [name for name in sorted(round_one) if name != own]
    follow_up = (
        ROUND2_TEMPLATE.read_text()
        .replace("{peer_a}", round_one[peers[0]].strip())
        .replace("{peer_b}", round_one[peers[1]].strip())
    )
    return [
        {"role": "user", "content": render_prompt(persona_path, item)},
        {"role": "assistant", "content": round_one[own].strip()},
        {"role": "user", "content": follow_up},
    ]


@dataclass
class DebateRecord:
    """One persona's round-2 answer to one item.

    Attributes:
        item_id: ``<item>::<persona>::round2``.
        problem_id: Parent problem.
        label: Ground truth, kept for diagnostics only; pairs never read it.
        persona: Which persona answered.
        verdict: The written verdict after revision, or ``None``.
        round_one_verdict: The same persona's verdict before seeing peers.
        logprob_score: P(yes) at the verdict token, for diagnostics.
        text_score: The written 0-10 score.
        finish_reason: Why generation stopped.
        response: The full round-2 text.
        backend: Where it was generated.
        model: Which model generated it.
    """

    item_id: str
    problem_id: str
    label: int
    persona: str
    verdict: str | None
    round_one_verdict: str | None
    logprob_score: float | None
    text_score: int | None
    finish_reason: str
    response: str
    backend: str
    model: str


def run_debate(
    split: SplitName = "train",
    resume: bool = True,
    backend_name: str = "vllm",
    limit: int | None = None,
) -> Path:
    """Run debate round 2 for every item. The entry point ``main.py debate`` calls.

    Resumable: items already in the output file are skipped, and all three
    personas' answers for an item are written together, so an interruption
    never leaves an item half-debated.

    Args:
        split: Which split to debate.
        resume: Skip items already in the output file.
        backend_name: ``"vllm"`` or ``"mlx"``.
        limit: Debate only the first N eligible items (smoke tests).

    Returns:
        The transcript file.
    """
    round_one = load_round_one(split)
    items = [item for item in load_split(split) if item.item_id in round_one]
    if limit is not None:
        items = items[:limit]

    suffix = f"__limit{limit}" if limit is not None else ""
    out_path = config.GENERATIONS_DIR / f"debate__{split}{suffix}.json"
    records = read_json_records(out_path) if resume else []
    done = {r["item_id"].split("::", 1)[0] for r in records}
    pending = [item for item in items if item.item_id not in done]
    logger.info(
        "debate round 2: %d items to debate (%d already done), 3 personas each",
        len(pending), len(items) - len(pending),
    )

    backend = get_backend(backend_name)
    sampling = config.SAMPLING

    def debate_one(item: Item) -> list[dict[str, Any]]:
        rows = []
        for persona_path in config.PERSONA_PROMPTS:
            messages = round_two_messages(item, persona_path, round_one[item.item_id])
            try:
                generation = backend.generate(messages, 1, sampling)[0]
                text, finish = generation.text, generation.finish_reason
                logprob = read_verdict_probability(generation)
            except PromptTooLongError:
                text, finish, logprob = "", "prompt_too_long", None
            rows.append(asdict(DebateRecord(
                item_id=f"{item.item_id}::{persona_path.stem}::round2",
                problem_id=item.problem_id,
                label=item.label,
                persona=persona_path.stem,
                verdict=parse_verdict(text),
                round_one_verdict=parse_verdict(
                    round_one[item.item_id][persona_path.stem]
                ),
                logprob_score=logprob,
                text_score=parse_text_score(text),
                finish_reason=finish,
                response=text,
                backend=getattr(backend, "name", "unknown"),
                model=getattr(backend, "model_id", "unknown"),
            )))
        return rows

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=backend.max_concurrency) as pool:
        for count, rows in enumerate(pool.map(debate_one, pending), start=1):
            records.extend(rows)
            if count % 25 == 0 or count == len(pending):
                write_json_records(out_path, records)
            if count % 100 == 0:
                elapsed = time.perf_counter() - started
                logger.info("  %d/%d items · %.1f s/item", count, len(pending),
                            elapsed / count)
    write_json_records(out_path, records)
    logger.info("wrote %d round-2 answers -> %s", len(records), out_path)
    return out_path


# --------------------------------------------------------------------------
# Consensus and preference pairs
# --------------------------------------------------------------------------


def consensus_pairs(
    answers: Sequence[dict[str, Any]], max_pairs: int
) -> tuple[str | None, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Split one item's round-2 answers by majority verdict and pair them.

    Args:
        answers: Round-2 records for one item.
        max_pairs: Cap on pairs contributed by one item.

    Returns:
        ``(majority_verdict, [(chosen, rejected), ...])``. The majority is
        ``None`` when fewer than two answers are readable or no verdict holds a
        strict majority; pairs are empty when every readable answer agrees.
    """
    readable = [
        a for a in answers
        if a.get("verdict") is not None and canonical_response(a["response"])
    ]
    if len(readable) < 2:
        return None, []
    counts = Counter(a["verdict"] for a in readable)
    verdict, top = counts.most_common(1)[0]
    if top * 2 <= len(readable):
        return None, []
    agree = [a for a in readable if a["verdict"] == verdict]
    dissent = [a for a in readable if a["verdict"] != verdict]
    pairs = list(itertools.product(agree, dissent))[:max_pairs]
    return verdict, pairs


def build_preference_pairs(
    split: SplitName = "train", limit: int | None = None
) -> Path:
    """Turn debate transcripts into (chosen, rejected) preference pairs.

    The entry point ``main.py build-pairs`` calls. Writes the pairs and a
    statistics sidecar. The sidecar includes how often the round-2 majority
    matches the true label -- a diagnostic of consensus quality, computed here
    for the report and never written into a pair.

    Args:
        split: Which split's transcripts to read.
        limit: Match a ``debate --limit`` run.

    Returns:
        The pairs file.
    """
    suffix = f"__limit{limit}" if limit is not None else ""
    transcripts = config.GENERATIONS_DIR / f"debate__{split}{suffix}.json"
    records = read_json_records(transcripts)
    by_item: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_item.setdefault(record["item_id"].split("::", 1)[0], []).append(record)

    pairs, stats = [], Counter()
    majority_correct = majority_total = 0
    for item_id, answers in sorted(by_item.items()):
        stats["items"] += 1
        round_one = [a["round_one_verdict"] for a in answers]
        round_two = [a["verdict"] for a in answers]
        stats["unanimous_round1"] += len(set(round_one)) == 1 and None not in round_one
        stats["unanimous_round2"] += len(set(round_two)) == 1 and None not in round_two
        stats["changed_verdicts"] += sum(
            a["verdict"] != a["round_one_verdict"] for a in answers
        )

        verdict, item_pairs = consensus_pairs(answers, config.DPO.max_pairs_per_item)
        if verdict is None:
            stats["no_majority"] += 1
            continue
        majority_total += 1
        majority_correct += (verdict == "yes") == (answers[0]["label"] == 1)
        if item_pairs:
            stats["items_with_pairs"] += 1
        for chosen, rejected in item_pairs:
            pairs.append({
                "item_id": item_id,
                "problem_id": chosen["problem_id"],
                "majority": verdict,
                "chosen": canonical_response(chosen["response"]),
                "rejected": canonical_response(rejected["response"]),
                "chosen_persona": chosen["persona"],
                "rejected_persona": rejected["persona"],
            })

    out_path = config.TRAINING_DIR / f"dpo_pairs__{split}{suffix}.jsonl"
    write_jsonl(out_path, pairs)

    items = max(1, stats["items"])
    summary = {
        "items_debated": stats["items"],
        "pairs": len(pairs),
        "items_with_pairs": stats["items_with_pairs"],
        "unanimous_round1": round(stats["unanimous_round1"] / items, 4),
        "unanimous_round2": round(stats["unanimous_round2"] / items, 4),
        "verdicts_changed_in_round2": stats["changed_verdicts"],
        "items_without_majority": stats["no_majority"],
        "diagnostic_majority_matches_label": round(
            majority_correct / max(1, majority_total), 4
        ),
    }
    out_path.with_suffix(".stats.json").write_text(json.dumps(summary, indent=2))
    logger.info("pairs: %s", summary)
    return out_path


def load_pairs(
    split: SplitName = "train", limit: int | None = None
) -> list[dict[str, Any]]:
    """Read preference pairs back.

    Args:
        split: Which split's pairs to read.
        limit: Match a ``build-pairs --limit`` run.

    Returns:
        Pair records.

    Raises:
        FileNotFoundError: If pairs have not been built.
    """
    suffix = f"__limit{limit}" if limit is not None else ""
    path = config.TRAINING_DIR / f"dpo_pairs__{split}{suffix}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python main.py build-pairs`")
    return list(read_jsonl(path))
