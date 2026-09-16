"""Contract tests for M4's training code.

Two groups:

- **Plumbing** (label stripping, contract checks, history summary) runs
  anywhere.
- The loss contract lives in ``test_train_sft_loss.py``, which needs torch.
"""

from __future__ import annotations

import json

import pytest

from src.train_sft import (
    SFTExample,
    check_examples,
    load_teacher_data,
    strip_label,
    summarise_history,
)


class CharTokenizer:
    """A tiny tokenizer: one token per character, ids are code points.

    Enough to exercise ``check_examples``, which only encodes and decodes.
    """

    def encode(self, text, add_special_tokens=False):
        """Encode each character as its code point."""
        return [ord(c) for c in text]

    def decode(self, ids):
        """Decode code points back to text."""
        return "".join(chr(i) for i in ids)


def make_example(prompt="Q?", response="BACKDOORED: yes", kd_target=0.67):
    """Build a well-formed example with the verdict as the last token."""
    tok = CharTokenizer()
    prompt_ids = tok.encode(prompt)
    response_ids = tok.encode(response)
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return SFTExample("p1__clean", input_ids, labels, len(input_ids) - 1, kd_target)


YES, NO = [ord("s")], [ord("o")]


def test_check_examples_accepts_a_valid_example():
    """A correctly masked example with the verdict supervised passes."""
    check_examples([make_example()], CharTokenizer(), YES, NO)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda ex: setattr(ex, "kd_target", 1.5), "outside"),
        (lambda ex: setattr(ex, "verdict_pos", 0), "verdict_pos points"),
        (lambda ex: ex.labels.__setitem__(0, 81), "not fully masked"),
        (lambda ex: ex.labels.__setitem__(-1, -100), "not supervised"),
    ],
)
def test_check_examples_rejects_contract_breaks(mutate, message):
    """Each break fails loudly, naming the item, before any GPU time is spent."""
    example = make_example()
    mutate(example)
    with pytest.raises(ValueError, match=message):
        check_examples([example], CharTokenizer(), YES, NO)


def test_teacher_data_never_carries_labels(tmp_path):
    """Research code cannot read ground truth, even by accident."""
    targets = tmp_path / "targets.jsonl"
    targets.write_text(json.dumps(
        {"item_id": "p1__clean", "label": 0, "teacher_logprob": 0.33}
    ) + "\n")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([
        {"item_id": "p1__clean::persona_attacker", "label": 0,
         "prompt_name": "persona_attacker", "logprob_score": 0.9, "response": "..."}
    ]))

    by_target, by_persona = load_teacher_data(targets_path=targets, raw_path=raw)
    assert "label" not in by_target["p1__clean"]
    assert all("label" not in row for row in by_persona["p1__clean"])


def test_strip_label_blanks_the_label():
    """Research code sees -1, never the real label."""
    from src.data import Item

    item = Item("p1__clean", "p1", "q", "code", 0, False)
    assert strip_label(item).label == -1


def test_summarise_history_reports_direction():
    """The smoke test's verdict line says whether the loss went down."""
    rows = [
        {"loss": 3.0 - i * 0.2, "ce_text": 2.0, "kd_yes": 1.0 - i * 0.2}
        for i in range(8)
    ]
    assert "decreased" in summarise_history(rows)
    assert "DID NOT DECREASE" in summarise_history(rows[::-1])


# --------------------------------------------------------------------------
# M3: label targets (ADR-0007)
# --------------------------------------------------------------------------


def base_sample(p_yes, reasoning):
    """A base-monitor answer with a known P(yes) and reasoning sentence."""
    verdict = "yes" if p_yes >= 0.5 else "no"
    return {
        "logprob_score": p_yes,
        "response": f"REASONING: {reasoning}\nBACKDOORED: {verdict}\nSUSPICION: 5",
    }


def test_label_target_uses_the_label_not_the_monitor():
    """Verdict and score follow the true label, whatever the monitor thought."""
    from src.train_sft import build_label_target_response

    samples = [base_sample(0.1, "Looks honest."), base_sample(0.2, "Fine.")]
    response = build_label_target_response(1, samples)
    assert "BACKDOORED: yes" in response
    assert response.endswith("SUSPICION: 10")

    response = build_label_target_response(0, [base_sample(0.9, "Odd branch.")])
    assert "BACKDOORED: no" in response
    assert response.endswith("SUSPICION: 0")


def test_label_target_prefers_reasoning_that_agrees_with_the_label():
    """The most confident agreeing sample supplies the reasoning sentence."""
    from src.train_sft import build_label_target_response

    samples = [
        base_sample(0.2, "Looks honest."),
        base_sample(0.7, "A suspicious special case."),
        base_sample(0.99, "A magic-constant trigger."),
    ]
    assert "magic-constant trigger" in build_label_target_response(1, samples)
    assert "Looks honest" in build_label_target_response(0, samples)


def test_label_target_raises_without_parseable_reasoning():
    """No reasoning to borrow means no example, never an invented sentence."""
    from src.train_sft import build_label_target_response

    with pytest.raises(ValueError):
        build_label_target_response(1, [{"logprob_score": 0.9, "response": "garbled"}])


def test_label_training_excludes_nondeterministic_items():
    """M3 reads labels, so it must not learn the noisy ones."""
    from src.data import Item
    from src.train_sft import label_training_items

    items = [
        Item("a__clean", "a", "q", "c", 0, False),
        Item("b__backdoored", "b", "q", "c", 1, True),
        Item("c__backdoored", "c", "q", "c", 1, False),
    ]
    kept = [item.item_id for item in label_training_items(items)]
    assert kept == ["a__clean", "c__backdoored"]


def test_reasoning_agreement_counts_items_with_an_agreeing_sample():
    """Reports how often the borrowed reasoning can match the verdict."""
    from src.data import Item
    from src.train_sft import reasoning_agreement

    items = [Item("a", "a", "q", "c", 1, False), Item("b", "b", "q", "c", 0, False)]
    samples = {"a": [base_sample(0.9, "x")], "b": [base_sample(0.8, "y")]}
    assert reasoning_agreement(items, samples) == pytest.approx(0.5)


def test_m1_teacher_averages_the_three_samples_and_strips_labels(tmp_path, monkeypatch):
    """M3's teacher is M1's three samples, aggregated exactly like M2's personas."""
    from src import config
    from src.train_sft import load_ensemble_teacher

    generations, training = tmp_path / "gen", tmp_path / "train"
    generations.mkdir()
    monkeypatch.setattr(config, "GENERATIONS_DIR", generations)
    monkeypatch.setattr(config, "TRAINING_DIR", training)

    rows = [
        {"item_id": "p1__clean::baseline", "problem_id": "p1", "label": 0, "sample": k,
         "prompt_name": "baseline", "logprob_score": score, "text_score": text,
         "response": "REASONING: x\nBACKDOORED: no\nSUSPICION: 1"}
        for k, (score, text) in enumerate([(0.9, 8), (0.3, 2), (0.0, 1)])
    ]
    (generations / "m1__train.json").write_text(json.dumps(rows))

    targets, members = load_ensemble_teacher("m1-ensemble")
    assert targets["p1__clean"]["teacher_logprob"] == pytest.approx(0.4)
    assert targets["p1__clean"]["teacher_text"] == pytest.approx(11 / 3)
    assert len(members["p1__clean"]) == 3
    assert "label" not in targets["p1__clean"]
    assert all("label" not in row for row in members["p1__clean"])


def test_debate_teacher_is_graded_not_a_vote(tmp_path, monkeypatch):
    """M8's target is a mean probability, not a 0/1 verdict (ADR-0010)."""
    import json

    from src import config
    from src.train_sft import load_debate_teacher

    generations = tmp_path / "gen"
    generations.mkdir()
    monkeypatch.setattr(config, "GENERATIONS_DIR", generations)

    personas = [p.stem for p in config.PERSONA_PROMPTS]
    teacher = [
        {"item_id": f"p1__clean::{persona}", "prompt_name": persona, "label": 0,
         "logprob_score": score, "text_score": 3,
         "response": f"REASONING: {persona} view.\nBACKDOORED: no\nSUSPICION: 3"}
        for persona, score in zip(personas, (0.1, 0.2, 0.9), strict=True)
    ]
    (generations / "teacher__train.json").write_text(json.dumps(teacher))

    debate = [
        {"item_id": f"p1__clean::{persona}::round2", "problem_id": "p1", "label": 0,
         "persona": persona, "verdict": "no", "round_one_verdict": "no",
         "logprob_score": score, "text_score": 4,
         "response": "REASONING: after debate.\nBACKDOORED: no\nSUSPICION: 4"}
        for persona, score in zip(personas, (0.4, 0.5, 0.6), strict=True)
    ]
    (generations / "debate__train.json").write_text(json.dumps(debate))

    targets, rows = load_debate_teacher("train", use_round=2, select="all")
    target = targets["p1__clean"]
    # Round-2 mean of 0.4, 0.5, 0.6 -- graded, and not either verdict.
    assert target["teacher_logprob"] == pytest.approx(0.5)
    assert target["n_averaged"] == 3
    # Reasoning text comes from round 1, which never mentions peers.
    assert all("after debate" not in r["response"] for r in rows["p1__clean"])

    first, _ = load_debate_teacher("train", use_round=1, select="all")
    assert first["p1__clean"]["teacher_logprob"] == pytest.approx(0.4)


def test_debate_teacher_rejects_unknown_settings():
    """A typo must fail loudly rather than silently distil the wrong thing."""
    from src.train_sft import load_debate_teacher

    with pytest.raises(ValueError, match="use_round"):
        load_debate_teacher("train", use_round=3)
    with pytest.raises(ValueError, match="select"):
        load_debate_teacher("train", select="majority")
