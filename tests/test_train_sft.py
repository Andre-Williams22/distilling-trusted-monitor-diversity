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
