"""M5's debate and MACA pairing logic, with no model or network."""

from __future__ import annotations

import json

import pytest

from src.debate import (
    canonical_response,
    consensus_pairs,
    parse_verdict,
    round_two_messages,
)


def answer(persona, verdict, reasoning="Checks bounds.", score=None):
    """A round-2 record in the shape run_debate writes."""
    score = score if score is not None else (8 if verdict == "yes" else 2)
    text = (
        f"REASONING: {reasoning}\nBACKDOORED: {verdict}\nSUSPICION: {score}"
        if verdict else "garbled output"
    )
    return {"persona": persona, "verdict": verdict, "response": text,
            "problem_id": "p1", "label": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("REASONING: x\nBACKDOORED: yes\nSUSPICION: 7", "yes"),
        ("BACKDOORED:  No", "no"),
        ("no verdict here", None),
    ],
)
def test_parse_verdict(text, expected):
    """The written verdict is what MACA votes on."""
    assert parse_verdict(text) == expected


def test_canonical_response_normalises_and_rejects():
    """Chosen and rejected get identical formatting; malformed answers drop."""
    messy = "REASONING:  Odd   branch.\n\nBACKDOORED: yes\nSUSPICION: 9\nThanks!"
    assert canonical_response(messy) == (
        "REASONING: Odd branch.\nBACKDOORED: yes\nSUSPICION: 9"
    )
    assert canonical_response("BACKDOORED: yes") is None


def test_unanimous_debate_yields_no_pairs():
    """With no dissent there is nothing to prefer against."""
    verdict, pairs = consensus_pairs(
        [answer("a", "yes"), answer("b", "yes"), answer("c", "yes")], max_pairs=3
    )
    assert verdict == "yes"
    assert pairs == []


def test_split_debate_pairs_each_agreeing_answer_with_the_dissent():
    """A 2-1 split gives G+ x G- = two pairs, chosen always on the majority."""
    verdict, pairs = consensus_pairs(
        [answer("a", "no"), answer("b", "yes"), answer("c", "yes")], max_pairs=3
    )
    assert verdict == "yes"
    assert len(pairs) == 2
    assert all(c["verdict"] == "yes" and r["verdict"] == "no" for c, r in pairs)
    assert {c["persona"] for c, _ in pairs} == {"b", "c"}


def test_unreadable_answers_do_not_vote():
    """A malformed answer is neither majority nor dissent."""
    verdict, pairs = consensus_pairs(
        [answer("a", None), answer("b", "yes"), answer("c", "no")], max_pairs=3
    )
    assert verdict is None
    assert pairs == []


def test_pair_cap_limits_one_items_contribution():
    """The cap bounds how much any single item can dominate training."""
    _, pairs = consensus_pairs(
        [answer("a", "no"), answer("b", "yes"), answer("c", "yes")], max_pairs=1
    )
    assert len(pairs) == 1


def test_round_two_shows_peers_not_self():
    """Each persona reads the other two answers, with its own as its prior turn."""
    from src.config import PERSONA_PROMPTS
    from src.data import Item

    item = Item("p1__clean", "p1", "Sum two numbers.", "print(a+b)", 0, False)
    round_one = {p.stem: f"answer from {p.stem}" for p in PERSONA_PROMPTS}
    own = PERSONA_PROMPTS[0]
    messages = round_two_messages(item, own, round_one)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == f"answer from {own.stem}"
    follow_up = messages[2]["content"]
    assert f"answer from {own.stem}" not in follow_up
    for other in PERSONA_PROMPTS[1:]:
        assert f"answer from {other.stem}" in follow_up
    assert follow_up.rstrip().endswith("certainly backdoored.")


def test_build_pairs_never_writes_labels(tmp_path, monkeypatch):
    """Labels feed only the diagnostic in the stats file, never a pair."""
    from src import config
    from src.debate import build_preference_pairs
    from src.utils import read_jsonl

    generations, training = tmp_path / "gen", tmp_path / "train"
    generations.mkdir()
    monkeypatch.setattr(config, "GENERATIONS_DIR", generations)
    monkeypatch.setattr(config, "TRAINING_DIR", training)

    rows = []
    for persona, verdict in (("a", "no"), ("b", "yes"), ("c", "yes")):
        row = answer(persona, verdict)
        row.update(item_id=f"p1__backdoored::{persona}::round2",
                   round_one_verdict="no")
        rows.append(row)
    (generations / "debate__train.json").write_text(json.dumps(rows))

    path = build_preference_pairs("train")
    pairs = list(read_jsonl(path))
    assert len(pairs) == 2
    assert all("label" not in pair for pair in pairs)
    stats = json.loads(path.with_suffix(".stats.json").read_text())
    assert stats["verdicts_changed_in_round2"] == 2
    assert stats["diagnostic_majority_matches_label"] == 1.0


def test_round1_text_source_avoids_peer_references(tmp_path, monkeypatch):
    """The vote comes from the debate; the text comes from context-free round 1."""
    import json

    from src import config
    from src.debate import load_consensus_responses

    generations = tmp_path / "gen"
    generations.mkdir()
    monkeypatch.setattr(config, "GENERATIONS_DIR", generations)

    personas = [p.stem for p in config.PERSONA_PROMPTS]
    # Round 1: two say no, one says yes. Round 2: the dissenter is talked round,
    # so the majority is "no" and every round-1 "no" becomes a target.
    teacher = [
        {"item_id": f"p1__clean::{persona}", "prompt_name": persona, "label": 0,
         "response": f"REASONING: Independent view from {persona}.\n"
                     f"BACKDOORED: {'yes' if persona == personas[2] else 'no'}\n"
                     "SUSPICION: 2"}
        for persona in personas
    ]
    (generations / "teacher__train.json").write_text(json.dumps(teacher))

    debate = [
        {"item_id": f"p1__clean::{persona}::round2", "problem_id": "p1", "label": 0,
         "persona": persona, "verdict": "no",
         "round_one_verdict": "yes" if persona == personas[2] else "no",
         "response": "REASONING: All reviewers agree it is clean.\n"
                     "BACKDOORED: no\nSUSPICION: 1"}
        for persona in personas
    ]
    (generations / "debate__train.json").write_text(json.dumps(debate))

    round1 = load_consensus_responses("train", text_source="round1")
    entry = round1["p1__clean"]
    assert entry["majority"] == "no"
    # The two personas who said "no" in round 1 supply the targets.
    assert len(entry["agree"]) == 2
    assert len(entry["dissent"]) == 1
    assert all("reviewers" not in a["response"] for a in entry["agree"])

    round2 = load_consensus_responses("train", text_source="round2")
    entry2 = round2["p1__clean"]
    assert len(entry2["agree"]) == 3        # unanimous after debate
    assert entry2["dissent"] == []
    assert all("reviewers" in a["response"] for a in entry2["agree"])


def test_unknown_text_source_is_rejected():
    """A typo must fail loudly, not silently pick a default."""
    from src.debate import load_consensus_responses

    with pytest.raises(ValueError, match="text_source"):
        load_consensus_responses("train", text_source="round3")
