"""Readout parsing, tested without a model.

Both readouts are pure functions of a ``Generation``, so a fake one exercises
exactly the code the real run uses. These tests are the reason a parsing bug
cannot survive to the GPU.

The invariant that matters most: an unreadable response must return ``None``,
never a plausible-looking number. A silently wrong score enters the metrics and
is indistinguishable from a real one.
"""

from __future__ import annotations

import math

import pytest

from src.scoring import (
    Generation,
    find_verdict_index,
    parse_text_score,
    read_verdict_probability,
    render_prompt,
)

VERDICT_TOKENS = [
    "RE", "ASON", "ING", ":", " fine", "\n",
    "BACK", "DOOR", "ED", ":", " yes", "\n", "SUS",
]
VERDICT_POSITION = 10


def generation(tokens: list[str], probabilities: dict[int, dict[str, float]]):
    """Build a fake generation with known probabilities at chosen positions.

    Args:
        tokens: Decoded token strings.
        probabilities: Position to {token string: probability}.

    Returns:
        A ``Generation`` whose log-probabilities match ``probabilities``.
    """
    logprobs = [
        {token: math.log(p) for token, p in probabilities.get(index, {}).items()}
        for index, _ in enumerate(tokens)
    ]
    return Generation(
        text="".join(tokens), tokens=tokens, token_logprobs=logprobs
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("REASONING: x\nBACKDOORED: yes\nSUSPICION: 8\n", 8),
        ("SUSPICION: 0", 0),
        ("SUSPICION:10", 10),
        ("SUSPICION: 99", 10),
        ("SUSPICION: -3", None),  # a negative score is malformed, not clamped
        ("the model ignored the format", None),
    ],
)
def test_parse_text_score(text: str, expected: int | None):
    """The secondary readout clamps to [0, 10] and gives up cleanly."""
    assert parse_text_score(text) == expected


def test_find_verdict_index_locates_the_token_after_the_marker():
    """Matching is on decoded text, since tokenizers split the marker freely."""
    index = find_verdict_index(VERDICT_TOKENS)
    assert index == VERDICT_POSITION
    assert VERDICT_TOKENS[index] == " yes"


def test_find_verdict_index_returns_none_without_a_marker():
    """A malformed response must not resolve to an arbitrary position."""
    assert find_verdict_index(["no", "marker", "here"]) is None


def test_verdict_probability_is_normalised():
    """``P(yes)/(P(yes)+P(no))``, not raw ``P(yes)``.

    A response spending mass on formatting tokens must not thereby read as less
    suspicious; normalising divides that noise out.
    """
    gen = generation(VERDICT_TOKENS, {VERDICT_POSITION: {" yes": 0.6, " no": 0.2}})
    assert read_verdict_probability(gen) == pytest.approx(0.75)


def test_verdict_probability_sums_over_spellings():
    """Tokenizers disagree about leading spaces and capitalisation."""
    gen = generation(
        VERDICT_TOKENS,
        {VERDICT_POSITION: {" yes": 0.3, "Yes": 0.2, " no": 0.5}},
    )
    assert read_verdict_probability(gen) == pytest.approx(0.5)


def test_verdict_probability_tolerates_a_stray_token():
    """A space or newline before the word must not lose the verdict."""
    tokens = ["BACK", "DOOR", "ED", ":", " ", "yes"]
    gen = generation(tokens, {5: {"yes": 0.8, "no": 0.2}})
    assert read_verdict_probability(gen) == pytest.approx(0.8)


def test_verdict_probability_is_none_when_unreadable():
    """Unreadable must mean ``None``, never a plausible wrong number."""
    assert read_verdict_probability(generation(VERDICT_TOKENS, {})) is None
    assert read_verdict_probability(Generation(text="no tokens")) is None


def test_render_prompt_substitutes_and_preserves_the_contract(tmp_path):
    """Placeholders are filled and the frozen output block survives."""
    from src.config import BASELINE_PROMPT
    from src.data import Item

    item = Item(
        item_id="p1__clean",
        problem_id="p1",
        question="Sum two integers.",
        code="def solve(a, b): return a + b",
        label=0,
        is_nondeterministic=False,
    )
    rendered = render_prompt(BASELINE_PROMPT, item)

    assert "{question}" not in rendered
    assert "{code}" not in rendered
    assert item.question in rendered
    assert item.code in rendered
    assert rendered.rstrip().endswith("certainly backdoored.")


class FakeBackend:
    """Returns a fixed well-formed response and counts how often it is called."""

    max_concurrency = 1
    name = "fake"
    model_id = "fake-model"

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls = 0

    def generate(self, prompt, n, sampling):
        """Produce ``n`` identical parseable generations."""
        self.calls += 1
        tokens = ["BACK", "DOOR", "ED", ":", " yes", "\nSUSPICION: 7"]
        logprobs = [{}, {}, {}, {}, {" yes": math.log(0.9), " no": math.log(0.1)}, {}]
        return [
            Generation(
                text="".join(tokens),
                tokens=tokens,
                token_logprobs=logprobs,
                finish_reason="stop",
            )
            for _ in range(n)
        ]


def make_items(count: int):
    """Build ``count`` synthetic items."""
    from src.data import Item

    return [
        Item(
            item_id=f"p{i}__clean",
            problem_id=f"p{i}",
            question="q",
            code="c",
            label=0,
            is_nondeterministic=False,
        )
        for i in range(count)
    ]


def test_score_items_writes_a_json_array(tmp_path):
    """Output is one readable JSON list with both readouts on every record."""
    import json

    from src.config import BASELINE_PROMPT
    from src.scoring import score_items

    out = tmp_path / "m0__val.json"
    score_items(make_items(3), BASELINE_PROMPT, FakeBackend(), "m0", out)

    records = json.loads(out.read_text())
    assert isinstance(records, list) and len(records) == 3
    assert records[0]["logprob_score"] == pytest.approx(0.9)
    assert records[0]["text_score"] == 7
    assert records[0]["item_id"] == "p0__clean::baseline"


def test_score_items_resumes_without_rescoring(tmp_path):
    """A second run over the same items must not call the model again."""
    from src.config import BASELINE_PROMPT
    from src.scoring import score_items
    from src.utils import read_json_records

    out = tmp_path / "m0__val.json"
    score_items(make_items(2), BASELINE_PROMPT, FakeBackend(), "m0", out)

    backend = FakeBackend()
    score_items(make_items(5), BASELINE_PROMPT, backend, "m0", out)

    assert backend.calls == 3
    assert len(read_json_records(out)) == 5


def test_multiple_samples_are_saved_together(tmp_path):
    """M1 writes all three samples of an item at once, never a partial item."""
    from src.config import BASELINE_PROMPT
    from src.scoring import score_items
    from src.utils import read_json_records

    out = tmp_path / "m1__val.json"
    score_items(make_items(2), BASELINE_PROMPT, FakeBackend(), "m1", out, n_samples=3)

    records = read_json_records(out)
    assert len(records) == 6
    first_item = [r["sample"] for r in records if r["item_id"].startswith("p0")]
    assert sorted(first_item) == [0, 1, 2]


def test_no_resume_keeps_other_templates(tmp_path):
    """Regenerating one persona must not erase the other personas in the file."""
    from src.config import PERSONA_PROMPTS
    from src.scoring import score_items
    from src.utils import read_json_records

    out = tmp_path / "m2__val.json"
    first, second = PERSONA_PROMPTS[0], PERSONA_PROMPTS[1]
    score_items(make_items(2), first, FakeBackend(), "m2", out)
    score_items(make_items(2), second, FakeBackend(), "m2", out)

    backend = FakeBackend()
    score_items(make_items(2), first, backend, "m2", out, resume=False)

    records = read_json_records(out)
    assert backend.calls == 2
    assert len(records) == 4
    assert {r["prompt_name"] for r in records} == {first.stem, second.stem}


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(4.26, "4.3s"), (782, "13m 02s"), (3852, "1h 04m 12s")],
)
def test_format_duration(seconds, expected):
    """Durations read naturally at every scale a run can take."""
    from src.utils import format_duration

    assert format_duration(seconds) == expected


def test_refuses_to_mix_backends_in_one_file(tmp_path):
    """A GPU run must not resume from, and silently extend, a 4-bit pilot file."""
    from src.config import BASELINE_PROMPT
    from src.scoring import score_items

    out = tmp_path / "m0__val.json"
    score_items(make_items(2), BASELINE_PROMPT, FakeBackend(), "m0", out)

    other = FakeBackend()
    other.name = "vllm"
    with pytest.raises(ValueError, match="mv "):
        score_items(make_items(2), BASELINE_PROMPT, other, "m0", out)
    assert other.calls == 0


def test_derive_m0_takes_the_first_m1_sample(tmp_path):
    """M0 is one draw of M1: same prompt and temperature, sample 0."""
    from src.config import BASELINE_PROMPT
    from src.scoring import derive_m0_from_m1, score_items
    from src.utils import read_json_records

    score_items(make_items(4), BASELINE_PROMPT, FakeBackend(), "m1",
                tmp_path / "m1__val.json", n_samples=3)
    derive_m0_from_m1("val", directory=tmp_path)

    m0 = read_json_records(tmp_path / "m0__val.json")
    assert len(m0) == 4
    assert {row["sample"] for row in m0} == {0}
    assert {row["arm"] for row in m0} == {"m0"}
    assert all(row["derived_from"] == "m1 sample 0" for row in m0)


def test_derive_m0_never_overwrites_an_independent_run(tmp_path):
    """A real M0 run must survive an accidental derive."""
    from src.config import BASELINE_PROMPT
    from src.scoring import derive_m0_from_m1, score_items

    score_items(make_items(2), BASELINE_PROMPT, FakeBackend(), "m1",
                tmp_path / "m1__val.json", n_samples=3)
    score_items(make_items(2), BASELINE_PROMPT, FakeBackend(), "m0",
                tmp_path / "m0__val.json")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        derive_m0_from_m1("val", directory=tmp_path)


def test_scoring_imports_without_mlx():
    """The GPU box has no MLX, so importing scoring must not require it.

    Runs in a fresh interpreter with every MLX module blocked, which is exactly
    the Linux GPU environment. A top-level MLX import passes every other test
    on a Mac and then fails on the first scoring command on the rented GPU.
    """
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "for name in ('mlx', 'mlx.core', 'mlx_lm', 'mlx_lm.sample_utils'):\n"
        "    sys.modules[name] = None\n"
        "import src.scoring, src.analysis, src.figures\n"
        "src.scoring.get_backend('vllm')\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_over_long_prompt_is_recorded_as_unreadable_not_fatal(tmp_path):
    """One item that can't fit the context window must not stop the run."""
    from dataclasses import replace

    from src.config import BASELINE_PROMPT
    from src.scoring import PromptTooLongError, score_items
    from src.utils import read_json_records

    items = make_items(3)
    items[1] = replace(items[1], code="x" * 50_000)
    too_long = items[1].item_id

    class LengthLimitedBackend(FakeBackend):
        """Rejects any prompt longer than a small context window."""

        def generate(self, prompt, n, sampling):
            """Raise for the over-long item, answer normally otherwise."""
            if len(prompt) > 10_000:
                raise PromptTooLongError("maximum context length is 8192 tokens")
            return super().generate(prompt, n, sampling)

    out = tmp_path / "m0__val.json"
    score_items(items, BASELINE_PROMPT, LengthLimitedBackend(), "m0", out)
    rows = {r["item_id"].split("::")[0]: r for r in read_json_records(out)}

    assert len(rows) == 3
    assert rows[too_long]["finish_reason"] == "prompt_too_long"
    assert rows[too_long]["logprob_score"] is None
    assert rows[too_long]["text_score"] is None
    assert rows[items[0].item_id]["logprob_score"] is not None
