"""Contract tests for ``two_term_loss`` (ADR-0005).

Marked ``xfail`` while the function still raises ``NotImplementedError``, so the
suite stays green and each test becomes a real pass or failure the moment the
body is written.

Needs torch, which the laptop's requirements leave out; for CPU-only testing on
macOS, ``pip install torch``. Without it this file is skipped.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="pip install torch to test the loss")

# --------------------------------------------------------------------------
# two_term_loss (Andre)
# --------------------------------------------------------------------------


def _loss():
    """Import lazily so collection works before the function exists."""
    from src.train_sft import two_term_loss

    return two_term_loss


def _implemented(fn, *args):
    """Call the loss, turning NotImplementedError into an expected failure."""
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.xfail("two_term_loss not written yet")


@pytest.fixture
def batch():
    """A small synthetic batch: two sequences, vocabulary of 50."""
    torch.manual_seed(0)
    b, length, vocab = 2, 12, 50
    labels = torch.randint(0, vocab, (b, length))
    labels[:, :5] = -100
    return {
        "shape": (b, length, vocab),
        "labels": labels,
        "verdict_pos": torch.tensor([8, 9]),
        "kd_target": torch.tensor([0.9, 0.2]),
        "yes_ids": [3, 4],
        "no_ids": [7],
    }


def test_zero_weight_is_exactly_shifted_cross_entropy(batch):
    """kd_weight = 0 must reduce to plain text SFT, the declared fallback."""
    from torch.nn import functional

    b, length, vocab = batch["shape"]
    logits = torch.randn(b, length, vocab, dtype=torch.bfloat16, requires_grad=True)
    loss, parts = _implemented(
        _loss(), logits, batch["labels"], batch["verdict_pos"], batch["kd_target"],
        batch["yes_ids"], batch["no_ids"], 0.0,
    )
    expected = functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, vocab),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(loss.float(), expected, atol=1e-4)
    assert set(parts) >= {"ce_text", "kd_yes", "p_yes_mean"}


def test_full_weight_is_finite_on_bf16_and_backpropagates(batch):
    """Bf16 logits give a finite loss and gradients that reach the logits."""
    b, length, vocab = batch["shape"]
    logits = torch.randn(b, length, vocab, dtype=torch.bfloat16, requires_grad=True)
    loss, _ = _implemented(
        _loss(), logits, batch["labels"], batch["verdict_pos"], batch["kd_target"],
        batch["yes_ids"], batch["no_ids"], 1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_kd_reads_normalised_p_yes_one_position_before_the_verdict(batch):
    """KD_yes is lowest when the student's P(yes) already equals the target.

    The probability is planted at ``verdict_pos - 1``. Reading any other
    position, or skipping the yes/no normalisation, fails the last assertion.
    """
    b, length, vocab = batch["shape"]

    def logits_with_p_yes(p):
        out = torch.full((b, length, vocab), -30.0)
        for i in range(b):
            row = out[i, batch["verdict_pos"][i] - 1]
            row[batch["yes_ids"]] = torch.log(p[i] / len(batch["yes_ids"]))
            row[batch["no_ids"]] = torch.log(1 - p[i])
        return out

    args = (batch["labels"], batch["verdict_pos"], batch["kd_target"],
            batch["yes_ids"], batch["no_ids"], 1.0)
    _, matched = _implemented(_loss(), logits_with_p_yes(batch["kd_target"]), *args)
    _, flipped = _implemented(_loss(), logits_with_p_yes(1 - batch["kd_target"]), *args)
    assert matched["kd_yes"] < flipped["kd_yes"]
    assert abs(matched["p_yes_mean"] - batch["kd_target"].mean().item()) < 1e-3
