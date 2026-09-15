"""DPO maths and M5's training loop. Needs torch; the loop test also needs peft."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="pip install torch to test DPO")


def test_dpo_loss_is_log2_when_policy_equals_reference():
    """Before any learning the implicit rewards are zero, so loss = log 2."""
    from src.train_dpo import dpo_loss

    logps = torch.tensor([-5.0, -7.0])
    loss, parts = dpo_loss(logps, logps, logps, logps, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert parts["reward_margin"] == pytest.approx(0.0)


def test_dpo_loss_falls_when_chosen_gains_on_rejected():
    """Raising the chosen answer relative to the reference lowers the loss."""
    from src.train_dpo import dpo_loss

    reference = torch.tensor([-5.0, -5.0])
    better, _ = dpo_loss(reference + 2, reference - 2, reference, reference, 0.1)
    worse, parts = dpo_loss(reference - 2, reference + 2, reference, reference, 0.1)
    assert better.item() < math.log(2) < worse.item()
    assert parts["reward_accuracy"] == 0.0


def test_logps_count_only_answer_tokens():
    """Prompt and padding positions contribute nothing to a sequence's log-prob."""
    from src.train_dpo import logps_from_logits

    vocab = 5
    logits = torch.zeros(1, 4, vocab)
    logits[0, 1, 3] = 10.0  # position 1 predicts token 3 almost surely
    labels = torch.tensor([[-100, -100, 3, -100]])
    logp = logps_from_logits(logits, labels)
    expected = math.log(math.exp(10) / (math.exp(10) + vocab - 1))
    assert logp.item() == pytest.approx(expected, abs=1e-4)


def test_pad_batch_masks_padding():
    """Padding is invisible to attention and to the loss."""
    from src.train_dpo import pad_batch

    batch = pad_batch([[1, 2, 3], [4]], [[-100, 2, 3], [-100]], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"][1].tolist() == [-100, -100, -100]


def test_training_learns_to_prefer_chosen_answers():
    """On a tiny model, DPO raises reward accuracy and lowers loss."""
    pytest.importorskip("peft", reason="pip install peft to run the loop test")
    import random

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from src.train_dpo import (
        PairExample,
        reference_logps,
        summarise_dpo_history,
        train_dpo_epochs,
    )

    torch.manual_seed(0)
    random.seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
    ))
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    examples = []
    for i in range(16):
        prompt = [random.randrange(10, 64) for _ in range(8)]
        chosen, rejected = [5, 6, 7], [8, 9, 10]
        mask = [-100] * len(prompt)
        examples.append(PairExample(
            f"e{i}", prompt + chosen, mask + chosen, prompt + rejected, mask + rejected,
        ))

    references = reference_logps(model, examples, batch_size=4, pad_id=0)
    history = train_dpo_epochs(
        model, examples, references, epochs=6, beta=0.5, learning_rate=5e-3,
        batch_size=4, grad_accum=1, seed=1, pad_id=0,
    )
    assert "loss decreased" in summarise_dpo_history(history)
    assert history[-1]["reward_accuracy"] >= history[0]["reward_accuracy"]
