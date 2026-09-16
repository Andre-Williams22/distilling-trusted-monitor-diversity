"""KTO maths and M7's training loop. Needs torch; the loop test also needs peft."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="pip install torch to test KTO")


def test_balanced_weight_pulls_ratio_into_ktos_range():
    """An 18.6:1 split is reweighted to the middle of KTO's [1, 4/3] band."""
    from src.train_kto import KTO_RATIO_RANGE, balanced_undesirable_weight

    weight = balanced_undesirable_weight(4468, 240)
    ratio = (4468 * 1.0) / (240 * weight)
    assert KTO_RATIO_RANGE[0] <= ratio <= KTO_RATIO_RANGE[1]


def test_balanced_weight_is_neutral_when_a_class_is_empty():
    """With one class missing the ratio is undefined; the weight stays put."""
    from src.train_kto import balanced_undesirable_weight

    assert balanced_undesirable_weight(100, 0) == 1.0
    assert balanced_undesirable_weight(0, 100) == 1.0


def test_desirable_answers_want_higher_logprob_than_the_reference():
    """Raising a desirable answer above the baseline lowers the loss."""
    from src.train_kto import kto_loss

    reference = torch.tensor([-5.0])
    desirable = torch.tensor([True])
    z_ref = torch.zeros(())
    better, _ = kto_loss(reference + 2, reference, desirable, z_ref, 1.0, 1.0, 1.0)
    worse, _ = kto_loss(reference - 2, reference, desirable, z_ref, 1.0, 1.0, 1.0)
    assert better.item() < worse.item()


def test_dissenting_answers_want_lower_logprob_than_the_reference():
    """The undesirable class pushes the other way -- the sign is not shared."""
    from src.train_kto import kto_loss

    reference = torch.tensor([-5.0])
    dissent = torch.tensor([False])
    z_ref = torch.zeros(())
    better, _ = kto_loss(reference - 2, reference, dissent, z_ref, 1.0, 1.0, 1.0)
    worse, _ = kto_loss(reference + 2, reference, dissent, z_ref, 1.0, 1.0, 1.0)
    assert better.item() < worse.item()


def test_class_weights_change_what_the_loss_cares_about():
    """Upweighting dissent makes a wrong dissenting answer cost more."""
    from src.train_kto import kto_loss

    policy = torch.tensor([-3.0, -3.0])       # both above their reference
    reference = torch.tensor([-5.0, -5.0])
    desirable = torch.tensor([True, False])   # the second should be pushed down
    z_ref = torch.zeros(())
    light, _ = kto_loss(policy, reference, desirable, z_ref, 1.0, 1.0, 1.0)
    heavy, _ = kto_loss(policy, reference, desirable, z_ref, 1.0, 1.0, 10.0)
    assert heavy.item() > light.item()


def test_kl_baseline_never_goes_negative():
    """z_ref is clamped at zero, as in the KTO paper."""
    from src.train_kto import kl_baseline

    drifted = kl_baseline(torch.tensor([-2.0, -3.0]), torch.tensor([-5.0, -5.0]))
    assert drifted.item() == pytest.approx(2.5)
    assert kl_baseline(torch.tensor([-9.0]), torch.tensor([-5.0])).item() == 0.0


def test_kl_baseline_is_detached():
    """The baseline shifts the sigmoid; it is not itself optimised."""
    from src.train_kto import kl_baseline

    policy = torch.tensor([-2.0], requires_grad=True)
    assert not kl_baseline(policy, torch.tensor([-5.0])).requires_grad


def test_mismatched_batch_pairs_each_prompt_with_another_answer():
    """The KL term must see pairings the training data never contains."""
    from src.train_kto import KTOExample, mismatched_batch

    examples = [
        KTOExample("a", [1, 2, 10, 11], [-100, -100, 10, 11], True),
        KTOExample("b", [3, 4, 20, 21], [-100, -100, 20, 21], False),
    ]
    sequences, labels = mismatched_batch(examples, [0, 1], tokenizer=None)
    assert sequences[0] == [1, 2, 20, 21]     # prompt A, answer B
    assert sequences[1] == [3, 4, 10, 11]     # prompt B, answer A
    assert labels[0] == [-100, -100, 20, 21]


def test_accuracy_counts_each_class_on_its_own_side():
    """A desirable answer above the baseline and a dissent below it are both right."""
    from src.train_kto import kto_loss

    policy = torch.tensor([-3.0, -7.0])
    reference = torch.tensor([-5.0, -5.0])
    desirable = torch.tensor([True, False])
    _, parts = kto_loss(
        policy, reference, desirable, torch.zeros(()), 1.0, 1.0, 1.0
    )
    assert parts["accuracy"] == 1.0
    assert parts["reward_desirable"] == pytest.approx(2.0)
    assert parts["reward_undesirable"] == pytest.approx(-2.0)


def test_training_learns_to_separate_the_classes():
    """On a tiny model, KTO lowers the loss and raises accuracy."""
    pytest.importorskip("peft", reason="pip install peft to run the loop test")
    import random

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from src.train_kto import (
        KTOExample,
        reference_logps_for,
        summarise_kto_history,
        train_kto_epochs,
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
        agreed = i % 2 == 0
        answer = [5, 6, 7] if agreed else [8, 9, 10]
        examples.append(KTOExample(
            f"e{i}", prompt + answer, [-100] * len(prompt) + answer, agreed,
        ))

    references = reference_logps_for(model, examples, batch_size=4, pad_id=0)
    history = train_kto_epochs(
        model, examples, references, epochs=6, beta=0.5,
        desirable_weight=1.0, undesirable_weight=1.0, learning_rate=5e-3,
        batch_size=4, grad_accum=1, kl_batch_size=4, seed=1, pad_id=0,
    )
    assert "loss decreased" in summarise_kto_history(history)
    assert history[-1]["accuracy"] >= history[0]["accuracy"]


def test_overrides_reach_the_training_loop_and_the_run_name(monkeypatch, tmp_path):
    """A recorded hyperparameter must be the one actually used.

    Guards a real failure: `learning_rate` was computed and written into the run
    metadata while the *default* was passed to the training loop, and `tag`
    never reached the run directory, so two diagnostic runs overwrote each
    other and one claimed a learning rate it never used.
    """
    import src.train_kto as train_kto

    seen = {}

    def fake_train_epochs(*args, **kwargs):
        seen.update(kwargs)
        return [{"loss": 1.0, "accuracy": 0.5, "reward_desirable": 0.0,
                 "reward_undesirable": 0.0, "z_ref": 0.0}]

    def fake_run_dir(name, root=None):
        seen["run_name"] = name
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(train_kto, "train_kto_epochs", fake_train_epochs)
    monkeypatch.setattr(train_kto, "run_dir", fake_run_dir)
    monkeypatch.setattr(train_kto, "load_student", lambda: object())
    monkeypatch.setattr(train_kto, "reference_logps_for", lambda *a, **k: [0.0, 0.0])
    monkeypatch.setattr(train_kto, "save_run",
                        lambda model, history, directory, meta: directory / "adapter")
    monkeypatch.setattr(train_kto, "load_split", lambda split: [])
    monkeypatch.setattr(train_kto, "load_consensus_responses", lambda *a, **k: {})

    class _Tok:
        pad_token_id = 0

    monkeypatch.setattr(train_kto, "load_tokenizer", lambda: _Tok())
    monkeypatch.setattr(
        train_kto, "build_kto_examples",
        lambda items, consensus, tokenizer: (
            [train_kto.KTOExample("a", [1, 2], [-100, 2], True),
             train_kto.KTOExample("b", [3, 4], [-100, 4], False)], 0),
    )

    train_kto.train(learning_rate=2e-5, undesirable_weight=3.0, tag="lowlr")
    assert seen["learning_rate"] == 2e-5, "override never reached the loop"
    assert seen["undesirable_weight"] == 3.0
    assert seen["run_name"].endswith("-lowlr"), "tag never reached the run name"
