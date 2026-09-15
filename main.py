"""Single entry point for every stage of the experiment.

This file contains argparse wiring and function calls, and nothing else. If a
subcommand needs an ``if``, that ``if`` belongs in ``src/`` — otherwise this
grows to 800 lines and stops being readable, which is the well-known failure
mode of the dispatcher pattern.

Every stage is independently runnable and resumable. The pipeline is never run
end to end in one go: generation stages take hours on interruptible instances,
so each is invoked on its own and appends results as it goes.

Imports of stage modules are **deliberately lazy**, inside each handler. That
keeps ``python main.py --help`` working before the modules are implemented, and
lets the laptop run data and analysis stages without importing anything from
the GPU dependency tier.

Usage::

    python main.py build-data
    python main.py score --arm m0 --split val
    python main.py diversity-check --split val
    python main.py debate --split train
    python main.py build-pairs
    python main.py teacher-scores --split train
    python main.py train-sft
    python main.py train-dpo
    python main.py analyse
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

ARM_CHOICES = ("m0", "m1", "m2", "m3", "m4", "m5")
SPLIT_CHOICES = ("train", "val", "test")


def cmd_build_data(args: argparse.Namespace) -> int:
    """Load, filter, pair and split the dataset into three frozen JSONL files."""
    from src import config
    from src.data import build_all

    paths = build_all(config.DATASET, config.SPLITS, config.SPLITS_DIR)
    for split, path in paths.items():
        print(f"{split:>5}: {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score one split with one arm, appending results as they complete."""
    from src.scoring import score_arm

    path = score_arm(
        arm=args.arm,
        split=args.split,
        adapter=args.adapter,
        resume=not args.no_resume,
        backend_name=args.backend,
        limit=args.limit,
    )
    print(f"scores -> {path}")
    return 0


def cmd_derive_m0(args: argparse.Namespace) -> int:
    """Take M0's scores from the first of M1's three samples."""
    from src.scoring import derive_m0_from_m1

    path = derive_m0_from_m1(split=args.split, limit=args.limit)
    print(f"scores -> {path}")
    return 0


def cmd_diversity_check(args: argparse.Namespace) -> int:
    """Score a split with the other model families, to test ADR-0002.

    Compares model diversity against prompt diversity before the debate
    generations commit us to the latter. Val only.
    """
    from src import config
    from src.scoring import score_alternate_models

    path = score_alternate_models(
        split=args.split,
        models=config.DIVERSITY_CHECK_MODELS,
        resume=not args.no_resume,
        backend_name=args.backend,
    )
    print(f"scores -> {path}")
    return 0


def cmd_debate(args: argparse.Namespace) -> int:
    """Run the two-round persona debate and persist full traces."""
    from src.debate import run_debate

    path = run_debate(split=args.split, resume=not args.no_resume)
    print(f"transcripts -> {path}")
    return 0


def cmd_build_pairs(args: argparse.Namespace) -> int:
    """Turn debate transcripts into DPO preference pairs under the baseline prompt."""
    from src.debate import build_preference_pairs

    path = build_preference_pairs(bins=args.bins)
    print(f"pairs -> {path}")
    return 0


def cmd_teacher_scores(args: argparse.Namespace) -> int:
    """Generate M4's regression targets: the personas' unrounded mean score."""
    from src.scoring import build_teacher_scores

    path = build_teacher_scores(
        split=args.split, resume=not args.no_resume, backend_name=args.backend
    )
    print(f"targets -> {path}")
    return 0


def cmd_train_sft(args: argparse.Namespace) -> int:
    """Train M4 (ensemble targets) or M3 (true labels) with the two-term loss."""
    from src.train_sft import train

    run = train(smoke=args.smoke, kd_weight=args.kd_weight, targets=args.targets)
    print(f"run -> {run}")
    return 0


def cmd_train_dpo(args: argparse.Namespace) -> int:
    """Train M5 on debate-consensus preference pairs (ADR-0003)."""
    from src.train_dpo import train

    run = train(smoke=args.smoke)
    print(f"run -> {run}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Compute metrics, test the hypotheses, draw figures and write the report."""
    from src.analysis import run_analysis

    report = run_analysis(
        split=args.split, limit=args.limit, resamples=args.resamples
    )
    print(f"report -> {report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser and every subcommand."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Distilling trusted-monitor diversity. Stages are run "
            "individually and are resumable; see project-plan.md section 4."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    def add_backend_flag(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--backend",
            default="vllm",
            choices=("vllm", "mlx"),
            help=(
                "Where to send requests. 'vllm' is the served CUDA box; 'mlx' "
                "runs a quantised model in-process on Apple Silicon, for "
                "pilots only -- 4-bit local scores are NOT comparable to bf16 "
                "served scores."
            ),
        )

    def add_resume_flag(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--no-resume",
            action="store_true",
            help=(
                "Regenerate from scratch instead of skipping already-completed "
                "items. Resuming is the default because generation runs on "
                "interruptible instances."
            ),
        )

    add("build-data", cmd_build_data, "Build the three frozen splits.")

    sub = add("score", cmd_score, "Score one split with one arm.")
    sub.add_argument("--arm", required=True, choices=ARM_CHOICES)
    sub.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    sub.add_argument(
        "--adapter",
        default=None,
        help=(
            "Served name of a LoRA adapter (the NAME in vLLM's --lora-modules "
            "NAME=PATH). Required for m3, m4 and m5, ignored otherwise."
        ),
    )
    sub.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N items. For pilots; writes a separate file.",
    )
    add_backend_flag(sub)
    add_resume_flag(sub)

    sub = add(
        "derive-m0",
        cmd_derive_m0,
        "Take M0's scores from the first of M1's samples (same prompt and "
        "temperature, so no re-scoring).",
    )
    sub.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    sub.add_argument("--limit", type=int, default=None)

    sub = add(
        "diversity-check",
        cmd_diversity_check,
        "Score with Llama and Mistral to compare model vs prompt diversity.",
    )
    sub.add_argument("--split", default="val", choices=SPLIT_CHOICES)
    add_backend_flag(sub)
    add_resume_flag(sub)

    sub = add("debate", cmd_debate, "Run the two-round persona debate.")
    sub.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    add_resume_flag(sub)

    sub = add("build-pairs", cmd_build_pairs, "Build DPO preference pairs.")
    sub.add_argument(
        "--bins",
        default="binary",
        choices=("binary", "quintiles"),
        help=(
            "Consensus binning. Binary unless val agreement exceeds 85%%, a "
            "rule fixed in advance rather than tuned after seeing results."
        ),
    )

    sub = add("teacher-scores", cmd_teacher_scores, "Generate M4's targets.")
    sub.add_argument("--split", default="train", choices=SPLIT_CHOICES)
    add_backend_flag(sub)
    add_resume_flag(sub)

    sub = add("train-sft", cmd_train_sft, "Train M4, or M3 with --targets labels.")
    sub.add_argument(
        "--targets",
        default="ensemble",
        choices=("ensemble", "labels"),
        help=(
            "ensemble: M4, trained on M2's judgements (no labels). labels: M3, "
            "trained on the true labels of deterministic train items (ADR-0007)."
        ),
    )
    sub.add_argument(
        "--smoke",
        action="store_true",
        help="Run on ~50 items to prove the loop executes and loss decreases.",
    )
    sub.add_argument(
        "--kd-weight",
        type=float,
        default=None,
        help=(
            "Override lambda in `ce_text + lambda * kd_yes`. Pass 0.0 for the "
            "declared fallback to plain text SFT (ADR-0005)."
        ),
    )

    sub = add("train-dpo", cmd_train_dpo, "Train M5.")
    sub.add_argument("--smoke", action="store_true", help="Run on ~50 items.")

    sub = add(
        "analyse", cmd_analyse, "Compute metrics, draw figures, write the report."
    )
    sub.add_argument("--split", default="val", choices=SPLIT_CHOICES)
    sub.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyse the files a `score --limit N` run wrote.",
    )
    sub.add_argument(
        "--resamples",
        type=int,
        default=None,
        help="Bootstrap draws (default 10,000). Lower it for a quick look.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected stage, and report how long it took.

    The timing line prints even when the stage fails or is interrupted, so a
    crashed overnight run still says how far it got.
    """
    from datetime import datetime

    from src.utils import format_duration

    args = build_parser().parse_args(argv)
    started_at = datetime.now()
    started = time.perf_counter()
    try:
        return args.handler(args)
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"\n[{args.command}] started {started_at:%Y-%m-%d %H:%M:%S} · "
            f"finished {datetime.now():%H:%M:%S} · took {format_duration(elapsed)}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
