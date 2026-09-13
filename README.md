# distilling-trusted-monitor-diversity

Can inference-time monitor diversity be distilled into a single trusted monitor
— recovering most of a 3-monitor ensemble's backdoor-detection gain at 1x
inference cost, using no labels and no stronger teacher?

Two distillation methods compared side by side: DPO on MACA-style
consensus-derived preference pairs, and supervised distillation from the
ensemble's mean scores. Evaluated as static classification on the
[ControlArena APPS backdoor dataset](https://huggingface.co/datasets/RoganInglis/apps-control-arena).

See `project-plan.md` for the full design, `CONTEXT.md` for vocabulary, and
`docs/adr/` for the decisions and their trade-offs.

---

## Setup

**Python 3.11 or newer is required.** Check before anything else — an older
interpreter fails with a confusing `Could not find a version that satisfies`
error rather than a clear one:

```bash
python3 --version
```

If that prints 3.10 or lower, use an explicit newer interpreter below
(`python3.12`, `python3.13`, …) wherever it says `python3.12`.

```bash
git clone <this repo> && cd distilling-trusted-monitor-diversity

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt      # ~5 min

pytest -q                            # 47 passed, 9 skipped
python main.py build-data            # writes data/splits/
```

That is the whole setup. One requirements file, one virtualenv, no extras to
choose between.

### How one file works on every machine

`requirements.txt` carries environment markers, so pip installs the right thing
per platform without being told which:

| Packages | Marker | Installed on |
|---|---|---|
| datasets, numpy, pandas, scipy, scikit-learn, matplotlib, pytest, ruff | none | everywhere |
| vllm, torch, transformers, trl, peft, bitsandbytes, accelerate | `sys_platform == 'linux'` | the CUDA box |
| mlx, mlx-lm | `sys_platform == 'darwin'` | Apple Silicon |

This is not a style choice: **`vllm` and `bitsandbytes` publish no macOS wheels
at all**, so a flat unmarked list cannot install on a laptop. The Linux-only
marks on torch and friends are for weight rather than possibility — nothing on
a Mac uses them, since the local pilot runs through MLX.

`control-arena` is deliberately absent — only its dataset is used. See
[ADR-0004](docs/adr/0004-static-classification-not-a-live-control-protocol.md).

### GPU box

Identical, plus a served model:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # markers pull in vllm, torch, CUDA wheels

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.85 --max-model-len 8192

# second shell:
python main.py score --arm m0 --split test
```

vLLM is served out of process on purpose: paged attention and continuous
batching make it several times faster than a `transformers` generate loop, and
a crashed scoring script leaves the 15 GB model load standing rather than
paying another cold start on resume.

### GPU run on vast.ai

Three scripts in `infra/vast/` cover the whole round trip:

| Where | Command | Does |
|---|---|---|
| VM | `bash infra/vast/remote_setup.sh` | installs the env, rebuilds and **hash-checks** the splits, starts vLLM in tmux, smoke-tests 4 items |
| VM | `tmux new -s run 'bash infra/vast/run_untrained_arms.sh'` | M1 → M0 (derived) → M2 on val then test, then `analyse` for both |
| laptop | `bash infra/vast/pull_results.sh` | copies `results/`, `data/generations/` and `logs/` back |

The host driver must support CUDA 13 (driver ≥ 580): the pinned torch uses
CUDA 13 libraries, so an offer showing CUDA 12.8 will fail at the first model
load. Search with `cuda_vers>=13.0`.

### Local pilot (Apple Silicon)

The MLX packages are already installed by the same file:

```bash
python main.py score --arm m0 --split val --backend mlx --limit 100
```

First run downloads ~4.5 GB of 4-bit weights. **These scores are not comparable
to bf16 served scores** — the pilot answers "does the prompt parse and is the
model above chance", not "what is the gate number".

### Regenerating `requirements.txt`

It is pinned and generated; edit `pyproject.toml`, then re-export. `pip freeze`
is *not* a substitute — it captures only the current platform and would drop
every Linux marker:

```bash
uv lock && uv export --format requirements-txt --no-hashes \
  --no-emit-project -o requirements.txt
```

uv is needed only for that regeneration step, never to install or run the
project.

## Pipeline

Every stage is independently runnable and resumable — generation jobs run for
hours on interruptible instances and append each result as it completes.

```bash
python main.py build-data                        # splits (no GPU)
python main.py score --arm m0 --split val        # the 14 Sep gate
python main.py score --arm m1 --split test
python main.py score --arm m2 --split test
python main.py diversity-check --split val       # ADR-0002
python main.py teacher-scores --split train      # M4's targets
python main.py debate --split train              # M3's transcripts
python main.py build-pairs
python main.py train-sft                         # M4
python main.py train-dpo                         # M3
python main.py analyse --split test                     # report.md + charts
```

`python main.py --help` lists every stage.

## Artifacts

| Directory | Git | Holds |
|---|---|---|
| `data/` | ignored | splits, raw generations, derived training files |
| `runs/` | ignored | one dir per run: `DATE__arm__githash/` with adapter, logs, config |
| `results/<split>/` | **committed** | `report.md`, `summary.csv`, `comparisons.csv`, `hypotheses.csv`, `figures/*.png` |

`results/` is committed on purpose: it is a few hundred KB and it is what the
write-up cites, so every reported number stays timestamped and attributable to
a commit.
