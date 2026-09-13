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

The full round trip, split by where each command runs. Replace every
`UPPERCASE_PLACEHOLDER` with the real value. **Don't type angle brackets:**
zsh reads `<` as a file redirect and fails with `parse error near '\n'`.

`vastai` is not installed globally; every command runs it through `uvx`, which
fetches it on demand.

#### 1. On the laptop, once per machine

```bash
# API key. This repo's .env stores it as fastai_key.
uvx vastai set api-key "$(sed -n 's/^fastai_key=//p' .env)"
chmod 600 ~/.config/vastai/vast_api_key

# Register the SSH public key the VM will accept.
uvx vastai create ssh-key "$(cat ~/.ssh/arena_key.pub)"
```

#### 2. On the laptop, at the start of every session

**Log in with 2FA.** The account requires it, and the API key alone returns
`401 ... requires you to have logged in using Two Factor Authentication`.
Use whichever method the account was set up with:

```bash
# Email: the first command prints a secret, and the code arrives by email.
uvx vastai tfa send-email
uvx vastai tfa login --method-type email --secret SECRET_FROM_TERMINAL -c CODE_FROM_EMAIL

# Authenticator app: the 6-digit code shown for Vast.ai
uvx vastai tfa login --method-type totp -c CODE_FROM_APP

chmod 600 ~/.config/vastai/vast_tfa_key
```

**Rent a GPU.** Search for 48 GB cards whose driver supports CUDA 13:

```bash
uvx vastai search offers \
  'gpu_name=L40S num_gpus=1 rentable=true reliability>0.99 cuda_vers>=13.0 disk_space>=200 inet_down>200' \
  -o dph
# RTX_A6000 also works; use gpu_name=RTX_A6000.

uvx vastai create instance OFFER_ID \
  --image nvidia/cuda:12.8.1-devel-ubuntu22.04 --disk 200 --ssh --direct \
  --label monitor-distillation
```

> **CUDA 13 is required.** The pinned torch ships CUDA 13 libraries, which need
> driver ≥ 580. An offer listing CUDA 12.8 rents fine and then fails at the
> first model load.

**Wait for it to boot, then get its address:**

```bash
uvx vastai show instances              # wait until STATUS says running (a few minutes)
uvx vastai ssh-url INSTANCE_ID         # prints ssh://root@HOST:PORT
```

**Connect.** Use the host and port from `ssh-url`, which is the direct address.
The `ssh3.vast.ai` proxy address listed in `show instances` can reject the key
for the first few minutes.

```bash
ssh -i ~/.ssh/arena_key -p PORT root@HOST
```

#### Reconnecting to a VM that already exists

If the instance is already rented, skip renting and just reconnect. Log in
with 2FA if the session has expired, find your instance's address, and SSH in.
The `tmux attach` puts you back inside a run that was left going.

```bash
uvx vastai show instances                      # note the ID of the running instance
uvx vastai ssh-url INSTANCE_ID                 # prints ssh://root@HOST:PORT
ssh -i ~/.ssh/arena_key -p PORT root@HOST      # connect
tmux ls                                        # on the VM: list sessions (run, vllm)
tmux switch-client -t run                      # rejoin the scoring run (you land inside vast's auto-tmux)
```

#### 3. On the VM

**Set up once per instance.** This installs everything, verifies the data and
starts the model server. It takes roughly 10–20 minutes, mostly downloading
dependencies and the ~15 GB model.

```bash
curl -fsSL https://raw.githubusercontent.com/Andre-Williams22/distilling-trusted-monitor-diversity/main/infra/vast/remote_setup.sh -o remote_setup.sh
bash remote_setup.sh
```

What `infra/vast/remote_setup.sh` does, in order:

| Step | Command | Why |
|---|---|---|
| System packages | `apt-get install tmux git curl rsync build-essential ninja-build` | tmux keeps runs alive after SSH drops; rsync brings results home; **gcc is required** because vLLM compiles GPU kernels on first start |
| Installer | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | uv, used only as a faster pip |
| Code | `git init` + `git checkout origin/main` directly in `~` | the code lives in the home directory, not a subfolder; it's the pushed `main`, so push before renting |
| Environment | `uv venv --python 3.12 .venv` then `uv pip install -r requirements.txt` | the same file as the laptop; Linux markers pull in vllm, torch and CUDA 13 |
| Data | `python main.py build-data`, then SHA-256 checks | stops if the splits differ by a single byte from the laptop's |
| Model server | `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20` in tmux session `vllm` | waits until `localhost:8000/health` answers |
| Smoke test | `python main.py score --arm m0 --split val --limit 4` | stops unless all 4 items produce both readouts |

It ends with `Ready.` Anything else means a step failed; the message says which.

**Start the real run** in its own tmux session, so it survives a dropped
connection. vast.ai already puts you inside tmux when you log in (the
"auto-tmux" banner), so create the session detached with `-d`; a plain
`tmux new` fails with `sessions should be nested with care`.

```bash
cd ~
curl -fs localhost:8000/health && echo "vLLM ok"               # server must be up
tmux new -d -s run 'bash infra/vast/run_untrained_arms.sh'     # start in the background
tmux switch-client -t run                                      # jump in to watch
```

`infra/vast/run_untrained_arms.sh` scores M1, derives M0 from M1's first
sample, and scores M2, on val first and then test (test thresholds come from
val). It then runs `analyse` for both splits. Every step resumes, so after an
interruption the same command finishes only the remaining items.

**While it runs:**

| To | Do |
|---|---|
| Leave it running and disconnect | `exit`; the `run` session keeps going |
| Come back to it | `ssh -i ~/.ssh/arena_key -p PORT root@HOST`, then `tmux switch-client -t run` |
| Go back to your login shell | `Ctrl-b` then `s`, and pick the other session |
| Watch progress without attaching | `tail -f ~/logs/run_untrained_*.log` (prints s/item and time left every 50 items) |
| Check the model server | `tmux switch-client -t vllm`, or `curl localhost:8000/health` |
| Check the GPU | `nvidia-smi` |

It's finished when the log's last line reads
`Done. From the laptop: bash infra/vast/pull_results.sh`.

#### 4. Back on the laptop

```bash
# Bring results/, data/generations/ and logs/ home
VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh INSTANCE_ID

```

Only when the pull prints **`Pulled and verified. Safe to destroy`**, destroy
the instance. Run it as its own command, after reading the pull output:
destroying deletes the VM's disk, and nothing on it can be recovered.

```bash
# Stop paying. vast.ai keeps charging for a stopped instance's disk, so destroy it.
uvx vastai destroy instance INSTANCE_ID
uvx vastai show instances              # confirm nothing is left running
```

Then open `results/test/report.md`.

#### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `zsh: parse error near '\n'` | Angle brackets were typed literally. Use the real value with no `<` `>` |
| `zsh: command not found: vastai` | Run it as `uvx vastai …` |
| `401 … Two Factor Authentication` | The 2FA session is missing or expired; repeat the login in step 2 |
| `Permission denied (publickey)` | Use the direct address from `uvx vastai ssh-url`, pass `-i ~/.ssh/arena_key`, or wait a minute for the key to reach a new instance |
| `CUDA driver version is insufficient` | The offer's driver is older than CUDA 13. Destroy it and rent one with `cuda_vers>=13.0` |
| `… already holds scores from mlx, but this run uses vllm` | A 4-bit laptop file is in the way. Move it to `data/generations/mlx_pilot/` |
| `Failed to find C compiler` in `logs/vllm.log` | The image has no gcc. `apt-get install -y build-essential`, then re-run `bash remote_setup.sh` |
| `No such file or directory: 'ninja'` from `flashinfer/jit` | FlashInfer's sampler tries to compile CUDA kernels. Start vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`, as `remote_setup.sh` now does |
| `sessions should be nested with care, unset $TMUX to force` | vast.ai's auto-tmux is already running. Start with `tmux new -d -s run …` and join with `tmux switch-client -t run` |
| `rsync: unrecognized option '--info=stats1'` | An old copy of `pull_results.sh`; macOS's openrsync rejects GNU options. Update the script. **Don't destroy the instance until the pull verifies** |
| `SPLIT MISMATCH` during setup | The dataset or split code changed. Don't score anything until the splits match |

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
