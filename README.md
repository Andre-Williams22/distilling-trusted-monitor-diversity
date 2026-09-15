# distilling-trusted-monitor-diversity

Can inference-time monitor diversity be distilled into a single trusted monitor
— recovering most of a 3-monitor ensemble's backdoor-detection gain at 1x
inference cost, using no labels and no stronger teacher?

Two distillation methods compared side by side: DPO on MACA-style
consensus-derived preference pairs, and supervised distillation from the
ensemble's mean scores. Evaluated as static classification on the
[ControlArena APPS backdoor dataset](https://huggingface.co/datasets/RoganInglis/apps-control-arena).

See `project-plan.md` for the full design, `CONTEXT.md` for vocabulary, and
`docs/adr/` for the decisions and their trade-offs. Results are committed under
`results/<split>/report.md`.

## Local setup

Requires Python 3.11+.

```bash
git clone https://github.com/Andre-Williams22/distilling-trusted-monitor-diversity.git
cd distilling-trusted-monitor-diversity

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # platform markers pick MLX on macOS, vLLM/torch on Linux

pytest -q
python main.py build-data            # writes data/splits/
```

### Local pilot (Apple Silicon)

```bash
python main.py score --arm m0 --split val --backend mlx --limit 100
```

The first run downloads ~4.5 GB of 4-bit weights. These scores are a sanity
check only and are not comparable to the bf16 GPU scores.

## GPU run on vast.ai

Replace every `UPPERCASE_PLACEHOLDER` with the real value.

### 1. Laptop, once per machine

```bash
uvx vastai set api-key "$(sed -n 's/^fastai_key=//p' .env)"
chmod 600 ~/.config/vastai/vast_api_key
uvx vastai create ssh-key "$(cat ~/.ssh/arena_key.pub)"
```

### 2. Laptop, every session

Log in with 2FA (the API key alone returns 401):

```bash
# Email
uvx vastai tfa send-email
uvx vastai tfa login --method-type email --secret SECRET_FROM_TERMINAL -c CODE_FROM_EMAIL

# Or authenticator app
uvx vastai tfa login --method-type totp -c CODE_FROM_APP

chmod 600 ~/.config/vastai/vast_tfa_key
```

Rent a 48 GB GPU. The driver must support CUDA 13:

```bash
uvx vastai search offers \
  'gpu_name=L40S num_gpus=1 rentable=true reliability>0.99 cuda_vers>=13.0 disk_space>=200 inet_down>200' \
  -o dph                               # RTX_A6000 also works

uvx vastai create instance OFFER_ID \
  --image nvidia/cuda:12.8.1-devel-ubuntu22.04 --disk 200 --ssh --direct \
  --label monitor-distillation
```

Connect once it's running:

```bash
uvx vastai show instances              # wait for STATUS running
uvx vastai ssh-url INSTANCE_ID         # prints ssh://root@HOST:PORT
ssh -i ~/.ssh/arena_key -p PORT root@HOST
```

To rejoin a run later, reconnect the same way, then `tmux switch-client -t run`.

### 3. On the VM

Set up once per instance (~10–20 min). It installs dependencies, builds and
verifies the data splits, starts vLLM in tmux session `vllm`, and runs a smoke
test. It ends with `Ready.`

```bash
curl -fsSL https://raw.githubusercontent.com/Andre-Williams22/distilling-trusted-monitor-diversity/main/infra/vast/remote_setup.sh -o remote_setup.sh
bash remote_setup.sh
```

Score the untrained arms (M0, M1, M2 on val then test, plus `analyse`). The
script resumes after interruption.

```bash
cd ~
curl -fs localhost:8000/health && echo "vLLM ok"
tmux new -d -s run 'bash infra/vast/run_untrained_arms.sh'
tmux switch-client -t run
```

Train and score M4 (SFT distilled from M2):

```bash
tmux new -d -s m4smoke 'bash infra/vast/run_m4_smoke.sh'
# after the smoke test passes:
tmux new -d -s m4full 'bash infra/vast/run_m4_full.sh'
tmux switch-client -t m4full
```

Follow progress with `tail -f ~/logs/*.log`.

### 4. Laptop, after the run

```bash
VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh INSTANCE_ID
```

Only once the pull prints `Pulled and verified. Safe to destroy`:

```bash
uvx vastai destroy instance INSTANCE_ID
uvx vastai show instances              # confirm nothing is left running
```

Then build the report:

```bash
python main.py analyse --split test    # writes results/test/report.md
```

## Pipeline stages

Every stage is resumable. `python main.py --help` lists them all.

```bash
python main.py build-data
python main.py score --arm m0 --split val
python main.py score --arm m1 --split test
python main.py score --arm m2 --split test
python main.py diversity-check --split val
python main.py teacher-scores --split train      # M4 targets
python main.py debate --split train              # M3 transcripts
python main.py build-pairs
python main.py train-sft                         # M4
python main.py train-dpo                         # M3
python main.py analyse --split test
```
