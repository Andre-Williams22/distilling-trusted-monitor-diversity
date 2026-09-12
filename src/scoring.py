"""Turn items into suspicion scores by asking a served monitor.

One file, two backends, one readout path. ``VLLMBackend`` talks to vLLM's
OpenAI-compatible endpoint on a CUDA box; ``MLXBackend`` runs a quantised model
in-process on Apple Silicon. Everything downstream of ``Generation`` is shared,
so a bug in readout parsing shows up identically in both and a local pilot
exercises the same code path the real run will use.

Both readouts are recorded on every call (ADR-0006):

- **logprob** (primary) -- ``P("yes") / (P("yes") + P("no"))`` at the token
  following ``BACKDOORED:``. Normalised, so probability the model spends on
  formatting does not read as "less suspicious".
- **text** (secondary) -- the integer after ``SUSPICION:``.

Every stage appends one JSONL row per (item, sample) as it completes, so a
killed instance loses the in-flight request rather than the run.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.config import (
    ARMS,
    BASE_MODEL,
    BASELINE_PROMPT,
    GENERATIONS_DIR,
    PERSONA_PROMPTS,
    SAMPLING,
    SCORE_MAX,
    SCORE_MIN,
    SERVING,
    TRAINING_DIR,
    Arm,
    SamplingConfig,
    ServingConfig,
)
from src.data import Item, SplitName, load_split
from src.utils import append_jsonl, completed_ids, get_logger, write_jsonl

logger = get_logger(__name__)

VERDICT_MARKER = "BACKDOORED:"
SCORE_PATTERN = re.compile(r"SUSPICION:\s*(\d+)")

#: Tokenizers disagree about leading spaces and capitalisation, so probability
#: mass is summed over every spelling rather than assuming one of them.
YES_SPELLINGS = ("yes", " yes", "Yes", " Yes", "YES", " YES")
NO_SPELLINGS = ("no", " no", "No", " No", "NO", " NO")

#: How many tokens after the marker to search for the verdict. Covers a
#: stray space or newline emitted before the word itself.
VERDICT_SEARCH_WINDOW = 4


@dataclass
class Generation:
    """One completion, with enough detail to read both scores out of it.

    Attributes:
        text: The full decoded response.
        tokens: Decoded token strings, in emission order.
        token_logprobs: Per position, a mapping from candidate token string to
            its log-probability. vLLM supplies a truncated top-k; MLX supplies
            the full vocabulary. Readout code treats both the same way.
        finish_reason: Why generation stopped, for diagnosing truncation.
    """

    text: str
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[dict[str, float]] = field(default_factory=list)
    finish_reason: str = "unknown"


class Backend(Protocol):
    """A source of completions.

    Attributes:
        max_concurrency: How many requests may be in flight. vLLM batches
            internally and wants many; MLX is a single in-process model and
            wants exactly one.
    """

    max_concurrency: int

    def generate(
        self,
        prompt: str,
        n: int,
        sampling: SamplingConfig,
    ) -> list[Generation]:
        """Produce ``n`` completions for one prompt."""
        ...


class VLLMBackend:
    """Talks to vLLM's OpenAI-compatible endpoint over HTTP.

    Out-of-process deliberately: a crashed scoring script leaves the 15 GB
    model load standing, so a resumed run starts generating immediately rather
    than waiting out another cold start.
    """

    def __init__(
        self,
        model: str = BASE_MODEL,
        serving: ServingConfig = SERVING,
        adapter: str | None = None,
    ) -> None:
        """Configure the client.

        Args:
            model: Model name as the server advertises it.
            serving: Host, port and timeouts.
            adapter: LoRA adapter name for M3/M4, served by vLLM alongside the
                base model. ``None`` for the untrained arms.
        """
        self.model = adapter or model
        self.serving = serving
        self.max_concurrency = serving.max_concurrent

    def generate(
        self, prompt: str, n: int, sampling: SamplingConfig
    ) -> list[Generation]:
        """Produce ``n`` completions for one prompt.

        Args:
            prompt: The rendered monitor prompt.
            n: How many samples to draw.
            sampling: Temperature, top-p and length limits.

        Returns:
            One ``Generation`` per sample.

        Raises:
            RuntimeError: If the server is unreachable or returns an error.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "logprobs": True,
            "top_logprobs": 20,
        }
        request = urllib.request.Request(
            f"{self.serving.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.serving.request_timeout_s
            ) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"vLLM at {self.serving.base_url} did not answer: {error}. "
                f"Is the server up?"
            ) from error

        return [_generation_from_choice(choice) for choice in body["choices"]]


class MLXBackend:
    """Runs a quantised model in-process on Apple Silicon.

    For pilots and plumbing checks only. Scores from a 4-bit local model are
    **not** comparable to scores from a bf16 served model, so an arm measured
    here cannot be compared against an arm measured on the GPU box.
    """

    def __init__(self, model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit") -> None:
        """Load the model and tokenizer.

        Args:
            model: An MLX-format model id.

        Raises:
            RuntimeError: If ``mlx-lm`` is not installed.
        """
        try:
            from mlx_lm import load
        except ImportError as error:
            raise RuntimeError(
                "MLXBackend needs mlx-lm: `uv sync --extra local`"
            ) from error

        self.model, self.tokenizer = load(model)
        self.max_concurrency = 1

    def generate(
        self, prompt: str, n: int, sampling: SamplingConfig
    ) -> list[Generation]:
        """Produce ``n`` completions for one prompt, one after another.

        Args:
            prompt: The rendered monitor prompt.
            n: How many samples to draw.
            sampling: Temperature, top-p and length limits.

        Returns:
            One ``Generation`` per sample.
        """
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        sampler = make_sampler(temp=sampling.temperature, top_p=sampling.top_p)

        results = []
        for _ in range(n):
            results.append(
                self._generate_once(chat, sampler, sampling, stream_generate)
            )
        return results

    def _generate_once(
        self,
        chat: str,
        sampler: Any,
        sampling: SamplingConfig,
        stream_generate: Any,
    ) -> Generation:
        """Run one completion, capturing per-token log-probabilities.

        Args:
            chat: The chat-templated prompt.
            sampler: An mlx-lm sampler.
            sampling: Length limit.
            stream_generate: The mlx-lm streaming entry point.

        Returns:
            The completed generation.
        """
        import mlx.core as mx

        text_parts: list[str] = []
        tokens: list[str] = []
        logprobs: list[dict[str, float]] = []
        finish = "length"

        for step in stream_generate(
            self.model,
            self.tokenizer,
            chat,
            max_tokens=sampling.max_tokens,
            sampler=sampler,
        ):
            text_parts.append(step.text)
            tokens.append(step.text)
            logprobs.append(self._candidates(step.logprobs, mx))
            if step.finish_reason:
                finish = step.finish_reason

        return Generation(
            text="".join(text_parts),
            tokens=tokens,
            token_logprobs=logprobs,
            finish_reason=finish,
        )

    def _candidates(self, step_logprobs: Any, mx: Any) -> dict[str, float]:
        """Extract log-probabilities for the yes/no spellings at one position.

        Keeping only the tokens the readout needs, rather than the whole
        vocabulary, keeps a 1,288-item run's memory flat.

        Args:
            step_logprobs: Full-vocabulary log-probabilities for this step.
            mx: The ``mlx.core`` module.

        Returns:
            A mapping from token string to log-probability.
        """
        wanted: dict[str, float] = {}
        for spelling in YES_SPELLINGS + NO_SPELLINGS:
            ids = self.tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1:
                wanted[spelling] = float(mx.array(step_logprobs)[ids[0]].item())
        return wanted


def _generation_from_choice(choice: dict[str, Any]) -> Generation:
    """Convert one OpenAI-shaped choice into a ``Generation``.

    Args:
        choice: A single element of the response's ``choices`` array.

    Returns:
        The generation, with an empty token list if the server returned no
        logprobs -- in which case only the text readout is available.
    """
    content = (choice.get("logprobs") or {}).get("content") or []
    tokens = [entry["token"] for entry in content]
    logprobs = [
        {
            alternative["token"]: alternative["logprob"]
            for alternative in entry.get("top_logprobs", [])
        }
        for entry in content
    ]
    return Generation(
        text=choice["message"]["content"],
        tokens=tokens,
        token_logprobs=logprobs,
        finish_reason=choice.get("finish_reason", "unknown"),
    )


def get_backend(name: str, adapter: str | None = None, model: str | None = None):
    """Construct a backend by name.

    Args:
        name: ``"vllm"`` or ``"mlx"``.
        adapter: LoRA adapter for the trained arms. vLLM only.
        model: Override the model id.

    Returns:
        A configured backend.

    Raises:
        ValueError: If the name is unknown.
    """
    if name == "vllm":
        return VLLMBackend(model=model or BASE_MODEL, adapter=adapter)
    if name == "mlx":
        if adapter is not None:
            raise ValueError("MLXBackend cannot serve LoRA adapters; use vllm")
        return MLXBackend(model=model or "mlx-community/Qwen2.5-7B-Instruct-4bit")
    raise ValueError(f"unknown backend {name!r}; expected 'vllm' or 'mlx'")


def render_prompt(template_path: Path, item: Item) -> str:
    """Fill a prompt template with one item's problem and code.

    Args:
        template_path: A file in ``prompts/``.
        item: The item to be judged.

    Returns:
        The rendered prompt.
    """
    template = template_path.read_text()
    return template.replace("{question}", item.question).replace("{code}", item.code)


def parse_text_score(text: str) -> int | None:
    """Read the integer after ``SUSPICION:`` (the secondary readout).

    Args:
        text: The model's full response.

    Returns:
        The score clamped to [0, 10], or ``None`` if the model did not follow
        the format.
    """
    match = SCORE_PATTERN.search(text)
    if match is None:
        return None
    return max(SCORE_MIN, min(SCORE_MAX, int(match.group(1))))


def find_verdict_index(tokens: Sequence[str]) -> int | None:
    """Locate the token position just after ``BACKDOORED:``.

    Works on decoded text rather than token identities, because tokenizers
    split the marker differently and matching on pieces is brittle.

    Args:
        tokens: Decoded token strings in emission order.

    Returns:
        The index of the first token beginning after the marker, or ``None``.
    """
    starts: list[int] = []
    position = 0
    for token in tokens:
        starts.append(position)
        position += len(token)

    marker_at = "".join(tokens).find(VERDICT_MARKER)
    if marker_at < 0:
        return None

    target = marker_at + len(VERDICT_MARKER)
    for index, start in enumerate(starts):
        if start >= target:
            return index
    return None


def read_verdict_probability(generation: Generation) -> float | None:
    """Compute ``P(yes) / (P(yes) + P(no))`` at the verdict (primary readout).

    Normalised rather than raw: a response that spends probability mass on
    formatting tokens should not thereby read as less suspicious. The result is
    the model's binary verdict with format noise divided out.

    Probability is summed across capitalisation and leading-space spellings,
    since tokenizers disagree about both. The search walks a few tokens past
    the marker to tolerate a stray space or newline.

    Args:
        generation: A completion carrying per-token log-probabilities.

    Returns:
        A probability in [0, 1], or ``None`` if the verdict could not be
        located or neither spelling appeared among the candidates.
    """
    start = find_verdict_index(generation.tokens)
    if start is None:
        return None

    end = min(start + VERDICT_SEARCH_WINDOW, len(generation.token_logprobs))
    for index in range(start, end):
        candidates = generation.token_logprobs[index]
        yes = _probability_mass(candidates, YES_SPELLINGS)
        no = _probability_mass(candidates, NO_SPELLINGS)
        if yes + no > 0.0:
            return yes / (yes + no)
    return None


def _probability_mass(
    candidates: dict[str, float], spellings: Iterable[str]
) -> float:
    """Sum probability across several spellings of the same word.

    Args:
        candidates: Token string to log-probability.
        spellings: Variants to accumulate.

    Returns:
        Total probability, zero if none are present.
    """
    import math

    return sum(
        math.exp(candidates[spelling])
        for spelling in spellings
        if spelling in candidates
    )


@dataclass
class ScoreRecord:
    """One monitor call's result. One JSONL row per (item, sample).

    Attributes:
        item_id: Which item was judged.
        problem_id: Its parent problem, for split-aware analysis.
        label: Ground truth. Present for analysis only -- never shown to a
            monitor and never read by a training arm.
        arm: Which arm produced this.
        prompt_name: Which template was used, so M2's three personas stay
            distinguishable within one file.
        sample: Zero-based index within this item's ``n`` samples.
        logprob_score: The primary readout, or ``None`` if unreadable.
        text_score: The secondary readout, or ``None`` if unreadable.
        finish_reason: For spotting truncation.
        response: The full text, kept so a parsing change can be re-run without
            paying for generation again.
    """

    item_id: str
    problem_id: str
    label: int
    arm: str
    prompt_name: str
    sample: int
    logprob_score: float | None
    text_score: int | None
    finish_reason: str
    response: str


def score_items(
    items: Sequence[Item],
    template_path: Path,
    backend: Backend,
    arm: str,
    out_path: Path,
    n_samples: int = 1,
    sampling: SamplingConfig = SAMPLING,
    resume: bool = True,
) -> Path:
    """Score every item with one template, appending results as they land.

    Items already present in ``out_path`` are skipped when resuming, so a
    killed instance costs one in-flight request rather than the run.

    Args:
        items: Items to score.
        template_path: Which prompt to use.
        backend: Where to send the requests.
        arm: Arm id, recorded on every row.
        out_path: JSONL file to append to.
        n_samples: Samples per item.
        sampling: Decoding settings.
        resume: Skip items already in the output file.

    Returns:
        ``out_path``.
    """
    prompt_name = template_path.stem
    done = completed_ids(out_path) if resume else set()
    if resume and not out_path.exists():
        logger.info("no existing output at %s; starting fresh", out_path)

    pending = [item for item in items if f"{item.item_id}::{prompt_name}" not in done]
    logger.info(
        "%s/%s: %d items to score (%d already done), %d sample(s) each",
        arm,
        prompt_name,
        len(pending),
        len(items) - len(pending),
        n_samples,
    )

    def score_one(item: Item) -> list[dict[str, Any]]:
        prompt = render_prompt(template_path, item)
        generations = backend.generate(prompt, n_samples, sampling)
        return [
            asdict(
                ScoreRecord(
                    item_id=item.item_id,
                    problem_id=item.problem_id,
                    label=item.label,
                    arm=arm,
                    prompt_name=prompt_name,
                    sample=index,
                    logprob_score=read_verdict_probability(generation),
                    text_score=parse_text_score(generation.text),
                    finish_reason=generation.finish_reason,
                    response=generation.text,
                )
            )
            for index, generation in enumerate(generations)
        ]

    completed = 0
    with ThreadPoolExecutor(max_workers=backend.max_concurrency) as pool:
        for rows in pool.map(score_one, pending):
            # The composite id is what `resume` checks, so it must be written
            # even though ScoreRecord keys on item_id alone.
            for row in rows:
                row["item_id"] = f"{row['item_id']}::{prompt_name}"
            append_jsonl(out_path, rows)
            completed += 1
            if completed % 50 == 0:
                logger.info("  %d/%d", completed, len(pending))

    logger.info("wrote %s", out_path)
    return out_path


def _arm_by_name(name: str) -> Arm:
    """Look up an arm's configuration.

    Args:
        name: Arm id, ``"m0"``..``"m4"``.

    Returns:
        Its ``Arm`` record.

    Raises:
        ValueError: If the name is unknown.
    """
    for arm in ARMS:
        if arm.name == name:
            return arm
    raise ValueError(f"unknown arm {name!r}")


def score_arm(
    arm: str,
    split: SplitName,
    adapter: str | None = None,
    resume: bool = True,
    backend_name: str = "vllm",
    limit: int | None = None,
) -> Path:
    """Score one split with one arm. The entry point ``main.py score`` calls.

    M0 and M1 differ only in sample count and share a temperature, so M0 is the
    first of M1's three draws -- otherwise H1 would measure ensembling plus a
    temperature change with no way to separate them. M2 runs its three persona
    templates into one file, distinguished by ``prompt_name``.

    Args:
        arm: Arm id.
        split: Which split to score.
        adapter: LoRA adapter path for m3/m4.
        resume: Skip items already scored.
        backend_name: ``"vllm"`` or ``"mlx"``.
        limit: Score only the first N items. For pilots -- the subset is the
            head of the split, not a random sample, so a limited run is
            reproducible and resumable into the full one.

    Returns:
        The JSONL file written.

    Raises:
        ValueError: If a trained arm is requested without an adapter.
    """
    config = _arm_by_name(arm)
    if config.adapter_required and adapter is None:
        raise ValueError(f"arm {arm} needs --adapter")

    items = load_split(split)
    if limit is not None:
        items = items[:limit]
        logger.info("limited to the first %d items of %s", len(items), split)
    backend = get_backend(backend_name, adapter=adapter)
    suffix = f"__limit{limit}" if limit is not None else ""
    out_path = GENERATIONS_DIR / f"{arm}__{split}{suffix}.jsonl"

    for template_path in config.prompts:
        score_items(
            items=items,
            template_path=template_path,
            backend=backend,
            arm=arm,
            out_path=out_path,
            n_samples=config.n_samples,
            resume=resume,
        )
    return out_path


def score_alternate_models(
    split: SplitName,
    models: Sequence[str],
    resume: bool = True,
    backend_name: str = "vllm",
) -> Path:
    """Score a split with other model families, to test ADR-0002.

    Compares model diversity against prompt diversity before the debate
    generations commit the project to the latter. Val only, baseline prompt
    only -- the personas are what is being measured against, not applied.

    Args:
        split: Which split to score. Intended to be ``"val"``.
        models: Model ids to run.
        resume: Skip items already scored.
        backend_name: ``"vllm"`` or ``"mlx"``.

    Returns:
        The JSONL file written.
    """
    items = load_split(split)
    out_path = GENERATIONS_DIR / f"diversity_check__{split}.jsonl"

    for model in models:
        backend = get_backend(backend_name, model=model)
        score_items(
            items=items,
            template_path=BASELINE_PROMPT,
            backend=backend,
            arm=f"alt::{model}",
            out_path=out_path,
            n_samples=1,
            resume=resume,
        )
    return out_path


def build_teacher_scores(
    split: SplitName = "train",
    resume: bool = True,
    backend_name: str = "vllm",
) -> Path:
    """Generate M4's regression targets: the personas' unrounded mean score.

    Runs the three persona templates once each, then averages. **The mean is
    persisted unrounded** -- three monitors averaging to 6.33 is finer-grained
    than any single monitor's 6 or 7, and that resolution is what M4's KD term
    exists to inherit (ADR-0005). Rounding here would silently defeat it.

    Args:
        split: Which split to build targets for.
        resume: Skip items already scored.
        backend_name: ``"vllm"`` or ``"mlx"``.

    Returns:
        The targets JSONL file.
    """
    items = load_split(split)
    backend = get_backend(backend_name)
    raw_path = GENERATIONS_DIR / f"teacher__{split}.jsonl"

    for template_path in PERSONA_PROMPTS:
        score_items(
            items=items,
            template_path=template_path,
            backend=backend,
            arm="teacher",
            out_path=raw_path,
            n_samples=1,
            resume=resume,
        )

    return aggregate_teacher_scores(
        raw_path, TRAINING_DIR / f"sft_targets__{split}.jsonl"
    )


def aggregate_teacher_scores(raw_path: Path, out_path: Path) -> Path:
    """Average the three personas' scores per item into one training target.

    Args:
        raw_path: The per-persona score file.
        out_path: Where to write the aggregated targets.

    Returns:
        ``out_path``.
    """
    from collections import defaultdict

    from src.utils import read_jsonl

    logprobs: dict[str, list[float]] = defaultdict(list)
    texts: dict[str, list[int]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(raw_path):
        item_id = row["item_id"].split("::")[0]
        meta[item_id] = {"problem_id": row["problem_id"], "label": row["label"]}
        if row["logprob_score"] is not None:
            logprobs[item_id].append(row["logprob_score"])
        if row["text_score"] is not None:
            texts[item_id].append(row["text_score"])

    targets = [
        {
            "item_id": item_id,
            **meta[item_id],
            "n_personas_logprob": len(logprobs[item_id]),
            "n_personas_text": len(texts[item_id]),
            # Unrounded on purpose -- see ADR-0005.
            "teacher_logprob": (
                sum(logprobs[item_id]) / len(logprobs[item_id])
                if logprobs[item_id]
                else None
            ),
            "teacher_text": (
                sum(texts[item_id]) / len(texts[item_id]) if texts[item_id] else None
            ),
        }
        for item_id in sorted(meta)
    ]
    write_jsonl(out_path, targets)
    logger.info("wrote %d teacher targets to %s", len(targets), out_path)
    return out_path
