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

Every stage writes one record per (item, sample) to a ``.json`` array in
``data/generations/``, rewriting the file atomically after each item. A killed
run keeps every finished item and resumes from there.
"""

from __future__ import annotations

import json
import math
import re
import time
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
from src.utils import (
    format_duration,
    get_logger,
    read_json_records,
    write_json_records,
    write_jsonl,
)

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


#: A single user prompt, or a whole conversation (debate round 2).
Prompt = str | list[dict[str, str]]


def as_messages(prompt: Prompt) -> list[dict[str, str]]:
    """Turn a prompt into chat messages.

    Args:
        prompt: A single user prompt, or a conversation already in chat form.

    Returns:
        Chat messages.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


class Backend(Protocol):
    """A source of completions.

    Attributes:
        max_concurrency: How many requests may be in flight. vLLM batches
            internally and wants many; MLX is a single in-process model and
            wants exactly one.
    """

    max_concurrency: int
    name: str
    model_id: str

    def generate(
        self,
        prompt: Prompt,
        n: int,
        sampling: SamplingConfig,
    ) -> list[Generation]:
        """Produce ``n`` completions for one prompt."""
        ...


class PromptTooLongError(RuntimeError):
    """The prompt plus the answer budget exceeds the model's context window.

    Raised instead of a generic failure so a single over-long item is recorded
    as unreadable and skipped, rather than stopping a whole scoring run.
    """


#: Phrases vLLM uses when a request exceeds ``--max-model-len``.
CONTEXT_LENGTH_MARKERS = (
    "maximum context length",
    "longer than the maximum model length",
    "max_model_len",
    "too long",
)


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
            adapter: LoRA adapter name for M3, M4 and M5, served by vLLM alongside the
                base model. ``None`` for the untrained arms.
        """
        self.model = adapter or model
        self.serving = serving
        self.max_concurrency = serving.max_concurrent
        self.name = "vllm"
        self.model_id = self.model

    def generate(
        self, prompt: Prompt, n: int, sampling: SamplingConfig
    ) -> list[Generation]:
        """Produce ``n`` completions for one prompt.

        Args:
            prompt: The rendered monitor prompt, or a whole conversation.
            n: How many samples to draw.
            sampling: Temperature, top-p and length limits.

        Returns:
            One ``Generation`` per sample.

        Raises:
            RuntimeError: If the server is unreachable or returns an error.
        """
        payload = {
            "model": self.model,
            "messages": as_messages(prompt),
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
        except urllib.error.HTTPError as error:
            # The server answered, so "is it up?" is the wrong question. Surface
            # vLLM's own reason, and single out over-long prompts.
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code == 400 and any(
                marker in detail.lower() for marker in CONTEXT_LENGTH_MARKERS
            ):
                raise PromptTooLongError(detail) from error
            raise RuntimeError(
                f"vLLM rejected the request (HTTP {error.code}): {detail}"
            ) from error
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
        self.name = "mlx"
        self.model_id = model

    def generate(
        self, prompt: Prompt, n: int, sampling: SamplingConfig
    ) -> list[Generation]:
        """Produce ``n`` completions for one prompt, one after another.

        Args:
            prompt: The rendered monitor prompt, or a whole conversation.
            n: How many samples to draw.
            sampling: Temperature, top-p and length limits.

        Returns:
            One ``Generation`` per sample.
        """
        # Imported here, never at module level: MLX exists only on Apple
        # Silicon, and a top-level import makes this whole module unimportable
        # on the Linux GPU box even when the vLLM backend is the one in use.
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        chat = self.tokenizer.apply_chat_template(
            as_messages(prompt),
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
        backend: ``"mlx"`` or ``"vllm"``. Reports print it, because 4-bit local
            scores and bf16 served scores are not comparable.
        model: The model id that produced the score.
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
    backend: str
    model: str


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
    """Score every item with one template, saving after each item.

    The output is a JSON array. It cannot be appended to, so the whole file is
    rewritten after every item -- atomically, so a run killed mid-write leaves
    the last complete file in place. At ~2.6 s per generation the rewrite is
    negligible.

    One file can hold several templates (M2 writes its three personas to one
    file), so rows are always loaded and kept for *other* templates. With
    ``resume=False`` only this template's rows are dropped and regenerated.

    Args:
        items: Items to score.
        template_path: Which prompt to use.
        backend: Where to send the requests.
        arm: Arm id, recorded on every row.
        out_path: ``.json`` file holding the records.
        n_samples: Samples per item.
        sampling: Decoding settings.
        resume: Skip items this template has already scored.

    Returns:
        ``out_path``.
    """
    prompt_name = template_path.stem
    records = read_json_records(out_path)
    _refuse_mixed_backends(records, backend, out_path)
    if not resume:
        records = [row for row in records if row.get("prompt_name") != prompt_name]

    done = {row["item_id"] for row in records if row.get("prompt_name") == prompt_name}
    pending = [item for item in items if _record_id(item, prompt_name) not in done]
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
        try:
            generations = backend.generate(prompt, n_samples, sampling)
        except PromptTooLongError:
            # Recorded, not skipped: the item counts as done so a resume does
            # not retry it forever, and both readouts are None so it can never
            # enter a metric or a training target as a made-up score.
            logger.warning(
                "%s/%s: prompt exceeds the context window; recorded as unreadable",
                item.item_id, prompt_name,
            )
            generations = [
                Generation(text="", finish_reason="prompt_too_long")
                for _ in range(n_samples)
            ]
        return [
            asdict(
                ScoreRecord(
                    item_id=_record_id(item, prompt_name),
                    problem_id=item.problem_id,
                    label=item.label,
                    arm=arm,
                    prompt_name=prompt_name,
                    sample=index,
                    logprob_score=read_verdict_probability(generation),
                    text_score=parse_text_score(generation.text),
                    finish_reason=generation.finish_reason,
                    response=generation.text,
                    backend=getattr(backend, "name", "unknown"),
                    model=getattr(backend, "model_id", "unknown"),
                )
            )
            for index, generation in enumerate(generations)
        ]

    started = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=backend.max_concurrency) as pool:
        for rows in pool.map(score_one, pending):
            # All samples for an item land in one write, so a kill can never
            # leave an item half-scored and then skip it on resume.
            records.extend(rows)
            write_json_records(out_path, records)
            completed += 1
            if completed % 50 == 0:
                _log_progress(completed, len(pending), time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    per_item = elapsed / completed if completed else 0.0
    logger.info(
        "%s/%s: scored %d items in %s (%.2f s/item) -> %s",
        arm,
        prompt_name,
        completed,
        format_duration(elapsed),
        per_item,
        out_path,
    )
    return out_path


def _refuse_mixed_backends(
    records: Sequence[dict[str, Any]], backend: Backend, out_path: Path
) -> None:
    """Stop a run from adding scores to a file made by a different backend.

    A pilot on the Mac and the real run on the GPU write to the same file name.
    Without this check, the GPU run would resume from the pilot file, skip
    every item already scored in 4-bit, and silently produce a mixed file whose
    numbers match neither setup.

    Args:
        records: What the output file already holds.
        backend: The backend about to score.
        out_path: The output file, named in the error.

    Raises:
        ValueError: If the file holds scores from another backend.
    """
    current = getattr(backend, "name", None)
    existing = {row["backend"] for row in records if row.get("backend")}
    if current is None or not existing or existing == {current}:
        return
    tag = "_".join(sorted(existing))
    pilot_name = out_path.with_name(f"{out_path.stem}__{tag}.json")
    raise ValueError(
        f"{out_path.name} already holds scores from {', '.join(sorted(existing))}, "
        f"but this run uses {current}. Scores from different backends are not "
        f"comparable and must not share a file. Move the old file aside first:\n"
        f"  mv {out_path} {pilot_name}"
    )


def _record_id(item: Item, prompt_name: str) -> str:
    """Build the id a record is stored and resumed under.

    Includes the template name, so M2's three personas scoring the same item
    are three distinct records rather than one overwriting another.

    Args:
        item: The item scored.
        prompt_name: The template's file stem.

    Returns:
        ``"<item_id>::<prompt_name>"``.
    """
    return f"{item.item_id}::{prompt_name}"


def _log_progress(completed: int, total: int, elapsed: float) -> None:
    """Log progress with a rate and a rough time remaining.

    Args:
        completed: Items finished so far.
        total: Items in this run.
        elapsed: Seconds since the run started.
    """
    per_item = elapsed / completed
    remaining = per_item * (total - completed)
    logger.info(
        "  %d/%d · %s elapsed · %.2f s/item · ~%s left",
        completed,
        total,
        format_duration(elapsed),
        per_item,
        format_duration(remaining),
    )


def _arm_by_name(name: str) -> Arm:
    """Look up an arm's configuration.

    Args:
        name: Arm id, ``"m0"``..``"m5"``.

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
        adapter: LoRA adapter name for m3, m4 and m5.
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
    out_path = GENERATIONS_DIR / f"{arm}__{split}{suffix}.json"

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


def derive_m0_from_m1(
    split: SplitName,
    limit: int | None = None,
    directory: Path = GENERATIONS_DIR,
) -> Path:
    """Build M0's scores from the first of M1's three samples.

    This is the preregistered design, not a shortcut (``SamplingConfig``): M0
    and M1 use the same prompt and temperature, so M0 is exactly one draw of
    M1. Taking it from M1 keeps the two arms paired item for item and saves
    re-scoring the whole split.

    Every derived row records ``derived_from`` so the report's provenance is
    visible in the data.

    Args:
        split: Which split's M1 scores to read.
        limit: The ``--limit`` the M1 run used, if any.
        directory: Where generation files live.

    Returns:
        The M0 file written.

    Raises:
        FileNotFoundError: If M1 has not been scored for this slice.
        ValueError: If an independently scored M0 file already exists, which
            this must never overwrite.
    """
    suffix = f"__limit{limit}" if limit is not None else ""
    source = directory / f"m1__{split}{suffix}.json"
    target = directory / f"m0__{split}{suffix}.json"

    rows = read_json_records(source)
    if not rows:
        raise FileNotFoundError(
            f"{source} not found; run `python main.py score --arm m1 "
            f"--split {split}` first"
        )

    existing = read_json_records(target)
    if existing and any("derived_from" not in row for row in existing):
        raise ValueError(
            f"{target} holds an independently scored M0; refusing to overwrite it"
        )

    derived = [
        {**row, "arm": "m0", "derived_from": "m1 sample 0"}
        for row in rows
        if row["sample"] == 0
    ]
    write_json_records(target, derived)
    logger.info("derived %d M0 records from %s -> %s", len(derived), source, target)
    return target


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
    out_path = GENERATIONS_DIR / f"diversity_check__{split}.json"

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
    raw_path = GENERATIONS_DIR / f"teacher__{split}.json"

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
        raw_path: The per-persona ``.json`` score file.
        out_path: Where to write the aggregated targets.

    Returns:
        ``out_path``.
    """
    from collections import defaultdict

    logprobs: dict[str, list[float]] = defaultdict(list)
    texts: dict[str, list[int]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}

    for row in read_json_records(raw_path):
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
