"""Rampart on-device NER model loader and inference.

Ported from the reference Python implementation in ``arcjet-sensitive-info-rampart``
(https://pypi.org/project/arcjet-sensitive-info-rampart/), adapted to work
as a Hermes plugin without the ``arcjet`` SDK dependency.

The model is a 14.7 MB ONNX token-classification model (MiniLM-L6-H384
fine-tuned with a 35-label BIO head). It runs entirely on-device via
``onnxruntime``; no data leaves the host.

Falls back gracefully — if dependencies are unavailable or the model
download fails, ``detect`` returns an empty list and the plugin continues
with regex-only detection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

RAMPART_ENTITIES = [
    "GIVEN_NAME", "SURNAME", "EMAIL", "PHONE_NUMBER", "URL",
    "TAX_ID", "BANK_ACCOUNT", "ROUTING_NUMBER", "GOVERNMENT_ID",
    "PASSPORT", "DRIVERS_LICENSE", "BUILDING_NUMBER", "STREET_NAME",
    "SECONDARY_ADDRESS", "CITY", "STATE", "ZIP_CODE",
]

DEFAULT_KEEP = {"CITY", "STATE", "ZIP_CODE"}

_ENTITY_TO_TYPE = {
    "GIVEN_NAME": "NAME", "SURNAME": "NAME", "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE", "URL": "URL", "TAX_ID": "TAXID",
    "BANK_ACCOUNT": "BANK", "ROUTING_NUMBER": "ROUTING",
    "GOVERNMENT_ID": "GOVID", "PASSPORT": "PASSPORT",
    "DRIVERS_LICENSE": "DL", "BUILDING_NUMBER": "ADDR",
    "STREET_NAME": "ADDR", "SECONDARY_ADDRESS": "ADDR",
    "CITY": "CITY", "STATE": "STATE", "ZIP_CODE": "ZIP",
}

# ---------------------------------------------------------------------------
# Model file layout on HuggingFace
# ---------------------------------------------------------------------------

_MODEL_ID = "nationaldesignstudio/rampart"
_MODEL_FILES = [
    "onnx/model_q4.onnx",
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
]

# ---------------------------------------------------------------------------
# Tokenization constants
# ---------------------------------------------------------------------------

MAX_SEQUENCE_TOKENS = 512
MAX_INPUT_CHARS = 480
CHUNK_OVERLAP = 64
CHUNK_OVERLAP_TOKENS = 64
DEFAULT_THRESHOLD = 0.5

_BIO_PREFIX = re.compile(r"^[BI]-", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RawToken:
    """A single token classified by the model, with character offsets."""
    entity: str
    score: float
    start: int
    end: int


@dataclass(slots=True)
class DetectedSpan:
    """A detected PII span in text."""
    start: int
    end: int
    type: str


# ---------------------------------------------------------------------------
# Pure functions (ported from the reference)
# ---------------------------------------------------------------------------


def _normalize_label(label: str) -> Optional[str]:
    """Normalize a raw model label to our placeholder type tag."""
    stripped = _BIO_PREFIX.sub("", label).upper()
    if stripped == "O" or stripped == "":
        return None
    return _ENTITY_TO_TYPE.get(stripped)


def _is_whitespace(value: str) -> bool:
    return value == "" or value.isspace()


def aggregate_tokens(
    value: str,
    tokens: Sequence[RawToken],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[DetectedSpan]:
    """Aggregate per-token model output into entity spans."""
    spans: list[DetectedSpan] = []
    current: DetectedSpan | None = None

    for token in tokens:
        entity_type = _normalize_label(token.entity)
        if entity_type is None or token.score < threshold:
            if current is not None:
                spans.append(current)
                current = None
            continue

        is_begin = token.entity[:2].lower() == "b-"
        if (
            current is not None
            and current.type == entity_type
            and not is_begin
            and _is_whitespace(value[current.end : token.start])
        ):
            current.end = token.end
            continue

        if current is not None:
            spans.append(current)
        current = DetectedSpan(start=token.start, end=token.end, type=entity_type)

    if current is not None:
        spans.append(current)
    return spans


def _merge_windowed_spans(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    """Union overlapping same-type spans from overlapping windows."""
    if not spans:
        return spans
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[DetectedSpan] = []
    running: dict[str, DetectedSpan] = {}
    for span in ordered:
        current = running.get(span.type)
        if current is not None and span.start < current.end:
            if span.end > current.end:
                current.end = span.end
        else:
            new = DetectedSpan(start=span.start, end=span.end, type=span.type)
            merged.append(new)
            running[span.type] = new
    return merged


def _plan_windows(
    word_ids: Sequence[Optional[int]], budget: int, overlap: int
) -> list[tuple[int, int]]:
    """Split a token sequence into overlapping [start, end) windows."""
    count = len(word_ids)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + budget, count)
        windows.append((start, end))
        if end == count:
            break
        next_start = end - overlap
        while (
            next_start - 1 > start
            and word_ids[next_start] == word_ids[next_start - 1]
        ):
            next_start -= 1
        start = next_start
    return windows


# ---------------------------------------------------------------------------
# Model loader (lazy, cached, thread-safe)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, "_LoadedModel"] = {}
_LOAD_LOCK = threading.Lock()


class _LoadedModel:
    """A loaded ONNX session, tokenizer, and label map."""

    def __init__(
        self, model_path: str, providers: Sequence[str]
    ) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        self.id2label: dict[int, str] = {
            int(k): v for k, v in config["id2label"].items()
        }
        # tokenizer.json is what Tokenizer.from_file expects (NOT vocab.txt)
        self.tokenizer = Tokenizer.from_file(
            os.path.join(model_path, "tokenizer.json")
        )
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.max_tokens: int = config.get(
            "max_position_embeddings", MAX_SEQUENCE_TOKENS
        )
        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")
        if cls_id is None or sep_id is None:
            raise ValueError("Rampart tokenizer is missing [CLS] or [SEP]")
        self.cls_id: int = cls_id
        self.sep_id: int = sep_id
        self.window_budget = self.max_tokens - self.tokenizer.num_special_tokens_to_add(
            False
        )
        if self.window_budget < 1:
            raise ValueError(
                f"Rampart model allows {self.max_tokens} positions, "
                "which leaves no room for input tokens beside [CLS] and [SEP]"
            )
        self.session = onnxruntime.InferenceSession(
            os.path.join(model_path, "onnx", "model_q4.onnx"),
            providers=list(providers),
        )
        self.input_names = {i.name for i in self.session.get_inputs()}


def _load_model(model_path: str, providers: Sequence[str]) -> "_LoadedModel":
    """Load (or return a cached) model."""
    key = json.dumps({"model_path": model_path, "providers": list(providers)})
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        loaded = _LoadedModel(model_path, providers)
        _MODEL_CACHE[key] = loaded
        return loaded


def _classify_window(
    model: _LoadedModel,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> list[RawToken]:
    """Classify tokens [start, end) of a tokenized chunk into raw tokens."""
    import numpy as np

    ids = [model.cls_id, *token_ids[start:end], model.sep_id]
    if len(ids) > model.max_tokens:
        raise ValueError(
            f"Rampart window of {len(ids)} tokens exceeds the model limit of "
            f"{model.max_tokens}"
        )

    feed: dict[str, Any] = {}
    if "input_ids" in model.input_names:
        feed["input_ids"] = np.array([ids], dtype=np.int64)
    if "attention_mask" in model.input_names:
        feed["attention_mask"] = np.ones((1, len(ids)), dtype=np.int64)
    if "token_type_ids" in model.input_names:
        feed["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)

    outputs = model.session.run(None, feed)
    logits = np.asarray(outputs[0])[0]  # [seq, num_labels]
    # Softmax for per-token confidence scores
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    label_ids = probs.argmax(axis=-1)
    scores = probs.max(axis=-1)

    tokens: list[RawToken] = []
    for i, (token_start, token_end) in enumerate(offsets[start:end], 1):
        if token_end <= token_start:
            continue
        tokens.append(
            RawToken(
                entity=model.id2label.get(int(label_ids[i]), "O"),
                score=float(scores[i]),
                start=int(token_start),
                end=int(token_end),
            )
        )
    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RampartEngine:
    """Lazy-loading wrapper around the Rampart ONNX NER model."""

    def __init__(
        self,
        cache_dir: Path,
        threshold: float = DEFAULT_THRESHOLD,
        keep_labels: Optional[set[str]] = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._threshold = threshold
        self._keep = keep_labels if keep_labels is not None else set(DEFAULT_KEEP)
        self._loaded = False
        self._load_failed = False

    def _download_model(self) -> Path:
        """Download model files from HuggingFace Hub if not cached."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self._cache_dir / "rampart"
        model_dir.mkdir(parents=True, exist_ok=True)

        if all((model_dir / f).exists() for f in _MODEL_FILES):
            return model_dir

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("huggingface-hub not installed")

        import shutil

        for fname in _MODEL_FILES:
            local = model_dir / fname
            if local.exists():
                continue
            logger.info("Downloading Rampart model file: %s", fname)
            downloaded = hf_hub_download(
                repo_id=_MODEL_ID,
                filename=fname,
                cache_dir=str(self._cache_dir / "hf_cache"),
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, local)

        return model_dir

    def _load(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        try:
            model_dir = self._download_model()
            self._model = _load_model(
                str(model_dir),
                ("CPUExecutionProvider",),
            )
            self._loaded = True
            logger.info("Rampart NER model loaded (%s)", model_dir)
            return True
        except Exception as exc:
            logger.warning("Failed to load Rampart model: %s; regex-only mode", exc)
            self._load_failed = True
            return False

    def detect(self, text: str) -> list[tuple[str, str, int, int]]:
        """Detect PII spans in text.

        Returns list of (matched_text, type_tag, start, end).
        Returns empty list if model is not loaded or detection fails.
        """
        if not text or not text.strip():
            return []

        if not self._load():
            return []

        try:
            return self._run_model(text)
        except Exception as exc:
            logger.warning("Rampart inference error: %s", exc)
            return []

    def _run_model(
        self, value: str
    ) -> list[tuple[str, str, int, int]]:
        """Run the full detection pipeline (windowing + inference + aggregation)."""
        model = self._model
        threshold = self._threshold
        budget = model.window_budget
        overlap = min(CHUNK_OVERLAP_TOKENS, budget - 1)

        def scan(
            chunk: str, offset: int, spans: list[DetectedSpan]
        ) -> int:
            """Scan one character window, splitting by tokens if needed."""
            encoding = model.tokenizer.encode(chunk, add_special_tokens=False)
            token_ids, offsets = encoding.ids, encoding.offsets
            windows = _plan_windows(encoding.word_ids, budget, overlap)
            for start, end in windows:
                tokens = _classify_window(model, token_ids, offsets, start, end)
                for span in aggregate_tokens(chunk, tokens, threshold):
                    spans.append(
                        DetectedSpan(
                            start=span.start + offset,
                            end=span.end + offset,
                            type=span.type,
                        )
                    )
            return len(windows)

        spans: list[DetectedSpan] = []
        if len(value) <= MAX_INPUT_CHARS:
            if scan(value, 0, spans) <= 1:
                return self._to_results(value, spans)
            return self._to_results(value, _merge_windowed_spans(spans))

        # Long input: overlapping character windows
        step = MAX_INPUT_CHARS - CHUNK_OVERLAP
        start = 0
        while True:
            scan(value[start : start + MAX_INPUT_CHARS], start, spans)
            if start + MAX_INPUT_CHARS >= len(value):
                break
            start += step
        return self._to_results(value, _merge_windowed_spans(spans))

    def _to_results(
        self, text: str, spans: list[DetectedSpan]
    ) -> list[tuple[str, str, int, int]]:
        """Convert DetectedSpan list to (matched, type_tag, start, end) tuples."""
        results: list[tuple[str, str, int, int]] = []
        for span in spans:
            if span.type in self._keep:
                continue
            if span.start >= len(text) or span.end > len(text):
                continue
            matched = text[span.start : span.end]
            if len(matched.strip()) < 2:
                continue
            type_tag = _ENTITY_TO_TYPE.get(span.type, span.type)
            results.append((matched, type_tag, span.start, span.end))
        return results
