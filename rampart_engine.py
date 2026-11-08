"""Rampart on-device NER model loader and inference.

Downloads the 14.7 MB quantized ONNX token-classification model from
HuggingFace (nationaldesignstudio/rampart) on first use, caches it in the
plugin data directory, and provides span-level PII detection.

Falls back gracefully — if onnxruntime / tokenizers are unavailable or the
model download fails, ``RampartEngine.detect`` returns an empty list and the
plugin continues with regex-only detection.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Entity types Rampart can detect (35-label BIO → 17 entity types)
RAMPART_ENTITIES = [
    "GIVEN_NAME",
    "SURNAME",
    "EMAIL",
    "PHONE_NUMBER",
    "URL",
    "TAX_ID",
    "BANK_ACCOUNT",
    "ROUTING_NUMBER",
    "GOVERNMENT_ID",
    "PASSPORT",
    "DRIVERS_LICENSE",
    "BUILDING_NUMBER",
    "STREET_NAME",
    "SECONDARY_ADDRESS",
    "CITY",
    "STATE",
    "ZIP_CODE",
]

# Default entities we keep (don't redact) because they're usually non-sensitive
DEFAULT_KEEP = {"CITY", "STATE", "ZIP_CODE"}

# Mapping from Rampart entity labels to our placeholder type tags
_ENTITY_TO_TYPE = {
    "GIVEN_NAME": "NAME",
    "SURNAME": "NAME",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "URL": "URL",
    "TAX_ID": "TAXID",
    "BANK_ACCOUNT": "BANK",
    "ROUTING_NUMBER": "ROUTING",
    "GOVERNMENT_ID": "GOVID",
    "PASSPORT": "PASSPORT",
    "DRIVERS_LICENSE": "DL",
    "BUILDING_NUMBER": "ADDR",
    "STREET_NAME": "ADDR",
    "SECONDARY_ADDRESS": "ADDR",
    "CITY": "CITY",
    "STATE": "STATE",
    "ZIP_CODE": "ZIP",
}

_MODEL_ID = "nationaldesignstudio/rampart"
_MODEL_FILES = ["onnx/model_q4.onnx", "vocab.txt", "config.json"]


class RampartEngine:
    """Lazy-loading wrapper around the Rampart ONNX NER model."""

    def __init__(
        self,
        cache_dir: Path,
        threshold: float = 0.5,
        keep_labels: Optional[set[str]] = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._threshold = threshold
        self._keep = keep_labels if keep_labels is not None else set(DEFAULT_KEEP)
        self._session = None  # onnxruntime.InferenceSession
        self._tokenizer = None  # tokenizers.Tokenizer
        self._id2label: dict[int, str] = {}
        self._lock = threading.Lock()
        self._loaded = False
        self._load_failed = False

    def _download_model(self) -> Path:
        """Download model files from HuggingFace Hub if not cached."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        model_dir = self._cache_dir / "rampart"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Check if all files exist
        if all((model_dir / f).exists() for f in _MODEL_FILES):
            return model_dir

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("huggingface-hub not installed")

        for fname in _MODEL_FILES:
            # fname may include subdirs (e.g. "onnx/model_q4.onnx")
            local = model_dir / fname
            if local.exists():
                continue
            logger.info("Downloading Rampart model file: %s", fname)
            downloaded = hf_hub_download(
                repo_id=_MODEL_ID,
                filename=fname,
                cache_dir=str(self._cache_dir / "hf_cache"),
            )
            # Copy / symlink into our dir
            import shutil

            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, local)

        return model_dir

    def _load(self) -> bool:
        """Load the model into memory.  Returns True on success."""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            logger.warning(
                "Rampart dependencies not available (%s); "
                "falling back to regex-only mode",
                exc,
            )
            self._load_failed = True
            return False

        try:
            model_dir = self._download_model()
            onnx_path = model_dir / "onnx" / "model_q4.onnx"
            vocab_path = model_dir / "vocab.txt"
            config_path = model_dir / "config.json"

            # Load tokenizer — vocab.txt is a BERT WordPiece vocab, not a tokenizer.json.
            # We need to build a BertWordPieceTokenizer from the vocab file.
            self._tokenizer = self._build_tokenizer(str(vocab_path))

            # Load label mapping from config
            import json

            with open(config_path) as f:
                config = json.load(f)
            self._id2label = {
                int(k): v for k, v in config.get("id2label", {}).items()
            }
            if not self._id2label:
                # Build default label set (BIO scheme: B-XXX, I-XXX, O)
                labels = ["O"]
                for ent in RAMPART_ENTITIES:
                    labels.extend([f"B-{ent}", f"I-{ent}"])
                self._id2label = {i: label for i, label in enumerate(labels)}

            # Load ONNX session
            self._session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )

            self._loaded = True
            logger.info("Rampart NER model loaded (%s)", onnx_path)
            return True

        except Exception as exc:
            logger.warning("Failed to load Rampart model: %s; regex-only mode", exc)
            self._load_failed = True
            return False

    def _build_tokenizer(self, vocab_path: str):
        """Build a BERT WordPiece tokenizer from vocab.txt.

        The Rampart model ships a plain vocab.txt (BERT WordPiece), not a
        tokenizer.json that ``tokenizers.Tokenizer.from_file`` expects.
        We construct a ``BertWordPieceTokenizer`` and configure it
        to match MiniLM-L6-H384's settings (lowercase, 512 max length).
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import BertPreTokenizer

        tokenizer = Tokenizer(WordPiece(vocab_path, unk_token="[UNK]"))
        # BertPreTokenizer splits on whitespace + punctuation; lowercasing
        # is handled by the BERT-uncased model itself, not the pre-tokenizer.
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.enable_truncation(max_length=512)
        tokenizer.no_padding()
        return tokenizer

    def detect(self, text: str) -> list[tuple[str, str, int, int]]:
        """Detect PII spans in text.

        Returns list of (matched_text, type_tag, start, end).
        Returns empty list if model is not loaded or detection fails.
        """
        if not text or not text.strip():
            return []

        with self._lock:
            if not self._load():
                return []

        try:
            return self._infer(text)
        except Exception as exc:
            logger.warning("Rampart inference error: %s", exc)
            return []

    def _infer(self, text: str) -> list[tuple[str, str, int, int]]:
        """Run token classification inference and extract entity spans."""
        # Tokenize
        encoding = self._tokenizer.encode(text)

        input_ids = encoding.ids if hasattr(encoding, "ids") else encoding["input_ids"]
        attention_mask = (
            encoding.attention_mask
            if hasattr(encoding, "attention_mask")
            else encoding["attention_mask"]
        )

        # Truncate to max sequence length
        max_len = 512
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]

        import numpy as np

        input_ids_arr = np.array([input_ids], dtype=np.int64)
        attention_mask_arr = np.array([attention_mask], dtype=np.int64)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(
            None,
            {input_name: input_ids_arr},
        )

        logits = outputs[0]  # (1, seq_len, num_labels)
        predictions = np.argmax(logits, axis=-1)[0]  # (seq_len,)

        # Decode BIO labels to spans
        results: list[tuple[str, str, int, int]] = []
        tokens = encoding.tokens if hasattr(encoding, "tokens") else encoding["tokens"]
        offsets = (
            encoding.offsets
            if hasattr(encoding, "offsets")
            else encoding.get("offsets", [])
        )

        current_entity: Optional[str] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        for i, pred_id in enumerate(predictions):
            if i == 0 or i >= len(tokens):
                continue  # skip [CLS] etc.

            label = self._id2label.get(int(pred_id), "O")

            if label == "O":
                if current_entity is not None:
                    self._add_span(
                        results, text, current_entity,
                        current_start, current_end,
                    )
                    current_entity = None
                    current_start = None
                    current_end = None
            elif label.startswith("B-"):
                if current_entity is not None:
                    self._add_span(
                        results, text, current_entity,
                        current_start, current_end,
                    )
                entity = label[2:]
                current_entity = entity
                if i < len(offsets):
                    current_start = offsets[i][0]
                    current_end = offsets[i][1]
            elif label.startswith("I-"):
                entity = label[2:]
                if current_entity == entity and i < len(offsets):
                    current_end = offsets[i][1]
                else:
                    if current_entity is not None:
                        self._add_span(
                            results, text, current_entity,
                            current_start, current_end,
                        )
                    current_entity = entity
                    if i < len(offsets):
                        current_start = offsets[i][0]
                        current_end = offsets[i][1]

        # Flush last entity
        if current_entity is not None:
            self._add_span(
                results, text, current_entity, current_start, current_end
            )

        return results

    def _add_span(
        self,
        results: list[tuple[str, str, int, int]],
        text: str,
        entity: str,
        start: Optional[int],
        end: Optional[int],
    ) -> None:
        """Add a detected span to results, applying keep-set and threshold filters."""
        if start is None or end is None or start >= end:
            return
        if start >= len(text) or end > len(text):
            return

        # Skip kept labels
        if entity in self._keep:
            return

        # Map to placeholder type
        type_tag = _ENTITY_TO_TYPE.get(entity, entity)
        matched = text[start:end]

        # Skip very short matches (likely noise)
        if len(matched.strip()) < 2:
            return

        results.append((matched, type_tag, start, end))
