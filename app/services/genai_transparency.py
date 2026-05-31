"""
DataStorm 2026 - Generative AI transparency log (append-only JSONL)
==================================================================
Records each LLM/rules narrative generation for finals auditability.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = ROOT / "data" / "genai_transparency.jsonl"
SAMPLE_LOG_PATH = ROOT / "samples" / "genai_transparency.example.jsonl"


def _log_path() -> Path:
    custom = os.environ.get("GENAI_TRANSPARENCY_LOG")
    return Path(custom) if custom else DEFAULT_LOG_PATH


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_genai_event(
    *,
    outlet_id: str,
    source: str,
    model_id: Optional[str] = None,
    prompt_chars: int = 0,
    prompt_hash: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one transparency record (never raises to callers)."""
    record: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outlet_id": outlet_id,
        "source": source,
        "model_id": model_id,
        "prompt_chars": prompt_chars,
        "prompt_hash": prompt_hash,
        "success": success,
        "error": error,
    }
    if extra:
        record["extra"] = extra

    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_narrative_request(
    outlet_id: str,
    prompt: str,
    source: str,
    model_id: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    log_genai_event(
        outlet_id=outlet_id,
        source=source,
        model_id=model_id,
        prompt_chars=len(prompt),
        prompt_hash=_hash_text(prompt),
        success=success,
        error=error,
    )
