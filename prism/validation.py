"""Validation gate: JSON schema constraint + numeric range verification on raw VLM output."""

from __future__ import annotations

import json
import math
import re

from .metrics import METRIC_KEYS, METRIC_NAMES

_ALIASES = {}
for k in METRIC_KEYS:
    _ALIASES[k] = k
    _ALIASES[METRIC_NAMES[k].lower().replace(" ", "_")] = k
    _ALIASES[METRIC_NAMES[k].lower()] = k
    _ALIASES[k.replace("_", "")] = k


def _json_candidates(text: str):
    """Yield every balanced {...} substring, last one first."""
    spans, stack, quote, escaped = [], [], None, False
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ('"', "'") and stack:
            quote = ch
            continue
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append((start, i + 1))
    for s, e in reversed(spans):
        yield text[s:e]


def validate_scores(obj):
    """Enforce exactly twelve unique canonical numeric scores; booleans/strings are not scores."""
    if not isinstance(obj, dict):
        return None, ["not_an_object"]
    scores, errors = {}, []
    for raw_key, value in obj.items():
        key = _ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            errors.append(f"unknown_metric:{raw_key}")
            continue
        if key in scores:
            errors.append(f"duplicate_metric:{key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"non_numeric:{key}")
        elif not math.isfinite(value) or not 0 <= value <= 1:
            errors.append(f"out_of_range:{key}")
        else:
            scores[key] = round(float(value), 4)
    missing = set(METRIC_KEYS) - set(scores)
    if missing:
        errors.append("missing:" + ",".join(sorted(missing)))
    return (None, errors) if errors else (scores, [])


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_output(text: str):
    """Return (scores | None, errors). scores has exactly the 12 metric keys as floats in [0, 1]."""
    errors = []
    if not text or "{" not in text:
        return None, ["no_json_object"]
    for cand in _json_candidates(text):
        cleaned = re.sub(r",\s*}", "}", cand.replace("'", '"'))
        try:
            obj = json.loads(cleaned, object_pairs_hook=_unique_pairs)
        except (json.JSONDecodeError, ValueError):
            errors.append("json_decode_error")
            continue
        scores, errs = validate_scores(obj)
        if not errs:
            return scores, []
        errors.extend(errs)
    return None, errors or ["no_valid_object"]
