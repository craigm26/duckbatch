"""Read `duck-feedback/0` (see docs/duck-feedback-0.md): validate strictly, split, learn.

    duckbatch feedback validate  <file-or-dir>...      every record, or the first reason one is refused
    duckbatch feedback report    <file-or-dir>...      router calibration on the HELD-OUT split
    duckbatch feedback export-decide <in>... --out f   GLiNER2 fine-tuning set from the TRAIN split

REFUSED, NOT SKIPPED. A record without consent, with an unknown kind, or with a label outside the
vocabulary it names is an error with a sentence, the same rule duckkit applies to a policy file.
A pipeline that silently drops what it does not understand trains on a subset nobody chose.

THE SPLIT IS FIXED FOREVER. A record is held out when the FNV-1a hash of its `id` lands in the
bottom fifth, so adding records never moves an old one between train and held-out, and a
threshold or fine-tune is always reported on examples it never saw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import router

FORMAT = "duck-feedback/0"
KINDS = ("route_correction", "policy_preference")
SHARES = ("local", "research", "public")
OUTCOMES = ("accepted", "edited", "rejected")
CHOICES = ("a", "b", "tie", "both_bad")
WHERE = ("sim", "phone_bench", "ar", "robot")
REASONS = ("steadier", "more natural", "faster", "follows the command", "fell", "jittery", "other")
VOCABULARIES = {"r001": {"action": router.ACTIONS, "speed": router.SPEEDS,
                         "head": router.HEADS, "sound": router.SOUNDS}}
HELD_OUT_SHARE = 0.2
_TIME = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")


class FeedbackError(ValueError):
    pass


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def is_held_out(record_id: str) -> bool:
    return (fnv1a(record_id) % 1000) < HELD_OUT_SHARE * 1000


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise FeedbackError(why)


def validate(rec: dict[str, Any]) -> dict[str, Any]:
    """Return the record unchanged, or raise FeedbackError with the first reason it is refused."""
    _need(isinstance(rec, dict), "not a JSON object")
    _need(rec.get("format") == FORMAT, f"format is {rec.get('format')!r}, not {FORMAT!r}")
    _need(isinstance(rec.get("id"), str) and len(rec["id"]) >= 8, "id missing or shorter than 8")
    _need(isinstance(rec.get("created"), str) and bool(_TIME.match(rec["created"])),
          "created is not UTC ISO-8601 with a trailing Z")
    consent = rec.get("consent") or {}
    _need(consent.get("opt_in") is True, "no opt-in: a record without consent is refused")
    _need(consent.get("share") in SHARES, f"consent.share must be one of {SHARES}")
    _need(consent["share"] != "local", "consent.share is 'local': it may not leave the device")
    kind = rec.get("kind")
    _need(kind in KINDS, f"kind {kind!r} is not one of {KINDS}")
    body = rec.get(kind)
    _need(isinstance(body, dict), f"missing the {kind} block")
    if kind == "route_correction":
        _validate_route(body)
    else:
        _validate_preference(body)
    return rec


def _check_labels(labels: Any, vocab: dict[str, dict[str, str]], where: str) -> None:
    _need(isinstance(labels, dict) and "action" in labels, f"{where}.labels needs an action")
    for head, value in labels.items():
        _need(head in vocab, f"{where}.labels has unknown head {head!r}")
        _need(value in vocab[head], f"{where}.labels.{head} = {value!r} is not in the vocabulary")


def _validate_route(b: dict[str, Any]) -> None:
    _need(isinstance(b.get("clause"), str) and b["clause"].strip(), "route_correction.clause is empty")
    r = b.get("router") or {}
    vocab = VOCABULARIES.get(r.get("vocabulary"))
    _need(vocab is not None, f"router.vocabulary {r.get('vocabulary')!r} is not one of {list(VOCABULARIES)}")
    _need(isinstance(r.get("model"), str), "router.model missing")
    proposed = b.get("proposed") or {}
    _check_labels(proposed.get("labels"), vocab, "proposed")
    conf = proposed.get("confidence") or {}
    _need(all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in conf.values()),
          "proposed.confidence values must be in [0, 1]")
    outcome = b.get("outcome")
    _need(outcome in OUTCOMES, f"outcome must be one of {OUTCOMES}")
    if outcome == "rejected":
        return
    final = (b.get("final") or {}).get("labels")
    _check_labels(final, vocab, "final")
    same = final == proposed["labels"]
    _need(same if outcome == "accepted" else not same,
          "outcome says accepted but labels changed" if outcome == "accepted"
          else "outcome says edited but no label changed")


def _validate_preference(b: dict[str, Any]) -> None:
    for side in ("a", "b"):
        s = b.get(side) or {}
        _need(isinstance(s.get("repo"), str), f"policy_preference.{side}.repo missing")
        _need(isinstance(s.get("fingerprint"), str) and s["fingerprint"].startswith("sha256:"),
              f"policy_preference.{side}.fingerprint must be 'sha256:…'")
    _need(b["a"]["fingerprint"] != b["b"]["fingerprint"], "a and b are the same network")
    shown = b.get("shown") or {}
    _need(shown.get("where") in WHERE, f"shown.where must be one of {WHERE}")
    _need(shown.get("order") in ("a_left", "b_left"), "shown.order must be 'a_left' or 'b_left'")
    _need(b.get("choice") in CHOICES, f"choice must be one of {CHOICES}")
    reasons = b.get("reasons", [])
    _need(isinstance(reasons, list) and all(r in REASONS for r in reasons),
          f"reasons must come from {REASONS}")


def read(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every record in the given .jsonl files or directories; raises on the first bad one,
    naming the file and line."""
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    out, seen = [], set()
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = validate(json.loads(line))
            except (json.JSONDecodeError, FeedbackError) as e:
                raise FeedbackError(f"{f}:{n}: {e}") from None
            if rec["id"] in seen:
                raise FeedbackError(f"{f}:{n}: duplicate id {rec['id']}")
            seen.add(rec["id"])
            out.append(rec)
    return out


# ---- route corrections -------------------------------------------------------------------------


@dataclass
class LabelOutcome:
    head: str
    confidence: float
    kept: bool  # the person's final label equals the proposed one


def label_outcomes(records: list[dict]) -> list[LabelOutcome]:
    out = []
    for rec in records:
        if rec["kind"] != "route_correction" or rec["route_correction"]["outcome"] == "rejected":
            continue
        b = rec["route_correction"]
        final = b["final"]["labels"]
        for head, label in b["proposed"]["labels"].items():
            if head in final:
                out.append(LabelOutcome(head, float(b["proposed"]["confidence"].get(head, 0.0)),
                                        final[head] == label))
    return out


def calibration(records: list[dict], thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9),
                target: float = 0.95, min_support: int = 20) -> dict[str, Any]:
    """On the HELD-OUT split: per threshold, how many labels would be auto-accepted and how many
    of those the person kept. The suggested threshold is the lowest whose kept-rate reaches
    `target` on at least `min_support` labels; `None` when the data cannot support one."""
    held = [r for r in records if is_held_out(r["id"])]
    outcomes = label_outcomes(held)
    rows = []
    for t in thresholds:
        acc = [o.kept for o in outcomes if o.confidence >= t]
        rows.append({"threshold": t, "auto_accepted": len(acc),
                     "share": round(len(acc) / len(outcomes), 4) if outcomes else None,
                     "kept_rate": round(sum(acc) / len(acc), 4) if acc else None})
    ok = [r for r in rows if r["kept_rate"] is not None and r["kept_rate"] >= target
          and r["auto_accepted"] >= min_support]
    by_head: dict[str, list[bool]] = {}
    for o in outcomes:
        by_head.setdefault(o.head, []).append(o.kept)
    return {"held_out_records": len(held), "held_out_labels": len(outcomes),
            "kept_rate_by_head": {h: round(sum(v) / len(v), 4) for h, v in sorted(by_head.items())},
            "by_threshold": rows, "target": target, "min_support": min_support,
            "suggested_threshold": ok[0]["threshold"] if ok else None}


def export_decide(records: list[dict], out: str | Path) -> dict[str, Any]:
    """GLiNER2 fine-tuning examples from the TRAIN split's accepted and edited corrections,
    built with gliner2's own types so its validator has already accepted the file."""
    from gliner2.training.data import Classification, InputExample, TrainingDataset

    examples = []
    for rec in records:
        if is_held_out(rec["id"]) or rec["kind"] != "route_correction":
            continue
        b = rec["route_correction"]
        if b["outcome"] == "rejected":
            continue
        vocab = VOCABULARIES[b["router"]["vocabulary"]]
        cls = [Classification(task=head, labels=list(vocab[head]), true_label=label,
                              label_descriptions=dict(vocab[head]))
               for head, label in b["final"]["labels"].items()]
        examples.append(InputExample(text=b["clause"], classifications=cls))
    ds = TrainingDataset(examples)
    ds.validate()
    ds.save(str(out))
    return {"examples": len(examples), "out": str(out)}
