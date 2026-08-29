"""
Coverage Score: how AI-ready is this product's content, before we ever run a query?

    Coverage = SUM over tiers ( tier_weight * tier_fill * quality_multiplier )

Two things stop this being a naive "count the filled boxes" metric:

  1. Provenance discount - inferred content is worth less than sourced content.
  2. Fluff penalty - a field stuffed with "premium innovative game-changing"
     and no number and no source scores near zero. Marketing words are not data.

Scored instantly, no API calls. This is your always-works fallback if the
simulator breaks during the demo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schema import (
    FIELDS,
    FIELDS_BY_NAME,
    PROVENANCE_MULTIPLIER,
    SCORED_TIERS,
    TIERS,
    ProductRecord,
)

FLUFF = {
    "premium", "innovative", "revolutionary", "game-changing", "game changing",
    "cutting-edge", "cutting edge", "world-class", "best-in-class", "next-level",
    "unparalleled", "ultimate", "amazing", "incredible", "state-of-the-art",
    "high-quality", "top quality", "superior", "advanced technology", "elevate",
    "unleash", "redefine", "seamlessly",
}

HAS_NUMBER = re.compile(r"\d")


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text_of(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_text_of(v) for v in value.values())
    return str(value)


def quality_multiplier(fv, spec) -> float:
    """Penalise vague, unsourced, thin content. Reward specificity."""
    if not fv.filled:
        return 0.0

    mult = PROVENANCE_MULTIPLIER.get(fv.provenance, 0.0)
    text = _text_of(fv.value)
    low = text.lower()

    fluff_hits = sum(1 for f in FLUFF if f in low)
    if fluff_hits:
        mult *= max(0.4, 1.0 - 0.2 * fluff_hits)

    # Specificity bonus: numbers or named entities beat adjectives.
    if HAS_NUMBER.search(text) and spec.kind in ("str", "list[str]", "list[obj]"):
        mult *= 1.05

    # Thin-list penalty: a one-item list where several are expected.
    if spec.kind in ("list[str]", "list[obj]") and isinstance(fv.value, list):
        if len(fv.value) == 1:
            mult *= 0.8

    # Empty object keys: {"feature": "Flyknit", "mechanism": "", "benefit": ""}
    if spec.kind == "list[obj]" and isinstance(fv.value, list) and fv.value:
        total_keys = filled_keys = 0
        for obj in fv.value:
            if isinstance(obj, dict):
                for k in spec.obj_keys:
                    total_keys += 1
                    if str(obj.get(k, "")).strip():
                        filled_keys += 1
        if total_keys:
            mult *= 0.5 + 0.5 * (filled_keys / total_keys)

    # Confidence taper.
    mult *= 0.6 + 0.4 * min(1.0, max(0.0, fv.confidence))

    return min(1.15, mult)


@dataclass
class TierScore:
    tier: int
    name: str
    weight: float
    filled: int
    total: int
    raw_fill: float          # fraction of fields with any value
    quality_fill: float      # fraction after quality discount
    contribution: float      # weight * quality_fill
    missing: List[str] = field(default_factory=list)
    weak: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    product_id: str
    label: str
    score: float                       # 0-100
    tiers: List[TierScore]
    top_gaps: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "label": self.label,
            "score": self.score,
            "tiers": [t.__dict__ for t in self.tiers],
            "top_gaps": self.top_gaps,
        }

    def tier_scores(self) -> Dict[str, float]:
        return {t.name: round(t.quality_fill * 100, 1) for t in self.tiers}


def score(rec: ProductRecord) -> CoverageReport:
    tier_scores: List[TierScore] = []
    total = 0.0

    for tier in SCORED_TIERS:
        specs = [f for f in FIELDS if f.tier == tier]
        if not specs:
            continue
        q_sum = 0.0
        filled = 0
        missing, weak = [], []
        for spec in specs:
            fv = rec.get(spec.name)
            q = quality_multiplier(fv, spec)
            q_sum += min(1.0, q)
            if fv.filled:
                filled += 1
                if q < 0.55:
                    weak.append(spec.name)
            else:
                missing.append(spec.name)

        raw_fill = filled / len(specs)
        quality_fill = q_sum / len(specs)
        weight = TIERS[tier]["weight"]
        contribution = weight * quality_fill
        total += contribution

        tier_scores.append(TierScore(
            tier=tier, name=TIERS[tier]["name"], weight=weight,
            filled=filled, total=len(specs),
            raw_fill=round(raw_fill, 3), quality_fill=round(quality_fill, 3),
            contribution=round(contribution * 100, 2),
            missing=missing, weak=weak,
        ))

    return CoverageReport(
        product_id=rec.product_id,
        label=rec.label,
        score=round(total * 100, 1),
        tiers=tier_scores,
        top_gaps=rank_gaps(rec),
    )


def rank_gaps(rec: ProductRecord, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Which missing field would raise the score most? Impact = tier weight divided
    across that tier's fields. This is what the generator works through, and it
    is the list you put on screen when the judge asks 'so what do we fix first'.
    """
    per_tier_count = {t: len([f for f in FIELDS if f.tier == t]) for t in SCORED_TIERS}
    gaps = []
    for spec in FIELDS:
        if spec.tier not in SCORED_TIERS:
            continue
        fv = rec.get(spec.name)
        q = quality_multiplier(fv, spec)
        if q >= 0.85:
            continue
        headroom = 1.0 - min(1.0, q)
        impact = TIERS[spec.tier]["weight"] / per_tier_count[spec.tier] * headroom * 100
        gaps.append({
            "field": spec.name,
            "tier": spec.tier,
            "tier_name": TIERS[spec.tier]["name"],
            "status": "missing" if not fv.filled else "weak",
            "description": spec.description,
            "impact_points": round(impact, 2),
        })
    gaps.sort(key=lambda g: -g["impact_points"])
    return gaps[:limit]


def compare(before: CoverageReport, after: CoverageReport) -> Dict[str, Any]:
    """The delta. This single object is the demo."""
    b = {t.name: t.quality_fill for t in before.tiers}
    a = {t.name: t.quality_fill for t in after.tiers}
    return {
        "before": before.score,
        "after": after.score,
        "delta": round(after.score - before.score, 1),
        "tiers": [
            {"tier": name, "before": round(b.get(name, 0) * 100, 1),
             "after": round(a.get(name, 0) * 100, 1),
             "delta": round((a.get(name, 0) - b.get(name, 0)) * 100, 1)}
            for name in b
        ],
    }
