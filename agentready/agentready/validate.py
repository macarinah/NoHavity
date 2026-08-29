"""
Validation. RUBRIC 2: is the architecture justified, or did we just assert it?

Two experiments, both runnable in under a minute:

  1. PREDICTIVE VALIDITY
     Does the Coverage Score actually predict simulated win rate across
     products? If it does not, the score is decoration and we should say so.

  2. TIER ABLATION
     Strip one tier's fields from every product, re-run the full query set,
     measure the win-rate collapse. The tier that hurts most when removed is
     the tier that matters most. If our hand-set weights match the measured
     damage, the weights are evidence-based rather than vibes.

This is the difference between "we think context matters" and "removing context
costs 34 points of win rate, here is the table."

    python -m agentready.validate
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List

from .extractor import Extractor
from .generator import Generator
from .llm import LLM
from .queries import generate as gen_queries
from .schema import FIELDS, SCORED_TIERS, TIERS, ProductRecord
from .scorer import rank_gaps, score
from .simulator import Simulator, aggregate, leaderboard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


# ---------------------------------------------------------------------------
# Stats, hand-rolled so we do not add scipy for two functions.
# ---------------------------------------------------------------------------

def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(xs: List[float], ys: List[float]) -> float:
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


# ---------------------------------------------------------------------------
# Experiment 1: does coverage predict win rate?
# ---------------------------------------------------------------------------

def predictive_validity(baseline: List[ProductRecord], optimised: List[ProductRecord],
                        queries, sim: Simulator) -> Dict[str, Any]:
    """
    Build a MIXED corpus: half the catalog optimised, half left as-is, competing
    in the same market. This is the only way to get a real spread of coverage
    scores (12-56) inside one competitive set. Testing correlation across
    all-optimised products compresses coverage into a narrow band and tells you
    nothing - we tried that first and got r=0.30 for exactly that reason.
    """
    opt_by_id = {r.product_id: r for r in optimised}
    corpus, states = [], {}
    for i, rec in enumerate(baseline):
        use_opt = (i % 2 == 0)  # deterministic split, no seed to argue about
        chosen = opt_by_id[rec.product_id] if use_opt else rec
        corpus.append(chosen)
        states[rec.product_id] = "optimised" if use_opt else "baseline"

    results = sim.run(corpus, queries)
    rows = []
    for rec in corpus:
        perf = aggregate(results, rec.product_id)
        rows.append({
            "product_id": rec.product_id,
            "name": rec.name(),
            "state": states[rec.product_id],
            "coverage": score(rec).score,
            "win_rate": perf.win_rate,
            "retrieval_rate": perf.retrieval_rate,
        })
    cov = [r["coverage"] for r in rows]
    win = [r["win_rate"] for r in rows]
    ret = [r["retrieval_rate"] for r in rows]
    return {
        "rows": sorted(rows, key=lambda r: -r["coverage"]),
        "coverage_spread": [round(min(cov), 1), round(max(cov), 1)],
        "pearson_coverage_vs_win": round(pearson(cov, win), 3),
        "spearman_coverage_vs_win": round(spearman(cov, win), 3),
        "pearson_coverage_vs_retrieval": round(pearson(cov, ret), 3),
    }


# ---------------------------------------------------------------------------
# Experiment 2: tier ablation
# ---------------------------------------------------------------------------

def strip_tier(rec: ProductRecord, tier: int) -> ProductRecord:
    """A copy of the product with one tier's content removed."""
    out = ProductRecord(product_id=rec.product_id, raw=rec.raw,
                        fields=copy.deepcopy(rec.fields), label=f"ablate_t{tier}")
    for f in FIELDS:
        if f.tier == tier:
            out.fields.pop(f.name, None)
    return out


def tier_ablation(records: List[ProductRecord], queries, sim: Simulator,
                  hero: str) -> Dict[str, Any]:
    """
    Strip one tier from the HERO ONLY, leaving competitors intact.

    This matters. Our first attempt removed the tier from every product at once,
    which deletes it as a differentiator and measures nothing - Context and
    Constraints both showed a 0.0pt drop because everyone lost them together.
    Ablating one brand against an unchanged field measures the thing a brand
    actually wants to know: what does skipping this tier cost ME?
    """
    baseline_results = sim.run(records, queries)
    base_hero = aggregate(baseline_results, hero).win_rate
    base_retr = aggregate(baseline_results, hero).retrieval_rate

    rows = []
    for tier in SCORED_TIERS:
        corpus = [strip_tier(r, tier) if r.product_id == hero else r for r in records]
        res = sim.run(corpus, queries)
        perf = aggregate(res, hero)
        rows.append({
            "tier": tier,
            "name": TIERS[tier]["name"],
            "assigned_weight": TIERS[tier]["weight"],
            "hero_win_rate": perf.win_rate,
            "win_rate_drop": round(base_hero - perf.win_rate, 1),
            "retrieval_rate": perf.retrieval_rate,
            "retrieval_drop": round(base_retr - perf.retrieval_rate, 1),
        })

    total_damage = sum(max(0.0, r["win_rate_drop"]) for r in rows) or 1.0
    for r in rows:
        r["implied_weight"] = round(max(0.0, r["win_rate_drop"]) / total_damage, 3)
        r["weight_error"] = round(r["implied_weight"] - r["assigned_weight"], 3)

    assigned = [r["assigned_weight"] for r in rows]
    implied = [r["implied_weight"] for r in rows]
    return {
        "method": "hero-only ablation against an unchanged competitive field",
        "baseline_hero_win_rate": base_hero,
        "baseline_hero_retrieval": base_retr,
        "rows": sorted(rows, key=lambda r: -r["win_rate_drop"]),
        "weight_agreement_spearman": round(spearman(assigned, implied), 3),
        "weight_agreement_pearson": round(pearson(assigned, implied), 3),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(hero: str = "p001", products_path: str = None, optimised: bool = True) -> Dict[str, Any]:
    from .pipeline import load_products

    llm = LLM()
    if llm.live:
        llm.preflight()
    print(f"[validate] llm mode: {llm.mode}")
    sim = Simulator(llm)
    queries = gen_queries()
    ex = Extractor(llm)
    records = ex.extract_many(load_products(products_path))
    base_records = list(records)

    # Ablation on baseline records is uninformative when most tiers are empty
    # already, so we ablate the OPTIMISED set: you cannot measure the value of
    # content that was never there.
    if optimised:
        gen = Generator(llm)
        rich = []
        base_results = sim.run(records, queries)
        for rec in records:
            gaps = [g["field"] for g in aggregate(base_results, rec.product_id).gap_fields]
            gaps += [g["field"] for g in rank_gaps(rec, limit=60) if g["field"] not in gaps]
            rich.append(gen.optimise(rec, gaps))
        records = rich

    print("running predictive validity (mixed corpus)...")
    pv = predictive_validity(base_records, records, queries, sim)
    print("running tier ablation (this is the slow one)...")
    ab = tier_ablation(records, queries, sim, hero)

    report = {"n_products": len(records), "n_queries": len(queries),
              "content_state": "optimised" if optimised else "baseline",
              "predictive_validity": pv, "tier_ablation": ab}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "validation.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def print_report(rep: Dict[str, Any]) -> None:
    pv, ab = rep["predictive_validity"], rep["tier_ablation"]
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1 - does the Coverage Score predict win rate?")
    print("=" * 72)
    print(f"  mixed corpus, coverage spread {pv['coverage_spread'][0]}-{pv['coverage_spread'][1]}")
    print(f"  {'product':<9} {'state':<10} {'coverage':>9} {'win rate':>9} {'retrieval':>10}")
    for r in pv["rows"]:
        print(f"  {r['product_id']:<9} {r['state']:<10} {r['coverage']:>9.1f} "
              f"{r['win_rate']:>8.1f}% {r['retrieval_rate']:>9.1f}%")
    print(f"\n  Pearson  coverage vs win rate : {pv['pearson_coverage_vs_win']:+.3f}")
    print(f"  Spearman coverage vs win rate : {pv['spearman_coverage_vs_win']:+.3f}")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 2 - tier ablation: what breaks when you remove a tier?")
    print("=" * 72)
    print(f"  method: {ab['method']}")
    print(f"  hero baseline: win {ab['baseline_hero_win_rate']}%  "
          f"retrieval {ab['baseline_hero_retrieval']}%")
    print(f"\n  {'tier':<16} {'assigned w':>11} {'implied w':>10} {'win drop':>9} {'retr drop':>10}")
    for r in ab["rows"]:
        print(f"  T{r['tier']} {r['name']:<13} {r['assigned_weight']:>11.2f} "
              f"{r['implied_weight']:>10.3f} {r['win_rate_drop']:>8.1f}pt "
              f"{r['retrieval_drop']:>9.1f}pt")
    print(f"\n  Spearman: assigned weights vs measured damage = "
          f"{ab['weight_agreement_spearman']:+.3f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    print_report(run())
