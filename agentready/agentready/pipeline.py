"""
The whole loop, in one command.

    python -m agentready.pipeline run --hero p001

Stages:
  1. extract   catalog -> Universal Product Schema (baseline)
  2. score     coverage per product
  3. simulate  every product against every query -> win rates + gaps
  4. optimise  fill the hero's gaps -> optimised record
  5. re-run    same query set, optimised record swapped in
  6. export    JSON-LD + markdown brief

Everything caches to out/. Run this the night before the demo so that on stage
you are loading JSON, not waiting on an API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from .exporter import export_bundle
from .extractor import Extractor
from .generator import Generator
from .llm import LLM
from .queries import generate as gen_queries, load as load_queries, save as save_queries
from .schema import ProductRecord
from .scorer import compare, rank_gaps, score
from .simulator import Simulator, aggregate, leaderboard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")


def _p(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_products(path: str = None) -> List[Dict[str, Any]]:
    with open(path or os.path.join(DATA, "products.json")) as fh:
        return json.load(fh)


def run(hero: str = "p001", per_pair: int = 1, top_k: int = 5,
        products_path: str = None, quiet: bool = False) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    llm = LLM()
    if llm.live:
        llm.preflight()          # one call, so a dead key fails now and not 40 calls in
    _p(f"LLM mode: {llm.mode}")

    products = load_products(products_path)
    _p(f"{len(products)} products loaded")

    # 1. extract
    ex = Extractor(llm)
    baseline = ex.extract_many(products)
    with open(os.path.join(OUT, "baseline_records.json"), "w") as fh:
        json.dump([r.to_dict() for r in baseline], fh, indent=1)
    _p("extraction complete")

    # 2. score
    reports = {r.product_id: score(r) for r in baseline}
    with open(os.path.join(OUT, "baseline_scores.json"), "w") as fh:
        json.dump({k: v.as_dict() for k, v in reports.items()}, fh, indent=1)
    if not quiet:
        for pid, rep in sorted(reports.items(), key=lambda x: -x[1].score):
            _p(f"  {pid} coverage {rep.score:5.1f}  ({next(r for r in baseline if r.product_id==pid).name()})")

    # 3. simulate
    queries = gen_queries(per_pair=per_pair)
    save_queries(queries)
    _p(f"{len(queries)} shopper queries")

    sim = Simulator(llm, top_k=top_k)
    results = sim.run(baseline, queries)
    _p(f"simulation complete (retriever: {sim.retriever.backend})")

    perf = {p["product_id"]: aggregate(results, p["product_id"]) for p in products}
    board = leaderboard(results, [p["product_id"] for p in products])
    with open(os.path.join(OUT, "baseline_performance.json"), "w") as fh:
        json.dump({k: v.__dict__ for k, v in perf.items()}, fh, indent=1)
    if not quiet:
        for row in board[:6]:
            _p(f"  {row['product_id']} win rate {row['win_rate']:5.1f}%  ({row['wins']} wins)")

    # 4. optimise hero
    hero_rec = next(r for r in baseline if r.product_id == hero)
    hero_perf = perf[hero]
    gap_fields = [g["field"] for g in hero_perf.gap_fields]
    gap_fields += [g["field"] for g in rank_gaps(hero_rec, limit=60)
                   if g["field"] not in gap_fields]
    _p(f"hero {hero}: filling {len(gap_fields[:48])} gap fields")

    gen = Generator(llm)
    optimised = gen.optimise(hero_rec, gap_fields)
    with open(os.path.join(OUT, "optimised_record.json"), "w") as fh:
        json.dump(optimised.to_dict(), fh, indent=1)

    after_report = score(optimised)
    delta = compare(reports[hero], after_report)
    _p(f"coverage {delta['before']} -> {delta['after']}  (+{delta['delta']})")

    # 5. re-run the SAME queries with the optimised record swapped in
    swapped = [optimised if r.product_id == hero else r for r in baseline]
    results2 = sim.run(swapped, queries)
    perf2 = aggregate(results2, hero)
    _p(f"win rate {hero_perf.win_rate}% -> {perf2.win_rate}%")

    # 5b. FAIR FIGHT control.
    # The 91%-style number above is a first-mover result: one optimised product
    # against eleven that were not. Judges will (rightly) poke at that. So we
    # also run the steady state where every competitor adopts the same pipeline.
    # If the hero still beats the 1/N random baseline there, the content is
    # genuinely better matched and not just longer.
    _p("running fair-fight control (all products optimised)...")
    all_opt = []
    for rec in baseline:
        gaps = [g["field"] for g in perf[rec.product_id].gap_fields]
        gaps += [g["field"] for g in rank_gaps(rec, limit=60) if g["field"] not in gaps]
        all_opt.append(gen.optimise(rec, gaps) if rec.product_id != hero else optimised)
    results3 = sim.run(all_opt, queries)
    perf3 = aggregate(results3, hero)
    random_baseline = round(100 / len(products), 1)
    _p(f"fair fight: hero {perf3.win_rate}% vs {random_baseline}% random baseline")

    # 6. export
    bundle = export_bundle(optimised, os.path.join(OUT, "export"))
    _p(f"exported {bundle['jsonld']}")

    summary = {
        "llm_mode": llm.mode,
        "retriever": sim.retriever.backend,
        "n_products": len(products),
        "n_queries": len(queries),
        "hero": hero,
        "coverage": delta,
        "win_rate_before": hero_perf.win_rate,
        "win_rate_after": perf2.win_rate,
        "win_rate_fair_fight": perf3.win_rate,
        "random_baseline": random_baseline,
        "retrieval_before": hero_perf.retrieval_rate,
        "retrieval_after": perf2.retrieval_rate,
        "top_gaps_closed": gap_fields[:12],
        "leaderboard_before": board,
        "leaderboard_after": leaderboard(results2, [p["product_id"] for p in products]),
        "rejection_reasons_before": hero_perf.rejection_reasons[:6],
        "rejection_reasons_after": perf2.rejection_reasons[:6],
        "lost_personas_before": hero_perf.lost_personas[:6],
        "lost_personas_after": perf2.lost_personas[:6],
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # Cache full state for the Streamlit app so the demo never waits on an API.
    with open(os.path.join(OUT, "demo_state.json"), "w") as fh:
        json.dump({
            "summary": summary,
            "baseline_records": [r.to_dict() for r in baseline],
            "optimised_record": optimised.to_dict(),
            "baseline_scores": {k: v.as_dict() for k, v in reports.items()},
            "optimised_score": after_report.as_dict(),
            "performance": {k: v.__dict__ for k, v in perf.items()},
            "performance_after": perf2.__dict__,
            "performance_fair_fight": perf3.__dict__,
            "optimised_all": [r.to_dict() for r in all_opt],
            "queries": [q.to_dict() for q in queries],
            "results_before": [r.__dict__ for r in results],
            "results_after": [r.__dict__ for r in results2],
        }, fh, indent=1)
    _p("demo_state.json written - the app now loads instantly")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentready")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="full pipeline")
    r.add_argument("--hero", default="p001")
    r.add_argument("--per-pair", type=int, default=1)
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--products", default=None)
    r.add_argument("--quiet", action="store_true")

    s = sub.add_parser("score", help="extract and score only")
    s.add_argument("--products", default=None)

    c = sub.add_parser("coldstart", help="run on an unseen category CSV, no code changes")
    c.add_argument("csv")
    c.add_argument("--hero", default=None)

    q = sub.add_parser("queries", help="regenerate the query set")
    q.add_argument("--per-pair", type=int, default=3)

    args = ap.parse_args(argv)

    if args.cmd == "run":
        summary = run(hero=args.hero, per_pair=args.per_pair, top_k=args.top_k,
                      products_path=args.products, quiet=args.quiet)
        print("\n" + "=" * 58)
        print(f"  coverage {summary['coverage']['before']} -> {summary['coverage']['after']}"
              f"   (+{summary['coverage']['delta']})")
        print(f"  win rate {summary['win_rate_before']}% -> {summary['win_rate_after']}%"
              f"  (first-mover, vs unoptimised field)")
        print(f"  fair fight {summary['win_rate_fair_fight']}% when everyone adopts"
              f"  (random baseline {summary['random_baseline']}%)")
        print("=" * 58)
    elif args.cmd == "score":
        ex = Extractor()
        for p in load_products(args.products):
            rep = score(ex.extract(p))
            bars = "  ".join(f"{t.name[:4]} {t.quality_fill*100:4.0f}" for t in rep.tiers)
            print(f"{p['product_id']}  {rep.score:5.1f}   {bars}")
    elif args.cmd == "coldstart":
        from .ingest import auto_personas, from_csv, install_personas, to_json
        prods = from_csv(args.csv)
        n = install_personas(auto_personas(prods, LLM()))
        print(f"[coldstart] {n} personas generated for vertical "
              f"'{prods[0].get('vertical')}' - no code changes")
        path = to_json(prods, os.path.join(DATA, "products_coldstart.json"))
        summary = run(hero=args.hero or prods[0]["product_id"],
                      products_path=path, quiet=True)
        print("\n" + "=" * 58)
        print(f"  UNSEEN CATEGORY: {prods[0].get('category','')}")
        print(f"  coverage {summary['coverage']['before']} -> {summary['coverage']['after']}")
        print(f"  win rate {summary['win_rate_before']}% -> {summary['win_rate_after']}%")
        print("=" * 58)
    elif args.cmd == "queries":
        qs = gen_queries(per_pair=args.per_pair)
        print(f"{len(qs)} queries -> {save_queries(qs)}")


if __name__ == "__main__":
    main(sys.argv[1:])
