"""
REST API. RUBRIC 5: the integration pathway, in machine form.

A brand's engineer should be able to wire this into their build pipeline in an
afternoon. Endpoints mirror exactly what the UI does, because a demo that only
works through a web page is a demo, not a product.

    pip install fastapi uvicorn
    uvicorn agentready.api:app --reload
    open http://localhost:8000/docs

Every endpoint works with no API key (mock mode), so a brand can evaluate the
whole thing before signing anything.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from fastapi import Body, FastAPI, HTTPException, Query as Q
    from fastapi.responses import PlainTextResponse
except ImportError:  # pragma: no cover
    raise SystemExit("pip install fastapi uvicorn")

from .adopt import ApprovalQueue, embed_snippet, integration_checklist, to_pim_csv, to_review_csv
from .console import Console
from .exporter import to_jsonld, to_markdown
from .extractor import Extractor
from .generator import Generator
from .llm import LLM
from .queries import generate as gen_queries
from .schema import ProductRecord
from .scorer import compare, rank_gaps, score
from .simulator import Simulator, aggregate

app = FastAPI(
    title="AgentReady API",
    version="1.0",
    description="Content readiness scoring for AI-mediated commerce.",
)

_llm = LLM()
_extractor = Extractor(_llm)
_generator = Generator(_llm)
_cache: Dict[str, Any] = {}


def _health() -> Dict[str, Any]:
    return {"status": "ok", "llm": _llm.mode, "auth_required": False}


@app.get("/health")
def health():
    return _health()


# ---------------------------------------------------------------------------
# Core: score a single product. Read-only, no commitment, no integration.
# This is the endpoint a brand tries first.
# ---------------------------------------------------------------------------

@app.post("/v1/score")
def score_product(product: Dict[str, Any] = Body(..., example={
    "product_id": "sku-123",
    "product_name": "Example Road Shoe",
    "brand": "ExampleCo",
    "category": "Footwear > Running > Road",
    "price": 179,
    "currency": "SGD",
    "pdp_text": "A lightweight road shoe with an engineered mesh upper...",
    "bullet_specs": ["Weight: 212g", "Drop: 8mm"],
})):
    """Coverage score + ranked content gaps. No side effects."""
    product.setdefault("product_id", "tmp")
    rec = _extractor.extract(product)
    rep = score(rec)
    return {
        "product_id": rec.product_id,
        "coverage_score": rep.score,
        "tiers": rep.tier_scores(),
        "gaps": rep.top_gaps,
        "fields_filled": len(rec.filled_names()),
        "fields_missing": len(rec.missing_names()),
    }


@app.post("/v1/optimise")
def optimise_product(
    product: Dict[str, Any] = Body(...),
    max_fields: int = Q(40, description="Cap on generated fields per call"),
):
    """
    Score, fill the gaps, re-score. Everything generated comes back flagged
    for approval - the response never implies it is publishable as-is.
    """
    product.setdefault("product_id", "tmp")
    rec = _extractor.extract(product)
    before = score(rec)
    gaps = [g["field"] for g in rank_gaps(rec, limit=60)]
    opt = _generator.optimise(rec, gaps, max_fields=max_fields)
    after = score(opt)
    queue = ApprovalQueue.from_records(rec, opt, coverage_gaps=before.top_gaps)
    _cache[rec.product_id] = {"baseline": rec, "optimised": opt, "queue": queue}
    return {
        "product_id": rec.product_id,
        "delta": compare(before, after),
        "approval_required": queue.stats(),
        "pending_fields": [i.to_dict() for i in queue.items],
        "note": "Generated fields are NOT published until approved. "
                "POST /v1/approve, then GET /v1/jsonld.",
    }


@app.post("/v1/approve/{product_id}")
def approve(product_id: str, decisions: List[Dict[str, Any]] = Body(..., example=[
    {"field": "environment_conditions", "state": "approved"},
    {"field": "not_suitable_for", "state": "edited", "value": "Not for trail use"},
])):
    """Record human decisions. Nothing reaches the published record without this."""
    entry = _cache.get(product_id)
    if not entry:
        raise HTTPException(404, "Run /v1/optimise for this product first")
    applied = 0
    for d in decisions:
        if entry["queue"].act(d.get("field", ""), d.get("state", "pending"),
                              value=d.get("value"), note=d.get("note", "")):
            applied += 1
    return {"applied": applied, "stats": entry["queue"].stats()}


@app.get("/v1/jsonld/{product_id}")
def jsonld(product_id: str, approved_only: bool = Q(True)):
    """Schema.org JSON-LD. Defaults to approved content only."""
    entry = _cache.get(product_id)
    if not entry:
        raise HTTPException(404, "Run /v1/optimise for this product first")
    rec = (entry["queue"].publishable(entry["baseline"]) if approved_only
           else entry["optimised"])
    return to_jsonld(rec)


@app.get("/v1/embed/{product_id}", response_class=PlainTextResponse)
def embed(product_id: str):
    """The whole integration: one script tag to paste into the PDP template."""
    entry = _cache.get(product_id)
    if not entry:
        raise HTTPException(404, "Run /v1/optimise for this product first")
    return embed_snippet(entry["queue"].publishable(entry["baseline"]))


@app.get("/v1/export/{product_id}.csv", response_class=PlainTextResponse)
def export_csv(product_id: str, kind: str = Q("pim", pattern="^(pim|review)$")):
    """CSV for the PIM, or the review sheet for a content team working in Excel."""
    entry = _cache.get(product_id)
    if not entry:
        raise HTTPException(404, "Run /v1/optimise for this product first")
    if kind == "review":
        return to_review_csv(entry["queue"])
    return to_pim_csv([entry["queue"].publishable(entry["baseline"])])


@app.get("/v1/checklist/{product_id}")
def checklist(product_id: str):
    """What the brand has to do, in order, with honest effort estimates."""
    entry = _cache.get(product_id)
    if not entry:
        raise HTTPException(404, "Run /v1/optimise for this product first")
    return integration_checklist(entry["optimised"], entry["queue"])


# ---------------------------------------------------------------------------
# Reasoning endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/ask")
def ask(payload: Dict[str, Any] = Body(..., example={
    "query": "I'm training for a half marathon in humid weather, under S$200",
    "products": [],
})):
    """
    Answer a natural-language shopper query against a supplied catalog.
    This is the endpoint that shows a brand what an assistant sees.
    """
    prods = payload.get("products") or []
    if not prods:
        from .pipeline import load_products
        prods = load_products()
    recs = _extractor.extract_many(prods)
    return Console(recs, _llm).ask(payload["query"])


@app.post("/v1/simulate")
def simulate(payload: Dict[str, Any] = Body(...)):
    """Full benchmark sweep: win rate and field-level gaps per product."""
    prods = payload.get("products") or []
    if not prods:
        from .pipeline import load_products
        prods = load_products()
    recs = _extractor.extract_many(prods)
    sim = Simulator(_llm)
    results = sim.run(recs, gen_queries())
    return {
        "n_queries": len(results),
        "products": [aggregate(results, r.product_id).__dict__ for r in recs],
    }


@app.post("/v1/ingest/csv")
def ingest_csv(payload: Dict[str, str] = Body(..., example={"csv": "Item Name,Price\n..."})):
    """
    Upload a raw catalog CSV. Columns are auto-mapped; unmapped ones are kept.
    Proves a brand does not have to reformat anything to try this.
    """
    import tempfile

    from .ingest import auto_personas, from_csv
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(payload["csv"])
        tmp = fh.name
    try:
        prods = from_csv(tmp, report=False)
        return {
            "products_parsed": len(prods),
            "detected_vertical": prods[0].get("vertical") if prods else None,
            "personas_generated": len(auto_personas(prods, _llm)),
            "products": prods,
        }
    finally:
        os.unlink(tmp)
