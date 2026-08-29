"""
Export to machine-readable formats brands can actually ship.

Two outputs:
  1. Schema.org Product JSON-LD  - drops straight into a PDP <script> tag, which
     is how crawlers and many agent stacks ingest product data today.
  2. agentready:* extensions     - the Tier 3/4/5 content that Schema.org has no
     vocabulary for. This is the "here is what the standard is missing" slide.

Build this. It takes two hours and it makes the whole project look shippable
instead of academic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .schema import ProductRecord

CONTEXT = {
    "@vocab": "https://schema.org/",
    "agentready": "https://agentready.dev/ns#",
}


def _v(rec: ProductRecord, name: str, default=None):
    fv = rec.get(name)
    return fv.value if fv.filled else default


def to_jsonld(rec: ProductRecord) -> Dict[str, Any]:
    price = rec.price()
    doc: Dict[str, Any] = {
        "@context": CONTEXT,
        "@type": "Product",
        "@id": _v(rec, "url") or f"urn:sku:{rec.product_id}",
        "sku": _v(rec, "sku", rec.product_id),
        "name": _v(rec, "product_name", rec.name()),
        "brand": {"@type": "Brand", "name": _v(rec, "brand", "")},
        "description": _v(rec, "semantic_summary") or _v(rec, "primary_function", ""),
        "category": " > ".join(_v(rec, "category_path", []) or []),
        "material": _v(rec, "materials", []),
        "color": _v(rec, "colors", []),
        "countryOfOrigin": _v(rec, "country_of_origin"),
        "weight": ({"@type": "QuantitativeValue", "value": _v(rec, "weight_grams"), "unitCode": "GRM"}
                   if _v(rec, "weight_grams") else None),
    }

    if price is not None:
        doc["offers"] = {
            "@type": "Offer",
            "price": price,
            "priceCurrency": _v(rec, "currency", "SGD"),
            "availability": f"https://schema.org/{_availability(_v(rec, 'availability', 'in_stock'))}",
        }

    if _v(rec, "rating") and _v(rec, "review_count"):
        doc["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": _v(rec, "rating"),
            "reviewCount": _v(rec, "review_count"),
        }

    qa = _v(rec, "anticipated_qa", []) or []
    if qa:
        doc["mainEntity"] = {
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": q.get("question", ""),
                 "acceptedAnswer": {"@type": "Answer", "text": q.get("answer", "")}}
                for q in qa if isinstance(q, dict)
            ],
        }

    # ---- the extensions Schema.org has no words for ----------------------
    ext = {
        "agentready:oneLiner": _v(rec, "agent_one_liner"),
        "agentready:useCases": _v(rec, "use_cases"),
        "agentready:environmentConditions": _v(rec, "environment_conditions"),
        "agentready:climateSuitability": _v(rec, "climate_suitability"),
        "agentready:personas": _v(rec, "personas"),
        "agentready:experienceLevel": _v(rec, "experience_level"),
        "agentready:timeCommitment": _v(rec, "time_commitment"),
        "agentready:bodyOrSkinType": _v(rec, "body_or_skin_type"),
        "agentready:pairsWellWith": _v(rec, "pairs_well_with"),
        "agentready:requires": _v(rec, "requires"),
        "agentready:notSuitableFor": _v(rec, "not_suitable_for"),
        "agentready:knownLimitations": _v(rec, "known_limitations"),
        "agentready:tradeoffs": _v(rec, "tradeoffs"),
        "agentready:contraindications": _v(rec, "contraindications"),
        "agentready:learningCurve": _v(rec, "learning_curve"),
        "agentready:durabilityExpectation": _v(rec, "durability_expectation"),
        "agentready:differentiators": _v(rec, "differentiators"),
        "agentready:whyChooseOver": _v(rec, "why_choose_over"),
        "agentready:valueArgument": _v(rec, "value_argument"),
        "agentready:objectionHandlers": _v(rec, "objection_handlers"),
        "agentready:provenance": {
            name: {"provenance": fv.provenance, "confidence": fv.confidence}
            for name, fv in rec.fields.items() if fv.filled
        },
    }
    doc.update({k: v for k, v in ext.items() if v})
    return {k: v for k, v in doc.items() if v not in (None, [], "", {})}


def _availability(token: str) -> str:
    return {
        "in_stock": "InStock",
        "low_stock": "LimitedAvailability",
        "preorder": "PreOrder",
        "out_of_stock": "OutOfStock",
    }.get(token, "InStock")


def to_markdown(rec: ProductRecord) -> str:
    """A human-readable brief the brand's content team can approve line by line."""
    lines = [f"# {rec.name()}", ""]
    one = _v(rec, "agent_one_liner")
    if one:
        lines += [f"> {one}", ""]
    summ = _v(rec, "semantic_summary")
    if summ:
        lines += ["## Agent summary", "", summ, ""]

    def block(title: str, names: List[str]):
        rows = []
        for n in names:
            fv = rec.get(n)
            if not fv.filled:
                continue
            flag = " *(generated - needs approval)*" if fv.provenance == "generated" else ""
            rows.append(f"- **{n.replace('_',' ')}**{flag}: {json.dumps(fv.value, default=str, ensure_ascii=False)[:400]}")
        if rows:
            lines.extend([f"## {title}", ""] + rows + [""])

    block("Where it fits", ["use_cases", "environment_conditions", "climate_suitability",
                            "personas", "experience_level", "time_commitment", "occasion", "season"])
    block("Where it does not fit", ["not_suitable_for", "known_limitations", "tradeoffs",
                                    "contraindications", "durability_expectation"])
    block("Why this one", ["differentiators", "why_choose_over", "value_argument", "price_justification"])
    block("Questions shoppers ask", ["anticipated_qa", "objection_handlers"])
    return "\n".join(lines)


def export_bundle(rec: ProductRecord, out_dir: str) -> Dict[str, str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, f"{rec.product_id}.jsonld")
    md = os.path.join(out_dir, f"{rec.product_id}.md")
    with open(jl, "w") as fh:
        json.dump(to_jsonld(rec), fh, indent=2, ensure_ascii=False)
    with open(md, "w") as fh:
        fh.write(to_markdown(rec))
    return {"jsonld": jl, "markdown": md}
