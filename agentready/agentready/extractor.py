"""
Extractor: raw product text -> Universal Product Schema.

Three passes, in order:
  A. Deterministic rules   (price, sku, weight, rating...) - no LLM, no risk
  B. Grouped LLM calls     (one per tier group, strict JSON Schema)
  C. Vocabulary normalisation (free text -> controlled terms)

THE CRITICAL RULE: "missing" is a first-class, desirable output. Most teams
fight hallucination. We harvest it. A confident null is a content gap, and
content gaps are the product.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import yaml

from .llm import LLM, MockHeuristics
from .schema import (
    EXTRACTION_GROUPS,
    FIELDS_BY_NAME,
    FieldValue,
    ProductRecord,
    field_menu,
    fields_for_tiers,
    json_schema_for,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SYSTEM = """You are a product data extraction engine for AI-mediated commerce.

You read raw product marketing copy and map it onto a fixed schema. Your output
feeds AI shopping assistants that must decide whether to recommend this product
to a specific shopper with a specific need.

Rules, in priority order:

1. If the source text does not support a value, set provenance to "missing" and
   value to null. This is the correct, expected, and useful answer. Missing
   fields are the entire purpose of this system. Do NOT invent, embellish, or
   pad. A wrong value is far worse than a missing one.
2. If the text states it directly, set provenance "verbatim" and put the exact
   supporting quote in "evidence".
3. If it follows necessarily from the text but is not stated (e.g. a 180g road
   shoe is lightweight), set provenance "inferred", evidence null, and lower
   confidence. Use this sparingly. Marketing adjectives are not evidence.
4. Never treat vague marketing language ("premium", "revolutionary",
   "game-changing", "innovative") as a value. If a claim has no number and no
   source, it is missing.
5. Confidence is your honest probability that the value is correct and
   defensible to a shopper who asks "how do you know that".
"""

USER_TEMPLATE = """SOURCE MATERIAL
===============
{source}

FIELDS TO EXTRACT
=================
{menu}

Extract every field listed. Return one envelope per field, including the ones
you must mark missing. Do not add fields that are not listed."""


class Extractor:
    def __init__(self, llm: Optional[LLM] = None, vocab_path: Optional[str] = None):
        self.llm = llm or LLM()
        path = vocab_path or os.path.join(DATA_DIR, "vocab.yaml")
        with open(path) as fh:
            self.vocab: Dict[str, Dict[str, List[str]]] = yaml.safe_load(fh)

    # -- public ------------------------------------------------------------

    def extract(self, product: Dict[str, Any], label: str = "baseline") -> ProductRecord:
        rec = ProductRecord(product_id=product["product_id"], raw=product, label=label)
        source = self._source_text(product)

        self._pass_a_deterministic(rec, product, source)
        if self.llm.live:
            self._pass_b_llm(rec, source)
        else:
            self._pass_b_mock(rec, product, source)
        self._pass_c_normalise(rec)
        return rec

    def extract_many(self, products: List[Dict[str, Any]], label: str = "baseline",
                     workers: int = 4) -> List[ProductRecord]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda p: self.extract(p, label), products))

    # -- source ------------------------------------------------------------

    @staticmethod
    def _source_text(product: Dict[str, Any]) -> str:
        parts = [
            f"PRODUCT NAME: {product.get('product_name','')}",
            f"BRAND: {product.get('brand','')}",
            f"PRICE: {product.get('currency','')} {product.get('price','')}",
            f"CATEGORY: {product.get('category','')}",
            "",
            "PAGE COPY:",
            product.get("pdp_text", ""),
        ]
        if product.get("bullet_specs"):
            parts += ["", "SPEC BULLETS:"] + [f"- {b}" for b in product["bullet_specs"]]
        if product.get("reviews"):
            parts += ["", "CUSTOMER REVIEWS:"] + [f"- {r}" for r in product["reviews"]]
        return "\n".join(parts)

    # -- Pass A ------------------------------------------------------------

    def _pass_a_deterministic(self, rec: ProductRecord, product: Dict[str, Any], source: str) -> None:
        """Rules only. Exact, free, and it removes 11 chances for the LLM to be wrong."""
        direct = {
            "sku": product.get("sku"),
            "gtin": product.get("gtin"),
            "brand": product.get("brand"),
            "product_name": product.get("product_name"),
            "url": product.get("url"),
            "price": product.get("price"),
            "currency": product.get("currency"),
            "rating": product.get("rating"),
            "review_count": product.get("review_count"),
        }
        for name, val in direct.items():
            if val not in (None, ""):
                rec.set(name, FieldValue(value=val, provenance="verbatim",
                                         evidence="catalog field", confidence=1.0))

        if product.get("category"):
            rec.set("category_path", FieldValue(
                value=[c.strip() for c in str(product["category"]).split(">")],
                provenance="verbatim", evidence="catalog field", confidence=1.0))

        weight = MockHeuristics.number_near(source, [
            r"(\d+(?:\.\d+)?)\s*g\b", r"(\d+(?:\.\d+)?)\s*grams",
        ])
        if weight:
            rec.set("weight_grams", FieldValue(
                value=weight, provenance="verbatim",
                evidence=MockHeuristics.sentence_with(source, f"{weight:g}g") or f"{weight:g}g",
                confidence=0.95))

        cap = re.search(r"(\d+\s?(?:ml|mL|L|oz|g)\b)", source)
        if cap and not rec.get("capacity").filled:
            rec.set("capacity", FieldValue(value=cap.group(1), provenance="verbatim",
                                           evidence=cap.group(0), confidence=0.8))

        colors = MockHeuristics.find_all(source, MockHeuristics.COLORS)
        if colors:
            rec.set("colors", FieldValue(value=colors, provenance="verbatim",
                                         evidence="colour terms in copy", confidence=0.85))

        avail = product.get("availability")
        if avail:
            rec.set("availability", FieldValue(value=avail, provenance="verbatim",
                                               evidence="catalog field", confidence=1.0))

    # -- Pass B (live) -----------------------------------------------------

    def _pass_b_llm(self, rec: ProductRecord, source: str) -> None:
        """One structured call per tier group, in parallel. Never one giant prompt."""

        def run(group_tiers):
            specs = [s for s in fields_for_tiers(group_tiers)]
            # Skip fields Pass A already nailed.
            specs = [s for s in specs if not rec.get(s.name).filled]
            if not specs:
                return {}
            prompt = USER_TEMPLATE.format(source=source, menu=field_menu(specs))
            result = self.llm.json(SYSTEM, prompt, schema=json_schema_for(specs))
            return result or {}

        with ThreadPoolExecutor(max_workers=len(EXTRACTION_GROUPS)) as pool:
            results = list(pool.map(run, EXTRACTION_GROUPS.values()))

        for res in results:
            for name, env in (res or {}).items():
                if name not in FIELDS_BY_NAME or not isinstance(env, dict):
                    continue
                fv = FieldValue.from_dict(env)
                if fv.filled:
                    rec.set(name, fv)
                else:
                    rec.set(name, FieldValue.missing())

    # -- Pass B (mock) -----------------------------------------------------

    def _pass_b_mock(self, rec: ProductRecord, product: Dict[str, Any], source: str) -> None:
        """
        Offline stand-in. Fills specs and function from keywords; leaves most of
        Tier 3-5 missing, which is what a real catalog looks like anyway.
        """
        low = source.lower()
        H = MockHeuristics

        mats = H.find_all(source, H.MATERIALS)
        if mats:
            rec.set("materials", FieldValue(value=mats, provenance="verbatim",
                                            evidence=H.sentence_with(source, mats[0]), confidence=0.8))

        first_sentence = re.split(r"(?<=[.!?])\s+", product.get("pdp_text", "").strip())
        if first_sentence and first_sentence[0]:
            rec.set("primary_function", FieldValue(
                value=first_sentence[0][:200], provenance="verbatim",
                evidence=first_sentence[0][:200], confidence=0.7))

        bullets = product.get("bullet_specs") or []
        feats = []
        for b in bullets[:6]:
            if ":" in b:
                head, tail = b.split(":", 1)
                feats.append({"feature": head.strip(), "mechanism": tail.strip(), "benefit": ""})
            else:
                feats.append({"feature": b.strip(), "mechanism": "", "benefit": ""})
        if feats:
            rec.set("key_features", FieldValue(value=feats, provenance="verbatim",
                                               evidence="spec bullets", confidence=0.75))

        if product.get("size_range"):
            rec.set("size_range", FieldValue(value=product["size_range"], provenance="verbatim",
                                             evidence="catalog field", confidence=1.0))

        for term, cues in [("certifications", ["certified", "b corp", "cosmos", "vegan", "leaping bunny"])]:
            hits = [c for c in cues if c in low]
            if hits:
                rec.set(term, FieldValue(value=[h.title() for h in hits], provenance="verbatim",
                                         evidence=H.sentence_with(source, hits[0]), confidence=0.7))

        # Tier 3: only picks up what the copy literally shouts. Usually thin.
        env_hits = self._vocab_hits("environment", source)
        if env_hits:
            rec.set("environment_conditions", FieldValue(
                value=env_hits, provenance="verbatim",
                evidence=H.sentence_with(source, env_hits[0]) or env_hits[0], confidence=0.6))

        if product.get("reviews"):
            rec.set("social_proof_summary", FieldValue(
                value=product["reviews"][0][:160], provenance="verbatim",
                evidence=product["reviews"][0][:160], confidence=0.6))

        price = product.get("price")
        if price is not None:
            tier = "budget" if price < 80 else ("mid" if price < 220 else "premium")
            rec.set("price_tier", FieldValue(value=tier, provenance="inferred",
                                             evidence=None, confidence=0.7))

        # Everything else stays missing. That is the honest baseline.

    # -- Pass C ------------------------------------------------------------

    def _vocab_hits(self, key: str, text: str) -> List[str]:
        low = text.lower()
        out = []
        for canon, syns in self.vocab.get(key, {}).items():
            if any(s in low for s in syns):
                out.append(canon)
        return out

    def _normalise_term(self, key: str, term: str) -> Optional[str]:
        t = str(term).lower().strip()
        table = self.vocab.get(key, {})
        for canon, syns in table.items():
            if t == canon:
                return canon
            if any(s in t or t in s for s in syns):
                return canon
        return None

    def _pass_c_normalise(self, rec: ProductRecord) -> None:
        """Map free text onto controlled vocabulary. This is what makes it *universal*."""
        for name, fv in list(rec.fields.items()):
            spec = FIELDS_BY_NAME.get(name)
            if not spec or not spec.vocab_key or not fv.filled:
                continue
            if spec.kind == "list[str]" and isinstance(fv.value, list):
                mapped, unmapped = [], []
                for item in fv.value:
                    canon = self._normalise_term(spec.vocab_key, item)
                    (mapped if canon else unmapped).append(canon or item)
                fv.value = sorted(set(mapped))
                if unmapped:
                    fv.confidence = min(fv.confidence, 0.8)
            elif spec.kind == "str" and isinstance(fv.value, str):
                canon = self._normalise_term(spec.vocab_key, fv.value)
                if canon:
                    fv.value = canon
                else:
                    fv.confidence = min(fv.confidence, 0.7)
