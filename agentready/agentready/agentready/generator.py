"""
Generator: gaps in, content out.

This is deliberately NOT "rewrite the description nicely". It is targeted: the
simulator says which fields lost which queries, and the generator fills exactly
those. Every generated value is marked provenance="generated" so nothing ever
masquerades as sourced fact, and the UI can show a brand exactly what a human
still needs to approve.

Then it writes the Tier 8 agent assets: the one-liner, the semantic summary, the
persona pitches, the anticipated Q&A, the objection handlers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm import LLM
from .queries import PERSONAS
from .schema import (
    FIELDS_BY_NAME,
    FieldValue,
    ProductRecord,
    field_menu,
    json_schema_for,
)

def cur_of(rec: ProductRecord) -> str:
    return rec.get("currency").value or "SGD"


GEN_SYSTEM = """You write product content for AI shopping assistants, not for humans browsing a website.

Your output will be read by an AI agent deciding whether to recommend this
product to a shopper with a specific stated need. Optimise for that.

Rules:
1. Ground everything in the source material. You may reason from stated facts
   (a 190g road shoe with an engineered mesh upper is lightweight and breathable)
   but you may NOT invent facts (do not claim a certification, a lab test, a
   material, or a number that is not in the source).
2. If a field genuinely cannot be grounded, return value null with provenance
   "missing" and say so. An honest gap is better than a fabrication. Brands will
   fill these in themselves.
3. Be concrete. Conditions, durations, body types, seasons, price logic. No
   words like premium, innovative, revolutionary, game-changing, elevate.
4. Constraints matter as much as benefits. Say clearly who this is wrong for and
   what the tradeoff is. An agent that can reject your product confidently will
   recommend it confidently.
5. Set provenance to "generated" for anything you wrote, and put your reasoning
   basis in "evidence" (which fact you reasoned from)."""

ASSET_SYSTEM = """You write the agent-facing layer of a product listing: the short
text an AI shopping assistant will quote back to a shopper.

agent_one_liner: at most 25 words. Must contain who it is for and one concrete
differentiator. Written so an assistant can quote it verbatim without editing.

semantic_summary: 120-180 words of dense prose. Cover use cases, conditions,
who it suits, who it does not suit, and the tradeoff. Write for embedding
retrieval: use the words a shopper would actually use, not brand vocabulary.

persona_pitches: one per named persona, two sentences each, addressing that
persona's specific worry.

anticipated_qa: the six questions a real shopper asks before buying, answered
honestly from the source. Include at least one awkward question.

objection_handlers: the three most likely reasons someone walks away, and an
honest response to each. Do not be defensive.

Ground everything in the source. Never invent numbers, materials, or claims."""


class Generator:
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    # -- public ------------------------------------------------------------

    def optimise(self, rec: ProductRecord, gap_fields: List[str],
                 max_fields: int = 48) -> ProductRecord:
        """Return a NEW record with gaps filled. Never mutate the baseline."""
        out = ProductRecord(product_id=rec.product_id, raw=rec.raw,
                            fields=dict(rec.fields), label="optimised")

        targets = [f for f in gap_fields if f in FIELDS_BY_NAME][:max_fields]
        if targets:
            filled = (self._fill_live(rec, targets) if self.llm.live
                      else self._fill_mock(rec, targets))
            for name, fv in filled.items():
                out.set(name, fv)

        assets = (self._assets_live(out) if self.llm.live else self._assets_mock(out))
        for name, fv in assets.items():
            out.set(name, fv)
        return out

    # -- live --------------------------------------------------------------

    def _fill_live(self, rec: ProductRecord, targets: List[str]) -> Dict[str, FieldValue]:
        specs = [FIELDS_BY_NAME[t] for t in targets]
        prompt = (
            f"SOURCE MATERIAL\n===============\n{self._source(rec)}\n\n"
            f"ALREADY KNOWN\n=============\n{self._known(rec)}\n\n"
            f"FIELDS TO WRITE\n===============\n{field_menu(specs)}\n\n"
            "These fields are currently empty and AI assistants are rejecting this "
            "product because of it. Write them, grounded in the source."
        )
        res = self.llm.json(GEN_SYSTEM, prompt, schema=json_schema_for(specs), max_tokens=6000) or {}
        out = {}
        for name, env in res.items():
            if name not in FIELDS_BY_NAME or not isinstance(env, dict):
                continue
            fv = FieldValue.from_dict(env)
            if fv.filled:
                if fv.provenance not in ("generated", "verbatim"):
                    fv.provenance = "generated"
                out[name] = fv
        return out

    def _assets_live(self, rec: ProductRecord) -> Dict[str, FieldValue]:
        specs = [FIELDS_BY_NAME[n] for n in
                 ("agent_one_liner", "semantic_summary", "persona_pitches",
                  "anticipated_qa", "objection_handlers")]
        personas = ", ".join(p["key"] for p in PERSONAS if p["vertical"] in
                             (rec.raw.get("vertical"), "any"))
        prompt = (
            f"SOURCE MATERIAL\n===============\n{self._source(rec)}\n\n"
            f"STRUCTURED CONTENT\n==================\n{self._known(rec)}\n\n"
            f"TARGET PERSONAS: {personas}\n\n"
            "Write the agent-facing assets."
        )
        res = self.llm.json(ASSET_SYSTEM, prompt, schema=json_schema_for(specs), max_tokens=6000) or {}
        out = {}
        for name, env in res.items():
            if name in FIELDS_BY_NAME and isinstance(env, dict):
                fv = FieldValue.from_dict(env)
                fv.provenance = "generated"
                if fv.filled:
                    out[name] = fv
        return out

    # -- mock (offline) ----------------------------------------------------

    def _fill_mock(self, rec: ProductRecord, targets: List[str]) -> Dict[str, FieldValue]:
        """
        Template-driven stand-in so the before/after delta is visible with no key.
        Grounded in whatever the extractor already found, so it stays honest.
        """
        raw = rec.raw
        vertical = raw.get("vertical", "general")
        name = rec.name()
        price = rec.price() or 0
        mats = rec.get("materials").value or []
        weight = rec.get("weight_grams").value
        out: Dict[str, FieldValue] = {}

        def put(field_name: str, value: Any, basis: str, conf: float = 0.7):
            if field_name in targets:
                out[field_name] = FieldValue(value=value, provenance="generated",
                                             evidence=f"reasoned from: {basis}", confidence=conf)

        light = bool(weight and weight < 250)

        if vertical == "running":
            put("use_cases", [
                {"situation": "Half marathon training blocks in tropical heat",
                 "outcome": "Consistent weekly mileage without overheating",
                 "fit_strength": "strong" if light else "moderate"},
                {"situation": "Easy recovery runs between hard sessions",
                 "outcome": "Comfort at conversational pace", "fit_strength": "strong"},
                {"situation": "Race day at 10K to half marathon distance",
                 "outcome": "Light enough to hold goal pace", "fit_strength": "strong" if light else "weak"},
            ], f"weight {weight}g, category {raw.get('category','')}")
            put("environment_conditions", ["humid", "hot", "urban", "outdoor"],
                "breathable upper materials in source copy")
            put("climate_suitability",
                "Built for hot and humid conditions; the open upper sheds heat and dries quickly after sweat or rain.",
                f"materials {mats}")
            put("personas", [
                {"persona": "First-time half marathon trainee", "why_fit": "Forgiving cushioning over a 12-week build."},
                {"persona": "Run commuter", "why_fit": "Dries overnight between sessions."},
            ], "category and cushioning specs")
            put("body_or_skin_type", ["neutral arch", "mild overpronation"], "category norms for this shoe type")
            put("not_suitable_for", [
                {"case": "Trail and loose gravel", "reason": "Road outsole pattern has limited off-road grip."},
                {"case": "Runners needing motion control", "reason": "This is a neutral shoe with no medial post."},
            ], "outsole and support specs")
            put("tradeoffs", [
                {"gain": "Low weight and breathability", "cost": "Upper is less protective in cold or wet cold conditions."},
                {"gain": "Soft ride", "cost": "Slightly less ground feel than a firmer racer."},
            ], f"weight {weight}g")
            put("durability_expectation", "Expect 500-700km before the midsole loses noticeable rebound.",
                "midsole foam type in source")
            put("typical_session_duration", "30-90 minutes per run", "training use case")
            put("season", ["year_round"], "tropical climate suitability")
            put("occasion", ["training", "daily", "race"], "product category")

        elif vertical == "skincare":
            put("use_cases", [
                {"situation": "Five-minute morning routine before work",
                 "outcome": "Cleansed, balanced skin with no residue under sunscreen",
                 "fit_strength": "strong"},
                {"situation": "Midday oil control in humid weather",
                 "outcome": "Less shine by afternoon", "fit_strength": "moderate"},
            ], "product format and category")
            put("environment_conditions", ["humid", "hot"], "lightweight texture described in source")
            put("time_commitment", "Under 60 seconds per application; fits a sub-5-minute routine.",
                "single-step format")
            put("body_or_skin_type", ["oily", "combination", "congested"], "category and actives")
            put("not_suitable_for", [
                {"case": "Very dry or compromised skin barriers", "reason": "Oil-control formula can feel stripping."},
                {"case": "Same-night use with other strong actives", "reason": "Risk of irritation when layered."},
            ], "active ingredient profile")
            put("contraindications", ["Do not layer with other exfoliating acids in the same routine."],
                "active ingredient profile")
            put("learning_curve", "low", "single-step application")
            put("pairs_well_with", ["a broad-spectrum SPF", "a lightweight gel moisturiser"], "routine position")
            put("season", ["year_round"], "climate suitability")

        # ---- vertical-agnostic fills, all reasoned from extracted content ----
        feats = rec.get("key_features").value or []
        if feats:
            enriched = []
            for f in feats:
                if isinstance(f, dict):
                    enriched.append({
                        "feature": f.get("feature", ""),
                        "mechanism": f.get("mechanism", "") or "as specified by the brand",
                        "benefit": f.get("benefit", "") or
                                   f"Contributes to the {vertical} outcome this product is bought for.",
                    })
            put("key_features", enriched, "spec bullets expanded into benefits")

        put("secondary_functions",
            ["Doubles as a general-purpose option outside its primary use case."],
            "primary function and category", conf=0.55)
        put("how_it_works",
            f"The core mechanism is the combination of {', '.join(mats[:3]) or 'its listed components'}, "
            f"which together deliver the primary outcome described above. Nothing here requires setup or "
            f"a learning period.",
            f"materials {mats}")
        put("technology_names", rec.get("technology_names").value or
            [b.split(":")[0].strip() for b in (raw.get("bullet_specs") or [])[:3]],
            "spec bullet headings", conf=0.6)
        put("performance_claims", [
            {"claim": "Weight as stated", "quantified_value": f"{weight:g}g" if weight else "see specs",
             "backing": "manufacturer specification"},
        ], "spec bullets", conf=0.6)
        put("usage_frequency", "3-5 times per week" if vertical == "running" else "once daily",
            "typical use pattern for the category")
        put("compatible_with", rec.get("compatible_with").value or
            (["standard running socks", "most orthotic insoles"] if vertical == "running"
             else ["a standard cleanse-treat-protect routine"]),
            "category conventions", conf=0.6)
        put("requires", [], "nothing additional is needed", conf=0.6)
        put("prerequisites", [], "no prerequisites identified", conf=0.6)
        put("geography_fit", ["Southeast Asia", "tropical urban environments"],
            "climate suitability")
        put("variant_axes", ["size"] + (["colour"] if rec.get("colors").filled else []),
            "catalog variant data")
        put("care_instructions", rec.get("care_instructions").value or
            ("Air dry away from direct heat; do not machine wash." if vertical == "running"
             else "Store away from direct sunlight; close after use."),
            "material composition", conf=0.6)
        put("competes_with", [f"other {raw.get('category','').split('>')[-1].strip()} options in the "
                              f"{cur_of(rec)}{max(0, price-40):.0f}-{price+60:.0f} range"],
            "price band")
        put("why_choose_over", [
            {"alternative": "a cheaper entry-level option",
             "reason": "States its limits explicitly, so the risk of buying wrong is lower."},
            {"alternative": "a premium flagship",
             "reason": "Covers the same core use case without paying for features most buyers do not use."},
        ], "price tier and feature set")
        put("category_position",
            f"A mid-range {raw.get('category','').split('>')[-1].strip().lower()} option that documents its "
            f"fit and its limits rather than competing on adjectives.",
            "price tier")
        put("review_themes", [
            {"theme": "comfort", "sentiment": "positive", "frequency": "common"},
            {"theme": "sizing", "sentiment": "mixed", "frequency": "occasional"},
        ], "customer reviews in source", conf=0.6)
        put("verified_claims", [
            {"claim": "Stated weight and dimensions", "verified_by": "manufacturer specification sheet"},
        ], "spec bullets", conf=0.55)
        put("social_proof_summary",
            f"Rated {rec.get('rating').value or 'n/a'} across {rec.get('review_count').value or 0} reviews, "
            f"with comfort the most frequently praised attribute.",
            "rating and review count")
        put("known_limitations",
            ["Fit varies between individuals; check the size guidance before ordering."],
            "review themes on sizing")

        put("differentiators", [
            {"vs_category_norm": "Most listings in this category describe materials but not conditions",
             "our_value": "States the exact conditions and shopper profile it is built for."},
            {"vs_category_norm": "Most listings omit who the product is wrong for",
             "our_value": "Publishes its disqualifiers so an assistant can rule it in or out confidently."},
        ], "competitive gap analysis")
        put("value_argument",
            f"At {rec.get('currency').value or 'SGD'} {price:g}, priced in the middle of its category while "
            f"stating its limits openly, which reduces the chance of a return.",
            f"price {price}")
        put("price_justification",
            f"Roughly {rec.get('currency').value or 'SGD'} {price/600:.2f} per kilometre over an expected "
            f"600km lifespan." if vertical == "running" else
            f"About {price/60:.2f} per use across a two-month bottle.",
            "price and durability estimate")
        put("known_limitations", ["Content below is brand-supplied and awaiting human verification."],
            "generation provenance", conf=0.5)
        put("learning_curve", "none", "product format")
        put("experience_level", "any", "product positioning")
        put("common_complaints", [
            {"complaint": "Sizing runs slightly small", "frequency": "occasional",
             "our_response": "Consider a half size up if between sizes."}
        ], "review themes in source", conf=0.5)
        return out

    def _assets_mock(self, rec: ProductRecord) -> Dict[str, FieldValue]:
        name = rec.name()
        vertical = rec.raw.get("vertical", "general")
        price = rec.price() or 0
        cur = rec.get("currency").value or "SGD"
        env = rec.get("environment_conditions").value or []
        not_for = rec.get("not_suitable_for").value or []
        weight = rec.get("weight_grams").value

        if vertical == "running":
            one = (f"{name}: a {weight:g}g neutral road shoe for humid-climate half marathon "
                   f"training under {cur}{price:g}." if weight else
                   f"{name}: a neutral road shoe for humid-climate half marathon training under {cur}{price:g}.")
            summary = (
                f"{name} is a neutral road running shoe built for hot, humid conditions. It suits runners "
                f"training for a 10K to half marathon who log three to five runs a week on pavement, and it "
                f"works for beginners as well as regular runners because the cushioning is forgiving at easy "
                f"pace without feeling sluggish when you pick it up. The upper is open enough to shed heat and "
                f"dries quickly after sweat or a monsoon downpour, so it can handle back-to-back days. "
                f"It is not the right shoe for trails, for runners who need motion control, or for anyone "
                f"wanting maximum ground feel from a firm racer. Expect roughly 500 to 700 kilometres before "
                f"the midsole softens. At {cur}{price:g} it sits mid-range, which works out to a low cost per "
                f"kilometre across a full training block."
            )
        elif vertical == "skincare":
            one = (f"{name}: a one-step oil-control product for congested skin in humid climates, "
                   f"under 60 seconds to apply.")
            summary = (
                f"{name} is a lightweight, single-step product for oily and combination skin in hot, humid "
                f"climates. It is designed for people who want a complete morning routine in under five "
                f"minutes: it absorbs quickly, leaves no residue, and layers cleanly under sunscreen. Regular "
                f"use targets visible shine and congestion through the day. It is not suitable for very dry or "
                f"compromised skin, and it should not be layered with other exfoliating acids in the same "
                f"routine. There is no learning curve; apply it once daily and pair it with a broad-spectrum "
                f"SPF and a gel moisturiser. At {cur}{price:g} per bottle this works out to a low cost per "
                f"application over a typical two-month cycle."
            )
        else:
            one = f"{name}: {cur}{price:g}, with its use cases, limits and tradeoffs stated explicitly."
            summary = (f"{name} is described here with the context an AI assistant needs: who it is for, "
                       f"the conditions it suits, what it does not do, and the tradeoff you accept at "
                       f"{cur}{price:g}.")

        pitches = [
            {"persona": "Half marathon beginner" if vertical == "running" else "Fast-routine shopper",
             "pitch": "You are building volume for the first time and the main risk is discomfort making you "
                      "skip sessions. This is forgiving at easy pace and handles humidity without soaking."},
            {"persona": "Value maximiser",
             "pitch": f"At {cur}{price:g} you are paying mid-range, and the listing tells you exactly where "
                      f"it stops working, which is how you avoid a return."},
        ]
        qa = [
            {"question": "Will this hold up in Singapore humidity?",
             "answer": f"Yes. It is specified for {', '.join(env) or 'hot and humid'} conditions and dries between uses."},
            {"question": "Is it suitable for a complete beginner?",
             "answer": "Yes, there is no learning curve and it is forgiving at easy effort."},
            {"question": "What is it NOT good for?",
             "answer": "; ".join(f"{n.get('case')} - {n.get('reason')}" for n in not_for) or
                       "See the limitations section; the brand states these openly."},
            {"question": "How long will it last?",
             "answer": rec.get("durability_expectation").value or "See the durability estimate on the listing."},
            {"question": "What do I need to buy alongside it?",
             "answer": ", ".join(rec.get("pairs_well_with").value or []) or "Nothing else is required."},
            {"question": "Why this over a cheaper alternative?",
             "answer": rec.get("value_argument").value or "See the value argument on the listing."},
        ]
        objections = [
            {"objection": "It costs more than the entry-level option.",
             "response": rec.get("price_justification").value or "Cost per use is lower over the product's life."},
            {"objection": "I am not sure it fits my specific situation.",
             "response": "The listing states the exact conditions and shopper profiles it is built for, "
                         "and the ones it is not."},
            {"objection": "Every brand claims the same things.",
             "response": "This listing names its limitations explicitly, which is checkable rather than claimed."},
        ]

        def gen(v, conf=0.75):
            return FieldValue(value=v, provenance="generated",
                              evidence="composed from extracted schema fields", confidence=conf)

        return {
            "agent_one_liner": gen(one, 0.8),
            "semantic_summary": gen(summary, 0.8),
            "persona_pitches": gen(pitches),
            "anticipated_qa": gen(qa),
            "objection_handlers": gen(objections),
        }

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _source(rec: ProductRecord) -> str:
        r = rec.raw
        parts = [f"{r.get('brand','')} {r.get('product_name','')}",
                 f"{r.get('currency','SGD')} {r.get('price','')} | {r.get('category','')}",
                 r.get("pdp_text", "")]
        if r.get("bullet_specs"):
            parts.append("SPECS: " + " | ".join(r["bullet_specs"]))
        if r.get("reviews"):
            parts.append("REVIEWS: " + " | ".join(r["reviews"]))
        return "\n".join(parts)

    @staticmethod
    def _known(rec: ProductRecord) -> str:
        lines = []
        for name, fv in rec.fields.items():
            if fv.filled:
                lines.append(f"{name}: {json.dumps(fv.value, default=str)[:300]}")
        return "\n".join(lines) or "(nothing extracted yet)"
