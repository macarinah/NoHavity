"""
Universal Product Schema (UPS)
==============================

One registry drives everything: extraction prompts, JSON Schema for structured
outputs, the coverage score, and the gap report. Add a field here and the whole
pipeline picks it up. Do not scatter field definitions anywhere else.

Every extracted value carries a provenance envelope:

    {"value": ..., "provenance": "verbatim|inferred|generated|missing",
     "evidence": "<exact quote from source>", "confidence": 0.0-1.0}

The envelope is not decoration. It is (a) the anti-hallucination story you tell
judges, (b) how the scorer discounts made-up content, and (c) how the UI shows
"here is the sentence this came from".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------
# Weights are the thesis of the project. Tier 3 (context) and Tier 4
# (constraints) dominate because that is what agent-mediated shopping needs and
# what no catalog on earth currently has. Say this out loud on stage.

TIERS: Dict[int, Dict[str, Any]] = {
    0: {"name": "Identity",      "weight": 0.05, "blurb": "Table stakes. Everyone already has this."},
    1: {"name": "Hard specs",    "weight": 0.10, "blurb": "Agents can filter on these, but cannot reason with them."},
    2: {"name": "Function",      "weight": 0.15, "blurb": "Feature -> mechanism -> benefit. Catalogs stop at feature."},
    3: {"name": "Context fit",   "weight": 0.25, "blurb": "Matches how humans actually phrase requests."},
    4: {"name": "Constraints",   "weight": 0.20, "blurb": "Lets an agent reject confidently, which is what makes it trust you."},
    5: {"name": "Positioning",   "weight": 0.15, "blurb": "Comparative reasoning against alternatives."},
    6: {"name": "Evidence",      "weight": 0.10, "blurb": "Trust signals the agent can cite back to the shopper."},
    7: {"name": "Commerce ops",  "weight": 0.00, "blurb": "Scored at zero: operational, not persuasive. Still extracted."},
    8: {"name": "Agent assets",  "weight": 0.00, "blurb": "Generated outputs. Scored via the simulator, not coverage."},
}

PROVENANCE = ("verbatim", "inferred", "generated", "missing")

# Provenance discount applied inside the coverage score.
PROVENANCE_MULTIPLIER = {
    "verbatim": 1.00,
    "inferred": 0.75,
    "generated": 0.85,   # generated-then-approved content still counts, slightly discounted
    "missing": 0.00,
}


@dataclass
class FieldSpec:
    name: str
    tier: int
    kind: str                       # "str" | "number" | "list[str]" | "list[obj]" | "obj"
    description: str
    example: str = ""
    vocab_key: Optional[str] = None  # key into vocab.yaml for normalisation
    obj_keys: tuple = ()             # for list[obj]/obj: the sub-keys expected
    deterministic: bool = False      # filled by Pass A rules, not the LLM


F = FieldSpec

# --------------------------------------------------------------------------
# THE REGISTRY  (~75 fields)
# --------------------------------------------------------------------------

FIELDS: List[FieldSpec] = [
    # ---- Tier 0: Identity -------------------------------------------------
    F("sku", 0, "str", "Internal stock keeping unit.", deterministic=True),
    F("gtin", 0, "str", "Global trade item number / barcode.", deterministic=True),
    F("brand", 0, "str", "Brand name.", deterministic=True),
    F("product_name", 0, "str", "Full product name including variant line.", deterministic=True),
    F("category_path", 0, "list[str]", "Breadcrumb from broad to narrow.", "['Footwear','Running','Road']"),
    F("price", 0, "number", "Numeric price in the listed currency.", deterministic=True),
    F("currency", 0, "str", "ISO currency code.", "SGD", deterministic=True),
    F("price_tier", 0, "str", "Where it sits in its category.", "mid", vocab_key="price_tier"),
    F("availability", 0, "str", "Stock status.", "in_stock", vocab_key="availability"),
    F("url", 0, "str", "Canonical product page URL.", deterministic=True),
    F("variant_axes", 0, "list[str]", "Dimensions the product varies on.", "['size','colour','width']"),

    # ---- Tier 1: Hard specs ----------------------------------------------
    F("materials", 1, "list[str]", "Materials or fabrics used, by component if stated."),
    F("weight_grams", 1, "number", "Weight in grams.", deterministic=True),
    F("dimensions", 1, "str", "Physical dimensions as stated."),
    F("colors", 1, "list[str]", "Colourways available."),
    F("size_range", 1, "str", "Sizes offered."),
    F("capacity", 1, "str", "Volume, capacity, or quantity per unit.", "50ml"),
    F("ingredients", 1, "list[str]", "Ingredients / INCI list for consumables and cosmetics."),
    F("active_ingredients", 1, "list[obj]", "Actives with concentration.", obj_keys=("name", "concentration", "purpose")),
    F("power_specs", 1, "str", "Battery, wattage, voltage, runtime."),
    F("country_of_origin", 1, "str", "Where it is made."),
    F("warranty_terms", 1, "str", "Warranty length and cover."),
    F("care_instructions", 1, "str", "How to clean, store, maintain."),
    F("certifications", 1, "list[str]", "Formal certifications only.", "['B Corp','COSMOS Organic']"),

    # ---- Tier 2: Function -------------------------------------------------
    F("primary_function", 2, "str", "The single core job in one plain sentence."),
    F("secondary_functions", 2, "list[str]", "Other jobs it also does."),
    F("key_features", 2, "list[obj]", "Feature, the mechanism behind it, and the benefit to the user.",
      obj_keys=("feature", "mechanism", "benefit")),
    F("performance_claims", 2, "list[obj]", "Quantified claims with their backing.",
      obj_keys=("claim", "quantified_value", "backing")),
    F("how_it_works", 2, "str", "Mechanism in one short paragraph a layperson understands."),
    F("technology_names", 2, "list[str]", "Branded tech names plus what each actually does."),

    # ---- Tier 3: Contextual fit  *** the money layer *** -------------------
    F("use_cases", 3, "list[obj]", "Jobs-to-be-done. Situation the shopper is in, and the outcome they want.",
      obj_keys=("situation", "outcome", "fit_strength")),
    F("environment_conditions", 3, "list[str]", "Conditions it is built for.",
      "['humid','hot','wet roads']", vocab_key="environment"),
    F("climate_suitability", 3, "str", "Best-suited climate, stated plainly."),
    F("personas", 3, "list[obj]", "Named shopper archetypes and why this fits them.",
      obj_keys=("persona", "why_fit")),
    F("experience_level", 3, "str", "Who it suits by skill level.", "beginner", vocab_key="experience_level"),
    F("time_commitment", 3, "str", "Time per use or per session.", "under 5 minutes"),
    F("usage_frequency", 3, "str", "How often it is meant to be used."),
    F("occasion", 3, "list[str]", "Occasions or settings.", vocab_key="occasion"),
    F("season", 3, "list[str]", "Seasons it suits.", vocab_key="season"),
    F("body_or_skin_type", 3, "list[str]", "Body types, foot shapes, skin types it is designed for."),
    F("compatible_with", 3, "list[str]", "Things it works alongside."),
    F("requires", 3, "list[str]", "Things you must already own or buy for it to work."),
    F("pairs_well_with", 3, "list[str]", "Complementary products, ours or generic."),
    F("typical_session_duration", 3, "str", "How long a typical use lasts.", "45-90 minutes"),
    F("geography_fit", 3, "list[str]", "Regions or terrains it is suited to."),

    # ---- Tier 4: Constraints & disqualifiers  *** secret weapon *** --------
    F("not_suitable_for", 4, "list[obj]", "Who or what this is explicitly wrong for, and why.",
      obj_keys=("case", "reason")),
    F("known_limitations", 4, "list[str]", "Honest functional limits."),
    F("tradeoffs", 4, "list[obj]", "What you gain and what it costs you.", obj_keys=("gain", "cost")),
    F("allergens", 4, "list[str]", "Allergens present."),
    F("contraindications", 4, "list[str]", "Conditions or combinations to avoid."),
    F("prerequisites", 4, "list[str]", "What the user needs before this is useful."),
    F("learning_curve", 4, "str", "How hard it is to get value on day one.", "none", vocab_key="learning_curve"),
    F("common_complaints", 4, "list[obj]", "Recurring complaints from reviews, with frequency.",
      obj_keys=("complaint", "frequency", "our_response")),
    F("durability_expectation", 4, "str", "Realistic lifespan or replacement interval."),

    # ---- Tier 5: Comparative positioning ----------------------------------
    F("competes_with", 5, "list[str]", "Named competing products or categories."),
    F("differentiators", 5, "list[obj]", "How it differs from the category norm.",
      obj_keys=("vs_category_norm", "our_value")),
    F("why_choose_over", 5, "list[obj]", "Direct head-to-head reasoning.",
      obj_keys=("alternative", "reason")),
    F("upgrade_from", 5, "str", "What a shopper typically owns before this."),
    F("downgrade_from", 5, "str", "Cheaper sibling or the step-down option."),
    F("value_argument", 5, "str", "Why the price is defensible in one sentence."),
    F("price_justification", 5, "str", "Cost per use, longevity, or bundled value."),
    F("category_position", 5, "str", "One line on where it sits in the market."),

    # ---- Tier 6: Evidence & trust -----------------------------------------
    F("rating", 6, "number", "Average review rating.", deterministic=True),
    F("review_count", 6, "number", "Number of reviews.", deterministic=True),
    F("review_themes", 6, "list[obj]", "Themes mined from reviews.",
      obj_keys=("theme", "sentiment", "frequency")),
    F("verified_claims", 6, "list[obj]", "Claims with a verification source.",
      obj_keys=("claim", "verified_by")),
    F("awards", 6, "list[str]", "Awards and editorial picks."),
    F("lab_test_refs", 6, "list[str]", "Lab or clinical test references."),
    F("expert_endorsements", 6, "list[str]", "Named expert or institutional endorsements."),
    F("social_proof_summary", 6, "str", "One line summarising what buyers consistently say."),

    # ---- Tier 7: Commerce ops (extracted, not scored) ----------------------
    F("shipping_options", 7, "list[str]", "Shipping methods offered."),
    F("delivery_estimate", 7, "str", "Typical delivery window."),
    F("return_policy", 7, "str", "Returns window and conditions."),
    F("subscription_available", 7, "str", "Whether a subscription exists."),
    F("bundle_options", 7, "list[str]", "Bundles or kits it belongs to."),

    # ---- Tier 8: Generated agent-facing assets -----------------------------
    F("agent_one_liner", 8, "str", "<=25 words. Written to be quoted verbatim by an assistant."),
    F("semantic_summary", 8, "str", "120-180 words of dense, embedding-friendly prose covering context and constraints."),
    F("persona_pitches", 8, "list[obj]", "One tailored pitch per persona.", obj_keys=("persona", "pitch")),
    F("anticipated_qa", 8, "list[obj]", "Questions a shopper would ask, answered.", obj_keys=("question", "answer")),
    F("objection_handlers", 8, "list[obj]", "Objection and the honest response.", obj_keys=("objection", "response")),
]

FIELDS_BY_NAME: Dict[str, FieldSpec] = {f.name: f for f in FIELDS}

# Tiers the LLM extracts, grouped for parallel calls. Tier 0/1 partly handled by
# deterministic rules first; the LLM fills whatever the rules missed.
EXTRACTION_GROUPS: Dict[str, List[int]] = {
    "specs":       [0, 1],
    "function":    [2],
    "context":     [3],
    "constraints": [4],
    "positioning": [5],
    "evidence":    [6, 7],
}

SCORED_TIERS = [t for t, meta in TIERS.items() if meta["weight"] > 0]


def fields_for_tiers(tiers: List[int]) -> List[FieldSpec]:
    return [f for f in FIELDS if f.tier in tiers]


# --------------------------------------------------------------------------
# Provenance envelope
# --------------------------------------------------------------------------

@dataclass
class FieldValue:
    value: Any = None
    provenance: str = "missing"
    evidence: Optional[str] = None
    confidence: float = 0.0

    @property
    def filled(self) -> bool:
        if self.provenance == "missing":
            return False
        if self.value is None:
            return False
        if isinstance(self.value, (list, str, dict)) and len(self.value) == 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FieldValue":
        if d is None:
            return cls()
        return cls(
            value=d.get("value"),
            provenance=d.get("provenance", "missing"),
            evidence=d.get("evidence"),
            confidence=float(d.get("confidence") or 0.0),
        )

    @classmethod
    def missing(cls) -> "FieldValue":
        return cls()


@dataclass
class ProductRecord:
    """A product at a point in time. Two of these (before/after) make the demo."""
    product_id: str
    raw: Dict[str, Any] = field(default_factory=dict)      # original catalog row + PDP text
    fields: Dict[str, FieldValue] = field(default_factory=dict)
    label: str = "baseline"                                 # "baseline" | "optimised"

    def get(self, name: str) -> FieldValue:
        return self.fields.get(name, FieldValue.missing())

    def set(self, name: str, fv: FieldValue) -> None:
        self.fields[name] = fv

    def filled_names(self) -> List[str]:
        return [n for n, v in self.fields.items() if v.filled]

    def missing_names(self, tiers: Optional[List[int]] = None) -> List[str]:
        out = []
        for f in FIELDS:
            if f.tier == 8:
                continue
            if tiers and f.tier not in tiers:
                continue
            if not self.get(f.name).filled:
                out.append(f.name)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "label": self.label,
            "raw": self.raw,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductRecord":
        return cls(
            product_id=d["product_id"],
            label=d.get("label", "baseline"),
            raw=d.get("raw", {}),
            fields={k: FieldValue.from_dict(v) for k, v in d.get("fields", {}).items()},
        )

    # Convenience for display / retrieval
    def name(self) -> str:
        return self.get("product_name").value or self.raw.get("product_name", self.product_id)

    def price(self) -> Optional[float]:
        v = self.get("price").value
        try:
            return float(v)
        except (TypeError, ValueError):
            return self.raw.get("price")


# --------------------------------------------------------------------------
# JSON Schema generation for LLM structured outputs
# --------------------------------------------------------------------------

def _leaf_schema(spec: FieldSpec) -> Dict[str, Any]:
    if spec.kind == "number":
        return {"type": ["number", "null"]}
    if spec.kind == "str":
        return {"type": ["string", "null"]}
    if spec.kind == "list[str]":
        return {"type": "array", "items": {"type": "string"}}
    if spec.kind == "list[obj]":
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in spec.obj_keys},
                "required": list(spec.obj_keys),
            },
        }
    if spec.kind == "obj":
        return {
            "type": "object",
            "properties": {k: {"type": "string"} for k in spec.obj_keys},
        }
    return {"type": ["string", "null"]}


def json_schema_for(specs: List[FieldSpec]) -> Dict[str, Any]:
    """Envelope-wrapped JSON Schema, ready for an LLM structured-output call."""
    props = {}
    for s in specs:
        props[s.name] = {
            "type": "object",
            "description": s.description,
            "properties": {
                "value": _leaf_schema(s),
                "provenance": {"type": "string", "enum": list(PROVENANCE)},
                "evidence": {
                    "type": ["string", "null"],
                    "description": "Exact quote from the source text. null if provenance is inferred or missing.",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["value", "provenance", "confidence"],
        }
    return {
        "type": "object",
        "properties": props,
        "required": [s.name for s in specs],
        "additionalProperties": False,
    }


def field_menu(specs: List[FieldSpec]) -> str:
    """Human-readable field list injected into the extraction prompt."""
    lines = []
    for s in specs:
        bits = [f"- {s.name} ({s.kind}): {s.description}"]
        if s.obj_keys:
            bits.append(f"  each object has keys: {', '.join(s.obj_keys)}")
        if s.example:
            bits.append(f"  example: {s.example}")
        lines.append("\n".join(bits))
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    total = len([f for f in FIELDS if f.tier != 8])
    print(f"{total} scored/extracted fields, {len(FIELDS)} total across {len(TIERS)} tiers")
    for t, meta in TIERS.items():
        n = len([f for f in FIELDS if f.tier == t])
        print(f"  T{t} {meta['name']:<14} w={meta['weight']:.2f}  {n:>2} fields")
    print(json.dumps(json_schema_for(fields_for_tiers([3]))["properties"]["use_cases"], indent=2)[:400])
