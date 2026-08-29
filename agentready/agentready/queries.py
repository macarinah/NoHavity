"""
Shopper query generation.

Personas x intent templates x constraint noise = a few hundred natural-language
queries. Generate these ONCE, cache to disk, never generate live on stage.

Each query carries the schema fields it implicitly interrogates. That mapping is
what turns "we lost this query" into "we lost this query because
environment_conditions is empty" - which is the whole gap-report mechanism.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class Query:
    qid: str
    text: str
    vertical: str
    persona: str
    intent: str
    probes: List[str]              # schema fields this query interrogates
    max_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Personas. Keep them Singapore-flavoured; the brief hands you that setting.
# --------------------------------------------------------------------------

PERSONAS: List[Dict[str, Any]] = [
    dict(key="half_marathon_beginner", vertical="running",
         who="a first-time half marathon trainee in Singapore",
         context="training four times a week in constant humidity",
         constraints=["under S$200", "lightweight", "forgiving for beginners"],
         probes=["environment_conditions", "experience_level", "use_cases", "climate_suitability"]),
    dict(key="heavy_heel_striker", vertical="running",
         who="a heavier runner who lands hard on the heel",
         context="knee pain after long runs on pavement",
         constraints=["maximum cushioning", "durable outsole"],
         probes=["body_or_skin_type", "not_suitable_for", "durability_expectation", "key_features"]),
    dict(key="race_day_pb", vertical="running",
         who="an experienced runner chasing a sub-1:45 half",
         context="wants a dedicated race-day shoe, trains in something else",
         constraints=["fast", "responsive", "willing to pay more"],
         probes=["performance_claims", "experience_level", "tradeoffs", "occasion"]),
    dict(key="treadmill_only", vertical="running",
         who="someone who only runs indoors on a treadmill",
         context="lives in a condo, runs before work",
         constraints=["quiet", "no outdoor grip needed"],
         probes=["environment_conditions", "occasion", "not_suitable_for"]),
    dict(key="wide_feet", vertical="running",
         who="a runner with wide feet who keeps getting blisters",
         context="every shoe pinches across the forefoot",
         constraints=["wide fit available", "seamless upper"],
         probes=["body_or_skin_type", "size_range", "variant_axes", "common_complaints"]),
    dict(key="sustainable_shopper", vertical="running",
         who="a shopper who buys only from credibly sustainable brands",
         context="distrusts greenwashing, wants proof",
         constraints=["recycled materials", "verifiable certification"],
         probes=["certifications", "materials", "verified_claims", "lab_test_refs"]),
    dict(key="rainy_commuter", vertical="running",
         who="a run-commuter during monsoon season",
         context="runs to the office in unpredictable downpours",
         constraints=["quick-drying", "grip on wet ground"],
         probes=["environment_conditions", "care_instructions", "key_features", "season"]),
    dict(key="budget_starter", vertical="running",
         who="a student starting to run for the first time",
         context="not sure they will stick with it, low budget",
         constraints=["under S$120", "does not need to be fancy"],
         probes=["price_tier", "experience_level", "value_argument", "learning_curve"]),

    dict(key="oily_skin_fast_routine", vertical="skincare",
         who="someone with oily, congested skin in a tropical climate",
         context="wants a full morning routine done in under five minutes",
         constraints=["under 5 minutes", "non-greasy", "sustainable"],
         probes=["time_commitment", "body_or_skin_type", "environment_conditions", "pairs_well_with"]),
    dict(key="sensitive_reactive", vertical="skincare",
         who="a person with reactive, easily irritated skin",
         context="has reacted badly to strong actives before",
         constraints=["fragrance-free", "gentle", "clear ingredient list"],
         probes=["allergens", "contraindications", "not_suitable_for", "ingredients"]),
    dict(key="actives_curious", vertical="skincare",
         who="someone new to retinol and nervous about it",
         context="read conflicting advice online, worried about peeling",
         constraints=["beginner-friendly strength", "clear instructions"],
         probes=["learning_curve", "experience_level", "active_ingredients", "anticipated_qa"]),
    dict(key="travel_minimalist", vertical="skincare",
         who="a frequent business traveller",
         context="cabin bag only, hates carrying bottles",
         constraints=["travel size", "multi-purpose"],
         probes=["capacity", "occasion", "secondary_functions", "compatible_with"]),
    dict(key="gift_buyer", vertical="any",
         who="someone buying a gift and unsure of the recipient's preferences",
         context="wants something safe that will not be returned",
         constraints=["broadly appealing", "easy returns"],
         probes=["personas", "return_policy", "not_suitable_for", "social_proof_summary"]),
    dict(key="value_maximiser", vertical="any",
         who="a careful shopper comparing three shortlisted options",
         context="wants to know exactly what the extra money buys",
         constraints=["justify the price"],
         probes=["price_justification", "differentiators", "why_choose_over", "tradeoffs"]),
    dict(key="sceptic", vertical="any",
         who="a shopper who distrusts marketing claims",
         context="wants evidence, not adjectives",
         constraints=["proof for every claim"],
         probes=["verified_claims", "lab_test_refs", "performance_claims", "common_complaints"]),
]

# --------------------------------------------------------------------------
# Intent templates. {who} {context} {constraint} get filled from the persona.
# --------------------------------------------------------------------------

INTENT_TEMPLATES: List[Dict[str, Any]] = [
    dict(key="need_fit", probes=["use_cases", "personas"],
         tmpl="I'm {who}, {context}. What should I get? I need something {constraint}."),
    dict(key="constraint_first", probes=["price_tier", "value_argument"],
         tmpl="Looking for something {constraint} for {who_short}. {context}. Any recommendations?"),
    dict(key="comparison", probes=["differentiators", "why_choose_over", "competes_with"],
         tmpl="I'm choosing between a few options as {who_short}. What actually makes one better than another when {context}?"),
    dict(key="disqualify", probes=["not_suitable_for", "known_limitations", "tradeoffs"],
         tmpl="I'm {who}. {context}. What should I avoid, and why?"),
    dict(key="evidence", probes=["verified_claims", "performance_claims", "lab_test_refs"],
         tmpl="As {who_short}, I don't trust marketing copy. Is there any actual evidence this works when {context}?"),
    dict(key="how_long", probes=["time_commitment", "typical_session_duration", "learning_curve"],
         tmpl="How much time and effort does this actually take? I'm {who}, {context}."),
    dict(key="compatibility", probes=["compatible_with", "pairs_well_with", "requires"],
         tmpl="What else do I need to make this work? I'm {who} and {context}."),
    dict(key="condition_specific", probes=["environment_conditions", "climate_suitability", "season"],
         tmpl="Does this hold up in Singapore conditions? I'm {who}, {context}, and I need it {constraint}."),
    dict(key="beginner_safe", probes=["experience_level", "learning_curve", "prerequisites"],
         tmpl="I'm completely new to this. I'm {who}. Is this a reasonable place to start given {context}?"),
    dict(key="longevity", probes=["durability_expectation", "care_instructions", "price_justification"],
         tmpl="How long will this last and is it worth the money? Context: {who}, {context}."),
]

PRICE_CAPS = {"under S$120": 120, "under S$200": 200, "under S$60": 60}


def _who_short(who: str) -> str:
    return who.replace("a ", "", 1).replace("an ", "", 1).replace("someone who ", "someone who ")


def generate(seed: int = 7, per_pair: int = 1) -> List[Query]:
    rng = random.Random(seed)
    queries: List[Query] = []
    n = 0
    for persona, intent in itertools.product(PERSONAS, INTENT_TEMPLATES):
        for _ in range(per_pair):
            constraint = rng.choice(persona["constraints"])
            text = intent["tmpl"].format(
                who=persona["who"],
                who_short=_who_short(persona["who"]),
                context=persona["context"],
                constraint=constraint,
            )
            cap = None
            for phrase, val in PRICE_CAPS.items():
                if phrase in constraint or phrase in " ".join(persona["constraints"]):
                    cap = val
                    break
            probes = sorted(set(persona["probes"]) | set(intent["probes"]))
            n += 1
            queries.append(Query(
                qid=f"q{n:04d}",
                text=text,
                vertical=persona["vertical"],
                persona=persona["key"],
                intent=intent["key"],
                probes=probes,
                max_price=cap,
            ))
    return queries


def save(queries: List[Query], path: Optional[str] = None) -> str:
    path = path or os.path.join(DATA_DIR, "queries.json")
    with open(path, "w") as fh:
        json.dump([q.to_dict() for q in queries], fh, indent=1)
    return path


def load(path: Optional[str] = None) -> List[Query]:
    path = path or os.path.join(DATA_DIR, "queries.json")
    if not os.path.exists(path):
        qs = generate()
        save(qs, path)
        return qs
    with open(path) as fh:
        return [Query(**d) for d in json.load(fh)]


if __name__ == "__main__":
    qs = generate(per_pair=3)
    p = save(qs)
    print(f"{len(qs)} queries -> {p}")
    for q in qs[:5]:
        print(f"  [{q.persona}/{q.intent}] {q.text}")
