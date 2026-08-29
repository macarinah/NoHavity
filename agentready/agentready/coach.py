"""
Coaching. Two audiences, opposite directions.

CONSUMER SIDE - `merchant_questions()`
    An assistant that cannot recommend confidently should not go quiet. It
    should hand the shopper the exact questions to put to the merchant. This
    turns a dead end into a next action, and it is also the mechanism by which
    demand for better content reaches brands: shoppers start asking for it.

MERCHANT SIDE - `improvement_brief()`
    "Improve your content" is useless advice. A brand needs to know which
    sentence to write, why it matters, and roughly what it should say. Each
    brief carries a ready-to-paste prompt so a content team can generate a first
    draft in their own tools and keep editorial control.

Both work offline. Live mode makes them specific to the actual product.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm import LLM
from .schema import FIELDS_BY_NAME, TIERS, ProductRecord

# What a shopper would actually say, per schema field. Written in shopper
# voice, not schema voice - "does this hold up in humidity" not
# "specify environment_conditions".
SHOPPER_PHRASING: Dict[str, str] = {
    "environment_conditions": "What conditions is this actually built for? Does it hold up in heat and humidity?",
    "climate_suitability": "Will this work in a tropical climate, or is it designed for somewhere cooler?",
    "use_cases": "What situations is this actually meant for? Is my use case one of them?",
    "personas": "Who do you make this for? Would you recommend it for someone like me?",
    "experience_level": "Is this suitable for a beginner, or is it aimed at experienced users?",
    "learning_curve": "How long before I get value out of this? Is there anything to learn first?",
    "time_commitment": "How much time does this take per use?",
    "typical_session_duration": "How long does a typical session with this last?",
    "not_suitable_for": "Who should NOT buy this? What are the cases where it is the wrong choice?",
    "known_limitations": "What are the honest limitations? What does it not do well?",
    "tradeoffs": "What am I giving up by choosing this over the alternatives?",
    "allergens": "Does this contain any common allergens?",
    "contraindications": "Is there anything I should avoid combining this with?",
    "prerequisites": "Do I need anything else before this is useful to me?",
    "body_or_skin_type": "Is this designed for my body type or skin type specifically?",
    "durability_expectation": "Realistically, how long will this last before I need to replace it?",
    "care_instructions": "How do I clean and maintain this?",
    "materials": "What exactly is this made from?",
    "ingredients": "Can you share the full ingredient list?",
    "active_ingredients": "What are the active ingredients and at what concentration?",
    "certifications": "Do you have any third-party certifications to back that up?",
    "verified_claims": "Is there independent evidence for these claims, or is it in-house testing?",
    "lab_test_refs": "Has this been lab tested or clinically tested? Can I see the results?",
    "performance_claims": "Do you have actual numbers for that, rather than adjectives?",
    "common_complaints": "What do customers most often complain about?",
    "review_themes": "What do buyers consistently say about this after a few months?",
    "differentiators": "What makes this different from the cheaper option in the same category?",
    "why_choose_over": "Why should I choose this over your competitor's version?",
    "competes_with": "What would you say this competes with?",
    "value_argument": "Why is this priced where it is?",
    "price_justification": "What does the extra cost actually buy me?",
    "compatible_with": "Will this work with what I already own?",
    "pairs_well_with": "What do you recommend using alongside this?",
    "requires": "Is there anything else I have to buy to make this work?",
    "size_range": "What sizes do you offer, and how does the sizing run?",
    "warranty_terms": "What does the warranty actually cover?",
    "return_policy": "What is the returns window if it is not right for me?",
    "season": "Is this a year-round product or seasonal?",
    "occasion": "What occasions is this meant for?",
    "weight_grams": "How much does it weigh?",
    "capacity": "How much product do I actually get?",
}

# What the merchant should write, per field. Concrete instruction, not a nag.
MERCHANT_GUIDANCE: Dict[str, str] = {
    "environment_conditions": "Name the specific conditions: temperature, humidity, wet or dry, indoor or outdoor. Avoid 'all conditions' - it reads as no information.",
    "climate_suitability": "State the climate plainly in one sentence, including where it does NOT suit.",
    "use_cases": "Write 3-4 situations in the shopper's words, each with the outcome they get. 'Half marathon training in humid weather' beats 'versatile performance'.",
    "personas": "Name 2-3 shopper types and one sentence each on why this fits them.",
    "experience_level": "Say beginner, intermediate, advanced, or any. One word, no hedging.",
    "learning_curve": "Say how long before a first-time user gets value. If there is none, say none.",
    "time_commitment": "Give a real number: minutes per use or per session.",
    "not_suitable_for": "List 2-3 cases where this is the wrong purchase, each with the reason. This increases sales; it does not reduce them.",
    "known_limitations": "State 2-3 honest functional limits. Buyers already find these in reviews.",
    "tradeoffs": "For each strength, name what it costs. 'Lighter, so less protective in cold' is a complete thought.",
    "allergens": "List allergens present, or state clearly that there are none.",
    "contraindications": "Name anything this should not be combined with.",
    "body_or_skin_type": "Name the body types, foot shapes, or skin types this is designed around.",
    "durability_expectation": "Give a realistic lifespan or replacement interval with a number.",
    "differentiators": "For each point, name the category norm first, then how you differ. Differences need a baseline to be meaningful.",
    "why_choose_over": "Name a real alternative and give an honest reason to pick yours.",
    "value_argument": "One sentence on why the price is defensible. No superlatives.",
    "price_justification": "Cost per use, or lifespan maths. Show the arithmetic.",
    "verified_claims": "For each claim, name who verified it. In-house counts if you say so.",
    "performance_claims": "Replace every adjective with a number or delete it.",
    "common_complaints": "State the recurring complaint and your honest response. This builds more trust than hiding it.",
    "pairs_well_with": "Name the products that complete the job, including ones you do not sell.",
    "compatible_with": "List what it works with, and name the common thing it does not.",
}

GENERIC_MERCHANT = ("State this plainly in the product copy. If it does not apply, "
                    "say so explicitly - an explicit 'not applicable' is information, "
                    "silence is not.")


def merchant_questions(rec: ProductRecord, unanswered: List[str],
                       llm: Optional[LLM] = None, limit: int = 5) -> List[Dict[str, str]]:
    """
    CONSUMER SIDE. The questions this shopper should put to the merchant,
    because the listing does not answer them.
    """
    fields = [f for f in unanswered if f in FIELDS_BY_NAME][:limit]
    if not fields:
        return []

    if llm and llm.live:
        out = llm.json(
            "You help shoppers get answers brands did not publish. For each missing "
            "field, write the one question the shopper should ask the merchant. "
            "Natural, specific, polite, first person. Return "
            '{"questions": [{"field": "...", "question": "...", "why": "..."}]}',
            f"PRODUCT: {rec.name()}\nCATEGORY: {rec.raw.get('category','')}\n"
            f"MISSING INFORMATION:\n" + "\n".join(
                f"- {f}: {FIELDS_BY_NAME[f].description}" for f in fields),
            max_tokens=1200,
        )
        if out and isinstance(out.get("questions"), list) and out["questions"]:
            return out["questions"][:limit]

    return [{
        "field": f,
        "question": SHOPPER_PHRASING.get(
            f, f"Can you tell me about {f.replace('_', ' ')} for this product?"),
        "why": f"The listing does not cover {f.replace('_', ' ')}, so no assistant "
               f"can confirm this product fits your situation.",
    } for f in fields]


def improvement_brief(rec: ProductRecord, gaps: List[Dict[str, Any]],
                      llm: Optional[LLM] = None, limit: int = 8) -> List[Dict[str, Any]]:
    """
    MERCHANT SIDE. Per gap: what to write, why it matters commercially, and a
    prompt they can paste into their own tools to draft it.
    """
    briefs = []
    for g in gaps[:limit]:
        name = g.get("field")
        spec = FIELDS_BY_NAME.get(name)
        if not spec:
            continue
        tier_w = TIERS[spec.tier]["weight"]
        briefs.append({
            "field": name,
            "tier": spec.tier,
            "tier_name": TIERS[spec.tier]["name"],
            "what_it_is": spec.description,
            "how_to_write_it": MERCHANT_GUIDANCE.get(name, GENERIC_MERCHANT),
            "why_it_matters": (
                f"{g.get('lost_queries', 0)} simulated shopper queries were lost because "
                f"this was missing." if g.get("lost_queries")
                else f"Worth {g.get('impact_points', tier_w * 100):.1f} coverage points, "
                     f"in the tier weighted {tier_w:.0%}."),
            "queries_lost": g.get("lost_queries", 0),
            "impact_points": g.get("impact_points", 0.0),
            "draft_prompt": _draft_prompt(rec, spec),
        })
    return briefs


def _draft_prompt(rec: ProductRecord, spec) -> str:
    """A prompt the merchant can paste anywhere to draft this field themselves."""
    known = []
    for key in ("primary_function", "materials", "weight_grams", "capacity",
                "key_features", "price", "category_path"):
        fv = rec.get(key)
        if fv.filled:
            known.append(f"{key}: {str(fv.value)[:160]}")
    return (
        f"You are writing one field of a product listing for AI shopping assistants.\n\n"
        f"PRODUCT: {rec.name()}\n"
        f"CATEGORY: {rec.raw.get('category', '')}\n"
        f"WHAT WE ALREADY PUBLISH:\n" + ("\n".join(known) or "(very little)") + "\n\n"
        f"WRITE THIS FIELD: {spec.name} ({spec.kind})\n"
        f"Definition: {spec.description}\n"
        f"Guidance: {MERCHANT_GUIDANCE.get(spec.name, GENERIC_MERCHANT)}\n\n"
        f"Rules: ground everything in the facts above. Do not invent certifications, "
        f"materials, test results or numbers. If you cannot support a claim, say so "
        f"instead of writing it. No marketing adjectives."
    )


def rewrite_suggestions(rec: ProductRecord, llm: Optional[LLM] = None) -> List[Dict[str, str]]:
    """
    Line-level critique of the existing description. Separate from gaps: this is
    about what is already written being weak, not about what is absent.
    """
    text = rec.raw.get("pdp_text", "") or ""
    if not text.strip():
        return [{"issue": "No product description", "fix": "Add a description. "
                 "Spec bullets alone give an assistant nothing to reason with.",
                 "excerpt": ""}]

    if llm and llm.live:
        out = llm.json(
            "You critique product copy for AI-assistant readability. Find up to 5 "
            "specific weaknesses: unsupported claims, marketing adjectives with no "
            "substance, vagueness where a number belongs, missing context. Quote the "
            "exact phrase. Return "
            '{"suggestions": [{"excerpt": "...", "issue": "...", "fix": "..."}]}',
            f"PRODUCT: {rec.name()}\n\nCOPY:\n{text}",
            max_tokens=1500,
        )
        if out and isinstance(out.get("suggestions"), list):
            return out["suggestions"][:5]

    from .scorer import FLUFF
    suggestions = []
    low = text.lower()
    for word in sorted(FLUFF):
        if word in low:
            i = low.index(word)
            suggestions.append({
                "excerpt": text[max(0, i - 30):i + len(word) + 30].strip(),
                "issue": f"'{word}' is an adjective with no supporting fact.",
                "fix": "Replace it with the number, material, or test result behind it, "
                       "or cut it. An assistant cannot repeat this to a shopper.",
            })
        if len(suggestions) >= 4:
            break

    if not any(ch.isdigit() for ch in text):
        suggestions.append({
            "excerpt": text[:70].strip() + "...",
            "issue": "The description contains no numbers at all.",
            "fix": "Add at least one concrete figure: weight, duration, capacity, "
                   "temperature, lifespan. Numbers are what an assistant quotes.",
        })

    for cue, msg in (("who", "who this is for"), ("not", "who this is not for")):
        pass
    if "not " not in low and "avoid" not in low:
        suggestions.append({
            "excerpt": "(absent from the copy)",
            "issue": "The copy never says who this is NOT for.",
            "fix": "Add one sentence naming a case where this is the wrong choice. "
                   "Assistants recommend more confidently when they can also rule out.",
        })
    return suggestions[:5]
