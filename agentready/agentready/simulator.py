"""
The Simulator. This is the moat.

Pipeline per query:
  1. Retrieve  - embed the query, cosine-sim against each product's agent view
  2. Judge     - an LLM plays shopping assistant over the top-K, picks one, and
                 gives a one-line rejection reason for each it did not pick
  3. Aggregate - retrieval_rate, selection_rate (the headline win rate), and
                 rejection reasons clustered back onto schema fields

Step 3 is the part nobody else will build. "You lost 61% of humid-weather
queries because environment_conditions is empty" is a sentence that wins
hackathons.

Retrieval backend degrades gracefully: sentence-transformers -> TF-IDF ->
token overlap. Never let a model download failure kill the demo.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLM
from .queries import Query
from .schema import FIELDS_BY_NAME, ProductRecord

TOP_K = 5


# --------------------------------------------------------------------------
# Agent view: what the retriever and the judge actually see
# --------------------------------------------------------------------------

def agent_view(rec: ProductRecord, include_generated: bool = True) -> str:
    """
    Flatten a product into the text an AI assistant would reason over. Missing
    fields simply do not appear, which is precisely why thin products lose.
    """
    parts: List[str] = [f"{rec.name()} by {rec.get('brand').value or rec.raw.get('brand','')}"]

    # Category anchoring. Without this, retrieval matches on incidental words -
    # a skincare product whose reviews mention "humid weather" outranked a
    # running shoe for a running query. Product type is the strongest signal a
    # shopper gives, so it is stated first and weighted by repetition.
    cat = rec.get("category_path").value or []
    if isinstance(cat, str):
        cat = [cat]
    if not cat and rec.raw.get("category"):
        cat = [c.strip() for c in str(rec.raw["category"]).split(">")]
    if cat:
        leaf = cat[-1]
        parts.append(f"Product type: {leaf}. Category: {' > '.join(cat)}. "
                     f"This is a {leaf} product. {leaf} {leaf}")

    price = rec.price()
    if price is not None:
        parts.append(f"Price: {rec.get('currency').value or 'SGD'} {price}")

    order = [
        "agent_one_liner", "primary_function", "semantic_summary",
        "use_cases", "environment_conditions", "climate_suitability", "personas",
        "experience_level", "time_commitment", "occasion", "season",
        "body_or_skin_type", "compatible_with", "requires", "pairs_well_with",
        "key_features", "performance_claims", "how_it_works", "technology_names",
        "materials", "weight_grams", "capacity", "size_range", "ingredients",
        "active_ingredients", "certifications", "care_instructions",
        "not_suitable_for", "known_limitations", "tradeoffs", "allergens",
        "contraindications", "prerequisites", "learning_curve",
        "durability_expectation", "common_complaints",
        "differentiators", "why_choose_over", "competes_with", "value_argument",
        "price_justification", "category_position",
        "rating", "review_count", "review_themes", "verified_claims", "awards",
        "lab_test_refs", "expert_endorsements", "social_proof_summary",
        "anticipated_qa", "objection_handlers", "persona_pitches",
    ]

    for name in order:
        spec = FIELDS_BY_NAME.get(name)
        if not spec:
            continue
        if not include_generated and spec.tier == 8:
            continue
        fv = rec.get(name)
        if not fv.filled:
            continue
        parts.append(f"{name.replace('_',' ').title()}: {_render(fv.value)}")
    return "\n".join(parts)


def _render(value: Any) -> str:
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, dict):
                out.append("; ".join(f"{k}={v[k]}" for k in v if str(v.get(k, "")).strip()))
            else:
                out.append(str(v))
        return " | ".join(out)
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items() if str(v).strip())
    return str(value)


# --------------------------------------------------------------------------
# Retrieval, with three fallback levels
# --------------------------------------------------------------------------

class Retriever:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.model = None
        self.vectorizer = None
        self.doc_matrix = None
        self.doc_ids: List[str] = []
        self._resolve_backend()

    def _resolve_backend(self):
        if self.backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "sentence-transformers"
                return
            except Exception:
                pass
        if self.backend in ("auto", "tfidf"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
                self.backend = "tfidf"
                return
            except Exception:
                pass
        self.backend = "overlap"

    def fit(self, docs: Dict[str, str]) -> None:
        self.doc_ids = list(docs.keys())
        texts = [docs[i] for i in self.doc_ids]
        if self.backend == "sentence-transformers":
            self.doc_matrix = self.model.encode(texts, normalize_embeddings=True)
        elif self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            self.doc_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.doc_matrix = [set(_tokens(t)) for t in texts]

    def search(self, query: str, k: int = TOP_K) -> List[Tuple[str, float]]:
        if self.backend == "sentence-transformers":
            import numpy as np
            qv = self.model.encode([query], normalize_embeddings=True)[0]
            sims = self.doc_matrix @ qv
            idx = sims.argsort()[::-1][:k]
            return [(self.doc_ids[i], float(sims[i])) for i in idx]
        if self.backend == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self.doc_matrix)[0]
            idx = sims.argsort()[::-1][:k]
            return [(self.doc_ids[i], float(sims[i])) for i in idx]
        qt = set(_tokens(query))
        scored = []
        for i, dt in enumerate(self.doc_matrix):
            inter = len(qt & dt)
            denom = math.sqrt(len(qt) * len(dt)) or 1
            scored.append((self.doc_ids[i], inter / denom))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP = set("the a an and or of for to in on with is are i im my me you your it this that be can what which how".split())


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an AI shopping assistant. A shopper describes their
situation and you have candidate products, each described only by the structured
content its brand published. You may use ONLY that content. You cannot browse.

Pick the single best product, or pick none if none of them supply enough
information to recommend responsibly.

For every candidate you did NOT pick, give one short reason, and name the single
most useful piece of information that was missing or unconvincing. Be specific:
"does not say whether it handles humidity" beats "not a good fit".

Return JSON:
{"winner": "<product_id or null>",
 "winner_reason": "<one sentence>",
 "rejections": [{"product_id": "...", "reason": "...", "missing_info": "..."}],
 "confidence": 0.0-1.0}"""


@dataclass
class QueryResult:
    qid: str
    query: str
    persona: str
    intent: str
    probes: List[str]
    retrieved: List[str]
    winner: Optional[str]
    winner_reason: str = ""
    rejections: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ProductPerformance:
    product_id: str
    queries_seen: int
    retrieved: int
    won: int
    retrieval_rate: float
    win_rate: float
    rejection_reasons: List[Dict[str, Any]]
    gap_fields: List[Dict[str, Any]]
    lost_personas: List[Dict[str, Any]]


class Simulator:
    def __init__(self, llm: Optional[LLM] = None, top_k: int = TOP_K):
        self.llm = llm or LLM()
        self.top_k = top_k
        self.retriever = Retriever()

    def run(self, records: List[ProductRecord], queries: List[Query],
            workers: int = 6, progress=None) -> List[QueryResult]:
        docs = {r.product_id: agent_view(r) for r in records}
        by_id = {r.product_id: r for r in records}
        self.retriever.fit(docs)

        def one(q: Query) -> QueryResult:
            hits = self.retriever.search(q.text, k=self.top_k)
            cand_ids = [pid for pid, _ in hits]
            if q.max_price:
                priced = [pid for pid in cand_ids
                          if (by_id[pid].price() or 1e9) <= q.max_price]
                cand_ids = priced or cand_ids
            verdict = self._judge(q, [by_id[p] for p in cand_ids], docs)
            return QueryResult(
                qid=q.qid, query=q.text, persona=q.persona, intent=q.intent,
                probes=q.probes, retrieved=cand_ids,
                winner=verdict.get("winner"),
                winner_reason=verdict.get("winner_reason", ""),
                rejections=verdict.get("rejections", []),
                confidence=float(verdict.get("confidence") or 0.0),
            )

        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(one, queries)):
                results.append(res)
                if progress:
                    progress(i + 1, len(queries))
        return results

    # -- judging -----------------------------------------------------------

    def _judge(self, q: Query, cands: List[ProductRecord], docs: Dict[str, str]) -> Dict[str, Any]:
        if not cands:
            return {"winner": None, "winner_reason": "no candidates", "rejections": []}
        if self.llm.live:
            block = "\n\n".join(f"### {c.product_id}\n{docs[c.product_id]}" for c in cands)
            out = self.llm.json(
                JUDGE_SYSTEM,
                f"SHOPPER: {q.text}\n\nCANDIDATES\n==========\n{block}",
                max_tokens=1500,
            )
            if out:
                return out
        return self._mock_judge(q, cands)

    def _mock_judge(self, q: Query, cands: List[ProductRecord]) -> Dict[str, Any]:
        """
        Offline judge. Scores each candidate on how many of the query's probed
        fields it actually answers. Crude, but it is the same causal story as
        the real judge: unanswered probes lose you the sale.
        """
        scored = []
        n_cands = len(cands)
        for rank, c in enumerate(cands):
            answered = [p for p in q.probes if c.get(p).filled]
            unanswered = [p for p in q.probes if not c.get(p).filled]
            # Answering the shopper's implicit questions dominates...
            base = 0.55 * (len(answered) / max(1, len(q.probes)))
            # ...but keyword relevance still counts, so thin products are not
            # automatically at zero. This keeps the baseline honest rather than
            # theatrical: catalogs do win some queries today, just not enough.
            base += 0.38 * (1.0 - rank / max(1, n_cands))
            # Depth of overall content, as a generic quality proxy.
            base += 0.06 * min(1.0, len(c.filled_names()) / 30.0)
            if c.get("semantic_summary").filled:
                base += 0.10
            if c.get("agent_one_liner").filled:
                base += 0.04
            if q.max_price and (c.price() or 0) > q.max_price:
                base -= 0.45
            # Trust discount. An assistant should weigh unverified brand-generated
            # claims below sourced ones, so a catalog cannot win simply by
            # generating everything. This is also what stops the Coverage Score
            # from being gameable.
            base -= 0.18 * _unverified_ratio(c)
            # Responsibility gate. When the shopper signals risk - a sensitivity,
            # an injury, a reaction, being a beginner - an assistant should not
            # recommend a product that cannot rule itself out. Missing constraint
            # content is not neutral here, it is disqualifying.
            # Category mismatch is disqualifying, not a penalty. Recommending a
            # running shoe to someone asking about coffee grinders is worse than
            # recommending nothing, so it must never be recoverable by a strong
            # score elsewhere.
            if not _category_plausible(q.text, c):
                base = -1.0
            if _risk_signalled(q.text):
                unruled = [f for f in RISK_FIELDS if not c.get(f).filled]
                base -= 0.16 * (len(unruled) / len(RISK_FIELDS))
            scored.append((base, c, answered, unanswered))
        scored.sort(key=lambda x: -x[0])

        best_score, best, answered, _ = scored[0]

        # An assistant must answer at least ONE of the shopper's implicit
        # questions before it recommends anything. Without this floor, keyword
        # similarity alone can carry a product over the threshold and it "wins"
        # having answered nothing - which is exactly the behaviour this project
        # exists to criticise.
        winner = (best.product_id
                  if best_score >= 0.40 and len(answered) >= 1
                  else None)
        if winner:
            reason = (f"Answers {len(answered)} of {len(q.probes)} things this shopper "
                      f"needs to know ({', '.join(answered[:3])}).")
        elif best_score <= -0.5:
            reason = ("Nothing in this catalog is the product type the shopper asked for. "
                      "Recommending the closest available item would be worse than "
                      "recommending nothing.")
        elif not any(a for _s, _c, a, _u in scored):
            reason = ("No listing answers a single one of this shopper's questions, so "
                      "there is nothing to base a recommendation on.")
        else:
            reason = "No candidate supplied enough information to recommend responsibly."

        rejections = []
        for s, c, _a, unanswered in scored:
            if winner and c.product_id == winner:
                continue
            miss = unanswered[0] if unanswered else "distinctiveness"
            rejections.append({
                "product_id": c.product_id,
                "reason": f"Cannot confirm fit: content does not address {_pretty(miss)}.",
                "missing_info": miss,
            })
        return {"winner": winner, "winner_reason": reason,
                "rejections": rejections, "confidence": round(min(0.95, best_score), 2)}


# Fields an assistant needs in order to responsibly rule a product IN or OUT
# when the shopper has signalled a risk factor.
RISK_FIELDS = ("not_suitable_for", "contraindications", "allergens",
               "known_limitations", "body_or_skin_type")

RISK_CUES = ("sensitive", "allergic", "allergy", "react", "irritat", "injur",
             "pain", "knee", "condition", "pregnan", "beginner", "first time",
             "new to", "never", "nervous", "worried", "safe", "avoid", "wide feet",
             "flat feet", "eczema", "rosacea", "asthma", "medication")


def _risk_signalled(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in RISK_CUES)


# Coarse product-type vocabulary. Deliberately not a fixed category list: it
# maps shopper words to words that appear in ANY catalog's category path, so a
# new vertical works without an entry here (it simply abstains).
TYPE_CUES = {
    "footwear": ["shoe", "shoes", "sneaker", "trainer", "running", "runner", "marathon",
                 "10k", "5k", "jog", "footwear", "boot"],
    "skincare": ["skin", "skincare", "serum", "cleanser", "moisturis", "moisturiz",
                 "spf", "sunscreen", "retinol", "routine", "face", "acne", "pores"],
    "coffee": ["coffee", "espresso", "brew", "grinder", "kettle", "barista", "pour over",
               "filter", "cup"],
}


def _category_plausible(query: str, rec: ProductRecord) -> bool:
    """
    Does this product even belong in the answer? A shopper asking about running
    shoes should never be shown a moisturiser, however well its copy scores.
    Abstains (returns True) when the query names no recognisable product type,
    so unseen categories are never wrongly excluded.
    """
    low = query.lower()
    wanted = {k for k, cues in TYPE_CUES.items() if any(c in low for c in cues)}
    if not wanted:
        return True
    cat = rec.get("category_path").value or []
    if isinstance(cat, str):
        cat = [cat]
    blob = (" ".join(str(c) for c in cat) + " " + str(rec.raw.get("category", "")) + " "
            + str(rec.raw.get("vertical", "")) + " " + rec.name()).lower()
    theirs = {k for k, cues in TYPE_CUES.items() if any(c in blob for c in cues)}
    if not theirs:
        return True
    return bool(wanted & theirs)


def _unverified_ratio(rec: ProductRecord) -> float:
    """Share of a product's filled fields that are model-generated and unapproved."""
    filled = [fv for fv in rec.fields.values() if fv.filled]
    if not filled:
        return 0.0
    gen = sum(1 for fv in filled if fv.provenance == "generated")
    return gen / len(filled)


def _pretty(field_name: str) -> str:
    return field_name.replace("_", " ")


# --------------------------------------------------------------------------
# Aggregation: turn losses into a field-level to-do list
# --------------------------------------------------------------------------

FIELD_LOOKUP = {f.lower(): f for f in FIELDS_BY_NAME}


def map_reason_to_field(missing_info: str, probes: List[str]) -> Optional[str]:
    """Cluster a free-text rejection back onto a schema field."""
    if not missing_info:
        return None
    key = missing_info.strip().lower().replace(" ", "_")
    if key in FIELD_LOOKUP:
        return FIELD_LOOKUP[key]
    text = missing_info.lower()
    for probe in probes:
        if probe.replace("_", " ") in text:
            return probe
    best, best_hits = None, 0
    for name in FIELD_LOOKUP.values():
        words = name.split("_")
        hits = sum(1 for w in words if len(w) > 3 and w in text)
        if hits > best_hits:
            best, best_hits = name, hits
    return best if best_hits else (probes[0] if probes else None)


def aggregate(results: List[QueryResult], product_id: str) -> ProductPerformance:
    seen = retrieved = won = 0
    reasons = Counter()
    gap_counter = Counter()
    lost_by_persona = defaultdict(lambda: {"seen": 0, "won": 0})

    for r in results:
        seen += 1
        in_set = product_id in r.retrieved
        if in_set:
            retrieved += 1
            lost_by_persona[r.persona]["seen"] += 1
        if r.winner == product_id:
            won += 1
            lost_by_persona[r.persona]["won"] += 1
        for rej in r.rejections:
            if rej.get("product_id") != product_id:
                continue
            reasons[rej.get("reason", "").strip()] += 1
            fld = map_reason_to_field(rej.get("missing_info", ""), r.probes)
            if fld:
                gap_counter[fld] += 1

    total_q = max(1, seen)
    return ProductPerformance(
        product_id=product_id,
        queries_seen=seen,
        retrieved=retrieved,
        won=won,
        retrieval_rate=round(retrieved / total_q * 100, 1),
        win_rate=round(won / total_q * 100, 1),
        rejection_reasons=[{"reason": k, "count": v} for k, v in reasons.most_common(10)],
        gap_fields=[{"field": k, "lost_queries": v,
                     "description": FIELDS_BY_NAME[k].description if k in FIELDS_BY_NAME else ""}
                    for k, v in gap_counter.most_common(12)],
        lost_personas=sorted(
            [{"persona": p, "seen": d["seen"], "won": d["won"],
              "win_rate": round(d["won"] / max(1, d["seen"]) * 100, 1)}
             for p, d in lost_by_persona.items()],
            key=lambda x: x["win_rate"]),
    )


def leaderboard(results: List[QueryResult], product_ids: List[str]) -> List[Dict[str, Any]]:
    wins = Counter(r.winner for r in results if r.winner)
    total = len(results)
    rows = [{"product_id": pid, "wins": wins.get(pid, 0),
             "win_rate": round(wins.get(pid, 0) / max(1, total) * 100, 1)}
            for pid in product_ids]
    rows.sort(key=lambda r: -r["wins"])
    return rows


def save_results(results: List[QueryResult], path: str) -> str:
    with open(path, "w") as fh:
        json.dump([r.__dict__ for r in results], fh, indent=1)
    return path


def load_results(path: str) -> List[QueryResult]:
    with open(path) as fh:
        return [QueryResult(**d) for d in json.load(fh)]
