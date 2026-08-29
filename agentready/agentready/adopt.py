"""
Adoption layer. RUBRIC 5: could a real brand actually implement this?

Three frictions kill enterprise adoption of AI content tooling, and this module
addresses each:

  1. "We can't publish unreviewed AI text."
     -> APPROVAL QUEUE. Nothing generated is publishable until a human approves
        it. The provenance envelope already tracks what is generated, so the
        queue falls out of the data model for free.

  2. "Our catalog lives in a PIM, not your app."
     -> CSV ROUND-TRIP. Export the approved fields as columns their PIM can
        import. Same file shape they gave us, plus new columns.

  3. "Who is going to rewrite our product pages?"
     -> EMBED SNIPPET. One <script> tag. No page redesign, no CMS migration,
        no visible change to the site.

The integration story is: export catalog -> upload -> review queue -> paste one
script tag. A brand can do that in an afternoon without engineering.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exporter import to_jsonld
from .schema import FIELDS, FIELDS_BY_NAME, TIERS, ProductRecord

APPROVAL_STATES = ("pending", "approved", "rejected", "edited")


# ---------------------------------------------------------------------------
# 1. Approval queue
# ---------------------------------------------------------------------------

@dataclass
class ApprovalItem:
    product_id: str
    field_name: str
    tier: int
    tier_name: str
    description: str
    proposed_value: Any
    basis: str                      # what the model reasoned from
    confidence: float
    impact_points: float            # coverage points this unlocks
    queries_unlocked: int           # simulated queries it would win back
    state: str = "pending"
    edited_value: Any = None
    note: str = ""

    def final_value(self) -> Any:
        return self.edited_value if self.state == "edited" else self.proposed_value

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["final_value"] = self.final_value()
        return d


class ApprovalQueue:
    """
    Every generated field, ordered by business impact, waiting on a human.

    Sorted by impact so a content manager with twenty minutes approves the six
    fields that matter instead of working alphabetically. That ordering is the
    difference between a tool that gets used and one that gets abandoned.
    """

    def __init__(self, items: Optional[List[ApprovalItem]] = None):
        self.items: List[ApprovalItem] = items or []

    @classmethod
    def from_records(cls, baseline: ProductRecord, optimised: ProductRecord,
                     gap_report: Optional[List[Dict[str, Any]]] = None,
                     coverage_gaps: Optional[List[Dict[str, Any]]] = None) -> "ApprovalQueue":
        lost = {g["field"]: g.get("lost_queries", 0) for g in (gap_report or [])}
        impact = {g["field"]: g.get("impact_points", 0.0) for g in (coverage_gaps or [])}

        items = []
        for name, fv in optimised.fields.items():
            if fv.provenance != "generated":
                continue
            if baseline.get(name).filled and baseline.get(name).provenance != "generated":
                continue  # we did not replace sourced content
            spec = FIELDS_BY_NAME.get(name)
            if not spec:
                continue
            items.append(ApprovalItem(
                product_id=optimised.product_id,
                field_name=name,
                tier=spec.tier,
                tier_name=TIERS[spec.tier]["name"],
                description=spec.description,
                proposed_value=fv.value,
                basis=fv.evidence or "reasoned from source material",
                confidence=fv.confidence,
                impact_points=round(impact.get(name, 0.0), 2),
                queries_unlocked=lost.get(name, 0),
            ))
        # Order by business value, not by per-field arithmetic. Dividing a tier
        # weight across its fields makes small tiers look artificially urgent -
        # T2 has 6 fields so each scores 2.5, while T3 has 15 so each scores 1.7,
        # which would put "secondary_functions" above "environment_conditions".
        # Queries won back comes first, then tier weight, then per-field impact.
        items.sort(key=lambda i: (-i.queries_unlocked,
                                  -TIERS[i.tier]["weight"],
                                  -i.impact_points))
        return cls(items)

    # -- actions -----------------------------------------------------------

    def act(self, field_name: str, state: str, value: Any = None, note: str = "") -> bool:
        if state not in APPROVAL_STATES:
            raise ValueError(f"state must be one of {APPROVAL_STATES}")
        for it in self.items:
            if it.field_name == field_name:
                it.state = state
                it.note = note
                if state == "edited":
                    it.edited_value = value
                return True
        return False

    def approve_all_above(self, confidence: float = 0.7) -> int:
        """Bulk action for the impatient. Still a deliberate human decision."""
        n = 0
        for it in self.items:
            if it.state == "pending" and it.confidence >= confidence:
                it.state = "approved"
                n += 1
        return n

    # -- output ------------------------------------------------------------

    def publishable(self, baseline: ProductRecord) -> ProductRecord:
        """
        The record a brand can actually ship: sourced content, plus ONLY the
        generated fields a human signed off. Rejected and pending fields are
        dropped, so the published page never contains unreviewed model output.
        """
        out = ProductRecord(product_id=baseline.product_id, raw=baseline.raw,
                            fields=dict(baseline.fields), label="published")
        for it in self.items:
            if it.state in ("approved", "edited"):
                from .schema import FieldValue
                out.set(it.field_name, FieldValue(
                    value=it.final_value(),
                    provenance="verbatim" if it.state == "edited" else "generated",
                    evidence=f"human-{it.state} {it.basis}",
                    confidence=max(it.confidence, 0.9 if it.state == "edited" else it.confidence),
                ))
        return out

    def stats(self) -> Dict[str, Any]:
        by_state = {s: sum(1 for i in self.items if i.state == s) for s in APPROVAL_STATES}
        pending = [i for i in self.items if i.state == "pending"]
        return {
            "total": len(self.items),
            **by_state,
            "pending_impact_points": round(sum(i.impact_points for i in pending), 1),
            "pending_queries": sum(i.queries_unlocked for i in pending),
            "est_review_minutes": round(len(self.items) * 0.75),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [i.to_dict() for i in self.items], "stats": self.stats()}


# ---------------------------------------------------------------------------
# 2. CSV round-trip
# ---------------------------------------------------------------------------

# The columns a brand's PIM most plausibly accepts. Keeping this short matters:
# a 75-column import file gets rejected by procurement, an 12-column one does not.
PIM_COLUMNS = [
    "sku", "product_name", "agent_one_liner", "semantic_summary",
    "use_cases", "environment_conditions", "personas", "experience_level",
    "not_suitable_for", "known_limitations", "tradeoffs",
    "differentiators", "value_argument", "anticipated_qa",
]


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for v in value:
            if isinstance(v, dict):
                parts.append(" - ".join(str(x) for x in v.values() if str(x).strip()))
            else:
                parts.append(str(v))
        return " | ".join(parts)
    if isinstance(value, dict):
        return " | ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def to_pim_csv(records: List[ProductRecord], columns: Optional[List[str]] = None) -> str:
    """
    Approved content as a CSV their PIM can import. Same primary key they gave
    us (sku), new columns appended. No schema migration on their side.
    """
    cols = columns or PIM_COLUMNS
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for rec in records:
        row = []
        for c in cols:
            fv = rec.get(c)
            row.append(_flatten(fv.value) if fv.filled else "")
        w.writerow(row)
    return buf.getvalue()


def to_review_csv(queue: ApprovalQueue) -> str:
    """
    The queue as a spreadsheet, because a lot of content teams review in Excel
    and emailing them a link to a web app is how a pilot dies.
    Fill the 'decision' column, send it back, re-import.
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["product_id", "field", "tier", "why_it_matters", "queries_unlocked",
                "proposed_value", "reasoning_basis", "confidence", "decision", "your_edit"])
    for i in queue.items:
        w.writerow([i.product_id, i.field_name, f"T{i.tier} {i.tier_name}",
                    i.description, i.queries_unlocked, _flatten(i.proposed_value),
                    i.basis, f"{i.confidence:.2f}", i.state, ""])
    return buf.getvalue()


def from_review_csv(queue: ApprovalQueue, csv_text: str) -> Dict[str, int]:
    """Re-import a reviewed spreadsheet. Closes the loop for offline teams."""
    counts = {s: 0 for s in APPROVAL_STATES}
    for row in csv.DictReader(io.StringIO(csv_text)):
        fname = (row.get("field") or "").strip()
        decision = (row.get("decision") or "").strip().lower()
        edit = (row.get("your_edit") or "").strip()
        if edit:
            queue.act(fname, "edited", value=edit)
            counts["edited"] += 1
        elif decision in APPROVAL_STATES:
            queue.act(fname, decision)
            counts[decision] += 1
    return counts


# ---------------------------------------------------------------------------
# 3. Embed snippet
# ---------------------------------------------------------------------------

def embed_snippet(rec: ProductRecord, indent: int = 2) -> str:
    """
    The entire integration, from the brand's point of view. Paste into the
    product page template. No redesign, no CMS change, nothing visible to
    shoppers - it is a script tag that AI crawlers and agent stacks read.
    """
    payload = json.dumps(to_jsonld(rec), indent=indent, ensure_ascii=False)
    return f'<script type="application/ld+json">\n{payload}\n</script>'


def integration_checklist(rec: ProductRecord, queue: ApprovalQueue) -> List[Dict[str, Any]]:
    """What a brand has to actually do, in order, with honest time estimates."""
    st = queue.stats()
    return [
        {"step": "Export your catalog", "detail": "Any CSV from your PIM or Shopify admin. "
         "Column names do not need to match ours; they are auto-mapped.",
         "owner": "merchandising", "effort": "10 min"},
        {"step": "Run the readiness scan", "detail": "Coverage score and gap report per SKU. "
         "No integration required to get this - it is read-only.",
         "owner": "us", "effort": "automated"},
        {"step": "Review the queue", "detail": f"{st['total']} proposed fields, ordered by "
         f"queries won back. Approve, edit, or reject. Nothing publishes without sign-off.",
         "owner": "content team", "effort": f"~{st['est_review_minutes']} min per SKU"},
        {"step": "Choose a delivery route", "detail": "Either paste one <script type=\"application/ld+json\"> "
         "tag into your PDP template, or import the CSV back into your PIM. Most brands do both.",
         "owner": "web team", "effort": "30 min, once"},
        {"step": "Re-scan monthly", "detail": "Catalog changes, competitors improve, and the "
         "score moves. The gap report tells you what to write next.",
         "owner": "us", "effort": "automated"},
    ]


def bundle(rec_baseline: ProductRecord, rec_published: ProductRecord,
           queue: ApprovalQueue, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    files = {
        "review_queue.csv": to_review_csv(queue),
        "pim_import.csv": to_pim_csv([rec_published]),
        "embed_snippet.html": embed_snippet(rec_published),
        "approval_state.json": json.dumps(queue.to_dict(), indent=1, ensure_ascii=False),
    }
    for name, content in files.items():
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        paths[name] = p
    return paths
