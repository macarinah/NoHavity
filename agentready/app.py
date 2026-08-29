"""
AgentReady - console.

    streamlit run app.py

Design principles for this file:
  - Plain English first, jargon second. Every schema term gets a human label.
  - One idea per panel. If a panel needs a paragraph to explain, it is too dense.
  - Colour means exactly one thing: green = good/won, amber = gap, rose = lost.
  - Every tab opens with one sentence saying what you are looking at.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentready.adopt import (ApprovalQueue, embed_snippet,  # noqa: E402
                              integration_checklist, to_pim_csv, to_review_csv)
from agentready.coach import (improvement_brief, merchant_questions,  # noqa: E402
                              rewrite_suggestions)
from agentready.console import compare_console  # noqa: E402
from agentready.exporter import to_jsonld, to_markdown  # noqa: E402
from agentready.extractor import Extractor  # noqa: E402
from agentready.generator import Generator  # noqa: E402
from agentready.llm import LLM  # noqa: E402
from agentready.queries import generate as gen_queries  # noqa: E402
from agentready.schema import FIELDS, FIELDS_BY_NAME, FieldValue, ProductRecord  # noqa: E402
from agentready.scorer import rank_gaps, score  # noqa: E402
from agentready.simulator import Simulator, agent_view, aggregate  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(ROOT, "data", "products.json")

st.set_page_config(page_title="AgentReady", page_icon="◆", layout="wide")

# ==========================================================================
# Plain-English vocabulary. The schema uses precise names; humans need labels.
# ==========================================================================

TIER_PLAIN = {
    0: ("What it is",          "Name, brand, price, category"),
    1: ("The specs",           "Weight, materials, size, ingredients"),
    2: ("What it does",        "Its job, features, and how they help"),
    3: ("Who it's for & when",  "Situations, climates, shopper types, timing"),
    4: ("Who it's NOT for",    "Limits, tradeoffs, allergens, wrong use cases"),
    5: ("Why this one",        "How it compares, why the price is fair"),
    6: ("Proof it works",      "Reviews, certifications, test results"),
}

TIER_WHY = {
    0: "Every catalog already has this. It gets you into the list, nothing more.",
    1: "An assistant can filter on these, but it can't reason about whether they suit you.",
    2: "Features only help if you also say what benefit they give the shopper.",
    3: "This is how people actually ask. Without it, the assistant can't match you to a situation.",
    4: "Sounds risky to publish, but it's the opposite: an assistant that can rule you out confidently can also rule you in confidently.",
    5: "When three products look similar, this is what breaks the tie.",
    6: "Lets the assistant cite a reason back to the shopper instead of just asserting.",
}

FIELD_PLAIN = {
    "environment_conditions": "What conditions it's built for",
    "climate_suitability": "Which climate it suits",
    "use_cases": "Situations it's meant for",
    "personas": "Types of shopper it fits",
    "experience_level": "Beginner or expert",
    "time_commitment": "How long it takes to use",
    "not_suitable_for": "Who should NOT buy it",
    "known_limitations": "Honest limitations",
    "tradeoffs": "What you give up",
    "allergens": "Allergens present",
    "contraindications": "What not to combine it with",
    "body_or_skin_type": "Body or skin types it suits",
    "durability_expectation": "How long it lasts",
    "differentiators": "How it differs from the norm",
    "why_choose_over": "Why pick this over alternatives",
    "value_argument": "Why the price is fair",
    "price_justification": "Cost per use",
    "pairs_well_with": "What to use alongside it",
    "compatible_with": "What it works with",
    "learning_curve": "How hard it is to start",
    "common_complaints": "What buyers complain about",
    "verified_claims": "Claims backed by a source",
    "performance_claims": "Claims with real numbers",
    "review_themes": "Recurring themes in reviews",
    "key_features": "Features and what they do for you",
    "primary_function": "Its main job",
    "materials": "What it's made of",
    "care_instructions": "How to look after it",
    "season": "Seasons it suits",
    "occasion": "Occasions it suits",
}


def plain(field_name: str) -> str:
    if field_name in FIELD_PLAIN:
        return FIELD_PLAIN[field_name]
    return field_name.replace("_", " ").capitalize()


def verdict(s: float) -> tuple:
    if s >= 65:
        return ("Ready", "An assistant has what it needs to recommend this confidently.", "good")
    if s >= 40:
        return ("Getting there", "An assistant can describe this, but still can't match it to a specific shopper.", "mid")
    if s >= 20:
        return ("Not ready", "An assistant knows what this is, but nothing about who it's for.", "bad")
    return ("Invisible", "An assistant has almost nothing to reason with. It will lose to competitors that do.", "bad")


# ==========================================================================
# Styling. Calmer than before: one accent for good, one for gaps, one for lost.
# ==========================================================================

CSS = """
<style>
:root{
  --ground:#10141F; --panel:#1A2030; --panel2:#151A28; --edge:#2B3244;
  --ink:#E9ECF2; --mute:#8B93A7; --faint:#5C6478;
  --good:#4FD1A5; --gap:#E8A33D; --bad:#D9647A; --info:#6BA3DD;
}
.stApp{background:var(--ground);}
html,body,[class*="css"]{color:var(--ink);}
h1,h2,h3{letter-spacing:-0.02em;font-weight:600;}
.stTabs [data-baseweb="tab"]{font-size:14px;}

.card{background:var(--panel);border:1px solid var(--edge);border-radius:12px;
  padding:20px 22px;margin-bottom:16px;}
.card.tight{padding:14px 16px;}

.lead{color:var(--mute);font-size:14px;line-height:1.6;margin:2px 0 18px;max-width:760px;}
.label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--faint);margin-bottom:10px;font-weight:600;}

.big{font-size:52px;font-weight:600;line-height:1;letter-spacing:-0.03em;}
.med{font-size:30px;font-weight:600;line-height:1;}
.unit{font-size:17px;color:var(--faint);font-weight:400;}
.good{color:var(--good);} .gap{color:var(--gap);} .bad{color:var(--bad);}
.num{font-variant-numeric:tabular-nums;}

/* tier rows */
.trow{margin:16px 0;}
.thead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}
.tname{font-size:15px;font-weight:500;}
.tsub{font-size:12.5px;color:var(--faint);margin-left:8px;font-weight:400;}
.tcount{font-size:13px;color:var(--mute);font-variant-numeric:tabular-nums;}
.bar{height:10px;background:#11151F;border-radius:5px;overflow:hidden;position:relative;}
.bar span{display:block;height:100%;border-radius:5px;}
.twhy{font-size:12.5px;color:var(--faint);margin-top:6px;line-height:1.5;}
.keyrow{border-left:2px solid var(--gap);padding-left:14px;}

/* verdict banner */
.verdict{border-radius:12px;padding:16px 20px;margin-bottom:16px;
  border:1px solid var(--edge);background:var(--panel);}
.vtitle{font-size:17px;font-weight:600;margin-bottom:4px;}
.vtext{font-size:14px;color:var(--mute);line-height:1.6;}

/* result blocks */
.res{border-radius:10px;padding:14px 16px;margin:10px 0;border:1px solid var(--edge);}
.res.won{background:#14251F;border-color:#2A5145;}
.res.lost{background:#241820;border-color:#4A2C38;}
.res.neutral{background:var(--panel2);}
.badge{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;padding:3px 9px;border-radius:5px;margin-bottom:8px;}
.badge.won{background:#1E4034;color:var(--good);}
.badge.lost{background:#3D2029;color:var(--bad);}
.badge.info{background:#1C2A3C;color:var(--info);}
.q{font-size:14.5px;line-height:1.6;}
.why{font-size:13px;color:var(--mute);margin-top:8px;line-height:1.55;}

/* gaps */
.gapitem{border-left:3px solid var(--gap);padding:12px 16px;margin:10px 0;
  background:var(--panel2);border-radius:0 8px 8px 0;}
.gtitle{font-size:14.5px;font-weight:500;}
.gmeta{font-size:12.5px;color:var(--faint);margin-top:3px;}

.chip{display:inline-block;background:#20283A;color:var(--mute);padding:4px 10px;
  border-radius:6px;font-size:12.5px;margin:0 6px 6px 0;}

.stripe{display:flex;gap:3px;flex-wrap:wrap;margin:12px 0 8px;}
.tick{width:13px;height:26px;border-radius:3px;background:#252C3C;}
.tick.verbatim{background:var(--good);}
.tick.inferred{background:var(--info);}
.tick.generated{background:var(--gap);}
.legend{font-size:12.5px;color:var(--mute);display:flex;gap:18px;
  margin-top:10px;flex-wrap:wrap;}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;}

.arrowcol{text-align:center;color:var(--faint);font-size:26px;padding-top:60px;}
hr{border-color:var(--edge);margin:22px 0;}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Session catalog
# ==========================================================================

def _load_seed():
    with open(CATALOG) as fh:
        return json.load(fh)


if "catalog" not in st.session_state:
    st.session_state.catalog = _load_seed()
if "edits" not in st.session_state:
    st.session_state.edits = {}
if "approvals" not in st.session_state:
    st.session_state.approvals = {}

llm = LLM()
if llm.live and "preflight_done" not in st.session_state:
    llm.preflight(verbose=False)
    st.session_state.preflight_done = True
    st.session_state.llm_disabled = llm.disabled_reason
if st.session_state.get("llm_disabled"):
    st.warning(f"Live API unavailable ({st.session_state.llm_disabled}). "
               f"Running in offline mode — everything still works.")


def catalog_key() -> str:
    blob = json.dumps(st.session_state.catalog, sort_keys=True, default=str)
    blob += json.dumps(st.session_state.edits, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def analyse(key: str, catalog_json: str, edits_json: str):
    products = json.loads(catalog_json)
    edits = json.loads(edits_json)
    ex, gen = Extractor(llm), Generator(llm)
    baseline = ex.extract_many(products)

    for rec in baseline:
        for fname, val in (edits.get(rec.product_id) or {}).items():
            rec.set(fname, FieldValue(value=val, provenance="verbatim",
                                      evidence="written by the brand", confidence=1.0))

    queries = gen_queries()
    if llm.live:
        queries = queries[::5]
    sim = Simulator(llm)
    res_before = sim.run(baseline, queries)

    optimised, perf = [], {}
    for rec in baseline:
        p = aggregate(res_before, rec.product_id)
        perf[rec.product_id] = p.__dict__
        gaps = [g["field"] for g in p.gap_fields]
        gaps += [g["field"] for g in rank_gaps(rec, limit=60) if g["field"] not in gaps]
        optimised.append(gen.optimise(rec, gaps))

    res_after = sim.run(optimised, queries)
    return {
        "baseline": [r.to_dict() for r in baseline],
        "optimised": [r.to_dict() for r in optimised],
        "scores": {r.product_id: score(r).as_dict() for r in baseline},
        "scores_after": {r.product_id: score(r).as_dict() for r in optimised},
        "perf": perf,
        "perf_after": {r.product_id: aggregate(res_after, r.product_id).__dict__
                       for r in optimised},
        "results": [r.__dict__ for r in res_before],
        "n_queries": len(queries),
    }


with st.spinner("Analysing your catalog..."):
    A = analyse(catalog_key(),
                json.dumps(st.session_state.catalog, default=str),
                json.dumps(st.session_state.edits, default=str))

BASE = {d["product_id"]: ProductRecord.from_dict(d) for d in A["baseline"]}
OPT = {d["product_id"]: ProductRecord.from_dict(d) for d in A["optimised"]}


# ==========================================================================
# Sidebar
# ==========================================================================

st.sidebar.markdown("### AgentReady")
st.sidebar.caption("Is your product content ready for AI shopping assistants?")
st.sidebar.markdown("---")

ids = list(BASE.keys())
sel = st.sidebar.selectbox("Viewing product", ids, key="selected_product",
                           format_func=lambda p: BASE[p].name()[:36])
rec, opt = BASE[sel], OPT[sel]
sc, sc_after = A["scores"][sel], A["scores_after"][sel]
perf, perf_after = A["perf"][sel], A["perf_after"][sel]
vlabel, vtext, vkind = verdict(sc["score"])

st.sidebar.markdown(
    f"<div class='card tight'><div class='label'>Readiness</div>"
    f"<div class='med num {'good' if vkind=='good' else ('gap' if vkind=='mid' else 'bad')}'>"
    f"{sc['score']:.0f}<span class='unit'>/100</span></div>"
    f"<div style='font-size:13px;color:var(--mute);margin-top:6px'>{vlabel}</div></div>"
    f"<div class='card tight'><div class='label'>Recommended by AI</div>"
    f"<div class='med num'>{perf['win_rate']:.0f}%</div>"
    f"<div style='font-size:13px;color:var(--mute);margin-top:6px'>"
    f"of {A['n_queries']} shopper questions</div></div>",
    unsafe_allow_html=True)

st.sidebar.caption(f"{len(BASE)} products loaded")
if st.sidebar.button("Reset to sample catalog"):
    st.session_state.catalog = _load_seed()
    st.session_state.edits, st.session_state.approvals = {}, {}
    st.rerun()


def stripe(r: ProductRecord) -> str:
    ticks = "".join(
        f'<div class="tick {r.get(f.name).provenance if r.get(f.name).filled else ""}" '
        f'title="{plain(f.name)}"></div>'
        for f in FIELDS if f.tier != 8)
    return (f'<div class="stripe">{ticks}</div>'
            '<div class="legend">'
            '<span><span class="sw" style="background:var(--good)"></span>you wrote it</span>'
            '<span><span class="sw" style="background:var(--info)"></span>we inferred it</span>'
            '<span><span class="sw" style="background:var(--gap)"></span>AI drafted it</span>'
            '<span><span class="sw" style="background:#252C3C"></span>still missing</span>'
            "</div>")


st.markdown(f"# {rec.name()}")
st.markdown(f"<div class='lead'>{rec.raw.get('category','')} · "
            f"{rec.get('currency').value or 'SGD'} {rec.price() or 0:g}</div>",
            unsafe_allow_html=True)

t_cat, t_live, t_ready, t_sim, t_gaps, t_edit, t_pub = st.tabs(
    ["1 · My catalog", "2 · Live query", "3 · Readiness", "4 · Simulation",
     "5 · Gaps & coaching", "6 · Edit content", "7 · Publish"])


# ==========================================================================
# 1. My catalog
# ==========================================================================
with t_cat:
    st.markdown("### Add your products")
    st.markdown("<div class='lead'>Everything you add here shows up in the "
                "<b>Viewing product</b> dropdown on the left, and flows through every "
                "other tab.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Add one product**")
        with st.form("add_product", clear_on_submit=True):
            name = st.text_input("Product name *", placeholder="Meridian Trail Runner 2")
            brand = st.text_input("Brand", placeholder="Meridian")
            cat = st.text_input("Category", placeholder="Footwear > Running > Trail")
            cc1, cc2 = st.columns(2)
            price = cc1.number_input("Price", min_value=0.0, value=0.0, step=1.0)
            curr = cc2.text_input("Currency", value="SGD")
            desc = st.text_area("Product description *", height=140,
                                placeholder="Paste your existing product page copy here...")
            bullets = st.text_area("Spec bullets (one per line)", height=80,
                                   placeholder="Weight: 265g\nDrop: 8mm")
            reviews = st.text_area("Customer reviews (one per line)", height=68)
            go = st.form_submit_button("Add to catalog", type="primary")

        if go:
            if not name.strip() or not desc.strip():
                st.error("Product name and description are both required.")
            else:
                n = len([p for p in st.session_state.catalog
                         if str(p.get("product_id", "")).startswith("user")]) + 1
                st.session_state.catalog.append({
                    "product_id": f"user{n:03d}", "product_name": name.strip(),
                    "brand": brand.strip() or "Unbranded",
                    "category": cat.strip() or "General",
                    "price": float(price) or None, "currency": curr.strip() or "SGD",
                    "pdp_text": desc.strip(),
                    "bullet_specs": [b.strip() for b in bullets.splitlines() if b.strip()],
                    "reviews": [r.strip() for r in reviews.splitlines() if r.strip()],
                    "vertical": (cat.split(">")[0].strip().lower() or "general"),
                    "availability": "in_stock"})
                st.success(f"Added. Pick it in the sidebar to see its score.")
                st.rerun()

    with c2:
        st.markdown("**Or upload a catalog CSV**")
        st.markdown("<div class='lead'>Your column names don't need to match ours. "
                    "\"Item Name\", \"Retail Price SGD\" and \"Long Description\" are all "
                    "recognised automatically.</div>", unsafe_allow_html=True)
        up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        replace = st.checkbox("Replace my catalog instead of adding to it")
        if up is not None and st.button("Import CSV"):
            import csv as _csv
            import io as _io
            import tempfile
            from agentready.ingest import from_csv, sniff_columns
            with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as fh:
                fh.write(up.getvalue()); tmp = fh.name
            try:
                headers = next(_csv.reader(_io.StringIO(up.getvalue().decode("utf-8-sig"))))
                mapping = sniff_columns(headers)
                prods = from_csv(tmp, report=False)
                for i, p in enumerate(prods):
                    p["product_id"] = f"up{i+1:03d}"
                st.session_state.catalog = prods if replace else st.session_state.catalog + prods
                st.session_state.edits = {}
                st.success(f"Imported {len(prods)} products. "
                           f"Matched {len(mapping)} of {len(headers)} columns automatically.")
                st.rerun()
            finally:
                os.unlink(tmp)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**In your catalog now**")
        for pid, r in BASE.items():
            s = A["scores"][pid]["score"]
            col = "good" if s >= 65 else ("gap" if s >= 40 else "bad")
            here = " ← viewing" if pid == sel else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:7px 0;"
                f"border-bottom:1px solid var(--edge);font-size:13.5px'>"
                f"<span>{r.name()[:38]}<span style='color:var(--info)'>{here}</span></span>"
                f"<span class='num {col}'>{s:.0f}</span></div>", unsafe_allow_html=True)


# ==========================================================================
# 2. Live query  — explicit before / after
# ==========================================================================
with t_live:
    st.markdown("### Ask what a shopper would ask")
    st.markdown("<div class='lead'>We run your question twice: once against your "
                "listing <b>as it is today</b>, and once against the <b>same listing with "
                "its gaps filled in</b>. Compare the two answers.</div>",
                unsafe_allow_html=True)

    EXAMPLES = [
        "I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200",
        "Sustainable skincare routine for oily skin that takes under 5 minutes every morning",
        "I have wide feet and keep getting blisters on long runs",
        "I have sensitive reactive skin and I'm nervous about starting retinol",
    ]
    pick = st.selectbox("Pick an example, or write your own below",
                        ["Write my own"] + EXAMPLES)
    query = st.text_area("Your question", value="" if pick == "Write my own" else pick,
                         height=76, label_visibility="collapsed")

    if st.button("Ask the AI assistant", type="primary") and query.strip():
        with st.spinner("Thinking..."):
            res = compare_console(list(BASE.values()), list(OPT.values()),
                                  query.strip(), sel, llm)

        st.markdown("<div class='label'>What the assistant needs to know to answer this</div>",
                    unsafe_allow_html=True)
        st.markdown("".join(f"<span class='chip'>{plain(p)}</span>"
                            for p in res["before"]["probes"]), unsafe_allow_html=True)
        if res["before"]["price_cap"]:
            st.caption(f"It also picked up your budget: under {res['before']['price_cap']:.0f}")
        st.markdown("<hr>", unsafe_allow_html=True)

        cB, cMid, cA = st.columns([5, 1, 5])
        for col, key, title, sub in (
                (cB, "before", "BEFORE", "Your listing as it is today"),
                (cA, "after", "AFTER", "Same product, gaps filled in")):
            with col:
                r = res[key]
                mine = (r["focus"] or {})
                won = r["winner"] == sel
                st.markdown(
                    f"<div class='label'>{title}</div>"
                    f"<div style='color:var(--mute);font-size:13px;margin-bottom:12px'>{sub}</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div class='res {'won' if won else 'lost'}'>"
                    f"<div class='badge {'won' if won else 'lost'}'>"
                    f"{'Your product recommended' if won else 'Your product not recommended'}</div>"
                    f"<div style='font-size:15px;font-weight:500'>"
                    f"Assistant picked: {r['winner_name'] or 'nothing at all'}</div>"
                    f"<div class='why'>{r['winner_reason'][:210]}</div></div>",
                    unsafe_allow_html=True)
                if mine:
                    pct = mine.get("coverage_of_query", 0)
                    st.markdown(
                        f"<div style='font-size:13px;color:var(--mute);margin-top:10px'>"
                        f"Your listing answers <b class='num "
                        f"{'good' if pct >= 60 else 'gap'}'>{pct}%</b> of what "
                        f"this shopper asked.</div>", unsafe_allow_html=True)
        cMid.markdown("<div class='arrowcol'>→</div>", unsafe_allow_html=True)

        if res["flipped"]:
            st.markdown(
                f"<div class='verdict' style='border-color:#2A5145;background:#14251F'>"
                f"<div class='vtitle good'>That's the difference</div>"
                f"<div class='vtext'>Same product, same question. It went from "
                f"<b>not recommended</b> to <b>recommended</b> purely because these "
                f"were written down: "
                f"{', '.join(plain(f) for f in res['gained_fields'][:5])}.</div></div>",
                unsafe_allow_html=True)

        mine_before = (res["before"].get("focus") or {})
        unanswered = mine_before.get("unanswered", [])
        if unanswered:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### If you're the shopper, ask the merchant this")
            st.markdown("<div class='lead'>The listing can't answer these. Send them to "
                        "the brand — and if enough shoppers do, the content gets "
                        "fixed.</div>", unsafe_allow_html=True)
            qs = merchant_questions(rec, unanswered, llm)
            for q in qs:
                st.markdown(f"<div class='res neutral'><div class='q'>“{q['question']}”</div>"
                            f"<div class='why'>{q.get('why','')[:150]}</div></div>",
                            unsafe_allow_html=True)
            st.download_button("Copy these questions",
                               "\n".join(f"- {q['question']}" for q in qs),
                               file_name="questions_for_merchant.txt")


# ==========================================================================
# 3. Readiness — tiers explained in plain English
# ==========================================================================
with t_ready:
    st.markdown("### How ready is this listing?")
    st.markdown("<div class='lead'>We check your product against 75 pieces of "
                "information an AI assistant needs. The score is how many of them you've "
                "actually published, weighted by how much each one matters.</div>",
                unsafe_allow_html=True)

    vcol = {"good": "good", "mid": "gap", "bad": "bad"}[vkind]
    st.markdown(f"<div class='verdict'><div class='vtitle {vcol}'>"
                f"{sc['score']:.0f}/100 — {vlabel}</div>"
                f"<div class='vtext'>{vtext}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='label'>Where you stand, category by category</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='lead'>The bars aren't equal in importance. The two "
                "highlighted rows are worth 45% of the score between them, and they're "
                "the two almost no brand publishes.</div>", unsafe_allow_html=True)

    rows = []
    for t in sc["tiers"]:
        tier = t["tier"]
        name, sub = TIER_PLAIN[tier]
        pct = t["quality_fill"] * 100
        colour = "var(--good)" if pct >= 60 else ("var(--gap)" if pct >= 25 else "var(--bad)")
        key = " keyrow" if tier in (3, 4) else ""
        rows.append(
            f"<div class='trow{key}'>"
            f"<div class='thead'><div><span class='tname'>{name}</span>"
            f"<span class='tsub'>{sub}</span></div>"
            f"<div class='tcount'>{t['filled']} of {t['total']} written · "
            f"{t['weight']*100:.0f}% of score</div></div>"
            f"<div class='bar'><span style='width:{pct:.0f}%;background:{colour}'></span></div>"
            f"<div class='twhy'>{TIER_WHY[tier]}</div></div>")
    st.markdown("<div class='card'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='label'>All 75 pieces of information at a glance</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='card'>{stripe(rec)}</div>", unsafe_allow_html=True)

    with st.expander("What the AI assistant actually reads about this product"):
        st.caption("This is the entire picture an assistant has. Anything not here "
                   "does not exist as far as it's concerned.")
        st.code(agent_view(rec), language=None)

    with st.expander("How is the score calculated?"):
        st.markdown("""
Three things go into it:

1. **Did you write it?** Each of the 75 pieces is either present or missing.
2. **How much does it matter?** Categories are weighted. "Who it's for & when"
   is worth 25%; "What it is" is worth 5%, because every catalog already has that.
3. **Is it actually specific?** Vague marketing words like *premium* or
   *revolutionary* score close to zero. Numbers and named conditions score full marks.

Content you write yourself scores higher than content the AI drafts for you,
because a person vouched for it.
        """)


# ==========================================================================
# 4. Simulation
# ==========================================================================
with t_sim:
    st.markdown("### Tested against shopper questions")
    st.markdown(f"<div class='lead'>We generated {A['n_queries']} realistic shopper "
                f"questions from {len(set(r['persona'] for r in A['results']))} different "
                f"types of buyer, then had an AI assistant choose a product for each one. "
                f"Here's how yours did.</div>", unsafe_allow_html=True)

    seen = sum(1 for r in A["results"] if sel in r["retrieved"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions where you were a candidate", f"{seen}",
              f"{perf['retrieval_rate']:.0f}% of all questions")
    c2.metric("Questions you won", f"{perf['won']}",
              f"{perf['win_rate']:.0f}% of all questions")
    c3.metric("If you filled your gaps", f"{perf_after['win_rate']:.0f}%",
              f"{perf_after['win_rate'] - perf['win_rate']:+.0f} points")

    st.markdown("<hr>", unsafe_allow_html=True)
    personas = sorted({r["persona"] for r in A["results"]})
    f1, f2 = st.columns([2, 1])
    pf = f1.selectbox("Filter by shopper type", ["Everyone"] + personas)
    only = f2.selectbox("Show", ["All questions", "Only the ones I lost",
                                 "Only the ones I won"])

    st.caption("Green = your product was recommended.  Red = it wasn't.")

    shown = 0
    for r in A["results"]:
        if pf != "Everyone" and r["persona"] != pf:
            continue
        if sel not in r["retrieved"]:
            continue
        won = r["winner"] == sel
        if only == "Only the ones I lost" and won:
            continue
        if only == "Only the ones I won" and not won:
            continue
        rej = next((x for x in r["rejections"] if x.get("product_id") == sel), None)
        name = BASE[r["winner"]].name() if r["winner"] in BASE else "nothing at all"
        st.markdown(
            f"<div class='res {'won' if won else 'lost'}'>"
            f"<div class='badge {'won' if won else 'lost'}'>"
            f"{'You won this' if won else 'You lost this'}</div>"
            f"<div class='q'>“{r['query']}”</div>"
            f"<div class='why'><b>Assistant chose:</b> {name}<br>{r['winner_reason'][:180]}"
            + (f"<br><span class='gap'>Why yours was passed over:</span> {rej['reason']}"
               if rej else "") +
            "</div></div>", unsafe_allow_html=True)
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        st.info("Nothing matches that filter. If you were never even a candidate, "
                "that's the finding — check the Gaps tab.")


# ==========================================================================
# 5. Gaps & coaching
# ==========================================================================
with t_gaps:
    st.markdown("### What's missing, and how to fix it")
    st.markdown("<div class='lead'>Every question you lost gets traced back to the one "
                "piece of information that would have answered it. This is your writing "
                "to-do list, in priority order.</div>", unsafe_allow_html=True)

    gaps = perf["gap_fields"] or sc["top_gaps"]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<div class='label'>Biggest gaps first</div>", unsafe_allow_html=True)
        for g in gaps[:7]:
            lost = g.get("lost_queries", 0)
            meta = (f"cost you {lost} shopper questions" if lost
                    else f"worth {g.get('impact_points', 0):.1f} points")
            st.markdown(f"<div class='gapitem'><div class='gtitle'>{plain(g['field'])}</div>"
                        f"<div class='gmeta'>{meta}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='label'>Shoppers you're losing</div>", unsafe_allow_html=True)
        for p in perf["lost_personas"][:5]:
            label = p["persona"].replace("_", " ").capitalize()
            st.markdown(f"<div class='gapitem'><div class='gtitle'>{label}</div>"
                        f"<div class='gmeta'>won {p['won']} of {p['seen']} "
                        f"({p['win_rate']:.0f}%)</div></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### How to write each one")
    st.caption("Open any gap for guidance, plus a prompt you can paste into ChatGPT "
               "or your own tools to draft it.")
    for b in improvement_brief(rec, gaps, llm):
        with st.expander(f"{plain(b['field'])}"):
            st.markdown(f"**What this means** — {b['what_it_is']}")
            st.markdown(f"**How to write it** — {b['how_to_write_it']}")
            st.markdown(f"**Why it's worth doing** — {b['why_it_matters']}")
            st.caption("Paste this into any AI tool to get a first draft:")
            st.code(b["draft_prompt"], language=None)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### What's weak in the copy you already have")
    st.caption("These are things you've written that an assistant can't use.")
    for s in rewrite_suggestions(rec, llm):
        st.markdown(f"<div class='gapitem'><div style='font-size:13.5px;color:var(--mute)'>"
                    f"“{s['excerpt']}”</div>"
                    f"<div class='gtitle gap' style='margin-top:8px'>{s['issue']}</div>"
                    f"<div class='gmeta'>{s['fix']}</div></div>", unsafe_allow_html=True)


# ==========================================================================
# 6. Edit content — now covers ALL metadata
# ==========================================================================
with t_edit:
    st.markdown("### Edit your product")
    st.markdown("<div class='lead'>Change anything here and the whole dashboard "
                "recalculates. Text you write yourself counts as <b>verified</b>, which "
                "scores higher than an AI draft — because a person stands behind it.</div>",
                unsafe_allow_html=True)

    raw = next((p for p in st.session_state.catalog if p.get("product_id") == sel), None)

    if not llm.live:
        st.info("**Offline mode.** Editing the basic details, specs and reviews updates "
                "the score right away. Rewriting the description won't move it much, "
                "because offline mode only keyword-matches your copy rather than truly "
                "reading it. Add an API key for full reading — or use the "
                "**Deeper fields** tab, which works fully offline.")

    e1, e2, e3 = st.tabs(["Basic details", "Description & specs", "Deeper fields"])

    with e1:
        if raw is None:
            st.info("This product came from a CSV import and has no editable source row.")
        else:
            with st.form("basics"):
                b1, b2 = st.columns(2)
                n_name = b1.text_input("Product name", value=raw.get("product_name", ""))
                n_brand = b2.text_input("Brand", value=raw.get("brand", ""))
                n_cat = st.text_input("Category", value=raw.get("category", ""),
                                      help="Use > between levels, e.g. Footwear > Running > Road")
                b3, b4, b5 = st.columns(3)
                n_price = b3.number_input("Price", min_value=0.0,
                                          value=float(raw.get("price") or 0))
                n_curr = b4.text_input("Currency", value=raw.get("currency", "SGD"))
                n_avail = b5.selectbox("Availability",
                                       ["in_stock", "low_stock", "preorder", "out_of_stock"],
                                       index=["in_stock", "low_stock", "preorder",
                                              "out_of_stock"].index(
                                                  raw.get("availability", "in_stock")))
                b6, b7 = st.columns(2)
                n_rating = b6.number_input("Average rating", 0.0, 5.0,
                                           float(raw.get("rating") or 0), step=0.1)
                n_rcount = b7.number_input("Number of reviews", 0,
                                           value=int(raw.get("review_count") or 0))
                n_size = st.text_input("Size range", value=raw.get("size_range", ""))
                if st.form_submit_button("Save details", type="primary"):
                    raw.update({"product_name": n_name, "brand": n_brand,
                                "category": n_cat, "price": n_price or None,
                                "currency": n_curr, "availability": n_avail,
                                "rating": n_rating or None,
                                "review_count": n_rcount or None,
                                "size_range": n_size})
                    st.success("Saved.")
                    st.rerun()

    with e2:
        if raw is None:
            st.info("No editable source row for this product.")
        else:
            st.markdown("**Product description**")
            st.caption("The main body copy from your product page.")
            n_desc = st.text_area("Description", value=raw.get("pdp_text", ""),
                                  height=170, label_visibility="collapsed")

            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Spec bullets**")
                st.caption("One per line, e.g. `Weight: 265g`")
                n_bul = st.text_area("Specs", height=170, label_visibility="collapsed",
                                     value="\n".join(raw.get("bullet_specs") or []))
            with s2:
                st.markdown("**Customer reviews**")
                st.caption("One per line. Real quotes work best.")
                n_rev = st.text_area("Reviews", height=170, label_visibility="collapsed",
                                     value="\n".join(raw.get("reviews") or []))

            if st.button("Save description, specs and reviews", type="primary"):
                raw["pdp_text"] = n_desc
                raw["bullet_specs"] = [b.strip() for b in n_bul.splitlines() if b.strip()]
                raw["reviews"] = [r.strip() for r in n_rev.splitlines() if r.strip()]
                st.success("Saved. Recalculating.")
                st.rerun()

    with e3:
        st.markdown("**The information an assistant is missing**")
        st.caption("Pick anything below. The AI's suggested draft is on the left; "
                   "write or edit your own version on the right.")

        gap_names = [g["field"] for g in (perf["gap_fields"] or sc["top_gaps"])][:12]
        others = [f.name for f in FIELDS if f.tier in (2, 3, 4, 5, 6)
                  and f.name not in gap_names]
        options = gap_names + others
        field_name = st.selectbox(
            "Which piece of information?", options,
            format_func=lambda f: f"{plain(f)}" + ("  — missing" if f in gap_names else ""))

        st.caption(FIELDS_BY_NAME[field_name].description)
        cur, draft = rec.get(field_name), opt.get(field_name)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("<div class='label'>AI suggested draft</div>", unsafe_allow_html=True)
            st.code(json.dumps(draft.value, indent=1, ensure_ascii=False)
                    if draft.filled else "(nothing suggested)", language="json")
        with d2:
            st.markdown("<div class='label'>Your version</div>", unsafe_allow_html=True)
            start = st.session_state.edits.get(sel, {}).get(field_name)
            if start is None:
                start = cur.value if cur.filled else (draft.value if draft.filled else "")
            txt = st.text_area("Value", height=190, label_visibility="collapsed",
                               key=f"ed_{sel}_{field_name}",
                               value=start if isinstance(start, str)
                               else json.dumps(start, indent=1, ensure_ascii=False))
            k1, k2 = st.columns(2)
            if k1.button("Save as verified", type="primary"):
                try:
                    val = json.loads(txt)
                except json.JSONDecodeError:
                    val = txt
                st.session_state.edits.setdefault(sel, {})[field_name] = val
                st.success("Saved.")
                st.rerun()
            if k2.button("Clear"):
                st.session_state.edits.get(sel, {}).pop(field_name, None)
                st.rerun()

        if st.session_state.edits.get(sel):
            st.markdown("<div class='label' style='margin-top:16px'>You've written</div>",
                        unsafe_allow_html=True)
            st.markdown("".join(f"<span class='chip'>{plain(k)}</span>"
                                for k in st.session_state.edits[sel]),
                        unsafe_allow_html=True)


# ==========================================================================
# 7. Publish
# ==========================================================================
with t_pub:
    st.markdown("### Review and publish")
    st.markdown("<div class='lead'>Nothing the AI wrote goes live until a person "
                "approves it.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Readiness score", f"{sc['score']:.0f}",
              f"{sc_after['score'] - sc['score']:+.0f} if you fill the gaps")
    c2.metric("Recommended by AI", f"{perf['win_rate']:.0f}%",
              f"{perf_after['win_rate'] - perf['win_rate']:+.0f} points")
    c3.metric("Information published", f"{len(rec.filled_names())}",
              f"+{len(opt.filled_names()) - len(rec.filled_names())} available")

    b1, b2 = st.columns(2)
    b1.markdown(f"<div class='card'><div class='label'>Today</div>{stripe(rec)}</div>",
                unsafe_allow_html=True)
    b2.markdown(f"<div class='card'><div class='label'>With gaps filled</div>"
                f"{stripe(opt)}</div>", unsafe_allow_html=True)

    queue = ApprovalQueue.from_records(rec, opt, gap_report=perf["gap_fields"],
                                       coverage_gaps=sc["top_gaps"])
    for fname, stt in (st.session_state.approvals.get(sel) or {}).items():
        queue.act(fname, stt)
    stats = queue.stats()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Waiting for your approval**")
    m1, m2, m3 = st.columns(3)
    m1.metric("To review", stats["pending"])
    m2.metric("Approved", stats["approved"])
    m3.metric("Roughly", f"{stats['est_review_minutes']} min")

    for it in queue.items[:6]:
        with st.expander(f"{plain(it.field_name)} — {it.state}"):
            st.write(it.proposed_value)
            st.caption(f"Based on: {it.basis[:120]}")
            a1, a2 = st.columns(2)
            if a1.button("Approve", key=f"ap_{sel}_{it.field_name}"):
                st.session_state.approvals.setdefault(sel, {})[it.field_name] = "approved"
                st.rerun()
            if a2.button("Reject", key=f"rj_{sel}_{it.field_name}"):
                st.session_state.approvals.setdefault(sel, {})[it.field_name] = "rejected"
                st.rerun()

    approved = queue.publishable(rec)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Take it away**")
    st.caption("Approved content only. Anything pending or rejected is left out.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Tag for your product page", embed_snippet(approved),
                       file_name=f"{sel}_embed.html", mime="text/html")
    d2.download_button("CSV for your catalog system", to_pim_csv([approved]),
                       file_name=f"{sel}_pim.csv", mime="text/csv")
    d3.download_button("Spreadsheet to review offline", to_review_csv(queue),
                       file_name=f"{sel}_review.csv", mime="text/csv")

    with st.expander("What a brand actually has to do"):
        for i, s in enumerate(integration_checklist(approved, queue), 1):
            st.markdown(f"**{i}. {s['step']}** — {s['detail']}  \n"
                        f"*{s['owner']} · {s['effort']}*")
    with st.expander("The machine-readable version"):
        st.code(json.dumps(to_jsonld(approved), indent=2)[:3000], language="json")
