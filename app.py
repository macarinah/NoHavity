"""
AgentReady - console.

    streamlit run app.py

Everything is driven by the product selected in the sidebar. Upload your own
catalog on the "My catalog" tab and it appears in that selector immediately.
No product is hardcoded.

Design note: the palette encodes provenance, it is not decoration.
  mint    = verbatim, sourced from the brand's own copy
  ice     = inferred, follows from the source
  magenta = generated, awaiting human approval
  slate   = missing, this is the gap
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
from agentready.schema import FIELDS, FieldValue, ProductRecord  # noqa: E402
from agentready.scorer import rank_gaps, score  # noqa: E402
from agentready.simulator import Simulator, agent_view, aggregate  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(ROOT, "data", "products.json")

st.set_page_config(page_title="AgentReady", page_icon="◆", layout="wide")

CSS = """
<style>
:root{
  --ground:#0E1428; --panel:#161E38; --edge:#243055;
  --ink:#E8ECF7; --mute:#7E8AAE;
  --verbatim:#4FD1A5; --inferred:#5AA9E6; --generated:#E0568A; --missing:#2A3454;
  --warn:#F0A02C;
}
.stApp{background:var(--ground);}
html,body,[class*="css"]{color:var(--ink);}
h1,h2,h3{font-family:ui-sans-serif,"Inter","Helvetica Neue",sans-serif;
  letter-spacing:-0.02em;font-weight:650;}
.mono{font-family:ui-monospace,"JetBrains Mono","SF Mono",Menlo,monospace;}
.panel{background:var(--panel);border:1px solid var(--edge);border-radius:10px;
  padding:18px 20px;margin-bottom:14px;}
.eyebrow{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--mute);margin-bottom:6px;}
.stripe{display:flex;gap:2px;flex-wrap:wrap;margin:10px 0 4px;}
.tick{width:10px;height:22px;border-radius:2px;background:var(--missing);}
.tick.verbatim{background:var(--verbatim);}
.tick.inferred{background:var(--inferred);}
.tick.generated{background:var(--generated);}
.legend{font-family:ui-monospace,monospace;font-size:11px;color:var(--mute);
  display:flex;gap:16px;margin-top:6px;flex-wrap:wrap;}
.swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;}
.bignum{font-family:ui-monospace,monospace;font-size:44px;font-weight:600;
  line-height:1;letter-spacing:-0.03em;}
.delta{color:var(--verbatim);font-family:ui-monospace,monospace;font-size:15px;}
.sub{color:var(--mute);font-size:12px;font-family:ui-monospace,monospace;
  letter-spacing:.08em;text-transform:uppercase;}
.tierrow{display:flex;align-items:center;gap:10px;margin:7px 0;}
.tiername{width:130px;font-size:12px;color:var(--mute);font-family:ui-monospace,monospace;}
.track{flex:1;height:16px;background:#111a33;border-radius:3px;position:relative;overflow:hidden;}
.fill-before{position:absolute;height:100%;background:var(--missing);}
.fill-after{position:absolute;height:100%;
  background:linear-gradient(90deg,var(--inferred),var(--verbatim));}
.tierval{width:106px;text-align:right;font-family:ui-monospace,monospace;font-size:12px;}
.gap{border-left:3px solid var(--warn);padding:8px 12px;margin:6px 0;
  background:#1a2242;border-radius:0 6px 6px 0;}
.gapf{font-family:ui-monospace,monospace;font-size:13px;color:var(--warn);}
.win{border-left:3px solid var(--verbatim);padding:10px 14px;background:#152a28;
  border-radius:0 6px 6px 0;margin:8px 0;}
.loss{border-left:3px solid var(--generated);padding:10px 14px;background:#2a1622;
  border-radius:0 6px 6px 0;margin:8px 0;}
.ask{border-left:3px solid var(--inferred);padding:10px 14px;background:#152036;
  border-radius:0 6px 6px 0;margin:8px 0;}
.qtext{font-size:14px;font-style:italic;}
.chip{background:#1a2242;padding:3px 8px;border-radius:4px;font-size:12px;
  margin-right:6px;font-family:ui-monospace,monospace;display:inline-block;margin-bottom:4px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Session catalog. Uploads live here, so every tab sees them immediately.
# ==========================================================================

def _load_seed():
    with open(CATALOG) as fh:
        return json.load(fh)


if "catalog" not in st.session_state:
    st.session_state.catalog = _load_seed()
if "edits" not in st.session_state:
    st.session_state.edits = {}          # {product_id: {field: value}}
if "approvals" not in st.session_state:
    st.session_state.approvals = {}      # {product_id: {field: state}}

llm = LLM()


def catalog_key() -> str:
    blob = json.dumps(st.session_state.catalog, sort_keys=True, default=str)
    blob += json.dumps(st.session_state.edits, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# ==========================================================================
# Analysis. Cached on catalog content, so an upload or edit invalidates it and
# every tab recomputes together. This is what makes the selector actually work.
# ==========================================================================

@st.cache_data(show_spinner=False)
def analyse(key: str, catalog_json: str, edits_json: str):
    products = json.loads(catalog_json)
    edits = json.loads(edits_json)

    ex, gen = Extractor(llm), Generator(llm)
    baseline = ex.extract_many(products)

    # Human edits override extraction and count as sourced content, because a
    # person typed them. Editing therefore raises the score more than
    # generating does, which is the incentive we want.
    for rec in baseline:
        for fname, val in (edits.get(rec.product_id) or {}).items():
            rec.set(fname, FieldValue(value=val, provenance="verbatim",
                                      evidence="entered by the brand", confidence=1.0))

    # Cost guard. This function re-runs on every upload and every edit. With a
    # live API, the full 150-query sweep across a whole catalog would be both
    # slow and expensive for an interactive UI, so live mode uses a sampled set.
    # The full sweep still runs offline via `python -m agentready.pipeline run`.
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
    perf_after = {r.product_id: aggregate(res_after, r.product_id).__dict__
                  for r in optimised}

    return {
        "baseline": [r.to_dict() for r in baseline],
        "optimised": [r.to_dict() for r in optimised],
        "scores": {r.product_id: score(r).as_dict() for r in baseline},
        "scores_after": {r.product_id: score(r).as_dict() for r in optimised},
        "perf": perf,
        "perf_after": perf_after,
        "results": [r.__dict__ for r in res_before],
        "n_queries": len(queries),
    }


with st.spinner("Analysing catalog..."):
    A = analyse(catalog_key(),
                json.dumps(st.session_state.catalog, default=str),
                json.dumps(st.session_state.edits, default=str))

BASE = {d["product_id"]: ProductRecord.from_dict(d) for d in A["baseline"]}
OPT = {d["product_id"]: ProductRecord.from_dict(d) for d in A["optimised"]}


# ==========================================================================
# Sidebar: ONE product selector that every tab reads.
# ==========================================================================

st.sidebar.markdown("<div class='eyebrow'>AgentReady</div>", unsafe_allow_html=True)
ids = list(BASE.keys())
sel = st.sidebar.selectbox(
    "Product", ids, key="selected_product",
    format_func=lambda p: f"{BASE[p].name()[:34]}",
)
rec, opt = BASE[sel], OPT[sel]
sc, sc_after = A["scores"][sel], A["scores_after"][sel]
perf, perf_after = A["perf"][sel], A["perf_after"][sel]

st.sidebar.markdown(
    f"<div class='panel' style='padding:14px'>"
    f"<div class='eyebrow'>Coverage</div>"
    f"<div class='bignum' style='font-size:30px'>{sc['score']:.0f}"
    f"<span style='color:var(--mute);font-size:16px'> → {sc_after['score']:.0f}</span></div>"
    f"<div class='sub' style='margin-top:8px'>win rate</div>"
    f"<div class='mono' style='font-size:18px'>{perf['win_rate']:.0f}% "
    f"<span style='color:var(--verbatim)'>→ {perf_after['win_rate']:.0f}%</span></div>"
    f"</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div class='sub'>{len(BASE)} products · {A['n_queries']} queries<br>"
    f"llm: {llm.mode}</div>", unsafe_allow_html=True)
if llm.live:
    st.sidebar.caption("Live mode samples the query set to keep the UI responsive. "
                       "Run the CLI pipeline for the full sweep.")

if st.sidebar.button("Reset catalog to samples"):
    st.session_state.catalog = _load_seed()
    st.session_state.edits, st.session_state.approvals = {}, {}
    st.rerun()


def stripe(r: ProductRecord) -> str:
    ticks = []
    for f in FIELDS:
        if f.tier == 8:
            continue
        prov = r.get(f.name).provenance if r.get(f.name).filled else "missing"
        ticks.append(f'<div class="tick {prov}" title="{f.name} ({prov})"></div>')
    legend = ('<div class="legend">'
              '<span><span class="swatch" style="background:var(--verbatim)"></span>sourced</span>'
              '<span><span class="swatch" style="background:var(--inferred)"></span>inferred</span>'
              '<span><span class="swatch" style="background:var(--generated)"></span>generated</span>'
              '<span><span class="swatch" style="background:var(--missing)"></span>missing</span>'
              "</div>")
    return f'<div class="stripe">{"".join(ticks)}</div>{legend}'


st.markdown(
    f"<div class='eyebrow'>content readiness for AI-mediated commerce</div>"
    f"<h1 style='margin-top:0'>{rec.name()}</h1>"
    f"<div class='sub'>{rec.raw.get('category','')} · "
    f"{rec.get('currency').value or 'SGD'} {rec.price() or 0:g}</div>",
    unsafe_allow_html=True)
st.write("")

t_cat, t_live, t_ready, t_sim, t_gaps, t_edit, t_pub = st.tabs(
    ["My catalog", "Live query", "Readiness", "Simulation", "Gaps & coaching",
     "Edit content", "Publish"])


# ==========================================================================
# My catalog - upload your own products
# ==========================================================================
with t_cat:
    st.markdown("<div class='eyebrow'>Add your own products</div>"
                "<p style='color:var(--mute)'>Anything you add appears in the sidebar "
                "selector immediately and flows through every tab.</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Paste one product**")
        with st.form("add_product", clear_on_submit=True):
            name = st.text_input("Product name *", placeholder="Meridian Trail Runner 2")
            brand = st.text_input("Brand", placeholder="Meridian")
            cat = st.text_input("Category", placeholder="Footwear > Running > Trail")
            cc1, cc2 = st.columns(2)
            price = cc1.number_input("Price", min_value=0.0, value=0.0, step=1.0)
            curr = cc2.text_input("Currency", value="SGD")
            desc = st.text_area("Product description *", height=150,
                                placeholder="Paste your existing product page copy here...")
            bullets = st.text_area("Spec bullets (one per line)", height=80,
                                   placeholder="Weight: 265g\nDrop: 8mm")
            reviews = st.text_area("Customer reviews (one per line)", height=68)
            go = st.form_submit_button("Add to catalog", type="primary")

        if go:
            if not name.strip() or not desc.strip():
                st.error("Product name and description are both required.")
            else:
                pid = f"user{len([p for p in st.session_state.catalog if str(p.get('product_id','')).startswith('user')]) + 1:03d}"
                st.session_state.catalog.append({
                    "product_id": pid,
                    "product_name": name.strip(),
                    "brand": brand.strip() or "Unbranded",
                    "category": cat.strip() or "General",
                    "price": float(price) or None,
                    "currency": curr.strip() or "SGD",
                    "pdp_text": desc.strip(),
                    "bullet_specs": [b.strip() for b in bullets.splitlines() if b.strip()],
                    "reviews": [r.strip() for r in reviews.splitlines() if r.strip()],
                    "vertical": (cat.split(">")[0].strip().lower() or "general"),
                    "availability": "in_stock",
                })
                st.success(f"Added as {pid}. Select it in the sidebar.")
                st.rerun()

    with c2:
        st.markdown("**Or upload a catalog CSV**")
        st.markdown("<div style='color:var(--mute);font-size:13px'>Column names are "
                    "auto-mapped — 'Item Name', 'Retail Price SGD' and 'Long Description' "
                    "all work. Unmapped columns are kept, not dropped.</div>",
                    unsafe_allow_html=True)
        up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        replace = st.checkbox("Replace catalog instead of appending")
        if up is not None and st.button("Import CSV"):
            import tempfile
            from agentready.ingest import from_csv, sniff_columns
            with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as fh:
                fh.write(up.getvalue())
                tmp = fh.name
            try:
                import csv as _csv
                import io as _io
                headers = next(_csv.reader(_io.StringIO(up.getvalue().decode("utf-8-sig"))))
                mapping = sniff_columns(headers)
                prods = from_csv(tmp, report=False)
                for i, p in enumerate(prods):
                    p["product_id"] = f"up{i+1:03d}"
                st.session_state.catalog = prods if replace else st.session_state.catalog + prods
                st.session_state.edits = {}
                st.success(f"Imported {len(prods)} products. "
                           f"Auto-mapped {len(mapping)} of {len(headers)} columns.")
                st.json({k: v for k, v in mapping.items()}, expanded=False)
                st.rerun()
            finally:
                os.unlink(tmp)

        st.markdown("<div class='eyebrow' style='margin-top:20px'>Current catalog</div>",
                    unsafe_allow_html=True)
        for pid, r in BASE.items():
            mark = "●" if pid == sel else "○"
            st.markdown(
                f"<div style='font-size:13px;margin:4px 0;font-family:ui-monospace,monospace'>"
                f"<span style='color:var(--verbatim)'>{mark}</span> {pid} · {r.name()[:30]} "
                f"<span style='color:var(--mute)'>· coverage "
                f"{A['scores'][pid]['score']:.0f}</span></div>",
                unsafe_allow_html=True)


# ==========================================================================
# Live query
# ==========================================================================
with t_live:
    st.markdown("<div class='eyebrow'>Ask anything a shopper would ask</div>"
                "<p style='color:var(--mute)'>Same reasoning path as the benchmark. The "
                "query's implicit questions are inferred from the text, so nothing is "
                "pre-scripted.</p>", unsafe_allow_html=True)

    EXAMPLES = [
        "I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200",
        "Sustainable skincare routine for oily skin that takes under 5 minutes every morning",
        "I have wide feet and keep getting blisters on long runs",
        "I have sensitive reactive skin and I'm nervous about starting retinol",
    ]
    pick = st.selectbox("Try one, or write your own", ["(write my own)"] + EXAMPLES)
    query = st.text_area("Shopper query", value="" if pick.startswith("(") else pick,
                         height=80, placeholder="Type any shopping question...")

    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("Retrieving and judging..."):
            res = compare_console(list(BASE.values()), list(OPT.values()),
                                  query.strip(), sel, llm)

        st.markdown("<div class='eyebrow'>Questions this query implicitly asks</div>"
                    + " ".join(f"<span class='chip'>{p}</span>"
                               for p in res["before"]["probes"]), unsafe_allow_html=True)
        if res["before"]["price_cap"]:
            st.markdown(f"<div class='sub' style='margin-top:8px'>price cap detected: "
                        f"{res['before']['price_cap']:.0f}</div>", unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns(2)
        for col, key, label in ((c1, "before", "Your content today"),
                                (c2, "after", "After AgentReady")):
            with col:
                r = res[key]
                st.markdown(
                    f"<div class='eyebrow'>{label}</div>"
                    f"<div class='{'win' if r['winner'] else 'loss'}'>"
                    f"<div style='font-size:16px;font-weight:600'>"
                    f"{r['winner_name'] or 'No recommendation'}</div>"
                    f"<div style='font-size:12px;color:var(--mute);margin-top:6px'>"
                    f"{r['winner_reason'][:200]}</div></div>", unsafe_allow_html=True)
                for b in r["candidates"][:4]:
                    st.markdown(
                        f"<div style='font-size:12px;margin:5px 0;"
                        f"font-family:ui-monospace,monospace'>"
                        f"<span style='color:var(--verbatim)'>"
                        f"{'●' if b['is_winner'] else '○'}</span> {b['name'][:30]} "
                        f"<span style='color:var(--mute)'>· answers "
                        f"{b['coverage_of_query']}%</span></div>", unsafe_allow_html=True)

        if res["flipped"]:
            st.markdown(f"<div class='win' style='margin-top:14px'><b>Flipped.</b> "
                        f"{rec.name()} went from not recommended to recommended. Fields "
                        f"that did it: <span class='mono'>"
                        f"{', '.join(res['gained_fields'][:6])}</span></div>",
                        unsafe_allow_html=True)

        # CONSUMER-SIDE COACHING
        focus = res["before"].get("focus") or {}
        unanswered = focus.get("unanswered", [])
        if unanswered:
            qs = merchant_questions(rec, unanswered, llm)
            st.markdown("<div class='eyebrow' style='margin-top:22px'>"
                        "What to ask the merchant</div>"
                        "<p style='color:var(--mute);font-size:13px'>This listing cannot "
                        "answer everything you asked. Send these to the brand — and if "
                        "enough shoppers do, the content gets fixed.</p>",
                        unsafe_allow_html=True)
            for q in qs:
                st.markdown(f"<div class='ask'><div class='qtext'>“{q['question']}”</div>"
                            f"<div style='font-size:11px;color:var(--mute);margin-top:6px;"
                            f"font-family:ui-monospace,monospace'>{q.get('field','')} · "
                            f"{q.get('why','')[:120]}</div></div>", unsafe_allow_html=True)
            st.download_button("Copy all questions",
                               "\n".join(f"- {q['question']}" for q in qs),
                               file_name="questions_for_merchant.txt")


# ==========================================================================
# Readiness
# ==========================================================================
with t_ready:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"<div class='panel'><div class='eyebrow'>Coverage score</div>"
                    f"<div class='bignum'>{sc['score']:.0f}"
                    f"<span style='font-size:20px;color:var(--mute)'>/100</span></div></div>"
                    f"<div class='panel'><div class='eyebrow'>Against "
                    f"{A['n_queries']} queries</div>"
                    f"<div class='bignum' style='font-size:32px'>"
                    f"{perf['win_rate']:.1f}%</div><div class='sub'>recommended</div>"
                    f"<div style='margin-top:10px;color:var(--mute);font-size:13px'>"
                    f"retrieved in {perf['retrieval_rate']:.0f}% of candidate sets</div>"
                    f"</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='eyebrow'>Coverage by tier</div>",
                    unsafe_allow_html=True)
        st.markdown("".join(
            f"<div class='tierrow'><div class='tiername'>T{t['tier']} {t['name']}</div>"
            f"<div class='track'><div class='fill-after' "
            f"style='width:{t['quality_fill']*100:.0f}%'></div></div>"
            f"<div class='tierval'>{t['quality_fill']*100:4.0f}% · w{t['weight']:.2f}</div>"
            f"</div>" for t in sc["tiers"]) + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='eyebrow'>Field provenance · "
                    "all 75 fields</div>" + stripe(rec) + "</div>", unsafe_allow_html=True)

    with st.expander("What the AI assistant actually reads"):
        st.code(agent_view(rec), language=None)


# ==========================================================================
# Simulation
# ==========================================================================
with t_sim:
    c1, c2, c3 = st.columns(3)
    c1.metric("Win rate", f"{perf['win_rate']:.1f}%", f"{perf['won']} wins")
    c2.metric("Retrieval rate", f"{perf['retrieval_rate']:.1f}%")
    c3.metric("After optimisation", f"{perf_after['win_rate']:.1f}%",
              f"{perf_after['win_rate'] - perf['win_rate']:+.1f}pt")

    st.markdown("<div class='eyebrow' style='margin-top:16px'>Queries involving this "
                "product</div>", unsafe_allow_html=True)
    personas = sorted({r["persona"] for r in A["results"]})
    pf = st.selectbox("Filter by persona", ["all"] + personas)

    shown = 0
    for r in A["results"]:
        if pf != "all" and r["persona"] != pf:
            continue
        if sel not in r["retrieved"]:
            continue
        rej = next((x for x in r["rejections"] if x.get("product_id") == sel), None)
        won = r["winner"] == sel
        name = BASE[r["winner"]].name() if r["winner"] in BASE else "nobody"
        st.markdown(
            f"<div class='{'win' if won else 'loss'}'><div class='qtext'>“{r['query']}”</div>"
            f"<div style='font-size:12px;color:var(--mute);margin-top:6px;"
            f"font-family:ui-monospace,monospace'>→ {name} · {r['winner_reason'][:120]}</div>"
            + (f"<div style='font-size:12px;color:var(--warn);margin-top:5px'>"
               f"rejected: {rej['reason']}</div>" if rej else "")
            + "</div>", unsafe_allow_html=True)
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        st.info("This product was never retrieved for that persona. That is itself the "
                "finding — check the Gaps tab.")


# ==========================================================================
# Gaps & coaching
# ==========================================================================
with t_gaps:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='eyebrow'>Missing fields, by queries lost</div>",
                    unsafe_allow_html=True)
        gaps = perf["gap_fields"] or sc["top_gaps"]
        for g in gaps[:8]:
            st.markdown(
                f"<div class='gap'><span class='gapf'>{g['field']}</span> "
                f"<span style='color:var(--mute);font-size:12px'>· "
                f"{g.get('lost_queries', 0)} queries lost</span>"
                f"<div style='font-size:12px;color:var(--mute);margin-top:4px'>"
                f"{g.get('description','')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='eyebrow'>Weakest personas</div>", unsafe_allow_html=True)
        for p in perf["lost_personas"][:6]:
            st.markdown(f"<div class='gap'><span class='gapf'>{p['persona']}</span>"
                        f"<div style='font-size:12px;color:var(--mute);margin-top:4px'>"
                        f"won {p['won']} of {p['seen']} · {p['win_rate']:.0f}%</div></div>",
                        unsafe_allow_html=True)

    st.markdown("<div class='eyebrow' style='margin-top:24px'>How to fix each gap</div>"
                "<p style='color:var(--mute);font-size:13px'>Each brief includes a prompt "
                "you can paste into your own tools to draft it.</p>",
                unsafe_allow_html=True)
    for b in improvement_brief(rec, gaps, llm):
        with st.expander(f"{b['field']}  ·  T{b['tier']} {b['tier_name']}"):
            st.markdown(f"**What it is** — {b['what_it_is']}")
            st.markdown(f"**How to write it** — {b['how_to_write_it']}")
            st.markdown(f"**Why it matters** — {b['why_it_matters']}")
            st.code(b["draft_prompt"], language=None)

    st.markdown("<div class='eyebrow' style='margin-top:20px'>What is weak in the copy "
                "you already have</div>", unsafe_allow_html=True)
    for s in rewrite_suggestions(rec, llm):
        st.markdown(f"<div class='gap'><span style='font-size:13px'>“{s['excerpt']}”</span>"
                    f"<div style='color:var(--warn);font-size:12px;margin-top:5px'>"
                    f"{s['issue']}</div>"
                    f"<div style='color:var(--mute);font-size:12px;margin-top:3px'>"
                    f"{s['fix']}</div></div>", unsafe_allow_html=True)


# ==========================================================================
# Edit content
# ==========================================================================
with t_edit:
    st.markdown("<div class='eyebrow'>Write it yourself</div>"
                "<p style='color:var(--mute)'>The model's draft is on the left. Edit it, "
                "or write your own. Anything you save counts as <b>sourced</b> content, "
                "not generated — so editing raises your score more than accepting a "
                "draft does.</p>", unsafe_allow_html=True)

    st.markdown("<div class='eyebrow'>Product description</div>", unsafe_allow_html=True)
    new_desc = st.text_area("Description", value=rec.raw.get("pdp_text", ""),
                            height=160, label_visibility="collapsed")
    if st.button("Save description"):
        for p in st.session_state.catalog:
            if p.get("product_id") == sel:
                p["pdp_text"] = new_desc
        st.success("Saved. Re-analysing.")
        st.rerun()

    st.markdown("<div class='eyebrow' style='margin-top:22px'>Fields</div>",
                unsafe_allow_html=True)
    gap_names = [g["field"] for g in (perf["gap_fields"] or sc["top_gaps"])][:10]
    others = [f.name for f in FIELDS if f.tier in (2, 3, 4, 5) and f.name not in gap_names]
    field_name = st.selectbox("Field to write",
                              gap_names + others,
                              format_func=lambda f: (f"{f}  ({'gap' if f in gap_names else 'filled'})"))

    cur = rec.get(field_name)
    draft = opt.get(field_name)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='eyebrow'>Suggested draft</div>", unsafe_allow_html=True)
        st.code(json.dumps(draft.value, indent=1, ensure_ascii=False)
                if draft.filled else "(no draft)", language="json")
        if draft.filled:
            st.markdown(f"<div class='sub'>basis: {(draft.evidence or '')[:90]}</div>",
                        unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='eyebrow'>Your version</div>", unsafe_allow_html=True)
        start = st.session_state.edits.get(sel, {}).get(field_name)
        if start is None:
            start = cur.value if cur.filled else (draft.value if draft.filled else "")
        txt = st.text_area("Value", value=json.dumps(start, indent=1, ensure_ascii=False)
                           if not isinstance(start, str) else start,
                           height=180, label_visibility="collapsed", key=f"ed_{sel}_{field_name}")
        cc1, cc2 = st.columns(2)
        if cc1.button("Save as sourced", type="primary"):
            try:
                val = json.loads(txt)
            except json.JSONDecodeError:
                val = txt
            st.session_state.edits.setdefault(sel, {})[field_name] = val
            st.success(f"Saved {field_name}.")
            st.rerun()
        if cc2.button("Clear this field"):
            st.session_state.edits.get(sel, {}).pop(field_name, None)
            st.rerun()

    if st.session_state.edits.get(sel):
        st.markdown("<div class='eyebrow' style='margin-top:18px'>Your edits</div>",
                    unsafe_allow_html=True)
        for k in st.session_state.edits[sel]:
            st.markdown(f"<span class='chip'>{k}</span>", unsafe_allow_html=True)


# ==========================================================================
# Publish
# ==========================================================================
with t_pub:
    delta = {
        "before": sc["score"],
        "after": sc_after["score"],
        "delta": round(sc_after["score"] - sc["score"], 1),
        "tiers": [{"tier": a["name"],
                   "before": round(b["quality_fill"] * 100, 1),
                   "after": round(a["quality_fill"] * 100, 1)}
                  for b, a in zip(sc["tiers"], sc_after["tiers"])],
    }

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='panel'><div class='eyebrow'>Coverage</div>"
                f"<div class='bignum'>{delta['before']:.0f} "
                f"<span style='color:var(--mute)'>→</span> "
                f"<span style='color:var(--verbatim)'>{delta['after']:.0f}</span></div>"
                f"<div class='delta'>+{delta['delta']:.1f}</div></div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div class='panel'><div class='eyebrow'>Win rate</div>"
                f"<div class='bignum'>{perf['win_rate']:.0f}% "
                f"<span style='color:var(--mute)'>→</span> "
                f"<span style='color:var(--verbatim)'>{perf_after['win_rate']:.0f}%</span>"
                f"</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='panel'><div class='eyebrow'>Fields</div>"
                f"<div class='bignum'>{len(rec.filled_names())} "
                f"<span style='color:var(--mute)'>→</span> "
                f"<span style='color:var(--verbatim)'>{len(opt.filled_names())}</span>"
                f"</div></div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    b1.markdown("<div class='panel'><div class='eyebrow'>Before</div>"
                + stripe(rec) + "</div>", unsafe_allow_html=True)
    b2.markdown("<div class='panel'><div class='eyebrow'>After</div>"
                + stripe(opt) + "</div>", unsafe_allow_html=True)

    queue = ApprovalQueue.from_records(rec, opt, gap_report=perf["gap_fields"],
                                       coverage_gaps=sc["top_gaps"])
    for fname, stt in (st.session_state.approvals.get(sel) or {}).items():
        queue.act(fname, stt)
    stats = queue.stats()

    st.markdown("<div class='eyebrow' style='margin-top:12px'>Review queue</div>"
                "<p style='color:var(--mute);font-size:13px'>Nothing generated is "
                "published without sign-off. Ordered by queries won back.</p>",
                unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Awaiting review", stats["pending"])
    m2.metric("Approved", stats["approved"])
    m3.metric("Est. review time", f"{stats['est_review_minutes']} min")

    for it in queue.items[:6]:
        with st.expander(f"{it.field_name} · T{it.tier} {it.tier_name} · "
                         f"{it.queries_unlocked} queries · {it.state}"):
            st.write(it.proposed_value)
            st.markdown(f"<div class='sub'>basis: {it.basis[:110]}</div>",
                        unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            if a1.button("Approve", key=f"ap_{sel}_{it.field_name}"):
                st.session_state.approvals.setdefault(sel, {})[it.field_name] = "approved"
                st.rerun()
            if a2.button("Reject", key=f"rj_{sel}_{it.field_name}"):
                st.session_state.approvals.setdefault(sel, {})[it.field_name] = "rejected"
                st.rerun()

    approved = queue.publishable(rec)
    st.markdown("<div class='eyebrow' style='margin-top:18px'>Deliverables</div>"
                "<p style='color:var(--mute);font-size:13px'>Approved content only. "
                "Pending and rejected fields are excluded.</p>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.download_button("PDP script tag", embed_snippet(approved),
                       file_name=f"{sel}_embed.html", mime="text/html")
    d2.download_button("PIM import CSV", to_pim_csv([approved]),
                       file_name=f"{sel}_pim.csv", mime="text/csv")
    d3.download_button("Review sheet", to_review_csv(queue),
                       file_name=f"{sel}_review.csv", mime="text/csv")

    st.markdown("<div class='eyebrow' style='margin-top:18px'>What the brand does</div>",
                unsafe_allow_html=True)
    for i, s in enumerate(integration_checklist(approved, queue), 1):
        st.markdown(f"<div style='display:flex;gap:12px;margin:8px 0'>"
                    f"<div class='mono' style='color:var(--inferred);min-width:22px'>{i}</div>"
                    f"<div><b>{s['step']}</b> <span class='mono' style='color:var(--mute);"
                    f"font-size:11px'>[{s['owner']} · {s['effort']}]</span><br>"
                    f"<span style='color:var(--mute);font-size:13px'>{s['detail']}</span>"
                    f"</div></div>", unsafe_allow_html=True)

    with st.expander("Schema.org JSON-LD (approved content only)"):
        st.code(json.dumps(to_jsonld(approved), indent=2)[:3500], language="json")
    with st.expander("Content brief (Markdown)"):
        st.markdown(to_markdown(approved)[:3000])
