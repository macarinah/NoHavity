# Rubric evidence

Every claim below has a command that reproduces it. Run them in front of a judge
if you want to; they all finish in under a minute except `validate`.

---

## 2. Solution Architecture

> *Is the underlying system well-thought, well designed, and the technical
> approach justified given the problem?*

### The design, and why each part is shaped that way

| Decision | Justification |
|---|---|
| **One field registry drives everything** (`schema.py`) | Extraction prompts, JSON Schema for structured outputs, scoring weights, gap ranking and JSON-LD export are all generated from `FIELDS`. Adding a field is one line and the whole pipeline picks it up. No definition is duplicated anywhere |
| **Three-pass extraction, not one prompt** | Pass A is deterministic rules (price, SKU, weight) — exact, free, and it removes 11 chances for the model to be wrong. Pass B is 6 *parallel* calls grouped by tier, because one 75-field prompt returns slop. Pass C normalises free text onto controlled vocabulary, which is what makes cross-product comparison possible |
| **`missing` is a first-class output** | The extractor's failures *are* the product. Extraction and gap detection are the same pass rather than two systems, so a gap report costs nothing extra to produce |
| **Provenance envelope on every value** | Does three jobs at once: anti-hallucination evidence, score discounting (inferred < sourced), and it makes the score ungameable — generating everything triggers a trust penalty in both scorer and simulator |
| **Weighted tiers, not flat field counting** | 45% of the score sits in Context and Constraints, the two tiers no catalog has. A flat count would say a spec-sheet product is 60% ready when an assistant cannot use it at all |
| **Graceful degradation everywhere** | Retrieval: sentence-transformers → TF-IDF → token overlap. LLM: strict schema → json_object → plain text. Provider: live → deterministic mock. Nothing in the demo path has a single point of failure |

### We tested whether the architecture is actually justified

Asserting that Context matters is cheap. So there's a validation suite:

```bash
python -m agentready.validate
```

**Experiment 1 — does the Coverage Score predict anything?**
Mixed corpus (half the catalog optimised, half left alone, competing in the same
market) to get a real coverage spread of 12–56.

```
Pearson  coverage vs win rate : +0.586
Spearman coverage vs win rate : +0.687
```

The score predicts simulated win rate. It is a metric, not decoration.

**Experiment 2 — tier ablation.** Strip one tier from the hero product only,
leaving competitors intact, and measure the damage.

```
tier              assigned w  implied w  win drop  retr drop
T3 Context fit          0.25      0.418     13.3pt      30.6pt
T6 Evidence             0.10      0.208      6.6pt      -4.0pt
T4 Constraints          0.20      0.189      6.0pt      -8.7pt
T5 Positioning          0.15      0.104      3.3pt      15.3pt
T0 Identity             0.05      0.082      2.6pt       0.0pt
T1 Hard specs           0.10      0.000     -7.4pt     -16.0pt
T2 Function             0.15      0.000     -8.7pt      -8.0pt

Spearman: assigned weights vs measured damage = +0.495
```

Context fit is the most expensive tier to lose, matching its highest assigned
weight. Constraints is third. The hand-set weights correlate positively with
measured damage.

### Two things we got wrong, and fixed

Say these out loud if asked. They demonstrate method.

**Our first ablation was invalid.** We removed each tier from *every* product
simultaneously. Context and Constraints both showed a 0.0pt drop — because
deleting a tier from everyone removes it as a differentiator. We switched to
hero-only ablation, which measures the question a brand actually asks: *what
does skipping this tier cost me?*

**Our first correlation test was meaningless.** Run across all-optimised
products, coverage compressed into 48–56 and we measured r=0.30. Restricted
range, not a weak metric. The mixed corpus fixed it.

**Still open:** Hard specs and Function show *negative* drop — removing them
slightly helps. That's a TF-IDF artifact (shorter documents concentrate term
frequency), not a real finding, and it's the first thing we'd fix with a
dense retriever.

---

## 3. AI Reasoning Quality

> *When given real intent-driven queries, does the system surface the right
> products?*

**The Live query tab takes free text.** Nothing is pre-scripted. A judge-typed
query goes through the identical path as the 150-query benchmark:
`infer_probes` derives which schema fields the query implicitly interrogates,
then retrieval and judging proceed normally.

Three behaviours to demonstrate:

**1. It surfaces the right product and says why.**
> *"I'm training for a half marathon in Singapore's humid weather and need
> lightweight shoes under S$200"*

→ Meridian Streamline 4. *Answers 7 of 7 things this shopper needs to know
(environment_conditions, weight_grams, climate_suitability).* Price cap of 200
is parsed from the text and enforced.

**2. It refuses rather than reaching.**
> *"Which coffee grinder is good for a total beginner"*

→ **No recommendation.** *"Nothing in this catalog is the product type the
shopper asked for. Recommending the closest available item would be worse than
recommending nothing."*

This one is worth demoing deliberately. An earlier build returned a running shoe
here, and separately returned a *moisturiser* for the half-marathon query,
because retrieval matched the word "humid" in a skincare review. We fixed it by
anchoring the category in the agent view and making category mismatch
disqualifying rather than a penalty.

**3. It refuses on safety grounds.**
When a shopper signals risk — sensitive skin, an injury, being a beginner — a
product that cannot rule itself out is penalised. Missing constraint content is
disqualifying, not neutral. This is the responsibility gate in `simulator.py`,
and it's why Tier 4 shows measurable value in the ablation above.

**The before/after flip is the demo.** Same query, two content states, side by
side. The UI names the exact fields that changed the outcome.

```bash
# reproduce any of the above
python -c "
from agentready.extractor import Extractor
from agentready.console import Console
from agentready.pipeline import load_products
c = Console(Extractor().extract_many(load_products()))
print(c.ask('YOUR QUERY HERE')['winner_name'])"
```

---

## 4. Scalability & Generalisability

> *Does the architecture work across different product categories and datasets,
> or only within a narrow pre-defined scope?*

### Proof: a category the system has never seen, from a raw CSV

```bash
python -m agentready.pipeline coldstart data/sample_unseen_category.csv
```

Coffee equipment. No personas written for it, no vocabulary entries, no code
changes, deliberately awkward column headers.

```
[ingest] 6 rows, 12 columns
  product_name    <- Item Name
  price           <- Retail Price SGD
  pdp_text        <- Long Description
  bullet_specs    <- Key Features
  ... 11 of 12 auto-mapped
  unmapped (kept as extras): Warranty

[coldstart] 8 personas generated for vertical 'home' - no code changes

  UNSEEN CATEGORY: Home > Coffee Equipment > Kettles
  coverage 13.0 -> 39.4
  win rate 2.6% -> 90.9%
```

### What makes it generalise

- **Vertical-agnostic schema.** `use_cases`, `not_suitable_for`, `tradeoffs` and
  `environment_conditions` mean something for shoes, serums, kettles and dog
  food. Nothing in `FIELDS` names a category.
- **Column auto-mapping** (`ingest.sniff_columns`). Token-overlap matching
  against alias lists. A brand reformats nothing.
- **Cold-start personas** (`ingest.auto_personas`). Live mode asks the model,
  grounded in the actual catalog. Offline mode instantiates eight
  category-agnostic archetypes against the real category name and price range.
  Either way personas carry schema-field probes, so gap clustering works
  immediately.
- **The category gate abstains on unknowns.** `_category_plausible` returns True
  when it doesn't recognise a product type, so a new vertical is never wrongly
  excluded. It only ever *blocks* a known mismatch.

### Honest limits

- Cold-start coverage reaches 39 rather than 56, because our *offline mock*
  generator has hand-written branches for running and skincare and falls through
  to a generic path otherwise. **With a live API there is no branching at all** —
  that ceiling is an artifact of the no-key demo mode, not the architecture.
- Scaling cost is the simulator, not extraction. Extraction is 6 parallel calls
  per product and embarrassingly parallel across products. For a 50k-SKU catalog:
  cache query embeddings, sample the query set, run the sweep nightly rather than
  on demand.
- TF-IDF retrieval is the weak link past a few hundred products. The `Retriever`
  interface already accepts a swap; we didn't need one for this scale and adding
  a vector DB for 12 products would have been theatre.

---

## 5. Brand Adoptability

> *Would a real brand actually implement this? Is there a clear integration
> pathway with minimal friction?*

Three objections kill enterprise adoption of AI content tools. Each has a
concrete answer in `adopt.py`.

### "We can't publish unreviewed AI text."

**Approval queue.** Nothing generated is publishable until a human signs off.
This falls out of the provenance envelope for free — we already track what is
generated versus sourced.

The queue is ordered by **queries won back**, then tier weight, then coverage
impact — so a content manager with twenty minutes approves the six fields that
matter rather than working alphabetically. `queue.publishable()` returns a record
containing sourced content plus *only* approved generated fields. Rejected and
pending fields never reach the published page.

We got the ordering wrong first: dividing tier weight across fields made small
tiers look urgent, ranking `secondary_functions` above `environment_conditions`.
Fixed.

### "Our catalog lives in a PIM, not your web app."

**CSV round-trip.** `to_pim_csv` emits 14 columns keyed on the SKU they already
gave us. `to_review_csv` emits the review queue as a spreadsheet with a
`decision` column, because a lot of content teams review in Excel and emailing
them a web app link is how a pilot dies. `from_review_csv` re-imports it.

### "Who's going to rewrite our product pages?"

**One script tag.**

```html
<script type="application/ld+json">
{ "@context": {...}, "@type": "Product", "agentready:useCases": [...] }
</script>
```

Paste into the PDP template. No redesign, no CMS migration, nothing visible to
shoppers. Schema.org `Product` for anything that already consumes it, plus
`agentready:*` extensions for the context and constraint fields the standard has
no vocabulary for.

### The integration path, with honest effort estimates

| # | Step | Owner | Effort |
|---|---|---|---|
| 1 | Export your catalog (any CSV, columns auto-mapped) | merchandising | 10 min |
| 2 | Run the readiness scan (read-only, no integration) | us | automated |
| 3 | Review the queue — approve, edit, or reject | content team | ~28 min per SKU |
| 4 | Paste the script tag, or import the CSV to your PIM | web team | 30 min, once |
| 5 | Re-scan monthly as the catalog and competitors move | us | automated |

Step 2 is the wedge. A brand gets a coverage score and a competitor benchmark
with **zero integration** — it's read-only on a CSV they already have. Nothing is
asked of them until they've seen a number they don't like.

### It's an API, not just a UI

```bash
pip install fastapi uvicorn
uvicorn agentready.api:app --reload    # http://localhost:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | Coverage + gaps for one product. Read-only, no commitment |
| `POST /v1/optimise` | Fill gaps, return everything flagged for approval |
| `POST /v1/approve/{id}` | Record human decisions |
| `GET /v1/jsonld/{id}` | Approved content only, by default |
| `GET /v1/embed/{id}` | The script tag |
| `GET /v1/export/{id}.csv` | PIM import or review sheet |
| `POST /v1/ingest/csv` | Raw catalog upload, auto-mapped |
| `POST /v1/ask` | Answer a shopper query against their catalog |

Every endpoint works with no API key, so a brand can evaluate the whole thing
before signing anything.

---

## Reproduce everything

```bash
pip install -r requirements.txt

python -m agentready.pipeline run --hero p001                    # the main loop
python -m agentready.validate                                    # rubric 2 evidence
python -m agentready.pipeline coldstart data/sample_unseen_category.csv   # rubric 4
python -m agentready.ingest data/sample_unseen_category.csv      # column mapping
streamlit run app.py                                             # rubric 3 live demo
uvicorn agentready.api:app --reload                              # rubric 5
```
