# AgentReady




cd agentready

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m agentready.pipeline run --hero p001     # ~2s in offline mode
streamlit run app.py
```

**If `pip install` fails**, you are probably on a system Python that refuses to
be written to. Either use the venv above, or add `--break-system-packages`.

**Optional extras** live in `requirements-optional.txt` and are deliberately
separate. pip installs a requirements file all-or-nothing, so one heavy package
that fails to build silently prevents every other package in the file from
installing. `sentence-transformers` pulls in PyTorch (~2 GB) and was doing
exactly that. You do not need it; the demo runs on TF-IDF.

```bash
pip install -r requirements-optional.txt   # REST API + upgraded retriever
```

The app has seven tabs, all driven by the single product selector in the sidebar:

| Tab | What it does |
|---|---|
| **My catalog** | Paste a product or upload a CSV. It appears in the selector immediately |
| **Live query** | Type any shopper question. Also lists what to ask the merchant when the listing falls short |
| **Readiness** | Coverage score, tier bars, provenance stripe |
| **Simulation** | The queries this product was in, won and lost |
| **Gaps & coaching** | What is missing, how to write it, and a paste-ready prompt per field |
| **Edit content** | Rewrite any field. Your text counts as *sourced*, so editing scores higher than accepting a draft |
| **Publish** | Approval queue, script tag, PIM CSV, review sheet |

Other entry points:

```bash
python -m agentready.llm                                       # check your key works
python -m agentready.validate                                  # architecture evidence
python -m agentready.pipeline coldstart data/sample_unseen_category.csv   # unseen category
python -m agentready.ingest data/sample_unseen_category.csv    # CSV auto-mapping
uvicorn agentready.api:app --reload                            # REST API + /docs
```

Runs with **no API key**. Without `OPENAI_API_KEY` the pipeline uses
deterministic keyword heuristics that are strong on identity and specs and weak
on context and constraints — which is exactly what a real catalog looks like, so
the offline demo still tells the truth.

To use your API:

```bash
export OPENAI_API_KEY=sk-...
export AGENTREADY_MODEL=gpt-4o-mini        # optional, this is the default
python -m agentready.llm                   # self-test: verifies key + schema
python -m agentready.pipeline run --hero p001
```

`python -m agentready.llm` is your first move after plugging the key in. It
confirms the client connects, converts a real schema to strict mode, validates
it, and makes one live call. If that passes, the pipeline will work.

### If your key is for a proxy, not OpenAI itself

Hackathon and school-issued keys are often OpenAI-*compatible* endpoints
(Azure gateway, OpenRouter, a local vLLM server) rather than OpenAI. Set:

```bash
export OPENAI_BASE_URL=https://your-proxy.example.com/v1
```



---

## Model choice

| Model | Use when |
|---|---|
| `gpt-4o-mini` | **Default.** Cheap and fast enough to run the full pipeline repeatedly while you iterate. Fine for the demo |
| `gpt-4o` / `gpt-4.1` | Final run before you present. Noticeably better Tier 3/4 reasoning — richer use cases, more honest `not_suitable_for` entries |
| a reasoning model | Only if you have budget to burn. The client auto-handles the `max_completion_tokens` rename, but it's slow and this task isn't reasoning-bound |


---

## Architecture

```
[1] INGEST      catalog rows + PDP text
       |
[2] EXTRACT     Pass A  deterministic rules   (price, sku, weight, rating)
       |        Pass B  6 parallel LLM calls  (one per tier group, strict JSON Schema)
       |        Pass C  vocabulary normalisation
       |        -> every field tagged verbatim | inferred | generated | MISSING
       |
[3] SCORE       Coverage Score, instant, no API calls
       |
[4] SIMULATE    150+ shopper queries -> retrieve -> LLM judge -> win rate
       |        + rejection reasons clustered back onto schema fields
       |
[5] GENERATE    fill exactly the fields that lost queries
       |
       +------> re-score, re-simulate, show the delta
```

Five modules, arrows go one way. `pipeline.py` runs the whole loop and caches
everything to `out/demo_state.json` so the UI never waits on an API.

### Files

| File | What it is |
|---|---|
| `agentready/schema.py` | **Start here.** 75-field Universal Product Schema, tier weights, provenance envelope, JSON Schema generation |
| `agentready/llm.py` | OpenAI client, strict-mode schema conversion, three-level degradation, offline mock |
| `agentready/extractor.py` | Three-pass extraction. The prompt that treats `missing` as a desirable output |
| `agentready/scorer.py` | Coverage Score, quality multipliers, fluff penalty, gap ranking |
| `agentready/queries.py` | 15 personas x 10 intent templates -> shopper queries, each tagged with the fields it probes |
| `agentready/simulator.py` | Retrieval, LLM judging, rejection clustering. The moat |
| `agentready/generator.py` | Gap-driven content fill + Tier 8 agent assets |
| `agentready/exporter.py` | Schema.org JSON-LD + `agentready:*` extensions |
| `agentready/console.py` | Live free-text query answering. Infers which schema fields a query probes |
| `agentready/validate.py` | Predictive-validity and tier-ablation experiments |
| `agentready/ingest.py` | CSV column auto-mapping, cold-start personas for unseen categories |
| `agentready/coach.py` | Shopper questions for the merchant, and merchant improvement briefs |
| `agentready/adopt.py` | Approval queue, PIM round-trip, embed snippet |
| `agentready/api.py` | FastAPI service |
| `app.py` | Streamlit console |

Only `llm.py` knows which provider you're on. Everything else calls `.json()`
and `.live` through the same interface, which is why swapping providers touched
exactly one file.

---


## The Universal Product Schema

75 fields in 9 tiers. The tiering is not decoration — it sets the scoring
weights, and the weights are the argument.

| Tier | Name | Weight | Why |
|---|---|---|---|
| 0 | Identity | 5% | Table stakes. Everyone has it |
| 1 | Hard specs | 10% | Agents can filter on these but cannot reason with them |
| 2 | Function | 15% | feature -> mechanism -> benefit. Catalogs stop at feature |
| **3** | **Context fit** | **25%** | **Matches how humans actually phrase requests** |
| **4** | **Constraints** | **20%** | **Lets an agent reject confidently** |
| 5 | Positioning | 15% | Comparative reasoning against alternatives |
| 6 | Evidence | 10% | Trust signals the agent can cite back |
| 7 | Commerce ops | 0% | Operational, not persuasive. Extracted, not scored |
| 8 | Agent assets | 0% | Generated. Measured by the simulator, not by coverage |

Tiers 3 and 4 carry 45% of the score and almost no catalog on earth has them.
That asymmetry is the whole thesis.

**Tier 4 is the secret weapon.** Almost nobody builds it. A product that lets an
agent rule itself out cleanly earns more trust and more correct placements than
one that claims to be for everyone.

> *An agent that can confidently reject a product is an agent that can
> confidently recommend it.*

Run `python -m agentready.pipeline score` and look at the Constraints and
Positioning columns. Zero for every product. That table is a slide.

### The provenance envelope

Every value carries:

```json
{"value": "...", "provenance": "verbatim|inferred|generated|missing",
 "evidence": "exact quote from the source", "confidence": 0.0}
```

Three jobs: it's the anti-hallucination answer, it powers the coverage discount
(inferred content is worth less than sourced), and it means a judge can click any
field and see the sentence it came from.

It also stops the score being gameable. A brand cannot win by generating
everything: the simulator applies a trust discount proportional to the share of
unverified generated content, and the scorer discounts it too.

---

## Why the honest number matters

The 91% first-mover win rate is real but it's measured against eleven
competitors who did nothing. A judge will poke at that, so the pipeline also runs
a **fair-fight control**: every competitor gets the same treatment, then the hero
is re-measured. It lands at 20.7% against an 8.3% random baseline — still 2.5x,
because the content is genuinely better matched to the queries rather than just
longer.

Present both numbers. Volunteering the weakness before the judge finds it is
worth more than the bigger number.

---



## Troubleshooting

| Symptom | Fix |
|---|---|
| `429 ... insufficient_quota` / `no credits remaining` | Your OpenAI account balance is zero. This is billing, not rate limiting — waiting does nothing. Add credits, or just run without a key: mock mode is fully functional |
| `429 ... rate limit reached` | Different problem, same code. Transient. The client backs off and retries 3×; if it persists, lower `workers` in `extract_many()` and `Simulator.run()` |
| `mode: mock` when you set a key | The env var isn't in the shell running Python. `echo $OPENAI_API_KEY`. In Streamlit, export it *before* launching |
| `strict schema rejected` in the log | Harmless. Your endpoint doesn't do structured outputs; it fell back to `json_object`. Extraction quality drops slightly |
| `call failed: 401` | Bad key, or you need `OPENAI_BASE_URL` because it's a proxy key |
| `call failed: 404 model not found` | Your endpoint doesn't have `gpt-4o-mini`. Set `AGENTREADY_MODEL` to whatever it does have |
| Responses truncated | `[llm] warning: hit the token cap`. Raise `MAX_TOKENS` in `llm.py` or cut `max_fields` in `generator.optimise()` |
| Extraction is slow | Expected: 6 calls/product. Use `gpt-4o-mini` while iterating, save the big model for the final run |
| Rate limited | Drop `workers` in `Extractor.extract_many()` and `Simulator.run()` from the defaults |