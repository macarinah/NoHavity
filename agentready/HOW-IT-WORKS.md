# How AgentReady works

A plain-English walkthrough of the whole thing: the problem, the design, what
each file does, how the dashboard works, and how it answers the brief.

Written so that anyone on the team can read it once and then explain the project
to a judge. No prior knowledge of the code needed.

- [1. The problem](#1-the-problem)
- [2. Our core idea](#2-our-core-idea)
- [3. How the system works](#3-how-the-system-works)
- [4. The 75-field schema](#4-the-75-field-schema)
- [5. What each file does](#5-what-each-file-does)
- [6. The dashboard, tab by tab](#6-the-dashboard-tab-by-tab)
- [7. How this answers the brief](#7-how-this-answers-the-brief)
- [8. Numbers to quote](#8-numbers-to-quote)
- [9. Glossary](#9-glossary)

---

## 1. The problem

### The old way

Someone types `running shoes size 10` into a search box. The retailer's job is
keyword matching and ranking. Product content gets written for a human who will
*look* at it: nice photos, "premium comfort", brand personality. **The human does
the reasoning** about whether it actually suits them.

### The new way

Someone tells an AI assistant a whole situation:

> *"I'm training for a half marathon in Singapore's humid weather and need
> lightweight shoes under S$200."*

Now **the machine has to do the reasoning**. It has to work out whether the shoe
is light enough, whether it survives humidity, whether it's under budget, whether
it suits a first-timer. Then it has to justify its pick out loud to the shopper.

### The gap

The assistant can only reason over what the brand actually published. And what
brands publish is two things, neither of which helps:

**Specs.** Weight 212g, 8mm drop, EVA midsole. A machine can *filter* on these,
but it can't reason with them. "8mm drop" does not answer "will this work for me
in Singapore?"

**Adjectives.** "Premium." "Revolutionary." "Innovative." These are worse than
useless, because the assistant can't repeat them to a shopper as a reason. There
is nothing behind them.

Think of it as a shop assistant who is only allowed to read the box. The box
lists dimensions and materials. The customer asks "will this work for me?" and
the assistant genuinely has nothing to say.

### The part most people miss

In keyword search, missing information means you rank a bit lower.

In agent commerce, missing information gets you **eliminated**. The assistant is
accountable for its recommendation, so if it cannot confirm the shoe handles
humidity, the safe move is to recommend something else. Silence doesn't cost you
a position. It costs you the sale.

So the gap isn't "brands write bad copy." It's that a specific, nameable category
of content doesn't exist anywhere in the industry:

- **Context** — who this is for, in what conditions, for how long
- **Constraints** — who it's wrong for, what you trade off, what it can't do

---

## 2. Our core idea

Three claims, in order.

### Claim 1: the missing content is nameable and measurable

We wrote down the 75 pieces of information an AI assistant needs before it can
recommend something responsibly, sorted into seven categories and weighted by how
much each matters. That gives any product a score out of 100.

### Claim 2: the AI's failures are the product

This is the design insight that shapes everything.

When we point a model at a product page and ask it to fill in "what conditions is
this built for?", it comes back empty. **That emptiness is the finding.** We don't
need a separate gap-detection system, because extraction and gap detection are
the same pass. A confident null is a content brief.

Most teams building on this brief will fight the model making things up. We
harvest the blanks instead. The extraction prompt explicitly says *"missing is
the correct and expected answer — do not invent."*

### Claim 3: the measurement layer is the product

Anyone can generate nicer copy. Nobody can prove it worked.

So we built the measurement first and made the generator a feature of it. On
stage we don't say "our copy is better." We say: *this product won 9% of 150
shopper questions, we filled its gaps, now it wins 72%, and here's the same
question failing and then passing.*

---

## 3. How the system works

Five stages in a loop.

```
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  1. INGEST   │ ──► │  2. EXTRACT  │ ──► │   3. SCORE   │
   │ catalog, CSV │     │ text into 75 │     │  out of 100  │
   │  or typed in │     │    fields    │     │  by category │
   └──────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
          ┌─────────────────────────────────────────┘
          ▼
   ┌──────────────┐     ┌──────────────┐
   │ 4. SIMULATE  │ ──► │ 5. GENERATE  │
   │  150 shopper │     │ write only   │
   │   questions  │     │ the fields   │
   │              │     │ that lost    │
   └──────────────┘     └──────┬───────┘
                               │
          re-score, re-test ───┘  ► show the difference
```

**Stage 1 — Ingest.** A product comes in as a catalog row, a CSV upload, or typed
into a form. CSV column names are matched automatically, so a brand reformats
nothing.

**Stage 2 — Extract.** Three passes, deliberately in this order:

- *Pass A, rules only.* Price, SKU, weight, rating. No AI involved. Exact, free,
  and it removes eleven chances for the model to be wrong.
- *Pass B, six parallel AI calls.* One per category of fields, not one giant
  prompt — a single 75-field prompt returns slop.
- *Pass C, normalisation.* Maps free text onto a controlled vocabulary, so
  "tropical", "muggy" and "high humidity" all become `humid`. This is what lets
  us compare products to each other.

**Stage 3 — Score.** Weighted by category, with a penalty for vague marketing
words and a discount for anything the AI inferred rather than read.

**Stage 4 — Simulate.** 150 realistic shopper questions, built from 15 shopper
types crossed with 10 question shapes. Each question is tagged with the fields it
implicitly asks about. We retrieve candidate products, an AI judge picks one and
explains why it rejected the others.

**Stage 5 — Generate.** The rejections cluster back onto specific fields. The
generator writes *exactly those*, marks everything as awaiting human approval,
and we re-run the identical question set to prove the improvement.

---

## 4. The 75-field schema

### The seven categories

| Category | Plain English | Weight | Why that weight |
|---|---|---|---|
| Identity | Name, brand, price | 5% | Every catalog has this. Gets you in the list, nothing more |
| Hard specs | Weight, materials, size | 10% | An assistant can filter on these but can't reason with them |
| Function | Its job, features, benefits | 15% | Features only help if you also state the benefit |
| **Context fit** | **Who it's for and when** | **25%** | **This is how people actually ask** |
| **Constraints** | **Who it's NOT for** | **20%** | **Lets an assistant rule you out cleanly** |
| Positioning | Why this over alternatives | 15% | Breaks the tie when three products look similar |
| Evidence | Reviews, certifications, tests | 10% | Lets the assistant cite a reason back |

**Context and Constraints are 45% of the score, and almost no catalog on Earth
has them.** Run `python -m agentready.pipeline score` and the Constraints column
is zero for every single product. That asymmetry is the entire thesis.

### The counterintuitive bit

The second-highest-weighted category is the one where a brand admits who
*shouldn't* buy. That sounds commercially insane. It isn't:

> An assistant that can confidently reject your product is an assistant that can
> confidently recommend it. Uncertainty is what kills you, not honesty.

We didn't just assert this. `python -m agentready.validate` strips each category
out and measures the damage. Context fit is the most expensive to lose, matching
its highest weight.

### The provenance envelope

Every extracted value carries four things, not one:

```json
{
  "value": ["humid", "hot", "outdoor"],
  "provenance": "verbatim",
  "evidence": "breathable mesh keeps feet cool in heat",
  "confidence": 0.85
}
```

`provenance` is one of:

- **verbatim** — the brand actually wrote this
- **inferred** — follows necessarily from what they wrote
- **generated** — our AI drafted it, awaiting human approval
- **missing** — nobody has written this. *This is the gap.*

This one design choice does four jobs at once:

1. It's the anti-hallucination answer. Click any field, see the source sentence.
2. It powers the score discount — inferred content is worth less than sourced.
3. It makes the score ungameable. A brand can't win by generating everything,
   because generated content is discounted in both the scorer and the simulator.
4. The approval queue falls out of it for free — we already know what's AI-written.

---

## 5. What each file does

Sixteen modules, about 5,300 lines. **You only need to be able to explain five.**

### The five that matter

| File | What it does |
|---|---|
| `schema.py` | The 75 fields and their weights. **The heart.** Everything else reads from it, so adding a field is one line and the whole system picks it up |
| `extractor.py` | Turns messy product text into filled-in fields. Explicitly told that "missing" is a good answer |
| `scorer.py` | Turns filled fields into a score out of 100. Penalises fluff, discounts inference |
| `simulator.py` | The 150 fake shoppers, the retrieval, and the AI judge that picks a winner and explains its rejections |
| `generator.py` | Writes the missing fields, marks everything as needing approval |

### Everything else, one line each

| File | What it does |
|---|---|
| `queries.py` | Builds the 150 shopper questions from 15 personas × 10 question shapes |
| `console.py` | Handles a live typed question. Works out which fields a free-text query is implicitly asking about |
| `coach.py` | Two directions: questions a shopper should ask the merchant, and writing guidance for the merchant |
| `adopt.py` | Approval queue, CSV export back to the brand's system, the embed snippet |
| `exporter.py` | Turns a product into Schema.org JSON-LD plus our extensions |
| `validate.py` | The two experiments proving the weights aren't arbitrary |
| `ingest.py` | CSV import with automatic column matching, and personas for unseen categories |
| `llm.py` | Talks to OpenAI. Falls back to offline mode cleanly if the key dies |
| `pipeline.py` | Runs the whole loop from the command line |
| `api.py` | Exposes everything as a REST service |
| `app.py` | The dashboard |

### How they connect

```
schema.py  ◄── everything reads the field definitions from here
    │
    ├── extractor.py ──► scorer.py ──► validate.py
    │                        │
    │                   simulator.py ◄── queries.py
    │                        │
    │                   generator.py ──► adopt.py ──► exporter.py
    │                        │
    └── ingest.py       console.py ──► coach.py
                             │
                    app.py + pipeline.py + api.py  (three ways to drive it)
```

The important structural point: **`schema.py` is the single source of truth.**
Add a field there and extraction, scoring, gap ranking, the simulator, the
generator and the export all pick it up with no other changes. That's what makes
the architecture defensible rather than just large.

---

## 6. The dashboard, tab by tab

Run it with `streamlit run app.py`. One product selector in the sidebar drives
every tab.

### 1 · My catalog

Add your own products, either by pasting one into a form or dropping a CSV.
Column names are auto-matched, so "Item Name" and "Retail Price SGD" work without
anyone reformatting anything.

*Proves: this isn't locked to our demo data.*

### 2 · Live query

Type a real shopper question. We run it twice, side by side:

- **BEFORE** — your listing as it is today
- **AFTER** — the same product with its gaps filled

Each side shows what the assistant picked, why, and what percentage of the
question your listing could answer. When your product flips from rejected to
recommended, we name the exact fields that did it.

Two behaviours worth demonstrating deliberately:

- **It refuses when it should.** Ask about a coffee grinder in a shoe catalog and
  it says nothing is suitable, rather than reaching for the closest thing.
- **It gives the shopper questions to send the merchant.** When the listing can't
  answer, you get *"Who should NOT buy this?"* with a download button. This is the
  mechanism by which demand for better content actually reaches brands.

### 3 · Readiness

The score for one product with a verdict sentence, then a breakdown by category.
Each row says how many of that category's fields you've written, what percentage
of the score it's worth, and one line on why it matters. The two rows worth 45%
are visually pulled out.

Below that, the **provenance stripe**: 75 ticks, colour-coded. Green means you
wrote it, grey means missing. The whole content state in one glance.

There's also an expander showing literally everything the assistant can see about
your product. Anything not in that box does not exist as far as it's concerned.

### 4 · Simulation

Which of the 150 questions your product was a candidate for, won, and lost.
Filterable by shopper type and by "only the ones I lost". Green means you were
recommended, red means you weren't, and each loss shows the reason.

### 5 · Gaps & coaching

Two halves.

**What's missing** — every lost question traced back to the one field that would
have answered it, in priority order, plus which shopper types you lose worst.
Open any gap for what it means, how to write it, why it's worth doing, and a
paste-ready prompt for drafting it in your own tools.

**What's weak** — problems in the copy you *already have*. This is where
"premium" and "revolutionary" get flagged as adjectives with nothing behind them.

### 6 · Edit content

Three sub-tabs:

- **Basic details** — name, brand, category, price, currency, availability,
  rating, review count, size range
- **Description & specs** — the body copy, spec bullets, and customer reviews
- **Deeper fields** — the AI's suggested draft on the left, your version on the
  right, for any of the 75 fields

One deliberate design choice: **text you write is marked verified, the AI's draft
is marked generated, and verified scores higher.** So editing beats accepting a
draft. The incentive points the right way.

### 7 · Publish

The approval queue — nothing AI-written goes live until a person ticks it, ordered
by how many shopper questions each field wins back. Then the actual deliverables:
a script tag for the product page, a CSV for the catalog system, a spreadsheet for
reviewing offline.

---

## 7. How this answers the brief

### The brief's five questions

**"What information does an AI agent need to confidently recommend a product?"**
We answered it concretely: 75 named fields in 7 weighted categories, in
`schema.py`. Not a philosophy, a checklist you can score against.

**"How should brands describe products beyond titles and specifications?"**
With context and constraints. Our weighting says those are 45% of what matters,
and the ablation experiment backs that up with measured evidence.

**"How can attributes, personas, use cases, comparisons and storytelling be
represented so AI can reason over them?"** As structured fields with provenance,
exported as Schema.org JSON-LD plus `agentready:*` extensions for the things the
standard has no vocabulary for.

**"How can brands measure whether their content is AI-ready?"** Two ways. A
static Coverage Score with no API calls, and a simulated win rate against 150
shopper questions. We tested that the first predicts the second: Spearman +0.687.

**"Can generative AI transform existing catalogs into agent-optimised content?"**
Yes, and the interesting part is that it should be *targeted*. We don't rewrite
descriptions. The simulator says which fields lost which questions, and the
generator fills exactly those.

### The brief's suggested outcomes

We built five of the six they listed:

- ✅ **AI Content Copilot** — `generator.py` plus the Edit tab
- ✅ **Content Readiness Score** — `scorer.py`, validated as predictive
- ✅ **Persona-Aware Content Generator** — persona pitches, 15 shopper archetypes
- ✅ **Simulation Platform** — `simulator.py`, 150 queries, field-level gap report
- ✅ **Structured Knowledge Layer** — `exporter.py`, JSON-LD plus extensions

### The rubric

| Criterion | Where we answer it |
|---|---|
| **1 · Problem comprehension** | The weighting argument. We can name exactly which content is missing and prove no catalog has it |
| **2 · Solution architecture** | One schema drives everything. Plus `validate.py`, which tests our own design rather than asserting it |
| **3 · AI reasoning quality** | The Live query tab. Judge types their own question, nothing scripted. It refuses when it should |
| **4 · Scalability** | `pipeline coldstart` runs a category we never designed for, from a raw CSV, with no code changes |
| **5 · Brand adoptability** | Approval queue, CSV round-trip, one script tag. Read-only scoring means zero integration to get value |

Full evidence with reproduction commands is in `RUBRIC.md`.

---

## 8. Numbers to quote

All reproducible. Run the commands in the right column.

| Number | What it means | Command |
|---|---|---|
| **16.9 → 56.4** | Readiness score before and after | `pipeline run --hero p001` |
| **9.3% → 72.0%** | Share of 150 shopper questions won | same |
| **21.3%** | Win rate when *every* competitor also adopts (random baseline 8.3%) | same |
| **+0.687** | Correlation: does the score predict the win rate? | `validate` |
| **+0.495** | Correlation: do our weights match measured damage? | `validate` |
| **13.0 → 39.9** | Same loop on a category we never designed for | `pipeline coldstart ...` |

### Be honest about the 72%

That's a first-mover number: one optimised product against eleven that did
nothing. A judge will poke at it, so volunteer it first. The fair-fight control
where everyone adopts lands at **21.3% against an 8.3% random baseline** — still
2.5×, because the content is genuinely better matched rather than just longer.

Volunteering the weakness before someone finds it is worth more than the bigger
number.

### Two things we got wrong and fixed

Say these out loud if asked about method. They demonstrate rigour better than a
clean result would.

**Our first ablation was invalid.** We removed each category from *every* product
at once. Context and Constraints both showed zero damage — because deleting a
category from everyone removes it as a differentiator. We switched to removing it
from one product only, which measures the question a brand actually asks: *what
does skipping this cost me?*

**Our first correlation test was meaningless.** Run across only-optimised
products, the scores compressed into a 48–56 band and we measured a weak +0.30.
Restricted range, not a weak metric. A mixed corpus fixed it.

**And a bug the dashboard surfaced.** We noticed products winning questions while
answering *zero* of the shopper's implicit questions — keyword similarity alone
was carrying them over the line. 19 of 150 queries. An assistant must now answer
at least one question before it recommends anything. That's now 0 of 150.

---

## 9. Glossary

Jargon you'll hear in the code, in plain terms.

| Term | Means |
|---|---|
| **Field** | One of the 75 pieces of information, e.g. "what conditions it's built for" |
| **Tier** | One of the 7 categories fields are grouped into |
| **Coverage score** | Out of 100. How much of the 75 you've published, weighted |
| **Provenance** | Where a value came from: you wrote it, we inferred it, AI drafted it, or it's missing |
| **Probe** | The fields a shopper question implicitly asks about |
| **Win rate** | Share of shopper questions where an assistant recommends your product |
| **Retrieval rate** | Share where you at least made the shortlist |
| **Persona** | One of 15 shopper archetypes, e.g. "half marathon beginner" |
| **Agent view** | Everything an assistant can see about your product |
| **Fair fight** | The control run where every competitor also adopts |
| **Ablation** | Deleting one category on purpose to measure what it was worth |
| **Cold start** | Running on a product category the system was never designed for |
| **Mock / offline mode** | Runs with no API key using keyword heuristics. Deliberately strong on specs and weak on context, which is what a real catalog looks like |

---

## Where to go next

- **`README.md`** — setup, install, troubleshooting, the 24-hour build plan
- **`RUBRIC.md`** — evidence for each rubric criterion, with commands
- **`DEMO.md`** — the timed 7-minute presentation script
- **`agentready/schema.py`** — if you read one code file, read this one
