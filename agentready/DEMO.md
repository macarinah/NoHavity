# Demo script — 7 minutes

Rehearse this three times. Out loud. On the laptop you'll actually use.
The person who owns this is not the backup role.

**Before you present:**

```bash
python -m agentready.pipeline run --hero p001   # regenerates out/demo_state.json
streamlit run app.py                            # leave running, tab open
```

Open on the **Readiness** tab with **Meridian Streamline 4** selected in the sidebar.
Have a second terminal ready with `python -m agentready.validate` already run, output
scrolled back. Don't run it live; it takes a minute.

Current numbers, so you can say them without checking:

```
coverage    16.9 → 56.4   (+39.5)
win rate     9.3% → 72.0%  (first mover)
fair fight  21.3%          (everyone adopts; random baseline 8.3%)
validation  Spearman +0.687 coverage↔win, +0.495 weights↔ablation
cold start  13.0 → 39.9 on a category we never designed for
```

---

## 0:00 — The problem, in one product (30s)

> "This is a real product page for a running shoe. Here's everything the brand published."

Readiness tab → expand *What the AI assistant actually reads*.

> "Materials, weight, drop, midsole foam. Everything a spec sheet needs, and nothing
> a person needs."

---

## 0:30 — The score, and the shape of the failure (40s)

> "We score it against a 75-field schema built for how assistants reason. It gets 17."

Point at the tier bars.

> "Identity and specs are fine. Context — who is this for, what conditions — near
> zero. Constraints — who is this wrong for — completely empty. Those two tiers carry
> 45% of the score."

Point at the provenance stripe.

> "Every one of the 75 fields. Green is sourced from the brand's own copy, grey is
> missing. That's the whole content state in one glance."

---

## 1:10 — Watch it lose, in a query they choose (80s) ← **the emotional beat**

Live query tab. **Invite a judge to type their own.** If nobody bites, use:

> *"I'm training for a half marathon in Singapore's humid weather and need lightweight
> shoes under S$200."*

> "Nothing here is scripted. We infer which schema fields the query is implicitly
> asking about, then it goes through the identical path as our benchmark."

Point at the inferred field chips and the detected price cap.

> "This shoe is 212 grams, S$179, breathable mesh upper. It is objectively the right
> answer. And it loses."

**Pause.** Then scroll to *What to ask the merchant*.

> "And here's the part we're proudest of. When the assistant can't answer, it doesn't
> go quiet — it hands the shopper the exact questions to send the brand. That's the
> mechanism by which demand for better content actually reaches merchants."

---

## 2:30 — The gap report is a work order (40s)

Gaps & coaching tab.

> "Every rejection clusters back onto the schema field that would have answered it.
> `environment_conditions`, 34 queries lost. `not_suitable_for`, 28."

Expand one brief.

> "And this isn't 'improve your content'. It's what to write, why it matters
> commercially, and a prompt they can paste into their own tools. Brands keep
> editorial control."

Scroll to the copy critique.

> "It also marks what's weak in the copy they already have. 'Premium.'
> 'Revolutionary.' Adjectives with nothing behind them — an assistant can't repeat
> those to a shopper."

---

## 3:10 — One click, then a human (80s) ← **the win**

Publish tab.

> "We fill exactly those fields, grounded in the source. The model can reason that a
> 212g shoe with an open mesh upper handles humidity. It cannot invent a
> certification."

Point at the two stripes.

> "Before. After. Every new field is pink — generated, awaiting approval. Nothing
> publishes without a human. The queue is ordered by queries won back, so twenty
> minutes of review buys the most possible."

Point at the numbers.

> "Coverage 17 to 56. Win rate 9% to 72%."

**Then switch to the Edit tab and change one field live.**

> "And if they'd rather write it themselves — this counts as sourced, not generated.
> Editing scores higher than accepting our draft. The incentive points the right way."

---

## 4:30 — Volunteer the weakness (40s) ← **the credibility move**

> "That 72% is a first-mover number — one optimised product against eleven that did
> nothing. So we ran the control: everyone adopts."

> "It lands at 21% against an 8.3% random baseline. Still two and a half times, because
> the content is genuinely better matched, not just longer. We'd rather show you that
> than have you find it."

---

## 5:10 — We tested our own architecture (45s) ← **rubric 2**

Switch to the terminal with validation output.

> "We didn't want to just assert our tier weights were right, so we tested them. Two
> experiments. Does the score predict win rate — Spearman 0.69, yes. And tier
> ablation: strip one tier from this product only, leaving competitors intact, and
> measure the damage."

> "Context fit is the most expensive tier to lose, matching its highest weight.
> Constraints is third. Both experiments were wrong the first time — the write-up is
> in RUBRIC.md."

*If asked how they were wrong:* "We first ablated a tier from every product at once,
which deletes it as a differentiator and showed 0.0 damage. And we measured
correlation across only-optimised products, where coverage compresses into a 48–56
band. Restricted range, not a weak metric."

---

## 5:55 — It isn't hardcoded to two categories (35s) ← **rubric 4**

My catalog tab. Upload `data/sample_unseen_category.csv`.

> "Coffee equipment. No personas written for it, no vocabulary entries, no code
> changes. Columns are called 'Item Name' and 'Retail Price SGD' — eleven of twelve
> auto-mapped, and the one that didn't got folded into the copy rather than dropped."

Point at the selector.

> "It's in the dropdown, scored, and every tab follows it."

---

## 6:30 — Shippable (30s) ← **rubric 5**

Publish tab, deliverables row.

> "Schema.org JSON-LD as one script tag — paste into the PDP template, no redesign.
> Or the CSV back into their PIM. Or the review sheet, because a lot of content teams
> work in Excel and emailing them a web app link is how a pilot dies."

> "Step two of the integration path is the wedge: a brand gets a score and a
> competitor benchmark with zero integration, read-only on a CSV they already have.
> We ask nothing of them until they've seen a number they don't like."

---

## 7:00 — Close (30s)

> "Three things. The measurement layer is the product — anyone can generate copy,
> nobody can prove it worked. The extractor's failures *are* the product: a confident
> null is a content brief. And the highest-weighted tier in our schema is the one that
> tells an agent who the product is *wrong* for, because an agent that can reject you
> confidently is one that can recommend you confidently."

---

# If something breaks

| Breaks | Do this |
|---|---|
| Wifi dies | Nothing changes. Mock mode needs no network |
| Streamlit won't start | `python -m agentready.pipeline run` prints the delta to the terminal. Present from that |
| A tab errors | Skip to Publish. That tab alone is the pitch |
| Upload fails live | You already have 12 products loaded. Move on, don't debug on stage |
| Everything is on fire | `cat out/summary.json` and read the numbers |

**Do not** regenerate content live. It's cached. If a judge asks whether it's live, say
it's cached and offer to run it after — that's a normal engineering answer.

---

# Slides — 6, no more

1. **The shift.** A search box vs a paragraph of natural language. No bullets.
2. **The gap.** A real PDP next to a real shopper query, with MISSING between them.
3. **The schema.** Tier table, 25% and 20% highlighted. *45% of the score is content no catalog has.*
4. **The loop.** Five-box architecture. Ten seconds, don't read it aloud.
5. **The numbers.** 17→56, 9%→72%, fair fight 21% vs 8.3%, Spearman 0.69. Nothing else.
6. **What's next.** JSON-LD as a standard extension. Category benchmarking. Per-SKU scoring.

Slide 5 is what they remember. Everything else sets it up.

---

# Rubric coverage, if you're asked directly

| Criterion | Where it lands in this demo |
|---|---|
| 1 · Problem comprehension | 0:00–1:10, and the tier weights argument |
| 2 · Solution architecture | 5:10 validation, plus RUBRIC.md |
| 3 · AI reasoning quality | 1:10 live query, judge types their own |
| 4 · Scalability | 5:55 cold-start upload |
| 5 · Brand adoptability | 3:10 approval queue, 6:30 deliverables |

Full evidence with reproduction commands: **RUBRIC.md**.
Plain-English explanation of the whole system: **HOW-IT-WORKS.md**.
