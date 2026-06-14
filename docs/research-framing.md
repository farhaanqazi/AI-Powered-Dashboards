# From Dashboard Generator to Research Question: Trust-Preserving, Intent-Adaptive Visualization

*A research framing of the AI-Powered Dashboard Generator (Python · FastAPI · React)*

## 1. What was built, and why it is a research apparatus rather than a product

The system ingests an arbitrary tabular dataset and produces a complete analytical
dashboard — KPIs, charts, and a written narrative — with no human configuration. Its
distinguishing property is not that it generates dashboards, but *how* it constrains the
generation. The pipeline is organized as four deterministic analysis layers (syntactic
profiling, semantic role classification, relational analysis, and a heuristic
interpreter/strategist) sitting beneath a **semantic contract layer**. The contract
compiles an immutable, fingerprinted description of every column — its role, its legal
aggregations, its sensitivity — and every downstream figure must trace back to it through
a provenance token.

A large language model is present, but deliberately fenced out of every decision that
touches a number. It may relabel, reorder, group, and narrate; it may not compute,
estimate, or invent a value, and any output referencing a column or quantity not already
present in the ground-truth payload is rejected and replaced by the deterministic
fallback. This is enforced as a standing architectural invariant: *deterministic numbers,
AI decorative.*

The consequence is a system with an unusually strong, *provable* trust property: every
displayed figure is computed by verifiable code and is traceable to its source. That
property is the asset. It is also, as the next section argues, the precise cause of the
system's central limitation — and that coupling is the research question.

## 2. The observed phenomenon: the trust guarantee *is* the context-blindness

The system generates dashboards; it does not generate the *right* dashboard for this data
in this context. Two failures are visible in the code path, and they are the same failure
seen from two angles.

First, chart selection is driven entirely by **data structure**, never by **user intent**.
The interpreter enumerates everything statistically valid — a histogram for each numeric
column, a bar chart for each low-cardinality categorical, a line for each datetime-measure
pair, a scatter for each significant correlation. There is no channel through which an
analytical goal ("which segment is underperforming, and why?") can shape what the
dashboard shows. Intent does exist in the system, but only in a *separate* conversational
"Ask-Your-Data" path that answers a question with numbers and never feeds back to
re-compose the dashboard. The two paths do not talk.

Second, even on the data side, the system reasons about what is **valid**, not about what
is **relevant**. It knows a column is monetary and additive and therefore sums it; it does
not reason about which of twenty legal views actually matters for a decision. It produces
a correct, exhaustive, undifferentiated cluster of charts.

The critical observation is that these limitations are not bugs or unfinished features.
They are the **direct cost of the trust guarantee**. The system is trustworthy *because*
the flexible component (the LLM) was forbidden from deciding what to show; the moment one
lets a language model adapt the dashboard to intent and relevance, one reintroduces
exactly the hallucination risk the architecture was built to eliminate. Trust and
adaptivity are, in the present design, two ends of a single lever.

## 3. Positioning: two camps, each failing differently, and an empty corner between them

The territory has a name — automated / context-aware visualization recommendation — and
the existing work falls into two camps that fail in opposite ways.

- **Rule and heuristic recommenders** (Draco, Voyager, Mackinlay's "Show Me") encode
  perceptual and statistical principles. They are trustworthy and reproducible, but
  context-rigid: they recommend from data properties and fixed objectives, not from a
  user's evolving analytical goal.

- **LLM-based generators** (LIDA, Chat2VIS, and the NL2VIS line of work over benchmarks
  such as nvBench) are flexible and intent-responsive, but they place the language model in
  the numeric and selection path simultaneously. They hallucinate, and — more importantly —
  their correctness cannot be guaranteed, only sampled.

What no existing system cleanly occupies is the corner that is both **trustworthy** and
**intent-adaptive**: a system where an LLM is given authority over *what to show* while a
verifiable substrate retains exclusive authority over *what each number is*. The dashboard
generator described here is an existence proof of one extreme — maximal trust, minimal
adaptivity — built with the substrate already in place. That makes it an ideal vehicle for
studying the frontier rather than merely asserting a point on it.

## 4. The research question

> **Can a language model be given authority over *what an analytical dashboard should show*
> — adapting to a user's stated intent and to what actually matters in the data — while a
> deterministic layer retains exclusive, verifiable authority over *what every number is*;
> and what is the achievable frontier between numeric trust and analytical adaptivity?**

Two sub-questions make this concrete and testable:

1. **Formalizing the guarantee.** What is a precise, checkable definition of *numeric
   integrity* (every displayed figure is computed by trusted code and is provenance-traced),
   and can adaptive, LLM-driven composition be admitted *without* weakening it?

2. **Mapping the frontier.** As an LLM is granted progressively more control over selection,
   relevance ranking, and intent-fit, how much adaptivity can be bought before the trust
   guarantee degrades — and is the trade-off a smooth frontier or a sharp cliff?

The novelty is deliberately bounded and stated honestly. Decoupling computation from
presentation is not itself new; tool-using agents and LIDA-style pipelines do versions of
it. The contribution is (a) treating numeric integrity as a *hard, formalized constraint*
rather than a best-effort behavior, and (b) *measuring* the trust–adaptivity frontier
empirically, rather than presenting a single system and claiming it works.

## 5. Proposed investigation

The question is studied empirically with a compact, reproducible apparatus:

- **A benchmark** of roughly 15–30 datasets, each paired with two to three explicit
  analytical intents expressed in natural language.
- **Three conditions** evaluated on each (dataset, intent) pair: the deterministic system
  (the trustworthy/rigid extreme), a naive LLM dashboard generator (the flexible/untrusted
  extreme), and a classical rule recommender as a baseline.
- **Two measurement axes.** *Trust* — the fraction of displayed figures that are incorrect
  or unverifiable (expected near zero for the deterministic system by construction, and
  measurably non-zero for the naive generator; that contrast is itself a result). *Adaptivity
  / relevance* — whether the output actually serves the stated intent, scored with a rubric
  via an LLM-as-judge and a small human-rating check.
- **The central artifact** is a trust-versus-relevance plot on which each system is a point.
  The deterministic system occupies the high-trust, low-relevance corner; the naive generator
  occupies its opposite; the high-trust *and* high-relevance corner is empty. That empty corner
  is the thesis objective made visible.

This design is achievable by a single researcher, yields an honest result even where the
present system *loses* (it should score poorly on relevance, and saying so plainly is the
point), and converts an engineering project into a characterization study.

## 6. Why this is a PhD-worthy program, not a one-off paper

The immediate output is a focused characterization / position paper: it formalizes the
trust–adaptivity dilemma in automated dashboarding, presents the system as an existence
proof of one extreme, and quantifies the gap that separates the two camps. That is
publishable on its own and is strong evidence of the ability to find and frame a real
problem.

The longer arc — the actual doctoral program — is filling the empty corner: **grounded,
intent-adaptive composition**, in which a language model is trusted to decide what is worth
showing while never being trusted with a single number. This connects directly to active,
fundable research themes — grounded and verifiable generation, hallucination mitigation,
LLM agents constrained by guaranteed-correct tools, and human-AI decision support — and
generalizes well beyond dashboards to any setting where a flexible model must drive
selection over a substrate whose outputs must remain provably correct.

The honest one-line summary, the kind a supervisor trusts: *"I built a dashboard generator,
characterized why its trustworthiness and its context-blindness are the same design
decision, and propose to dissolve that trade-off."* That is a far more credible position
than claiming the problem is solved — and it is the position this project is uniquely
equipped to argue.
