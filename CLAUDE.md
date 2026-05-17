# Agent Instructions — Mandatory

## Upgrade-plan progress tracking (automatic, no user trigger required)

`UPGRADE-PLAN.md` in the project root is the single source of truth for planned
work. Every step (`**S<n>**`) and phase header carries a **planned date**.

**This rule fires automatically. Do not wait to be asked.**

Whenever you make a code change (any `Edit`/`Write`/`NotebookEdit`, new file,
deletion, or migration) that implements — fully or partially — a step or phase
in `UPGRADE-PLAN.md`, you MUST, in the **same turn, before ending your
response**, edit `UPGRADE-PLAN.md` to record the **actual** date against the
planned one.

### Exact convention

- Use the **real current date** (today's actual date from the environment), in
  `YYYY-MM-DD`. Never reuse the planned date as the actual date.
- The recorded output MUST pair the two dates explicitly and read as:
  **`original planned date -- actual implementation date`**.
- A step fully implemented: append the marker to that step line as
  ` — ✅ <original-planned-date> -- <actual-implementation-date>`.
  Example:
  `- **S0.1** — 2026-05-19 — Rotate the Groq API key... — ✅ 2026-05-19 -- 2026-05-22`
  (here `2026-05-19` = original planned date, `2026-05-22` = actual date).
- A step started but not finished: append
  ` — 🚧 <original-planned-date> -- in progress (started <YYYY-MM-DD>)`.
  When it completes, replace with the `✅ <planned> -- <actual>` form. Keep one
  marker per line (update in place, do not stack).
- When **every** step in a phase is `✅`, append the same paired form to the
  phase header using the phase's planned span and the actual span:
  ` — ✅ <planned-start → planned-end> -- <first-actual-date → last-actual-date>`.
- The planned date inside the marker is a copy of the step's original planned
  date; the inline planned date earlier on the line stays untouched. Never edit
  or remove any planned date.
- If a step is delivered out of planned order, still record its real actual
  date — the actual date reflects reality, not the schedule.

### Scope rules

- Only mark a step done when its described deliverable actually exists and (where
  it has a test/validation criterion) that criterion passes. Do not mark done on
  intent.
- If a change touches code unrelated to any plan step, do not edit the plan.
- If a change spans multiple steps, update each affected step line in the same
  turn.
- This file's rule is not optional and is not gated on the user mentioning the
  plan, dates, or tracking.
