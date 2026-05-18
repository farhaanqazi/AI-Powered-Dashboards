# Dependency & Version-Bump Policy — Phase 12 S12.3

## Automation

- **Dependabot** (`.github/dependabot.yml`) opens grouped weekly PRs for
  `pip` and `npm` minor/patch, monthly for GitHub Actions. Scans run on
  GitHub infra (no Actions billing).
- **SBOM**: `python scripts/generate_sbom.py` emits CycloneDX SBOMs for the
  Python env and the frontend when the (optional) tooling is present. Run it
  before a release and attach the SBOMs to the release.
- **Merge gate**: `.github/workflows/ci.yml` (lint + pytest + AI eval +
  frontend test/build) is the enforced gate **once `CI_ENABLED=true`**. Until
  billing is enabled, the local `pytest` run is the gate of record.

## Minor / patch

Auto-grouped by Dependabot; merge once the gate (local pytest + frontend
`npm test`/`build`) is green. No manual cadence required.

## Major bumps — cadence-gated

`react`, `react-dom`, and `vite` majors are **excluded from Dependabot
auto-PRs** and handled on an explicit cadence:

- **Quarterly review.** Evaluate React/Vite majors once per quarter, not
  reactively.
- **One major per PR.** Never combine a React major with a Vite major; each
  gets its own branch, its own `npm test` + `npm run build` + Playwright
  smoke, and a manual dashboard click-through.
- **Plugin lockstep.** A Vite major requires a matching `@vitejs/plugin-react`
  major in the same PR (the v3→v4 bump done in S12.2 is the reference case:
  v3 broke Vitest's React preamble; v4 fixed it and kept the Vite 4 build
  green).
- **Backout plan.** A major bump must be revertible as a single commit; do
  not mix it with feature work.

Other ecosystem majors (Python libs, non-build JS libs) follow the standard
flow: Dependabot surfaces them, the gate must pass, reviewer approves.

## Security

Dependabot security alerts are not cadence-gated — patch promptly regardless
of the schedule above, smallest viable bump first.
