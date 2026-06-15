# Running the app locally — baby steps

Two ways to run it. Pick one:

- **Mode A — Dev (live reload).** Use this to *see UI changes as you edit*. Two
  processes: Vite (frontend, port 5173) + uvicorn (backend, port **8000**).
- **Mode B — Single server.** Build the frontend once, let the backend serve it.
  One process on port **7860**. Closest to production.

All commands are PowerShell, run from the project root `F:\AI Powered Dashboards`.

---

## 0. One-time setup

You only do this once (or after dependencies change).

```powershell
# Python deps (the real env is venv\)
venv\Scripts\python.exe -m pip install -r requirements.txt

# Frontend deps
cd frontend
npm install
cd ..
```

**Auth key (required — without it the page shows "Set VITE_CLERK_PUBLISHABLE_KEY").**
Open `frontend\.env.local` and make sure this line has a real value:

```
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here
```

**AI narration (optional).** For the LLM-written insights/provenance path, add to
`.env` in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Without it the app still runs — it falls back to the deterministic heuristic layer.

---

## Mode A — Dev (live reload)

Open **two** terminals.

**Terminal 1 — backend on port 8000** (the Vite proxy expects exactly this port):

```powershell
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — frontend dev server:**

```powershell
cd frontend
npm run dev
```

Open **<http://localhost:5173>**.

> Why port 8000: the dev server proxies every `/api/...` call to
> `http://localhost:8000` (`frontend/vite.config.js`). If the backend is on any
> other port, uploads fail with **405 / Method Not Allowed**.

Edit a React file → the browser hot-reloads. Edit Python → uvicorn `--reload`
restarts the backend.

---

## Mode B — Single server (production-like)

Build the SPA once, then run only the backend; it serves the built site + API
from one port.

```powershell
# 1. Build the frontend
cd frontend
npm run build
cd ..

# 2. Run the backend (serves the built SPA + API)
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 7860
```

Open **<http://localhost:7860>**.

Re-run `npm run build` whenever you change frontend code (no hot reload here).

---

## Stopping

Press `Ctrl + C` in each terminal.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **405 / Method Not Allowed** on upload | Backend not on port 8000 in Mode A | Start uvicorn with `--port 8000` |
| Page says **"Set VITE_CLERK_PUBLISHABLE_KEY"** | Missing auth key | Add a real key to `frontend\.env.local`, restart the dev server |
| **FileNotFoundError `''`** on upload | A blank path var in `.env` (e.g. `JOB_SPOOL_DIR=`) | Leave it unset or give it a real path — blank is treated as unset now |
| AI insights look generic / no provenance | No `GROQ_API_KEY` | Add the key to `.env`, **restart the backend** (env is read only at startup) |
| Frontend changes don't show | You're in Mode B | Re-run `npm run build`, or switch to Mode A |

---

## Quick reference

| | Mode A (dev) | Mode B (single server) |
|---|---|---|
| Backend port | **8000** | **7860** |
| Frontend | `npm run dev` (port 5173) | pre-built, served by backend |
| Open | <http://localhost:5173> | <http://localhost:7860> |
| Live reload | yes | no |
| Use when | editing UI/code | quick run / prod check |

For the advanced setup (out-of-process Arq worker + Redis + Postgres, env vars,
PII extras) see [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md).
