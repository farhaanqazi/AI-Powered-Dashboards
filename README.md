# ML Dashboard Project

An experimental project to build an **AI-powered dashboard generator**.

The idea:  
User uploads a structured dataset (e.g. CSV), and the app will eventually:
- Analyse the data
- Automatically generate KPIs
- Recommend suitable charts
- Render a dashboard layout

At the moment this is a **work in progress**.  
This README documents what has been done so far.

---

## ✅ Status – Day 1

What works right now:

- Project folder created at `dashboard_project/`
- Python virtual environment (`venv/`) set up
- Flask installed and configured
- Basic Flask app (`app.py`) running locally
- HTML template system using `templates/`
- `index.html` rendered as the homepage
- Working file upload form:
  - User can upload a `.csv` file
  - Backend receives the file and confirms upload

What does **not** exist yet (planned next steps):

- No real CSV parsing/analysis yet
- No KPIs or chart generation
- No machine learning models integrated
- No dashboard visualisation (charts) yet

---

## 🧱 Project Structure (current)

```text
dashboard_project/
├── app.py                # Flask app entry point
├── requirements.txt      # Python dependencies (Flask, pandas, etc.)
├── .gitignore            # Ignored files (venv, __pycache__, etc.)
├── venv/                 # Virtual environment (not committed ideally)
├── templates/
│   └── index.html        # Homepage with file upload form
└── src/
    └── data/             # (placeholder for future parser/analyzer modules)
