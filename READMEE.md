# ShiftWise AI

An AI-powered shift scheduler for workforce management. It builds a weekly roster, checks every change against labor-law and availability rules in real time, and uses **IBM Granite on watsonx.ai** to explain problems and recommend fixes.

Built with **IBM Bob**: Plan Mode for the constraint architecture, Code Mode for the API and dashboard (see [docs/BOB_WORKFLOW.md](docs/BOB_WORKFLOW.md)).

## Why it's built this way

The LLM never decides what is legal. A deterministic **constraint engine** is the single source of truth. The AI layer only does what language models are good at:

1. **Explain** the current problems in plain language and say what to fix first.
2. **Rank** replacement candidates that the engine has already verified.
3. **Answer** questions about the roster.

If watsonx isn't configured, or a call fails, every feature falls back to a built-in summary, so the app always works.

```
 Browser (HTML/CSS/JS)
        │  REST
 Flask API (backend/app.py)
   ├── constraints.py   rules, validation, coverage      ← source of truth
   ├── scheduler.py     auto-schedule (risk-aware), replacement finder
   ├── ml_model.py      call-out risk model (scikit-learn)
   ├── ai_engine.py     watsonx Granite + offline fallback
   └── store.py         JSON-file persistence
```

## Machine learning layer

A supervised model predicts the **probability that each scheduled person calls out of their shift**, so managers can add standby cover where it matters.

| Step | What happens | Where |
|---|---|---|
| Data | 5,000 synthetic attendance records (reproducible seed) | `generate_training_data` |
| Features | 11 workload and fatigue signals computed from the live schedule: streak, rest since last shift, night load, hours vs. contract, weekend, role… | `features_for` |
| Models | Logistic regression, random forest, gradient boosting | `_candidates` |
| Selection | 3-fold cross-validated ROC-AUC on the training split | `train` |
| Threshold | Best-F1 operating point from cross-validated predictions | `train` |
| Evaluation | Untouched 25% hold-out: ROC-AUC, recall, precision, accuracy, confusion matrix | `train` |
| Explainability | Permutation importance, shown in the UI | `train` |

**How it's used**

- The grid shows a risk % on every shift, plus a forecast row for the chance of at least one call-out per shift. Staffing is exact, so a single call-out leaves a gap.
- The shift popover shows predicted risk for each option *before* you assign.
- **Auto-schedule is risk-aware**: among eligible people it prefers whoever the model predicts is least likely to call out. On the demo week this cut average predicted risk from 4.0% to 3.4% and the worst shift from 29% to 21%, with zero rule violations either way.
- Risk context is passed to Granite, so its briefings include standby advice.

**Honest limits.** There is no real attendance data in this repo, so the training data is synthetic and encodes *assumed* fatigue drivers. The metrics show the pipeline works; they don't prove real-world accuracy. To go live, replace `generate_training_data` with an HR attendance export and keep the rest. The ML layer only informs decisions; legality is still decided by the constraint engine.

## Features

- **Color-coded weekly grid**: morning, evening and night shifts, hours bar per person, coverage row per shift.
- **Real-time constraint checking**: assigning a shift that breaks a rule is blocked with the exact reasons, and can be overridden deliberately.
- **Auto-schedule**: fairness-aware greedy scheduler that never introduces a rule problem and reports any slot it can't fill.
- **Suggest fix**: for any problem, get verified replacements, an AI-recommended pick with a reason, and one-click apply.
- **Ask the assistant**: free-form questions grounded in the live roster.

### Rules enforced

| Rule | Severity |
|---|---|
| Double booking (two shifts on one day) | Error |
| Minimum 11h rest between shifts | Error |
| Unavailable weekday/shift or approved leave | Error |
| More than 6 consecutive working days | Error |
| Over 48h per week (legal cap) | Error |
| Over contract hours (overtime) | Warning |
| More than 3 night shifts per week | Warning |
| Understaffed shift (by role) | Warning |

Rules and staffing needs live in `backend/constraints.py` (`DEFAULT_RULES`, `DEFAULT_REQUIREMENTS`). Change the numbers to match your jurisdiction and policy.

## Quick start

```bash
git clone <your-repo-url> shiftwise-ai && cd shiftwise-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backend/app.py
```

Open http://localhost:5000, click **Auto-schedule week**, then click any cell to edit and watch the rule check update.

### Enable IBM Granite (optional)

```bash
cp .env.example .env
# fill in WATSONX_API_KEY and WATSONX_PROJECT_ID from your IBM Cloud / watsonx.ai project
```

The header pill changes from "Offline demo mode" to "Granite · <model>". Check that `WATSONX_MODEL_ID` is available in your watsonx region, and set `WATSONX_URL` to your region's endpoint.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | Employees, assignments, violations, hours, coverage |
| POST | `/api/assignments` | Add a shift. `409` + reasons if it breaks a rule (send `force: true` to override) |
| DELETE | `/api/assignments` | Remove a shift |
| POST | `/api/assignments/swap` | Atomically remove one shift and add another |
| POST | `/api/schedule/auto` | Generate the week |
| POST | `/api/schedule/clear` | Clear all shifts |
| POST | `/api/reset` | Restore demo data |
| GET | `/api/ml/model` | Selected model, metrics, confusion matrix, feature importance |
| POST | `/api/ml/retrain` | Regenerate data, retrain and re-evaluate |
| POST | `/api/ml/what-if` | `{employee_id, date}` → predicted risk for each shift |
| POST | `/api/ai/explain` | Briefing on current problems |
| POST | `/api/ai/suggest` | `{violation_id}` → verified candidates + AI pick |
| POST | `/api/ai/ask` | `{question}` → grounded answer |

## Tests

```bash
python -m pytest -q
```

Covers each rule, the ML pipeline (reproducible data, beats chance on hold-out, fatigue raises risk, risk-aware scheduling stays legal), the real-time check, the scheduler (no employee-level problems on the demo week), and the API including the 409/override and fix-apply flows. Tests always run offline.

## Project structure

```
backend/    Flask app, constraint engine, scheduler, ML model, AI layer, store
frontend/   index.html, style.css, app.js (no build step)
tests/      pytest suite
docs/       Bob workflow and prompts
data/       runtime state (git-ignored)
```

## Roadmap

- Train on real attendance data; add leave, weather and seasonality features
- Multi-week planning and rolling fairness across weeks
- Shift swap requests between employees with manager approval
- Import employees from CSV; export roster to PDF/ICS
- Auth and a real database (swap `store.py`)
- Natural-language rule editing ("no more than 2 nights in a row")

## License

MIT
