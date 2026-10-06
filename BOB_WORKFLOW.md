# Building ShiftWise AI with IBM Bob

How the project maps onto Bob's two modes. Paste the prompts as-is, or adapt them. Use this file as the write-up for your submission.

## Plan Mode: architecture first

**Prompt 1: constraint model**
> Design the data model and rule engine for a hospital shift scheduler. Entities: employee (role, contract hours, weekday/shift unavailability, approved leave), assignment (employee, date, shift), staffing requirement (shift, role, headcount). Rules: no double booking, 11h minimum rest, max 6 consecutive days, 48h legal weekly cap, overtime above contract, night-shift limit, understaffing by role. Separate hard errors from soft warnings. Output the module boundaries and function signatures, no code yet.

**Prompt 2: AI boundary**
> Propose how an LLM (IBM Granite on watsonx.ai) should fit in without being trusted for correctness. The LLM may explain and rank, but a deterministic engine must generate and validate every option. Include the failure mode when watsonx is unavailable.

**Outcome:** the split between `constraints.py` (truth), `scheduler.py` (search), and `ai_engine.py` (language), plus the offline fallback.

## Code Mode: build in parallel

| Track | Prompt |
|---|---|
| Backend | "Implement `constraints.py` from the plan, with `employee_violations`, `coverage_violations`, `check_schedule`, and `new_violations_if_added` for real-time checks. Add pytest cases for every rule." |
| Scheduler | "Implement a fairness-aware greedy auto-scheduler that fills night shifts first and rejects any candidate that introduces a violation. Add `find_candidates` for replacement suggestions." |
| API | "Create the Flask endpoints in the README. Adding a shift must return 409 with reasons when it breaks a hard rule, unless `force` is set." |
| AI | "Implement the watsonx.ai chat client with IAM token caching. Every AI feature needs a deterministic fallback. `pick` must only return a candidate the engine verified." |
| Frontend | "Build a no-build-step dashboard: color-coded week grid, per-person hours bar, coverage row, rule-check panel with Suggest fix, and an assistant panel." |

| ML | "Add a scikit-learn call-out risk model: synthetic training data, features from the schedule, compare three classifiers with cross-validation, pick a threshold without touching the test set, evaluate once, add permutation importance. Expose it through the API and feed it into the auto-scheduler as a tie-break." |

## Review prompts (good for Bob's review/explain features)

- "Find any case where `auto_schedule` could produce a schedule that `check_schedule` flags with an error."
- "What happens at the week boundary? Which rules ignore the previous week, and how would we load it?"
- "Check `ml_model.py` for data leakage between training, threshold selection and evaluation."
- "Review `ai_engine.py` for prompt-injection risk from employee names or free-text questions."

## Known limits (be upfront in your submission)

- Rules only look inside one week; rest and consecutive-day checks don't read the previous week yet.
- Auto-scheduler is greedy, not optimal. It's fast and never breaks a rule, but may leave a gap an exact solver (OR-Tools CP-SAT) could fill.
- State is a single JSON file; one shared roster, no auth.
