"""Flask API + static frontend for ShiftWise AI."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from ai_engine import AIEngine  # noqa: E402  (after load_dotenv)
from ml_model import FEATURE_LABELS, RiskModel, risk_level  # noqa: E402
from constraints import (  # noqa: E402
    SHIFTS, check_schedule, coverage_table, new_violations_if_added, week_dates, weekly_hours,
)
from scheduler import auto_schedule, find_candidates, fix_target  # noqa: E402
from store import Store  # noqa: E402

app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
store = Store(Path(os.getenv("STATE_FILE", ROOT / "data" / "state.json")))
ai = AIEngine()
risk_model = RiskModel(Path(os.getenv("MODEL_FILE", ROOT / "data" / "model.joblib")))


def risk_snapshot(s):
    per = risk_model.risk_for_assignments(s["employees"], s["assignments"])
    slots = risk_model.slot_risk(per)
    top = sorted(slots.items(), key=lambda kv: kv[1]["p_any"], reverse=True)[:5]
    return {
        "assignments": {k: round(v, 4) for k, v in per.items()},
        "slots": slots,
        "top": [{"date": k.split("|")[0], "shift": k.split("|")[1], **v, "level": risk_level(v["p_any"])} for k, v in top],
        "avg": round(sum(per.values()) / len(per), 4) if per else 0,
    }


def risk_lines(r):
    return [f"{t['date']} {t['shift']}: {t['p_any']:.0%} chance of at least one call-out" for t in r["top"]]


def snapshot(extra=None):
    s = store.read()
    out = {
        **s,
        "shifts": SHIFTS,
        "violations": check_schedule(s["employees"], s["assignments"], s["rules"], s["requirements"], s["week_start"]),
        "hours": weekly_hours(s["assignments"]),
        "coverage": coverage_table(s["assignments"], s["employees"], s["requirements"], s["week_start"]),
        "ai": ai.status(),
        "risk": risk_snapshot(s),
        "ml": {"selected": risk_model.info["selected"], "metrics": risk_model.info["metrics"]},
    }
    out.update(extra or {})
    return out


def _slot(body):
    return {"employee_id": body.get("employee_id"), "date": body.get("date"), "shift": body.get("shift")}


def _validate_slot(state, slot):
    if slot["shift"] not in SHIFTS:
        return "Unknown shift."
    if slot["date"] not in week_dates(state["week_start"]):
        return "Date is outside the scheduled week."
    if not any(e["id"] == slot["employee_id"] for e in state["employees"]):
        return "Unknown employee."
    return None


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "ai": ai.status()})


@app.get("/api/state")
def get_state():
    return jsonify(snapshot())


@app.post("/api/assignments")
def add_assignment():
    """Add a shift. Blocks (409) if it would break a hard rule, unless force=true."""
    body = request.get_json(silent=True) or {}
    slot, force = _slot(body), bool(body.get("force"))
    state = store.read()
    problem = _validate_slot(state, slot)
    if problem:
        return jsonify({"error": problem}), 400
    if slot in state["assignments"]:
        return jsonify(snapshot())
    emp = next(e for e in state["employees"] if e["id"] == slot["employee_id"])
    mine = [a for a in state["assignments"] if a["employee_id"] == emp["id"]]
    errors = [v for v in new_violations_if_added(emp, mine, slot, state["rules"]) if v["severity"] == "error"]
    if errors and not force:
        return jsonify({"error": "This assignment breaks a rule.", "violations": errors}), 409
    store.update(lambda s: s["assignments"].append(slot))
    return jsonify(snapshot())


@app.delete("/api/assignments")
def remove_assignment():
    slot = _slot(request.get_json(silent=True) or {})
    store.update(lambda s: s.update(assignments=[a for a in s["assignments"] if a != slot]))
    return jsonify(snapshot())


@app.post("/api/assignments/swap")
def swap_assignment():
    """Atomically remove one assignment (optional) and add another. Used by 'Apply fix'."""
    body = request.get_json(silent=True) or {}
    add, remove = _slot(body.get("add") or {}), body.get("remove")
    state = store.read()
    problem = _validate_slot(state, add)
    if problem:
        return jsonify({"error": problem}), 400

    def apply(s):
        if remove:
            s["assignments"] = [a for a in s["assignments"] if a != _slot(remove)]
        if add not in s["assignments"]:
            s["assignments"].append(add)

    store.update(apply)
    return jsonify(snapshot())


@app.post("/api/schedule/auto")
def schedule_auto():
    s = store.read()
    assignments, unfilled = auto_schedule(s["employees"], s["rules"], s["requirements"], s["week_start"], risk_model=risk_model)
    store.update(lambda st: st.update(assignments=assignments))
    return jsonify(snapshot({"unfilled": unfilled}))


@app.post("/api/schedule/clear")
def schedule_clear():
    store.update(lambda s: s.update(assignments=[]))
    return jsonify(snapshot())


@app.post("/api/reset")
def reset():
    store.reset()
    return jsonify(snapshot())


@app.get("/api/ml/model")
def ml_model_info():
    return jsonify(risk_model.info)


@app.post("/api/ml/retrain")
def ml_retrain():
    body = request.get_json(silent=True) or {}
    seed = int(body.get("seed", 42))
    info = risk_model.retrain(seed=seed)
    return jsonify({"info": info, "state": snapshot()})


@app.post("/api/ml/what-if")
def ml_what_if():
    """Predicted call-out risk if this person worked each shift on this day."""
    body = request.get_json(silent=True) or {}
    s = store.read()
    emp = next((e for e in s["employees"] if e["id"] == body.get("employee_id")), None)
    if not emp or body.get("date") not in week_dates(s["week_start"]):
        return jsonify({"error": "Unknown employee or date."}), 400
    risks = risk_model.what_if(emp, s["assignments"], body["date"])
    return jsonify({k: {"p": round(v, 4), "level": risk_level(v)} for k, v in risks.items()})


@app.post("/api/ai/explain")
def ai_explain():
    s = snapshot()
    return jsonify(ai.explain(s["violations"], s["hours"], s["employees"], risk_lines(s["risk"])))


@app.post("/api/ai/suggest")
def ai_suggest():
    body = request.get_json(silent=True) or {}
    s = snapshot()
    violation = next((v for v in s["violations"] if v["id"] == body.get("violation_id")), None)
    if not violation:
        return jsonify({"error": "That problem no longer exists."}), 404
    day, shift, role, remove = fix_target(violation, s["employees"])
    candidates = find_candidates(s["employees"], s["assignments"], s["rules"], day, shift, role, remove)
    choice = ai.pick(violation, candidates)
    return jsonify({
        "violation": violation,
        "candidates": candidates,
        "choice": choice,
        "add": {"date": day, "shift": shift},
        "remove": remove,
    })


@app.post("/api/ai/ask")
def ai_ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Ask a question first."}), 400
    s = snapshot()
    return jsonify(ai.answer(question[:500], s["violations"], s["hours"], s["employees"], s["week_start"], risk_lines(s["risk"])))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG") == "1")
