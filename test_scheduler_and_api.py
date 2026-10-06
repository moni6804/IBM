import pytest

from constraints import check_schedule
from scheduler import auto_schedule, find_candidates
from store import seed_state

WS = "2026-10-05"


def test_auto_schedule_has_no_employee_level_problems():
    s = seed_state(WS)
    assignments, unfilled = auto_schedule(s["employees"], s["rules"], s["requirements"], WS)
    vs = check_schedule(s["employees"], assignments, s["rules"], s["requirements"], WS)
    assert [v for v in vs if v["code"] != "UNDERSTAFFED"] == []
    assert len(unfilled) <= 3, f"too many unfilled slots: {unfilled}"


def test_find_candidates_respects_rules():
    s = seed_state(WS)
    # Ananya (e3) is on leave Wed/Thu, so she must never be offered Wednesday.
    cands = find_candidates(s["employees"], [], s["rules"], "2026-10-07", "morning", "Nurse")
    assert cands and all(c["employee_id"] != "e3" for c in cands)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "state.json"))
    import importlib
    import app as app_module
    importlib.reload(app_module)
    app_module.store.reset()
    return app_module.app.test_client()


def test_state_endpoint(client):
    data = client.get("/api/state").get_json()
    assert len(data["employees"]) == 15 and data["ai"]["provider"] == "offline"


def test_assign_blocks_rule_break_unless_forced(client):
    state = client.get("/api/state").get_json()
    ws = state["week_start"]
    leave_day = state["employees"][2]["time_off"][0]  # e3
    body = {"employee_id": "e3", "date": leave_day, "shift": "morning"}
    r = client.post("/api/assignments", json=body)
    assert r.status_code == 409 and r.get_json()["violations"][0]["code"] == "UNAVAILABLE"
    r = client.post("/api/assignments", json={**body, "force": True})
    assert r.status_code == 200
    assert any(v["code"] == "UNAVAILABLE" for v in r.get_json()["violations"])
    assert ws  # week is set


def test_auto_then_suggest_fix_and_apply(client):
    data = client.post("/api/schedule/auto").get_json()
    assert not data["violations"]
    victim = data["assignments"][0]
    data = client.delete("/api/assignments", json=victim).get_json()
    gap = next(v for v in data["violations"] if v["code"] == "UNDERSTAFFED")
    r = client.post("/api/ai/suggest", json={"violation_id": gap["id"]}).get_json()
    assert r["candidates"] and r["choice"]["employee_id"]
    add = {**r["add"], "employee_id": r["choice"]["employee_id"]}
    after = client.post("/api/assignments/swap", json={"add": add, "remove": r["remove"]}).get_json()
    assert len(after["assignments"]) == len(data["assignments"]) + 1
    assert not after["violations"]


def test_ai_endpoints_work_offline(client):
    client.post("/api/schedule/auto")
    assert client.post("/api/ai/explain").get_json()["text"]
    assert client.post("/api/ai/ask", json={"question": "Who is busiest?"}).get_json()["text"]
    assert client.post("/api/ai/ask", json={"question": ""}).status_code == 400
