import importlib

import numpy as np
import pytest

from constraints import check_schedule
from ml_model import FEATURES, RiskModel, features_for, generate_training_data, risk_level, train
from scheduler import auto_schedule
from store import seed_state

WS = "2026-10-05"  # Monday


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return RiskModel(tmp_path_factory.mktemp("ml") / "m.joblib")


def a(day, shift, eid="e1"):
    return {"employee_id": eid, "date": f"2026-10-{5 + day:02d}", "shift": shift}


def test_training_data_shape_and_rate():
    X, y = generate_training_data(2000, seed=1)
    assert X.shape == (2000, len(FEATURES))
    assert 0.05 < y.mean() < 0.35  # realistic, imbalanced call-out rate


def test_training_data_is_reproducible():
    X1, y1 = generate_training_data(500, seed=7)
    X2, y2 = generate_training_data(500, seed=7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_model_beats_chance_on_holdout(model):
    m = model.info["metrics"]
    assert m["roc_auc"] > 0.7 and m["recall"] > 0.3
    assert model.info["selected"] in {c["name"] for c in model.info["comparison"]}
    assert model.info["data"]["synthetic"] is True


def test_feature_vector_is_computed_from_schedule():
    emp = seed_state(WS)["employees"][0]
    prior = [a(0, "morning"), a(1, "morning"), a(2, "night")]
    row = dict(zip(FEATURES, features_for(emp, prior, "2026-10-09", "morning")))  # Fri
    assert row["consecutive_days"] == 0           # Thursday was a day off
    assert row["weekly_hours_before"] == 24 and row["nights_this_week"] == 1
    row = dict(zip(FEATURES, features_for(emp, prior, "2026-10-08", "morning")))  # Thu after Wed night
    assert row["consecutive_days"] == 3
    assert row["rest_hours"] == 0  # Wed night ends 06:00 Thu, Thu morning starts 06:00


def test_fatigued_worker_is_riskier_than_fresh_worker(model):
    emp = seed_state(WS)["employees"][0]
    fresh = features_for(emp, [], "2026-10-06", "morning")
    tired = features_for(emp, [a(i, "night") for i in range(5)], "2026-10-10", "night")
    p_fresh, p_tired = model.predict_rows([fresh, tired])
    assert p_tired > p_fresh * 1.5


def test_slot_risk_math(model):
    slots = model.slot_risk({"e1|2026-10-05|morning": 0.2, "e2|2026-10-05|morning": 0.5})
    s = slots["2026-10-05|morning"]
    assert s["p_any"] == pytest.approx(1 - 0.8 * 0.5) and s["expected"] == pytest.approx(0.7)


def test_risk_levels():
    assert (risk_level(0.05), risk_level(0.15), risk_level(0.4)) == ("low", "medium", "high")


def test_risk_aware_schedule_is_still_legal_and_not_riskier(model):
    s = seed_state(WS)
    plain, _ = auto_schedule(s["employees"], s["rules"], s["requirements"], WS)
    aware, unfilled = auto_schedule(s["employees"], s["rules"], s["requirements"], WS, risk_model=model)
    problems = [v for v in check_schedule(s["employees"], aware, s["rules"], s["requirements"], WS)
                if v["code"] != "UNDERSTAFFED"]
    assert problems == [] and unfilled == []
    mean = lambda asg: np.mean(list(model.risk_for_assignments(s["employees"], asg).values()))
    assert mean(aware) <= mean(plain)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("MODEL_FILE", str(tmp_path / "model.joblib"))
    import app as app_module
    importlib.reload(app_module)
    app_module.store.reset()
    return app_module.app.test_client()


def test_api_exposes_risk_and_model(client):
    state = client.post("/api/schedule/auto").get_json()
    assert state["risk"]["assignments"] and state["risk"]["slots"] and state["risk"]["top"]
    info = client.get("/api/ml/model").get_json()
    assert info["importance"] and info["confusion"]["tp"] >= 0


def test_what_if_endpoint(client):
    day = client.get("/api/state").get_json()["week_start"]
    r = client.post("/api/ml/what-if", json={"employee_id": "e1", "date": day}).get_json()
    assert set(r) == {"morning", "evening", "night"} and all(0 <= v["p"] <= 1 for v in r.values())
    assert client.post("/api/ml/what-if", json={"employee_id": "nope", "date": day}).status_code == 400


def test_retrain_endpoint(client):
    r = client.post("/api/ml/retrain", json={"seed": 3}).get_json()
    assert r["info"]["data"]["seed"] == 3 and "risk" in r["state"]
