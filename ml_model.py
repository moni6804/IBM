"""Call-out risk model: predicts the probability that an employee calls out of a shift.

Pipeline (the classic supervised-learning workflow):
  1. Data       - synthetic attendance history (see generate_training_data)
  2. Features   - workload and fatigue signals computed from the schedule itself
  3. Models     - logistic regression, random forest, gradient boosting
  4. Selection  - 3-fold cross-validated ROC-AUC on the training split
  5. Threshold  - best-F1 operating point from cross-validated predictions
  6. Evaluation - untouched hold-out set: accuracy, precision, recall, F1, ROC-AUC
  7. Explain    - permutation importance (model-agnostic)

The model feeds the scheduler (risk-aware tie-breaks) and the UI (risk per shift).
It never changes whether a schedule is legal; that is the constraint engine's job.

IMPORTANT: no real attendance data ships with this repo, so training data is
synthetic and encodes *assumed* fatigue drivers. Swap `generate_training_data`
for a real HR attendance export to make predictions meaningful.
"""
from __future__ import annotations

import threading
import time
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constraints import SHIFTS, SHIFT_ORDER, parse_date, shift_window

FEATURES = [
    "is_weekend", "is_evening", "is_night", "is_nurse", "part_time",
    "consecutive_days", "weekly_hours_before", "hours_ratio_after",
    "rest_hours", "nights_this_week", "over_contract",
]
FEATURE_LABELS = {
    "is_weekend": "Weekend shift",
    "is_evening": "Evening shift",
    "is_night": "Night shift",
    "is_nurse": "Nurse role",
    "part_time": "Part-time contract",
    "consecutive_days": "Days worked in a row",
    "weekly_hours_before": "Hours already this week",
    "hours_ratio_after": "Load vs. contract",
    "rest_hours": "Rest since last shift",
    "nights_this_week": "Night shifts this week",
    "over_contract": "Beyond contract hours",
}
LOW_BELOW, HIGH_FROM = 0.12, 0.25  # risk level cut-offs on call-out probability
MODEL_VERSION = 2


def risk_level(p: float) -> str:
    return "high" if p >= HIGH_FROM else "medium" if p >= LOW_BELOW else "low"


# ---------------------------------------------------------------- features --
def features_for(emp: dict, prior: list[dict], day: str, shift: str) -> list[float]:
    """Feature vector for `emp` entering (day, shift), given their earlier assignments."""
    prior = sorted(prior, key=lambda a: (a["date"], SHIFT_ORDER[a["shift"]]))
    hours_before = sum(SHIFTS[a["shift"]]["hours"] for a in prior)
    max_hours = max(emp.get("max_hours", 40), 1)

    streak, cursor = 0, parse_date(day) - timedelta(days=1)
    worked = {a["date"] for a in prior}
    while cursor.isoformat() in worked:
        streak += 1
        cursor -= timedelta(days=1)

    rest = 48.0
    if prior:
        gap = (shift_window(day, shift)[0] - shift_window(prior[-1]["date"], prior[-1]["shift"])[1]).total_seconds() / 3600
        rest = float(min(max(gap, 0.0), 48.0))

    hours_after = hours_before + SHIFTS[shift]["hours"]
    return [
        1.0 if parse_date(day).weekday() >= 5 else 0.0,
        1.0 if shift == "evening" else 0.0,
        1.0 if shift == "night" else 0.0,
        1.0 if emp["role"] == "Nurse" else 0.0,
        1.0 if max_hours < 40 else 0.0,
        float(streak),
        float(hours_before),
        hours_after / max_hours,
        rest,
        float(sum(1 for a in prior if a["shift"] == "night")),
        1.0 if hours_after > max_hours else 0.0,
    ]


def prior_assignments(assignments: list[dict], employee_id: str, day: str, shift: str) -> list[dict]:
    key = (day, SHIFT_ORDER[shift])
    return [a for a in assignments
            if a["employee_id"] == employee_id and (a["date"], SHIFT_ORDER[a["shift"]]) < key]


# ----------------------------------------------------------- training data --
def generate_training_data(n: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic attendance history with assumed fatigue drivers plus noise.

    Drivers: long streaks, quick turnarounds, night work, overtime and weekends raise
    call-out odds. Noise keeps the problem realistic (perfect prediction is impossible).
    """
    rng = np.random.default_rng(seed)
    is_weekend = (rng.random(n) < 2 / 7).astype(float)
    shift_idx = rng.choice(3, size=n, p=[0.36, 0.34, 0.30])
    is_evening, is_night = (shift_idx == 1).astype(float), (shift_idx == 2).astype(float)
    is_nurse = (rng.random(n) < 0.6).astype(float)
    part_time = (rng.random(n) < 0.25).astype(float)
    max_hours = np.where(part_time == 1, 24.0, 40.0)
    consecutive = np.minimum(rng.poisson(2.2, n), 6).astype(float)
    shifts_before = np.minimum(rng.poisson(2.4, n), 6)
    hours_before = shifts_before * 8.0
    nights = np.minimum(rng.poisson(0.8, n), shifts_before).astype(float)
    rest = rng.choice([8.0, 12.0, 16.0, 24.0, 48.0], size=n, p=[0.10, 0.12, 0.28, 0.25, 0.25])
    hours_after = hours_before + 8.0
    ratio = hours_after / max_hours
    over = (hours_after > max_hours).astype(float)

    logit = (-4.3
             + 0.45 * consecutive
             + 1.20 * (rest < 11)
             + 0.80 * is_night + 0.20 * is_evening
             + 0.60 * is_weekend
             + 1.00 * over
             + 0.50 * nights
             + 0.90 * np.clip(ratio - 0.8, 0, None)
             - 0.20 * part_time
             + 0.10 * is_nurse
             + rng.normal(0, 0.35, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([is_weekend, is_evening, is_night, is_nurse, part_time, consecutive,
                         hours_before, ratio, rest, nights, over])
    return X, y


def _candidates(seed: int):
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random forest": RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=10,
                                                random_state=seed, n_jobs=1),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08,
                                                        random_state=seed),
    }


def train(n: int = 5000, seed: int = 42) -> dict:
    started = time.time()
    X, y = generate_training_data(n, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    comparison = []
    for name, model in _candidates(seed).items():
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        comparison.append({"name": name, "cv_auc": round(float(scores.mean()), 4), "cv_std": round(float(scores.std()), 4)})
    best_name = max(comparison, key=lambda c: c["cv_auc"])["name"]

    # Operating threshold from cross-validated predictions (never touches the test set).
    best = _candidates(seed)[best_name]
    cv_prob = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, cv_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    threshold = float(thr[int(np.argmax(f1s))])

    best.fit(X_tr, y_tr)
    prob = best.predict_proba(X_te)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=4, random_state=seed)
    importance = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "importance": round(max(float(v), 0.0), 4)}
         for f, v in zip(FEATURES, imp.importances_mean)),
        key=lambda d: d["importance"], reverse=True)

    return {
        "version": MODEL_VERSION,
        "model": best,
        "info": {
            "selected": best_name,
            "comparison": comparison,
            "threshold": round(threshold, 3),
            "metrics": {
                "roc_auc": round(float(roc_auc_score(y_te, prob)), 3),
                "accuracy": round(float(accuracy_score(y_te, pred)), 3),
                "precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
                "recall": round(float(recall_score(y_te, pred, zero_division=0)), 3),
                "f1": round(float(f1_score(y_te, pred, zero_division=0)), 3),
            },
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "importance": importance,
            "data": {"rows": n, "train": int(len(y_tr)), "test": int(len(y_te)),
                     "call_out_rate": round(float(y.mean()), 3), "synthetic": True, "seed": seed},
            "trained_seconds": round(time.time() - started, 1),
            "levels": {"low_below": LOW_BELOW, "high_from": HIGH_FROM},
        },
    }


# ------------------------------------------------------------------- facade --
class RiskModel:
    """Loads a saved model or trains one, and answers risk queries."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._bundle = None
        self._load_or_train()

    def _load_or_train(self):
        if self.path and self.path.exists():
            try:
                bundle = joblib.load(self.path)
                if bundle.get("version") == MODEL_VERSION:
                    self._bundle = bundle
                    return
            except Exception:
                pass
        self.retrain()

    def retrain(self, n: int = 5000, seed: int = 42) -> dict:
        bundle = train(n, seed)
        with self._lock:
            self._bundle = bundle
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(bundle, self.path)
            except OSError:
                pass
        return self.info

    @property
    def info(self) -> dict:
        return self._bundle["info"]

    def predict_rows(self, rows: list[list[float]]) -> list[float]:
        if not rows:
            return []
        with self._lock:
            model = self._bundle["model"]
        return [float(p) for p in model.predict_proba(np.asarray(rows, dtype=float))[:, 1]]

    def risk_for_assignments(self, employees: list[dict], assignments: list[dict]) -> dict[str, float]:
        """{'emp|date|shift': probability} for every scheduled shift."""
        emp_by_id = {e["id"]: e for e in employees}
        rows, keys = [], []
        for a in assignments:
            emp = emp_by_id.get(a["employee_id"])
            if not emp:
                continue
            prior = prior_assignments(assignments, a["employee_id"], a["date"], a["shift"])
            rows.append(features_for(emp, prior, a["date"], a["shift"]))
            keys.append(f"{a['employee_id']}|{a['date']}|{a['shift']}")
        return dict(zip(keys, self.predict_rows(rows)))

    def slot_risk(self, per_assignment: dict[str, float]) -> dict[str, dict]:
        """{'date|shift': {'p_any': P(at least one call-out), 'expected': E[call-outs]}}"""
        grouped: dict[str, list[float]] = {}
        for key, p in per_assignment.items():
            _, day, shift = key.split("|")
            grouped.setdefault(f"{day}|{shift}", []).append(p)
        out = {}
        for slot, ps in grouped.items():
            none = float(np.prod([1 - p for p in ps]))
            out[slot] = {"p_any": round(1 - none, 4), "expected": round(float(sum(ps)), 3)}
        return out

    def what_if(self, emp: dict, assignments: list[dict], day: str) -> dict[str, float]:
        """Risk if `emp` worked each shift on `day` (their other assignments unchanged)."""
        others = [a for a in assignments if not (a["employee_id"] == emp["id"] and a["date"] == day)]
        rows = [features_for(emp, prior_assignments(others, emp["id"], day, s), day, s) for s in SHIFTS]
        return dict(zip(SHIFTS, self.predict_rows(rows)))
