"""Heuristic auto-scheduler and replacement finder.

Greedy and fairness-aware: it fills the hardest shifts first, always picks the
least-loaded eligible person, and refuses any assignment that would introduce a
rule problem. Gaps it cannot fill are reported, never forced.
"""
from __future__ import annotations

from collections import defaultdict

from constraints import SHIFTS, new_violations_if_added, week_dates, parse_date

RISK_WEIGHT = 1.5  # how strongly predicted call-out risk steers candidate choice

FILL_ORDER = ["night", "morning", "evening"]


def _by_employee(assignments):
    grouped = defaultdict(list)
    for a in assignments:
        grouped[a["employee_id"]].append(a)
    return grouped


def auto_schedule(employees, rules, requirements, week_start, keep=None, risk_model=None):
    """Return (assignments, unfilled). `keep` assignments are preserved.

    With a `risk_model`, ties between eligible people are broken toward whoever the
    ML model predicts is least likely to call out of that shift.
    """
    assignments = list(keep or [])
    mine = _by_employee(assignments)
    unfilled = []
    role_of = {e["id"]: e["role"] for e in employees}

    for day in week_dates(week_start):
        for shift in FILL_ORDER:
            for role, need in requirements[shift].items():
                have = sum(1 for a in assignments
                           if a["date"] == day and a["shift"] == shift and role_of[a["employee_id"]] == role)
                for _ in range(need - have):
                    pick = _best_candidate(employees, mine, rules, day, shift, role, risk_model)
                    if pick is None:
                        unfilled.append({"date": day, "shift": shift, "role": role})
                        continue
                    a = {"employee_id": pick["id"], "date": day, "shift": shift}
                    assignments.append(a)
                    mine[pick["id"]].append(a)
    return assignments, unfilled


def _prev_day_same_shift(assigns, day, shift):
    prev = parse_date(day).toordinal() - 1
    return any(parse_date(a["date"]).toordinal() == prev and a["shift"] == shift for a in assigns)


def _best_candidate(employees, mine, rules, day, shift, role, risk_model=None):
    eligible = []
    for emp in employees:
        if emp["role"] != role:
            continue
        cand = {"employee_id": emp["id"], "date": day, "shift": shift}
        existing = mine.get(emp["id"], [])
        if new_violations_if_added(emp, existing, cand, rules):
            continue  # any new error OR warning disqualifies in auto mode
        eligible.append((emp, existing))
    if not eligible:
        return None

    risks = [0.0] * len(eligible)
    if risk_model is not None:
        from ml_model import features_for  # local import keeps scheduler usable without sklearn
        risks = risk_model.predict_rows([features_for(e, ex, day, shift) for e, ex in eligible])

    scored = []
    for (emp, existing), p in zip(eligible, risks):
        hours = sum(SHIFTS[a["shift"]]["hours"] for a in existing)
        score = hours / max(emp.get("max_hours", 40), 1)
        if shift == "night":
            score += 0.1 * sum(1 for a in existing if a["shift"] == "night")
        if _prev_day_same_shift(existing, day, shift):
            score -= 0.15  # keep people on a steady rotation
        score += RISK_WEIGHT * p
        scored.append((score, emp["name"], emp))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def find_candidates(employees, assignments, rules, day, shift, role, remove=None, limit=5):
    """People who could take (day, shift) without breaking any rule.

    `remove` is an assignment being replaced; its owner is excluded.
    """
    base = [a for a in assignments if a != remove]
    mine = _by_employee(base)
    out = []
    for emp in employees:
        if emp["role"] != role:
            continue
        if remove and emp["id"] == remove["employee_id"]:
            continue
        existing = mine.get(emp["id"], [])
        cand = {"employee_id": emp["id"], "date": day, "shift": shift}
        if cand in existing or new_violations_if_added(emp, existing, cand, rules):
            continue
        hours = sum(SHIFTS[a["shift"]]["hours"] for a in existing)
        out.append({
            "employee_id": emp["id"],
            "name": emp["name"],
            "role": emp["role"],
            "weekly_hours_after": hours + SHIFTS[shift]["hours"],
            "max_hours": emp.get("max_hours", 40),
        })
    out.sort(key=lambda c: (c["weekly_hours_after"] / c["max_hours"], c["name"]))
    return out[:limit]


def fix_target(violation, employees):
    """Translate a violation into (day, shift, role, assignment_to_remove)."""
    if violation["code"] == "UNDERSTAFFED":
        return violation["date"], violation["shift"], violation["role"], None
    emp = next(e for e in employees if e["id"] == violation["employee_id"])
    remove = {"employee_id": emp["id"], "date": violation["date"], "shift": violation["shift"]}
    return violation["date"], violation["shift"], emp["role"], remove
