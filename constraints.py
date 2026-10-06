"""Constraint engine: the single source of truth for what is a valid schedule.

The LLM never decides validity. It explains and ranks options, but every
assignment is checked here first.

An assignment is a dict: {"employee_id": str, "date": "YYYY-MM-DD", "shift": str}
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

SHIFTS = {
    "morning": {"label": "Morning", "start": 6, "hours": 8},
    "evening": {"label": "Evening", "start": 14, "hours": 8},
    "night": {"label": "Night", "start": 22, "hours": 8},  # ends 06:00 next day
}
SHIFT_ORDER = {"morning": 0, "evening": 1, "night": 2}
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

DEFAULT_RULES = {
    "min_rest_hours": 11,            # labor law: rest between shifts
    "max_consecutive_days": 6,       # labor law: at least one day off in seven
    "max_night_shifts_per_week": 3,  # fatigue policy
    "hard_weekly_hours_cap": 48,     # legal ceiling, even with overtime
}

# Staff needed per shift, by role (applies to every day of the week).
DEFAULT_REQUIREMENTS = {
    "morning": {"Nurse": 2, "Technician": 1},
    "evening": {"Nurse": 2, "Technician": 1},
    "night": {"Nurse": 1, "Technician": 1},
}


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def week_dates(week_start: str) -> list[str]:
    start = parse_date(week_start)
    return [(start + timedelta(days=i)).isoformat() for i in range(7)]


def shift_window(day: str, shift: str) -> tuple[datetime, datetime]:
    info = SHIFTS[shift]
    start = datetime.combine(parse_date(day), datetime.min.time()) + timedelta(hours=info["start"])
    return start, start + timedelta(hours=info["hours"])


def day_label(day: str) -> str:
    d = parse_date(day)
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"


def make_violation(code, severity, message, employee_id=None, day=None, shift=None, role=None):
    return {
        "id": f"{code}:{employee_id}:{day}:{shift}:{role}",
        "code": code,
        "severity": severity,  # "error" = breaks a hard rule, "warning" = should be reviewed
        "message": message,
        "employee_id": employee_id,
        "date": day,
        "shift": shift,
        "role": role,
    }


def _sort_key(a):
    return (a["date"], SHIFT_ORDER[a["shift"]])


def employee_violations(emp: dict, assigns: list[dict], rules: dict) -> list[dict]:
    """All rule problems for one employee, given only that employee's assignments."""
    out: list[dict] = []
    eid, name = emp["id"], emp["name"]
    ordered = sorted(assigns, key=_sort_key)
    if not ordered:
        return out

    # 1. Availability and approved leave
    unavailable = set(emp.get("unavailable", []))
    for a in ordered:
        wd = WEEKDAYS[parse_date(a["date"]).weekday()]
        if a["date"] in emp.get("time_off", []):
            out.append(make_violation(
                "UNAVAILABLE", "error",
                f"{name} is on approved leave on {day_label(a['date'])}.",
                eid, a["date"], a["shift"]))
        elif f"{wd}:all" in unavailable or f"{wd}:{a['shift']}" in unavailable:
            what = "that day" if f"{wd}:all" in unavailable else f"{SHIFTS[a['shift']]['label'].lower()} shifts"
            out.append(make_violation(
                "UNAVAILABLE", "error",
                f"{name} is not available for {what} on {day_label(a['date'])}.",
                eid, a["date"], a["shift"]))

    # 2. Double booking: more than one shift on the same calendar day
    by_day = defaultdict(list)
    for a in ordered:
        by_day[a["date"]].append(a)
    for day, items in by_day.items():
        for extra in items[1:]:
            out.append(make_violation(
                "DOUBLE_BOOKING", "error",
                f"{name} is booked on {len(items)} shifts on {day_label(day)}.",
                eid, day, extra["shift"]))

    # 3. Minimum rest between shifts on different days
    for prev, cur in zip(ordered, ordered[1:]):
        if prev["date"] == cur["date"]:
            continue
        gap = shift_window(cur["date"], cur["shift"])[0] - shift_window(prev["date"], prev["shift"])[1]
        gap_h = gap.total_seconds() / 3600
        if gap_h < rules["min_rest_hours"]:
            out.append(make_violation(
                "INSUFFICIENT_REST", "error",
                f"{name} gets only {gap_h:.0f}h rest before {SHIFTS[cur['shift']]['label'].lower()} "
                f"on {day_label(cur['date'])} (minimum {rules['min_rest_hours']}h).",
                eid, cur["date"], cur["shift"]))

    # 4. Weekly hours: contract limit (warning) and legal cap (error)
    hours = sum(SHIFTS[a["shift"]]["hours"] for a in ordered)
    last = ordered[-1]
    if hours > rules["hard_weekly_hours_cap"]:
        out.append(make_violation(
            "WEEKLY_CAP", "error",
            f"{name} is scheduled for {hours}h, above the legal cap of {rules['hard_weekly_hours_cap']}h.",
            eid, last["date"], last["shift"]))
    elif hours > emp.get("max_hours", 40):
        out.append(make_violation(
            "OVERTIME", "warning",
            f"{name} is on {hours}h, over their {emp.get('max_hours', 40)}h contract (overtime).",
            eid, last["date"], last["shift"]))

    # 5. Consecutive working days
    days = sorted({a["date"] for a in ordered})
    run = 1
    for prev, cur in zip(days, days[1:]):
        run = run + 1 if (parse_date(cur) - parse_date(prev)).days == 1 else 1
        if run > rules["max_consecutive_days"]:
            shift = next(a["shift"] for a in ordered if a["date"] == cur)
            out.append(make_violation(
                "CONSECUTIVE_DAYS", "error",
                f"{name} works {run} days in a row by {day_label(cur)} "
                f"(maximum {rules['max_consecutive_days']}).",
                eid, cur, shift))

    # 6. Night shift load
    nights = [a for a in ordered if a["shift"] == "night"]
    if len(nights) > rules["max_night_shifts_per_week"]:
        out.append(make_violation(
            "NIGHT_LIMIT", "warning",
            f"{name} has {len(nights)} night shifts this week (policy limit "
            f"{rules['max_night_shifts_per_week']}).",
            eid, nights[-1]["date"], "night"))
    return out


def coverage_table(assignments, employees, requirements, week_start):
    """{date: {shift: {role: {"have": n, "need": n}}}}"""
    role_of = {e["id"]: e["role"] for e in employees}
    have = defaultdict(int)
    for a in assignments:
        have[(a["date"], a["shift"], role_of.get(a["employee_id"]))] += 1
    table = {}
    for day in week_dates(week_start):
        table[day] = {}
        for shift, roles in requirements.items():
            table[day][shift] = {r: {"have": have[(day, shift, r)], "need": n} for r, n in roles.items()}
    return table


def coverage_violations(assignments, employees, requirements, week_start):
    out = []
    table = coverage_table(assignments, employees, requirements, week_start)
    for day, shifts in table.items():
        for shift, roles in shifts.items():
            for role, c in roles.items():
                if c["have"] < c["need"]:
                    missing = c["need"] - c["have"]
                    out.append(make_violation(
                        "UNDERSTAFFED", "warning",
                        f"{day_label(day)} {SHIFTS[shift]['label'].lower()} needs {missing} more "
                        f"{role}{'s' if missing > 1 else ''} ({c['have']}/{c['need']}).",
                        None, day, shift, role))
    return out


def check_schedule(employees, assignments, rules, requirements, week_start) -> list[dict]:
    """Full validation. Errors first, then warnings, then by date."""
    by_emp = defaultdict(list)
    for a in assignments:
        by_emp[a["employee_id"]].append(a)
    found = []
    for emp in employees:
        found.extend(employee_violations(emp, by_emp.get(emp["id"], []), rules))
    found.extend(coverage_violations(assignments, employees, requirements, week_start))
    found.sort(key=lambda v: (v["severity"] != "error", v["date"] or "", SHIFT_ORDER.get(v["shift"], 9)))
    return found


def new_violations_if_added(emp, emp_assigns, candidate, rules) -> list[dict]:
    """Violations that appear only because `candidate` is added (real-time check)."""
    before = {v["id"] for v in employee_violations(emp, emp_assigns, rules)}
    after = employee_violations(emp, emp_assigns + [candidate], rules)
    return [v for v in after if v["id"] not in before]


def weekly_hours(assignments) -> dict:
    hours = defaultdict(int)
    for a in assignments:
        hours[a["employee_id"]] += SHIFTS[a["shift"]]["hours"]
    return dict(hours)
