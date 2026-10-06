from constraints import (DEFAULT_RULES, DEFAULT_REQUIREMENTS, check_schedule, employee_violations,
                         new_violations_if_added)

WS = "2026-10-05"  # a Monday
RULES = dict(DEFAULT_RULES)


def emp(**kw):
    base = {"id": "e1", "name": "Test Nurse", "role": "Nurse", "max_hours": 40, "unavailable": [], "time_off": []}
    base.update(kw)
    return base


def a(day, shift, eid="e1"):
    return {"employee_id": eid, "date": f"2026-10-{5 + day:02d}", "shift": shift}


def codes(vs):
    return {v["code"] for v in vs}


def test_double_booking():
    vs = employee_violations(emp(), [a(0, "morning"), a(0, "evening")], RULES)
    assert "DOUBLE_BOOKING" in codes(vs)


def test_rest_violation_night_then_morning():
    vs = employee_violations(emp(), [a(0, "night"), a(1, "morning")], RULES)
    assert "INSUFFICIENT_REST" in codes(vs)


def test_rest_night_then_evening_is_only_8h():
    vs = employee_violations(emp(), [a(0, "night"), a(1, "evening")], RULES)
    assert "INSUFFICIENT_REST" in codes(vs)


def test_rest_ok_night_then_next_night():
    vs = employee_violations(emp(), [a(0, "night"), a(1, "night")], RULES)
    assert "INSUFFICIENT_REST" not in codes(vs)


def test_unavailable_weekday_and_leave():
    e = emp(unavailable=["tue:night"], time_off=["2026-10-07"])
    assert "UNAVAILABLE" in codes(employee_violations(e, [a(1, "night")], RULES))
    assert "UNAVAILABLE" in codes(employee_violations(e, [a(2, "morning")], RULES))
    assert not employee_violations(e, [a(1, "morning")], RULES)


def test_consecutive_days_limit():
    seven = [a(i, "morning") for i in range(7)]
    assert "CONSECUTIVE_DAYS" in codes(employee_violations(emp(max_hours=99), seven, RULES))


def test_overtime_warning_then_cap_error():
    six = [a(i, "morning") for i in range(6)]       # 48h: warning only
    vs = employee_violations(emp(), six, RULES)
    assert "OVERTIME" in codes(vs) and "WEEKLY_CAP" not in codes(vs)
    seven_ish = six + [a(6, "morning")]             # 56h: legal cap
    assert "WEEKLY_CAP" in codes(employee_violations(emp(), seven_ish, RULES))


def test_understaffed_reported_on_empty_schedule():
    vs = check_schedule([emp()], [], RULES, DEFAULT_REQUIREMENTS, WS)
    assert sum(v["code"] == "UNDERSTAFFED" for v in vs) == 7 * 3 * 2


def test_realtime_check_only_reports_new_problems():
    existing = [a(0, "night")]
    new = new_violations_if_added(emp(), existing, a(1, "morning"), RULES)
    assert codes(new) == {"INSUFFICIENT_REST"}
