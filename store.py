"""Tiny JSON-file store with a lock. Swap for a real DB later."""
from __future__ import annotations

import copy
import json
import threading
from datetime import date, timedelta
from pathlib import Path

from constraints import DEFAULT_REQUIREMENTS, DEFAULT_RULES


def next_monday(today: date | None = None) -> str:
    today = today or date.today()
    return (today + timedelta(days=(7 - today.weekday()) % 7)).isoformat()


def seed_state(week_start: str | None = None) -> dict:
    ws = week_start or next_monday()
    start = date.fromisoformat(ws)
    d = lambda n: (start + timedelta(days=n)).isoformat()
    emp = lambda i, name, role, max_hours=40, unavailable=(), time_off=(): {
        "id": f"e{i}", "name": name, "role": role, "max_hours": max_hours,
        "unavailable": list(unavailable), "time_off": list(time_off),
    }
    employees = [
        emp(1, "Priya Sharma", "Nurse", unavailable=["sun:all"]),
        emp(2, "Rohan Verma", "Nurse", unavailable=["wed:night"]),
        emp(3, "Ananya Iyer", "Nurse", time_off=[d(2), d(3)]),
        emp(4, "Kabir Singh", "Nurse", unavailable=["sat:all", "sun:all"]),
        emp(5, "Meera Nair", "Nurse"),
        emp(6, "Arjun Das", "Nurse", unavailable=["mon:morning"]),
        emp(7, "Sana Khan", "Nurse", unavailable=["fri:night"]),
        emp(8, "Neha Joshi", "Nurse", max_hours=24, unavailable=["tue:all", "thu:all"]),
        emp(9, "Vikram Rao", "Nurse", max_hours=24),
        emp(10, "Isha Patel", "Technician"),
        emp(11, "Dev Malhotra", "Technician"),
        emp(12, "Tara Menon", "Technician", unavailable=["sun:all"]),
        emp(13, "Yash Kulkarni", "Technician"),
        emp(14, "Zoya Ali", "Technician", time_off=[d(5)]),
        emp(15, "Rahul Bose", "Technician", max_hours=24, unavailable=["mon:all", "wed:all"]),
    ]
    return {
        "week_start": ws,
        "employees": employees,
        "assignments": [],
        "rules": dict(DEFAULT_RULES),
        "requirements": copy.deepcopy(DEFAULT_REQUIREMENTS),
    }


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return seed_state()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._state, indent=2))
        except OSError:
            pass  # persistence is best-effort in demo mode

    def read(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def update(self, fn):
        with self._lock:
            result = fn(self._state)
            self._save()
            return result

    def reset(self) -> None:
        with self._lock:
            self._state = seed_state()
            self._save()
