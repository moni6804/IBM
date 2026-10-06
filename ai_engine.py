"""AI reasoning layer on IBM watsonx.ai (Granite) with an offline fallback.

Design rule: the constraint engine proposes verified options; the LLM explains
them, ranks them, and answers questions. If watsonx is not configured or a call
fails, a deterministic fallback keeps every feature working.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
API_VERSION = "2024-10-08"

SYSTEM = (
    "You are the scheduling assistant for a hospital ward roster. You are given "
    "verified data from a constraint engine. Never invent employees, shifts, or "
    "rules. Be concise, concrete, and use plain language a shift manager would use."
)


class AIEngine:
    def __init__(self):
        self.api_key = os.getenv("WATSONX_API_KEY", "").strip()
        self.project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
        self.model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")
        wanted = os.getenv("AI_PROVIDER", "auto").lower()
        self.provider = "watsonx" if wanted in ("auto", "watsonx") and self.api_key and self.project_id else "offline"
        self._token = None
        self._token_exp = 0.0

    def status(self) -> dict:
        return {"provider": self.provider, "model": self.model_id if self.provider == "watsonx" else None}

    # ---- watsonx transport -------------------------------------------------
    def _iam_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        r = requests.post(
            IAM_URL,
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        r.raise_for_status()
        body = r.json()
        self._token = body["access_token"]
        self._token_exp = time.time() + int(body.get("expires_in", 3600))
        return self._token

    def _chat(self, user: str, max_tokens: int = 500) -> str:
        r = requests.post(
            f"{self.url}/ml/v1/text/chat?version={API_VERSION}",
            headers={"Authorization": f"Bearer {self._iam_token()}", "Content-Type": "application/json"},
            json={
                "model_id": self.model_id,
                "project_id": self.project_id,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _try(self, prompt: str, fallback: str, max_tokens: int = 500) -> dict:
        if self.provider == "watsonx":
            try:
                return {"text": self._chat(prompt, max_tokens), "source": "granite"}
            except Exception as exc:  # network, auth, quota, malformed reply
                return {"text": fallback, "source": "fallback", "note": f"watsonx call failed: {type(exc).__name__}"}
        return {"text": fallback, "source": "fallback"}

    # ---- features ----------------------------------------------------------
    def explain(self, violations: list[dict], hours: dict, employees: list[dict], risk_lines: list[str] | None = None) -> dict:
        errors = [v for v in violations if v["severity"] == "error"]
        warns = [v for v in violations if v["severity"] == "warning"]
        risk_lines = risk_lines or []
        if not violations and not risk_lines:
            return {"text": "No rule problems found. Every shift is covered and nobody breaks a limit.", "source": "fallback"}

        lines = [f"- [{v['severity']}] {v['message']}" for v in violations[:25]]
        risk_block = ("\n\nML call-out forecast (riskiest shifts; staffing is exact, so any call-out "
                      "leaves a gap):\n" + "\n".join(f"- {r}" for r in risk_lines)) if risk_lines else ""
        prompt = (
            "Here are the current schedule problems (errors break labor-law or availability rules; "
            "warnings are coverage gaps or soft limits):\n" + ("\n".join(lines) or "none") + risk_block +
            "\n\nWrite a short briefing: 1) what matters most and why, 2) the two or three actions "
            "that reduce the most risk, including standby cover for the riskiest shifts. "
            "Max 130 words, no headings."
        )
        fallback = (
            (f"{len(errors)} rule violation(s) and {len(warns)} warning(s). " if violations else "No rule problems. ")
            + (f"Fix the errors first: {errors[0]['message']} " if errors else "")
            + (f"Next, coverage: {warns[0]['message']} " if warns else "")
            + (f"Highest call-out risk: {risk_lines[0]}. Consider standby cover. " if risk_lines else "")
            + "Use 'Suggest fix' on any item to see verified replacements."
        )
        return self._try(prompt, fallback)

    def pick(self, violation: dict, candidates: list[dict]) -> dict:
        """Rank verified candidates. Always returns a candidate that exists."""
        if not candidates:
            return {"employee_id": None, "reason": "No one can take this without breaking another rule.", "source": "fallback"}
        best = candidates[0]
        fallback_reason = (
            f"{best['name']} would be at {best['weekly_hours_after']}h of {best['max_hours']}h "
            f"after this shift, the lightest load among eligible {best['role'].lower()}s."
        )
        prompt = (
            f"Problem: {violation['message']}\n"
            "These replacements were verified against every rule:\n"
            + json.dumps(candidates, indent=1)
            + '\n\nPick the best one for fairness and workload. Reply with JSON only: '
              '{"employee_id": "<id>", "reason": "<one sentence>"}'
        )
        result = self._try(prompt, "", 200)
        if result["source"] == "granite":
            match = re.search(r"\{.*?\}", result["text"], re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if any(c["employee_id"] == data.get("employee_id") for c in candidates):
                        return {"employee_id": data["employee_id"], "reason": str(data.get("reason", "")).strip(), "source": "granite"}
                except json.JSONDecodeError:
                    pass
        return {"employee_id": best["employee_id"], "reason": fallback_reason, "source": "fallback"}

    def answer(self, question: str, violations: list[dict], hours: dict, employees: list[dict], week_start: str, risk_lines: list[str] | None = None) -> dict:
        roster = [
            f"{e['name']} ({e['role']}): {hours.get(e['id'], 0)}h of {e.get('max_hours', 40)}h"
            for e in employees
        ]
        issues = [f"[{v['severity']}] {v['message']}" for v in violations[:20]]
        prompt = (
            f"Week starting {week_start}.\nStaff and weekly hours:\n" + "\n".join(roster)
            + "\n\nCurrent problems:\n" + ("\n".join(issues) or "none")
            + ("\n\nML call-out forecast:\n" + "\n".join(risk_lines or []) if risk_lines else "")
            + f"\n\nManager's question: {question}\nAnswer using only this data. Max 100 words."
        )
        top = sorted(employees, key=lambda e: hours.get(e["id"], 0), reverse=True)[:3]
        fallback = (
            f"There are {len(violations)} open problem(s). Highest loads this week: "
            + ", ".join(f"{e['name']} ({hours.get(e['id'], 0)}h)" for e in top)
            + ". Connect watsonx Granite for free-form answers."
        )
        return self._try(prompt, fallback, 300)
