from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import requests

from .config import DEFAULT_OPENAI_MODEL

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


SYSTEM_PROMPT = (
    "You are a credit review copilot. Only use FACTS provided by the user. "
    "Do not guess missing numeric values, policy terms, or financial product details. "
    "If information is insufficient, explicitly ask for more information. "
    "Output valid JSON only."
)


def _resolve_api_base(base_url: str | None) -> str:
    if not base_url:
        return "https://api.openai.com/v1"
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _chat_completion_via_requests(
    api_key: str,
    base_url: str | None,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    api_base = _resolve_api_base(base_url)
    url = f"{api_base}/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=45,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    payload = resp.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Empty content from chat completion response.")
    return str(content)


class DriverReasoning(BaseModel):
    feature: str
    value: float | int | str
    direction: str
    reasoning: str


class CopilotMemo(BaseModel):
    one_line_summary: str
    risk_grade: str
    risk_score_pd: float = Field(ge=0.0, le=1.0)
    top_risk_drivers: list[DriverReasoning]
    questions_to_ask: list[str]
    documents_to_request: list[str]
    action_suggestions: list[str]
    disclaimer: str


def _model_validate(payload: dict[str, Any]) -> CopilotMemo:
    if hasattr(CopilotMemo, "model_validate"):
        return CopilotMemo.model_validate(payload)  # type: ignore[attr-defined]
    return CopilotMemo.parse_obj(payload)  # type: ignore[attr-defined]


def _model_dump(model: CopilotMemo) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()  # type: ignore[attr-defined]


def build_facts(
    pd_score: float,
    grade: str,
    drivers: list[dict[str, Any]],
    key_inputs: dict[str, Any],
    optional_company_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    facts = {
        "risk_score_pd": float(pd_score),
        "risk_grade": grade,
        "drivers": drivers,
        "key_inputs": key_inputs,
    }
    if optional_company_profile:
        facts["company_profile"] = optional_company_profile
    return facts


def get_fallback_memo(
    pd_score: float,
    grade: str,
    drivers: list[dict[str, Any]],
    error_message: str,
) -> dict[str, Any]:
    top_drivers: list[dict[str, Any]] = []
    for item in drivers[:5]:
        top_drivers.append(
            {
                "feature": str(item.get("feature", "unknown_feature")),
                "value": item.get("value", "n/a"),
                "direction": str(item.get("direction", "unknown")),
                "reasoning": "LLM unavailable; reasoning requires human review.",
            }
        )

    while len(top_drivers) < 3:
        top_drivers.append(
            {
                "feature": "unknown_feature",
                "value": "n/a",
                "direction": "unknown",
                "reasoning": "No driver detail available.",
            }
        )

    memo = CopilotMemo(
        one_line_summary="Copilot generation fallback response.",
        risk_grade=grade,
        risk_score_pd=float(pd_score),
        top_risk_drivers=top_drivers,
        questions_to_ask=[
            "What explains revenue changes over the last 12 months?",
            "What is the near-term liquidity management plan?",
            "How concentrated are major customers and suppliers?",
            "Are there upcoming debt maturities or refinancing plans?",
            "Any expected changes in collateral or guarantees?",
        ],
        documents_to_request=[
            "Last 3 years financial statements",
            "Recent tax filing documents",
            "Major customer and supplier transaction summary",
            "Debt schedule and repayment plan",
            "12-month cash flow projection",
        ],
        action_suggestions=[
            "Define a plan to shorten receivable collection days.",
            "Review fixed-cost items and reduce break-even pressure.",
            "Rebalance debt tenor to reduce short-term liquidity stress.",
        ],
        disclaimer=f"Fallback JSON generated because LLM output failed validation or API unavailable: {error_message}",
    )
    return _model_dump(memo)


def parse_and_validate_json(raw_text: str) -> CopilotMemo:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    payload = json.loads(text)
    return _model_validate(payload)


def generate_copilot_memo(
    facts: dict[str, Any],
    model_name: str | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    load_dotenv()
    pd_score = float(facts.get("risk_score_pd", 0.0))
    grade = str(facts.get("risk_grade", "N/A"))
    drivers = facts.get("drivers", [])

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return get_fallback_memo(pd_score, grade, drivers, "OPENAI_API_KEY is not configured.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client: Any | None = None
    client_init_error = ""
    try:
        client = OpenAI(**client_kwargs)
    except Exception as exc:
        client_init_error = f"{type(exc).__name__}: {exc}"

    schema_hint = {
        "one_line_summary": "str",
        "risk_grade": "str",
        "risk_score_pd": "float",
        "top_risk_drivers": [
            {"feature": "str", "value": "number|string", "direction": "str", "reasoning": "str"}
        ],
        "questions_to_ask": ["str", "..."],
        "documents_to_request": ["str", "..."],
        "action_suggestions": ["str", "..."],
        "disclaimer": "str",
    }

    last_error = "Unknown error"
    for attempt in range(max_retries + 1):
        retry_note = ""
        if attempt > 0:
            retry_note = "You must output valid JSON only. No markdown, no backticks, no prose outside JSON."

        user_payload = {
            "instruction": retry_note,
            "facts": facts,
            "required_schema": schema_hint,
        }

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ]
            content = ""
            sdk_error = ""
            if client is not None:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        temperature=0.1,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or ""
                except Exception as exc:
                    sdk_error = f"{type(exc).__name__}: {exc}"
            else:
                sdk_error = client_init_error or "OpenAI client unavailable"

            if not content:
                content = _chat_completion_via_requests(
                    api_key=api_key,
                    base_url=base_url or None,
                    model=model,
                    messages=messages,
                    temperature=0.1,
                )
            validated = parse_and_validate_json(content)
            return _model_dump(validated)
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            last_error = (
                f"sdk_init={client_init_error or 'ok'}; "
                f"error={type(exc).__name__}: {exc}"
            )
            continue

    return get_fallback_memo(pd_score, grade, drivers, last_error)


def memo_to_markdown(memo: dict[str, Any]) -> str:
    lines = [
        f"### Summary\n{memo.get('one_line_summary', '')}",
        f"- Risk Grade: {memo.get('risk_grade', '')}",
        f"- Risk Score PD: {memo.get('risk_score_pd', '')}",
        "\n### Top Drivers",
    ]

    for item in memo.get("top_risk_drivers", []):
        lines.append(
            f"- {item.get('feature')}: {item.get('direction')} ({item.get('reasoning', '')})"
        )

    lines.append("\n### Questions To Ask")
    for q in memo.get("questions_to_ask", []):
        lines.append(f"- {q}")

    lines.append("\n### Documents To Request")
    for d in memo.get("documents_to_request", []):
        lines.append(f"- {d}")

    lines.append("\n### Action Suggestions")
    for a in memo.get("action_suggestions", []):
        lines.append(f"- {a}")

    lines.append(f"\n### Disclaimer\n{memo.get('disclaimer', '')}")
    return "\n".join(lines)
