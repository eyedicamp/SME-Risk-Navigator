import json

from src.copilot import CopilotMemo, get_fallback_memo, parse_and_validate_json


def test_fallback_json_matches_schema() -> None:
    fallback = get_fallback_memo(
        pd_score=0.23,
        grade="C",
        drivers=[{"feature": "debt_ratio", "value": 0.7, "direction": "risk_up"}],
        error_message="test",
    )

    model = CopilotMemo.model_validate(fallback) if hasattr(CopilotMemo, "model_validate") else CopilotMemo.parse_obj(fallback)
    assert model.risk_grade == "C"
    assert len(model.questions_to_ask) >= 5


def test_parse_and_validate_json() -> None:
    payload = {
        "one_line_summary": "summary",
        "risk_grade": "B",
        "risk_score_pd": 0.12,
        "top_risk_drivers": [
            {
                "feature": "debt_ratio",
                "value": 0.6,
                "direction": "risk_up",
                "reasoning": "High debt ratio may weaken repayment capacity.",
            }
        ],
        "questions_to_ask": ["q1", "q2", "q3", "q4", "q5"],
        "documents_to_request": ["d1", "d2", "d3", "d4", "d5"],
        "action_suggestions": ["a1", "a2", "a3"],
        "disclaimer": "demo",
    }
    parsed = parse_and_validate_json(json.dumps(payload, ensure_ascii=True))
    assert parsed.risk_grade == "B"
