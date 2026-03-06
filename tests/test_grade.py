from src.grade import pd_to_grade


def test_pd_to_grade_boundaries() -> None:
    assert pd_to_grade(0.00) == "A"
    assert pd_to_grade(0.0499) == "A"
    assert pd_to_grade(0.05) == "B"
    assert pd_to_grade(0.10) == "C"
    assert pd_to_grade(0.20) == "D"
    assert pd_to_grade(0.35) == "E"
    assert pd_to_grade(0.99) == "E"
