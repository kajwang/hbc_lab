from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_idea_plan_records_original_research_direction():
    source = (REPO_ROOT / "IDEA_PLAN.md").read_text()

    assert "SUGAR" in source
    assert "GRAIL" in source
    assert "Contact labels from VLM parsing" in source
    assert "Sparse object pose guidance" in source
    assert "scale RL" in source
    assert "High-level low-frequency command policy" in source
    assert "Low-level whole-body controller policy" in source
    assert "Height map" in source
    assert "Object shape encoding" in source
    assert "Residual force compensation" in source
    assert "Same box, different context" in source
    assert "Articulated object interaction" in source
    assert "Heterogeneous embodiment" in source
