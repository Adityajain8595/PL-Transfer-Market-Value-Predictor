import json

from src.monitoring.drift_detector import run_drift_pipeline
from src.utils.paths import REPORTS_DIR

SUMMARY_PATH = REPORTS_DIR / "drift" / "drift_summary.json"


# Evidently drift monitoring tests
def test_drift_summary_exists():
    if not SUMMARY_PATH.exists():
        run_drift_pipeline()
    assert SUMMARY_PATH.exists(), f"Drift summary missing: {SUMMARY_PATH}"


def test_drift_tests_executed():
    if not SUMMARY_PATH.exists():
        run_drift_pipeline()
    with open(SUMMARY_PATH, "r") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    assert len(tests) > 0, "No drift tests evaluated."
    share_test = next(
        (t for t in tests if "Share of Drifted Columns" in t.get("name", "")),
        None
    )
    assert share_test is not None, "Share drift test missing."
    assert share_test["status"] in ["SUCCESS", "FAIL", "WARNING"]