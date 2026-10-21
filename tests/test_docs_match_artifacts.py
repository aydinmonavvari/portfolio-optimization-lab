"""CI-enforced doc/artifact consistency (Phase 2, A9).

Runs the full ``scripts/verify_docs.py`` check battery inside the test suite
so that any future edit of README.md / docs/research_report.md that drifts
from the committed artifacts fails CI instead of shipping a stale number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_docs import FORBIDDEN, build_checks, run_checks  # noqa: E402


def test_doc_artifact_consistency_full_battery():
    failures = run_checks()
    assert failures == [], "\n".join(failures)


def test_check_battery_is_substantive():
    """The battery must stay comprehensive (guards against silent pruning)."""
    checks = build_checks()
    assert len(checks) >= 150
    docs = {doc for doc, _, _ in checks}
    assert docs == {"README", "report"}


def test_forbidden_claims_list_is_enforced():
    assert "Sharpe is affine" in FORBIDDEN
    assert "significantly worse than risk parity" in FORBIDDEN
