import pytest

from audit_env.env import AuditEnv
from audit_env.generator import generate_episode


@pytest.fixture(scope="session")
def error_episode():
    """An episode with several issues, including both interacting scenarios."""
    return generate_episode(9, "interacting")


@pytest.fixture(scope="session")
def clean_episode():
    return generate_episode(3, "clean")


@pytest.fixture
def env():
    return AuditEnv(mode="train")


def oracle_findings(truth, evidence_index=0):
    return [{
        "type": t["type"], "affected_ids": list(t["affected_ids"]), "claimed_amount_minor": t["expected_amount_minor"],
        "currency": t["currency"], "evidence_ids": list(t["evidence_sets"][evidence_index]), "explanation": "x",
    } for t in truth["issues"]]
