import pytest

from runtime.database import RuntimeDatabase
from runtime.schema import JobState


def make_job(job_id, *, priority=0, engine_key=None):
    return {
        "schema_version": 1,
        "job_id": job_id,
        "priority": priority,
        "trainer_argv": ["--network_dim=16"],
        "engine_policy": "allow_build",
        "metadata": {"engine_key": engine_key} if engine_key else {},
    }


def test_submit_and_claim_prefers_warm_engine_then_priority(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("cold", priority=10, engine_key="cold"), engine_key="cold")
    database.submit(make_job("warm", priority=10, engine_key="warm"), engine_key="warm")

    claimed = database.claim_next("worker-0", engine_key="warm")

    assert claimed.job_id == "warm"
    assert claimed.state is JobState.RUNNING
    assert claimed.worker_id == "worker-0"
    assert claimed.attempt == 1


def test_claim_falls_back_to_priority_when_no_warm_match(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("low", priority=1, engine_key="a"), engine_key="a")
    database.submit(make_job("high", priority=10, engine_key="b"), engine_key="b")

    claimed = database.claim_next("worker-0", engine_key="missing")

    assert claimed.job_id == "high"


def test_state_transitions_and_cancel_are_explicit(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("job-a"))

    running = database.claim_next("worker-0")
    assert database.request_cancel("job-a") is True
    assert database.get("job-a").state is JobState.CANCEL_REQUESTED

    database.finish("job-a", success=False, error_message="cancelled at step boundary")
    assert database.get("job-a").state is JobState.FAILED

    with pytest.raises(ValueError, match="invalid state transition"):
        database.set_state("job-a", JobState.RUNNING)


def test_cancel_queued_job_does_not_claim_it(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("job-a"))

    assert database.request_cancel("job-a") is True
    assert database.get("job-a").state is JobState.CANCELLED
    assert database.claim_next("worker-0") is None


def test_duplicate_job_id_is_rejected(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("job-a"))

    with pytest.raises(ValueError, match="already exists"):
        database.submit(make_job("job-a"))
