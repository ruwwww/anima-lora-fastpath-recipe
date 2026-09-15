import sys

from runtime.database import RuntimeDatabase
from runtime.worker import ColdWorker


def make_job(job_id, code):
    return {
        "schema_version": 1,
        "job_id": job_id,
        "priority": 0,
        "trainer_argv": [code],
        "engine_policy": "allow_build",
    }


def test_worker_runs_one_job_and_marks_success(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("job-a", "ok"), engine_key="engine-a")

    worker = ColdWorker(
        database,
        worker_id="worker-0",
        command_builder=lambda record: [sys.executable, "-c", "raise SystemExit(0)"],
        engine_cache_root=tmp_path / "engine-cache",
    )
    result = worker.run_once()

    assert result is not None
    assert result.state.value == "succeeded"
    assert result.attempt == 1
    assert database.claim_next("worker-0") is None


def test_worker_marks_nonzero_exit_as_failed(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("job-a", "fail"))

    worker = ColdWorker(
        database,
        command_builder=lambda record: [sys.executable, "-c", "raise SystemExit(3)"],
    )
    result = worker.run_once()

    assert result.state.value == "failed"
    assert "exit code 3" in (result.error_message or "")


def test_worker_passes_engine_cache_environment_to_child(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("job-a", "ok"), engine_key="engine-a")
    output_file = tmp_path / "env.txt"

    worker = ColdWorker(
        database,
        command_builder=lambda record: [
            sys.executable,
            "-c",
            "import os, pathlib; pathlib.Path(os.environ['TEST_RUNTIME_OUTPUT']).write_text(os.environ['TORCHINDUCTOR_CACHE_DIR'])",
        ],
        engine_cache_root=tmp_path / "engine-cache",
        extra_environment={"TEST_RUNTIME_OUTPUT": str(output_file)},
    )
    result = worker.run_once()

    assert result.state.value == "succeeded"
    assert output_file.read_text(encoding="utf-8").endswith("engine-a")
