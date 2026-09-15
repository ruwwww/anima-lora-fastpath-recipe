import sys
import json
import threading
import time

from runtime.database import RuntimeDatabase
from runtime.worker import ColdWorker, WarmWorker


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


def test_worker_retains_engine_affinity_between_jobs(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("warm-first", "ok"), engine_key="engine-a")
    database.submit(make_job("cold", "ok"), engine_key="engine-b")
    database.submit(make_job("warm-second", "ok"), engine_key="engine-a")
    seen = []

    worker = ColdWorker(
        database,
        command_builder=lambda record: (seen.append(record.job_id) or [sys.executable, "-c", "raise SystemExit(0)"]),
        poll_seconds=0.01,
    )
    worker.run_once()
    worker.run_once()

    assert seen == ["warm-first", "warm-second"]


def test_warm_worker_reuses_one_python_engine_for_same_engine_key(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("warm-first", "ok"), engine_key="engine-a")
    database.submit(make_job("warm-second", "ok"), engine_key="engine-a")
    output_file = tmp_path / "engine-pids.jsonl"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import json, os, pathlib\n"
        "with pathlib.Path(os.environ['TEST_RUNTIME_OUTPUT']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps({'pid': os.getpid(), 'job': os.environ['ANIMA_RUNTIME_JOB_ID']}) + '\\n')\n",
        encoding="utf-8",
    )

    worker = WarmWorker(
        database,
        worker_id="warm-worker",
        trainer_script=str(trainer),
        working_directory=tmp_path,
        engine_cache_root=tmp_path / "engine-cache",
        extra_environment={"TEST_RUNTIME_OUTPUT": str(output_file)},
        poll_seconds=0.01,
    )
    try:
        first = worker.run_once()
        second = worker.run_once()
    finally:
        worker._stop_engine()

    assert first is not None and first.state.value == "succeeded"
    assert second is not None and second.state.value == "succeeded"
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert [record["job"] for record in records] == ["warm-first", "warm-second"]
    assert records[0]["pid"] == records[1]["pid"]


def test_warm_only_without_engine_key_fails_before_starting_engine(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit({**make_job("warm-only", "ok"), "engine_policy": "warm_only"})

    worker = WarmWorker(database, trainer_script=str(tmp_path / "missing.py"))
    result = worker.run_once()

    assert result is not None
    assert result.state.value == "failed"
    assert "engine_key" in (result.error_message or "")


def test_warm_worker_cancels_at_protocol_boundary_and_discards_engine(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    database.submit(make_job("cancel-me", "loop"), engine_key="engine-a")
    trainer = tmp_path / "trainer.py"
    trainer.write_text("import time\nwhile True: time.sleep(0.01)\n", encoding="utf-8")
    worker = WarmWorker(
        database,
        worker_id="warm-worker",
        trainer_script=str(trainer),
        working_directory=tmp_path,
        poll_seconds=0.01,
    )
    result_holder = []
    thread = threading.Thread(target=lambda: result_holder.append(worker.run_once()))
    thread.start()
    try:
        for _ in range(100):
            if database.get("cancel-me").state.value == "running":
                break
            time.sleep(0.01)
        assert database.request_cancel("cancel-me") is True
        thread.join(timeout=10)
    finally:
        worker.close()

    assert not thread.is_alive()
    assert result_holder[0] is not None
    assert result_holder[0].state.value == "cancelled"
    assert worker._engine_process is None
