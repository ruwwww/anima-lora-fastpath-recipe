import json
import os
import sys

from runtime.engine import WarmPythonEngine, run_server


def test_warm_engine_runs_multiple_jobs_in_one_python_process(tmp_path):
    output_file = tmp_path / "jobs.jsonl"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).open('a', encoding='utf-8').write(json.dumps({'pid': os.getpid(), 'job': os.environ['ANIMA_RUNTIME_JOB_ID']}) + '\\n')\n",
        encoding="utf-8",
    )
    engine = WarmPythonEngine(working_directory=tmp_path)

    first = engine.run(str(trainer), [str(output_file)], job_id="job-a")
    second = engine.run(str(trainer), [str(output_file)], job_id="job-b")

    assert first.return_code == 0
    assert second.return_code == 0
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert [record["job"] for record in records] == ["job-a", "job-b"]
    assert records[0]["pid"] == records[1]["pid"] == os.getpid()


def test_warm_engine_converts_trainer_failure_to_reusable_false(tmp_path):
    trainer = tmp_path / "trainer.py"
    trainer.write_text("raise RuntimeError('expected failure')\n", encoding="utf-8")

    result = WarmPythonEngine().run(str(trainer), [], job_id="job-fail")

    assert result.return_code == 1
    assert result.reusable is False
    assert "expected failure" in (result.error or "")


def test_engine_server_has_dedicated_response_protocol(tmp_path):
    trainer = tmp_path / "trainer.py"
    trainer.write_text("import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ok')\n", encoding="utf-8")
    response_read, response_write = os.pipe()
    request_read, request_write = os.pipe()
    with os.fdopen(request_read, "r", encoding="utf-8") as input_stream:
        os.write(
            request_write,
            (json.dumps({"job_id": "job-a", "trainer_script": str(trainer), "trainer_argv": [str(tmp_path / "out")]}) + "\n").encode(),
        )
        os.write(request_write, b'{"operation":"shutdown"}\n')
        os.close(request_write)
        assert run_server(response_fd=response_write, input_stream=input_stream) == 0
    with os.fdopen(response_read, "r", encoding="utf-8") as response_stream:
        responses = [json.loads(line) for line in response_stream]

    assert responses[0]["event"] == "ready"
    assert responses[1]["event"] == "finished"
    assert responses[1]["return_code"] == 0
