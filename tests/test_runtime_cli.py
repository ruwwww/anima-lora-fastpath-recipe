import json

from runtime.cli import main


def test_cli_submit_and_status(tmp_path, capsys):
    database = tmp_path / "runtime.sqlite3"
    job_file = tmp_path / "job.json"
    job_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "job-a",
                "trainer_argv": ["--network_dim=16"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--database", str(database), "submit", str(job_file)]) == 0
    assert "job-a" in capsys.readouterr().out

    assert main(["--database", str(database), "status", "job-a"]) == 0
    output = capsys.readouterr().out
    assert "queued" in output
    assert "job-a" in output


def test_cli_rejects_missing_input_path(tmp_path, capsys):
    job_file = tmp_path / "job.json"
    job_file.write_text(
        json.dumps(
            {
                "job_id": "job-a",
                "trainer_argv": ["--pretrained_model_name_or_path=/missing/model.safetensors"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--database", str(tmp_path / "runtime.sqlite3"), "submit", str(job_file), "--preflight"]) == 2
    assert "preflight" in capsys.readouterr().err


def test_cli_derives_engine_key_from_engine_config(tmp_path, capsys):
    database = tmp_path / "runtime.sqlite3"
    job_file = tmp_path / "job.json"
    job_file.write_text(
        json.dumps(
            {
                "job_id": "job-a",
                "trainer_argv": ["--network_dim=16"],
                "metadata": {"engine_config": {"resolution": [832, 1216], "checkpoint_blocks": 1}},
            }
        ),
        encoding="utf-8",
    )

    assert main(["--database", str(database), "submit", str(job_file)]) == 0
    capsys.readouterr()
    assert main(["--database", str(database), "status", "job-a", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert status["engine_key"]
    assert len(status["engine_key"]) == 32


def test_cli_runs_one_job_through_warm_worker(tmp_path, capsys):
    database = tmp_path / "runtime.sqlite3"
    job_file = tmp_path / "job.json"
    trainer = tmp_path / "trainer.py"
    output = tmp_path / "output.txt"
    trainer.write_text(
        f"import os, pathlib\npathlib.Path({str(output)!r}).write_text(os.environ['ANIMA_RUNTIME_JOB_ID'])\n",
        encoding="utf-8",
    )
    job_file.write_text(
        json.dumps(
            {
                "job_id": "job-a",
                "trainer_argv": ["--noop"],
                "metadata": {"engine_config": {"model": "anima", "resolution": [832, 1216]}},
            }
        ),
        encoding="utf-8",
    )

    assert main(["--database", str(database), "submit", str(job_file)]) == 0
    capsys.readouterr()
    assert main(
        [
            "--database",
            str(database),
            "run",
            "--trainer",
            str(trainer),
            "--workdir",
            str(tmp_path),
            "--warm",
            "--once",
        ]
    ) == 0
    capsys.readouterr()

    assert output.read_text(encoding="utf-8") == "job-a"
    assert main(["--database", str(database), "status", "job-a"]) == 0
    assert "succeeded" in capsys.readouterr().out
