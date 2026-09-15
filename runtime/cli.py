"""Command-line interface for the local Anima training queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .database import RuntimeDatabase
from .engine_key import build_engine_signature
from .preflight import validate_input_paths
from .worker import ColdWorker, WarmWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anima-runtime")
    parser.add_argument("--database", default=".runtime/runtime.sqlite3", help="SQLite queue path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="submit a JSON job")
    submit.add_argument("job_file", type=Path)
    submit.add_argument("--engine-key", default=None)
    submit.add_argument("--preflight", action="store_true", help="validate known input paths before enqueueing")

    list_parser = subparsers.add_parser("list", help="list jobs")
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    status = subparsers.add_parser("status", help="show one job")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true", dest="as_json")

    cancel = subparsers.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("job_id")

    run = subparsers.add_parser("run", help="run a worker")
    run.add_argument("--trainer", required=True, help="trainer script, e.g. sd-scripts/anima_train_network.py")
    run.add_argument("--python", dest="python_executable", default=sys.executable)
    run.add_argument("--worker-id", default=None)
    run.add_argument("--engine-cache-root", default="engine-cache")
    run.add_argument("--poll-seconds", type=float, default=2.0)
    run.add_argument("--warm", action="store_true", help="keep one Python trainer engine for matching engine keys")
    run.add_argument("--engine-script", default=None, help="override the persistent engine script used by --warm")
    run.add_argument("--workdir", default=None, help="trainer working directory used by --warm")
    run.add_argument("--empty-cache", action="store_true", help="empty the CUDA allocator between warm jobs")
    run.add_argument("--once", action="store_true", help="claim at most one job and exit when idle")
    return parser


def _record_dict(record):
    return {
        "job_id": record.job_id,
        "priority": record.priority,
        "state": record.state.value,
        "engine_key": record.engine_key,
        "attempt": record.attempt,
        "worker_id": record.worker_id,
        "submitted_at": record.submitted_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "heartbeat_at": record.heartbeat_at,
        "error_message": record.error_message,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database = RuntimeDatabase(args.database)
    worker = None
    try:
        if args.command == "submit":
            try:
                raw_spec = json.loads(args.job_file.read_text(encoding="utf-8"))
                if args.preflight:
                    errors = validate_input_paths(raw_spec.get("trainer_argv", []))
                    if errors:
                        print("preflight failed:", file=sys.stderr)
                        for error in errors:
                            print(f"- {error}", file=sys.stderr)
                        return 2
                engine_key = args.engine_key
                if engine_key is None:
                    metadata = raw_spec.get("metadata", {})
                    engine_config = metadata.get("engine_config") if isinstance(metadata, dict) else None
                    if engine_config is not None:
                        engine_key = build_engine_signature(engine_config).key
                record = database.submit(raw_spec, engine_key=engine_key)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                print(f"submit failed: {error}", file=sys.stderr)
                return 2
            print(record.job_id)
            return 0

        if args.command == "list":
            records = [_record_dict(record) for record in database.list_jobs()]
            if args.as_json:
                print(json.dumps(records, indent=2, sort_keys=True))
            else:
                for record in records:
                    print(f"{record['job_id']}\t{record['state']}\tpriority={record['priority']}\tengine={record['engine_key'] or '-'}")
            return 0

        if args.command == "status":
            try:
                record = _record_dict(database.get(args.job_id))
            except KeyError as error:
                print(str(error), file=sys.stderr)
                return 2
            if args.as_json:
                print(json.dumps(record, indent=2, sort_keys=True))
            else:
                print(f"{record['job_id']}\t{record['state']}\tattempt={record['attempt']}\tengine={record['engine_key'] or '-'}")
                if record["error_message"]:
                    print(f"error: {record['error_message']}")
            return 0

        if args.command == "cancel":
            try:
                changed = database.request_cancel(args.job_id)
            except KeyError as error:
                print(str(error), file=sys.stderr)
                return 2
            print("cancel requested" if changed else "job already terminal")
            return 0

        if args.command == "run":
            worker_type = WarmWorker if args.warm else ColdWorker
            worker_kwargs = {
                "worker_id": args.worker_id,
                "trainer_script": args.trainer,
                "python_executable": args.python_executable,
                "engine_cache_root": args.engine_cache_root,
                "poll_seconds": args.poll_seconds,
            }
            if args.warm:
                worker_kwargs.update(
                    {
                        "engine_script": args.engine_script,
                        "working_directory": args.workdir,
                        "empty_cache": args.empty_cache,
                    }
                )
            worker = worker_type(database, **worker_kwargs)
            if args.once:
                worker.run_once()
            else:
                worker.run_forever()
            return 0

        raise RuntimeError(f"unknown command: {args.command}")
    finally:
        if worker is not None and hasattr(worker, "close"):
            worker.close()
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
