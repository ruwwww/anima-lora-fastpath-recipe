"""Persistent Python trainer engine used by the warm queue worker.

The engine deliberately keeps the trainer's existing command-line entry point
as the source of truth.  A job is executed with ``runpy`` in this process, so
imports, TorchInductor's in-process state, and the process-level CUDA context
survive between jobs.  Model/optimizer reuse is a later contract; this first
stage removes interpreter/import startup and gives compiler caches one stable
owner without changing the trainer lifecycle.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import runpy
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class EngineResult:
    return_code: int
    cancelled: bool = False
    reusable: bool = True
    error: str | None = None


class WarmPythonEngine:
    """Execute trainer scripts in one long-lived Python interpreter."""

    def __init__(self, *, working_directory: str | os.PathLike[str] | None = None, empty_cache: bool = False):
        self.working_directory = Path(working_directory).resolve() if working_directory else None
        self.empty_cache = empty_cache

    def run(self, trainer_script: str | os.PathLike[str], trainer_argv: list[str], *, job_id: str) -> EngineResult:
        script = Path(trainer_script).resolve()
        if not script.is_file():
            return EngineResult(2, error=f"trainer script does not exist: {script}", reusable=False)
        if not isinstance(trainer_argv, list) or not all(isinstance(arg, str) for arg in trainer_argv):
            return EngineResult(2, error="trainer_argv must be a list of strings", reusable=False)

        old_argv = sys.argv
        old_cwd = Path.cwd()
        old_job_id = os.environ.get("ANIMA_RUNTIME_JOB_ID")
        inserted_path = False
        try:
            script_parent = str(script.parent)
            if script_parent not in sys.path:
                sys.path.insert(0, script_parent)
                inserted_path = True
            if self.working_directory is not None:
                os.chdir(self.working_directory)
            sys.argv = [str(script), *trainer_argv]
            os.environ["ANIMA_RUNTIME_JOB_ID"] = job_id
            runpy.run_path(str(script), run_name="__main__")
            return EngineResult(0)
        except SystemExit as error:
            code = error.code
            if code is None:
                return EngineResult(0)
            if isinstance(code, bool):
                return EngineResult(int(code))
            if isinstance(code, int):
                return EngineResult(code, reusable=code == 0)
            return EngineResult(1, error=str(code), reusable=False)
        except KeyboardInterrupt:
            return EngineResult(130, cancelled=True, reusable=False, error="trainer interrupted")
        except BaseException as error:  # trainer failures must not kill the queue protocol
            traceback.print_exc()
            return EngineResult(1, reusable=False, error=f"{type(error).__name__}: {error}")
        finally:
            sys.argv = old_argv
            if inserted_path:
                try:
                    sys.path.remove(str(script.parent))
                except ValueError:
                    pass
            if old_cwd != Path.cwd():
                os.chdir(old_cwd)
            if old_job_id is None:
                os.environ.pop("ANIMA_RUNTIME_JOB_ID", None)
            else:
                os.environ["ANIMA_RUNTIME_JOB_ID"] = old_job_id
            # runpy's globals are released on return.  This makes a warm
            # process reusable while allowing the CUDA allocator to reclaim
            # blocks between jobs when the operator explicitly requests it.
            gc.collect()
            if self.empty_cache:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass


def _write_response(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def run_server(
    *,
    response_fd: int,
    working_directory: str | os.PathLike[str] | None = None,
    empty_cache: bool = False,
    input_stream: TextIO | None = None,
) -> int:
    """Serve newline-delimited job requests; responses use a dedicated fd.

    Trainer stdout/stderr intentionally remain untouched.  The dedicated
    response descriptor prevents normal training logs from corrupting the
    scheduler protocol.
    """

    input_stream = input_stream or sys.stdin
    response_stream = os.fdopen(response_fd, "w", buffering=1, encoding="utf-8", closefd=True)
    engine = WarmPythonEngine(working_directory=working_directory, empty_cache=empty_cache)
    _write_response(response_stream, {"event": "ready", "pid": os.getpid()})
    try:
        for line in input_stream:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                operation = request.get("operation", "run")
                if operation == "shutdown":
                    return 0
                if operation != "run":
                    raise ValueError(f"unknown operation: {operation}")
                job_id = request.get("job_id")
                trainer_script = request.get("trainer_script")
                trainer_argv = request.get("trainer_argv")
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError("job_id must be a non-empty string")
                if not isinstance(trainer_script, str) or not trainer_script:
                    raise ValueError("trainer_script must be a non-empty string")
                if not isinstance(trainer_argv, list) or not all(isinstance(arg, str) for arg in trainer_argv):
                    raise ValueError("trainer_argv must be a list of strings")
                result = engine.run(trainer_script, trainer_argv, job_id=job_id)
                _write_response(
                    response_stream,
                    {
                        "event": "finished",
                        "job_id": job_id,
                        "return_code": result.return_code,
                        "cancelled": result.cancelled,
                        "reusable": result.reusable,
                        "error": result.error,
                    },
                )
            except KeyboardInterrupt:
                # A signal during protocol handling is not safe for reuse.
                _write_response(response_stream, {"event": "finished", "return_code": 130, "cancelled": True, "reusable": False})
            except Exception as error:
                traceback.print_exc()
                _write_response(
                    response_stream,
                    {"event": "error", "return_code": 1, "reusable": False, "error": str(error)},
                )
    finally:
        response_stream.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anima-warm-engine")
    parser.add_argument("--response-fd", type=int, required=True)
    parser.add_argument("--working-directory", default=None)
    parser.add_argument("--empty-cache", action="store_true")
    args = parser.parse_args(argv)
    return run_server(
        response_fd=args.response_fd,
        working_directory=args.working_directory,
        empty_cache=args.empty_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())
