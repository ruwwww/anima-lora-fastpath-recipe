"""Cold worker implementation used before the warm engine session lands."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Mapping

from .compiler_cache import cache_environment_for_key
from .database import JobRecord, RuntimeDatabase
from .schema import JobState


CommandBuilder = Callable[[JobRecord], Sequence[str]]


class ColdWorker:
    """Run one queued trainer process at a time on an assigned GPU."""

    def __init__(
        self,
        database: RuntimeDatabase,
        *,
        worker_id: str | None = None,
        trainer_script: str | None = None,
        python_executable: str | None = None,
        engine_cache_root: str | os.PathLike[str] = "engine-cache",
        poll_seconds: float = 2.0,
        command_builder: CommandBuilder | None = None,
        extra_environment: Mapping[str, str] | None = None,
    ):
        if command_builder is None and not trainer_script:
            raise ValueError("trainer_script is required when command_builder is omitted")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.database = database
        self.worker_id = worker_id or f"worker-{os.getpid()}"
        self.trainer_script = trainer_script
        self.python_executable = python_executable or sys.executable
        self.engine_cache_root = engine_cache_root
        self.poll_seconds = poll_seconds
        self.command_builder = command_builder
        self.extra_environment = dict(extra_environment or {})
        self.current_engine_key: str | None = None

    def run_once(self) -> JobRecord | None:
        job = self.database.claim_next(self.worker_id, engine_key=self.current_engine_key)
        if job is None:
            return None
        if job.engine_key:
            self.current_engine_key = job.engine_key

        process: subprocess.Popen[bytes] | None = None
        cancel_sent = False
        try:
            command = list(self._build_command(job))
            if not command or not all(isinstance(item, str) for item in command):
                raise ValueError("worker command must be a non-empty list of strings")
            environment = os.environ.copy()
            if job.engine_key:
                environment.update(cache_environment_for_key(self.engine_cache_root, job.engine_key))
            environment.update(self.extra_environment)
            environment["ANIMA_RUNTIME_JOB_ID"] = job.job_id
            environment["ANIMA_RUNTIME_WORKER_ID"] = self.worker_id
            process = subprocess.Popen(command, env=environment)

            while True:
                try:
                    return_code = process.wait(timeout=self.poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    current = self.database.get(job.job_id)
                    if current.state is JobState.CANCEL_REQUESTED and not cancel_sent:
                        process.send_signal(signal.SIGINT)
                        cancel_sent = True
                    self.database.heartbeat(job.job_id, self.worker_id)
            current = self.database.get(job.job_id)
            if cancel_sent or current.state is JobState.CANCEL_REQUESTED:
                return self.database.finish(
                    job.job_id,
                    success=False,
                    cancelled=True,
                    error_message=f"cancelled with exit code {return_code}",
                )
            if return_code == 0:
                return self.database.finish(job.job_id, success=True)
            return self.database.finish(job.job_id, success=False, error_message=f"trainer exit code {return_code}")
        except Exception as error:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            return self.database.finish(job.job_id, success=False, error_message=str(error))

    def run_forever(self, *, stop_when_empty: bool = False) -> None:
        while True:
            result = self.run_once()
            if result is None:
                if stop_when_empty:
                    return
                time.sleep(self.poll_seconds)

    def _build_command(self, job: JobRecord) -> Sequence[str]:
        if self.command_builder is not None:
            return self.command_builder(job)
        return [self.python_executable, self.trainer_script, *job.trainer_argv]  # type: ignore[list-item]
