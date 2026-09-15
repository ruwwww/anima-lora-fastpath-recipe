"""Queue workers for isolated and persistent trainer execution."""

from __future__ import annotations

import os
import json
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
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


class WarmWorker(ColdWorker):
    """Keep one Python trainer engine alive across compatible jobs.

    The existing trainer CLI remains unchanged: the engine executes that
    script in-process with ``runpy``.  A process is reused only for jobs with
    the same non-empty engine key.  Jobs without a key and jobs marked
    ``cold`` are isolated by restarting the engine after completion.
    """

    def __init__(
        self,
        database: RuntimeDatabase,
        *,
        engine_script: str | os.PathLike[str] | None = None,
        working_directory: str | os.PathLike[str] | None = None,
        empty_cache: bool = False,
        **kwargs,
    ):
        if kwargs.get("command_builder") is not None:
            raise ValueError("WarmWorker requires trainer_script and does not support command_builder")
        super().__init__(database, **kwargs)
        self.engine_script = str(engine_script or (Path(__file__).with_name("engine.py")))
        self.working_directory = str(working_directory) if working_directory is not None else None
        self.empty_cache = empty_cache
        self._engine_process: subprocess.Popen[bytes] | None = None
        self._response_fd: int | None = None
        self._response_buffer = b""
        self._server_engine_key: str | None = None

    def run_once(self) -> JobRecord | None:
        job = self.database.claim_next(self.worker_id, engine_key=self.current_engine_key)
        if job is None:
            return None
        if job.engine_key:
            self.current_engine_key = job.engine_key

        if job.engine_policy == "warm_only" and not job.engine_key:
            return self.database.finish(
                job.job_id,
                success=False,
                error_message="warm_only jobs require a non-empty engine_key",
            )

        cancel_sent = False
        try:
            # A missing key is intentionally not considered compatible: the
            # scheduler cannot prove that two arbitrary CLI jobs have the
            # same graph topology or dtype policy.
            if self._server_engine_key != job.engine_key or not job.engine_key or job.engine_policy == "cold":
                self._stop_engine()
            self._ensure_engine(job)
            self._send_request(
                {
                    "operation": "run",
                    "job_id": job.job_id,
                    "trainer_script": self.trainer_script,
                    "trainer_argv": list(job.trainer_argv),
                }
            )

            response = None
            while response is None:
                response = self._read_response(timeout=self.poll_seconds)
                if response is not None:
                    break
                current = self.database.get(job.job_id)
                if current.state is JobState.CANCEL_REQUESTED and not cancel_sent:
                    self._signal_engine(signal.SIGINT)
                    cancel_sent = True
                self.database.heartbeat(job.job_id, self.worker_id)

            current = self.database.get(job.job_id)
            return_code = int(response.get("return_code", 1))
            cancelled = cancel_sent or current.state is JobState.CANCEL_REQUESTED or bool(response.get("cancelled"))
            reusable = bool(response.get("reusable", False))
            if cancelled:
                result = self.database.finish(
                    job.job_id,
                    success=False,
                    cancelled=True,
                    error_message=response.get("error") or f"cancelled with exit code {return_code}",
                )
            elif return_code == 0:
                result = self.database.finish(job.job_id, success=True)
            else:
                result = self.database.finish(
                    job.job_id,
                    success=False,
                    error_message=response.get("error") or f"trainer exit code {return_code}",
                )

            if not reusable or cancelled or not job.engine_key or job.engine_policy == "cold":
                self._stop_engine()
            return result
        except Exception as error:
            self._stop_engine()
            return self.database.finish(job.job_id, success=False, error_message=str(error))

    def run_forever(self, *, stop_when_empty: bool = False) -> None:
        try:
            super().run_forever(stop_when_empty=stop_when_empty)
        finally:
            self._stop_engine()

    def close(self) -> None:
        """Stop the persistent engine, if one is running."""

        self._stop_engine()

    def _ensure_engine(self, job: JobRecord) -> None:
        if self._engine_process is not None:
            if self._engine_process.poll() is not None:
                self._stop_engine()
            else:
                return

        if not self.trainer_script:
            raise ValueError("trainer_script is required for WarmWorker")
        response_read, response_write = os.pipe()
        environment = os.environ.copy()
        if job.engine_key:
            environment.update(cache_environment_for_key(self.engine_cache_root, job.engine_key))
        environment.update(self.extra_environment)
        environment["ANIMA_RUNTIME_WORKER_ID"] = self.worker_id
        environment["ANIMA_RUNTIME_ENGINE_KEY"] = job.engine_key or ""
        command = [self.python_executable, self.engine_script, "--response-fd", str(response_write)]
        if self.working_directory is not None:
            command.extend(["--working-directory", self.working_directory])
        if self.empty_cache:
            command.append("--empty-cache")
        try:
            self._engine_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                env=environment,
                cwd=self.working_directory,
                pass_fds=(response_write,),
            )
        finally:
            os.close(response_write)
        self._response_fd = response_read
        ready = self._read_response(timeout=max(30.0, self.poll_seconds * 4))
        if ready is None or ready.get("event") != "ready":
            raise RuntimeError(f"warm engine did not become ready: {ready}")
        self._server_engine_key = job.engine_key

    def _send_request(self, request: dict) -> None:
        if self._engine_process is None or self._engine_process.stdin is None:
            raise RuntimeError("warm engine is not running")
        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        self._engine_process.stdin.write(payload)
        self._engine_process.stdin.flush()

    def _read_response(self, *, timeout: float) -> dict | None:
        if self._response_fd is None:
            raise RuntimeError("warm engine response channel is not open")
        readable, _, _ = select.select([self._response_fd], [], [], timeout)
        if not readable:
            return None
        chunk = os.read(self._response_fd, 65536)
        if not chunk:
            raise RuntimeError("warm engine exited before sending a response")
        self._response_buffer += chunk
        if b"\n" not in self._response_buffer:
            return None
        line, self._response_buffer = self._response_buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def _signal_engine(self, signum: signal.Signals) -> None:
        if self._engine_process is not None and self._engine_process.poll() is None:
            self._engine_process.send_signal(signum)

    def _stop_engine(self) -> None:
        process = self._engine_process
        response_fd = self._response_fd
        self._engine_process = None
        self._response_fd = None
        self._response_buffer = b""
        self._server_engine_key = None
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write(b'{"operation":"shutdown"}\n')
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=max(5.0, self.poll_seconds * 4))
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if response_fd is not None:
            os.close(response_fd)
