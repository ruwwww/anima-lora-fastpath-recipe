"""SQLite-backed queue state for local, single-host runtime scheduling."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import JobSpec, JobState, validate_job_spec


_CLAIMABLE_STATES = (JobState.QUEUED.value, JobState.READY.value)
_ALLOWED_TRANSITIONS = {
    JobState.QUEUED: {JobState.PREPARING, JobState.CANCELLED},
    JobState.PREPARING: {JobState.READY, JobState.QUEUED, JobState.FAILED, JobState.CANCELLED},
    JobState.READY: {JobState.RUNNING, JobState.QUEUED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.CANCEL_REQUESTED: {JobState.CANCELLED, JobState.FAILED},
    JobState.SUCCEEDED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    schema_version: int
    priority: int
    trainer_argv: tuple[str, ...]
    engine_policy: str
    metadata: dict[str, Any]
    engine_key: str | None
    state: JobState
    attempt: int
    worker_id: str | None
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    heartbeat_at: float | None
    error_message: str | None


class RuntimeDatabase:
    """Small SQLite repository with atomic job claims and explicit transitions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    trainer_argv_json TEXT NOT NULL,
                    engine_policy TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    engine_key TEXT,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    submitted_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    heartbeat_at REAL,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_claim_idx
                    ON jobs(state, priority DESC, submitted_at ASC);
                CREATE INDEX IF NOT EXISTS jobs_engine_idx
                    ON jobs(state, engine_key, priority DESC, submitted_at ASC);
                """
            )

    def submit(self, raw_spec: Mapping[str, Any] | JobSpec, *, engine_key: str | None = None) -> JobRecord:
        spec = raw_spec if isinstance(raw_spec, JobSpec) else validate_job_spec(raw_spec)
        timestamp = time.time()
        metadata = dict(spec.metadata)
        if engine_key is None:
            engine_key = metadata.get("engine_key")
        with self._lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, schema_version, priority, trainer_argv_json,
                        engine_policy, metadata_json, engine_key, state,
                        submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.job_id,
                        spec.schema_version,
                        spec.priority,
                        json.dumps(list(spec.trainer_argv), separators=(",", ":")),
                        spec.engine_policy,
                        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                        engine_key,
                        JobState.QUEUED.value,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"job_id already exists: {spec.job_id}") from error
        return self.get(spec.job_id)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job_id: {job_id}")
        return self._row_to_record(row)

    def list_jobs(self, states: Iterable[JobState] | None = None) -> list[JobRecord]:
        with self._lock:
            if states is None:
                rows = self._connection.execute("SELECT * FROM jobs ORDER BY submitted_at ASC").fetchall()
            else:
                values = tuple(state.value for state in states)
                if not values:
                    return []
                placeholders = ",".join("?" for _ in values)
                rows = self._connection.execute(
                    f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY submitted_at ASC", values
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def claim_next(self, worker_id: str, *, engine_key: str | None = None) -> JobRecord | None:
        """Claim one job atomically; priority wins, warm affinity breaks ties."""

        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE state IN (?, ?)
                    ORDER BY priority DESC,
                             CASE WHEN ? IS NOT NULL AND engine_key = ? THEN 0 ELSE 1 END,
                             submitted_at ASC
                    LIMIT 1
                    """,
                    (*_CLAIMABLE_STATES, engine_key, engine_key),
                ).fetchone()
                if row is None:
                    self._connection.execute("COMMIT")
                    return None

                self._connection.execute(
                    """
                    UPDATE jobs
                    SET state = ?, worker_id = ?, attempt = attempt + 1,
                        started_at = COALESCE(started_at, ?), heartbeat_at = ?, error_message = NULL
                    WHERE job_id = ? AND state IN (?, ?)
                    """,
                    (JobState.RUNNING.value, worker_id, now, now, row["job_id"], *_CLAIMABLE_STATES),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return self.get(row["job_id"])

    def set_engine_key(self, job_id: str, engine_key: str) -> JobRecord:
        with self._lock:
            self._connection.execute("UPDATE jobs SET engine_key = ? WHERE job_id = ?", (engine_key, job_id))
        return self.get(job_id)

    def set_state(self, job_id: str, state: JobState, *, error_message: str | None = None) -> JobRecord:
        if not isinstance(state, JobState):
            raise TypeError("state must be a JobState")
        with self._lock:
            current = self.get(job_id)
            if state is not current.state and state not in _ALLOWED_TRANSITIONS[current.state]:
                raise ValueError(f"invalid state transition: {current.state.value} -> {state.value}")
            finished_at = time.time() if state.terminal else current.finished_at
            self._connection.execute(
                "UPDATE jobs SET state = ?, finished_at = ?, error_message = ? WHERE job_id = ?",
                (state.value, finished_at, error_message, job_id),
            )
        return self.get(job_id)

    def mark_preparing(self, job_id: str) -> JobRecord:
        return self.set_state(job_id, JobState.PREPARING)

    def mark_ready(self, job_id: str, *, engine_key: str | None = None) -> JobRecord:
        with self._lock:
            if engine_key is not None:
                self._connection.execute("UPDATE jobs SET engine_key = ? WHERE job_id = ?", (engine_key, job_id))
        return self.set_state(job_id, JobState.READY)

    def heartbeat(self, job_id: str, worker_id: str) -> JobRecord:
        with self._lock:
            current = self.get(job_id)
            if current.state not in {JobState.RUNNING, JobState.CANCEL_REQUESTED}:
                raise ValueError(f"cannot heartbeat job in state {current.state.value}")
            self._connection.execute(
                "UPDATE jobs SET heartbeat_at = ?, worker_id = ? WHERE job_id = ?",
                (time.time(), worker_id, job_id),
            )
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            current = self.get(job_id)
            if current.state in {JobState.QUEUED, JobState.PREPARING, JobState.READY}:
                self.set_state(job_id, JobState.CANCELLED)
                return True
            if current.state is JobState.RUNNING:
                self.set_state(job_id, JobState.CANCEL_REQUESTED)
                return True
            return False

    def finish(
        self,
        job_id: str,
        *,
        success: bool,
        error_message: str | None = None,
        cancelled: bool = False,
    ) -> JobRecord:
        if cancelled:
            return self.set_state(job_id, JobState.CANCELLED, error_message=error_message)
        return self.set_state(job_id, JobState.SUCCEEDED if success else JobState.FAILED, error_message=error_message)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            schema_version=row["schema_version"],
            priority=row["priority"],
            trainer_argv=tuple(json.loads(row["trainer_argv_json"])),
            engine_policy=row["engine_policy"],
            metadata=dict(json.loads(row["metadata_json"])),
            engine_key=row["engine_key"],
            state=JobState(row["state"]),
            attempt=row["attempt"],
            worker_id=row["worker_id"],
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            heartbeat_at=row["heartbeat_at"],
            error_message=row["error_message"],
        )
