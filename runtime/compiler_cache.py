"""Persistent compiler-cache bundle helpers.

The bundle is an optimization hint, never a source of truth for model weights.
It is invalidated by the engine signature and may still be rejected by PyTorch
when the local runtime is incompatible.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .engine_key import EngineSignature


class CompilerCache:
    def __init__(self, root: str | Path, signature: EngineSignature):
        self.root = Path(root)
        self.signature = signature
        self.directory = self.root / signature.key
        self.manifest_path = self.directory / "manifest.json"
        self.artifact_path = self.directory / "compiler_cache.bin"

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def is_loadable(self) -> bool:
        if not self.manifest_path.is_file() or not self.artifact_path.is_file():
            return False
        try:
            manifest = self.read_manifest()
        except (OSError, ValueError, TypeError):
            return False
        return manifest.get("schema_version") == 1 and manifest.get("engine_key") == self.signature.key and manifest.get(
            "engine_manifest"
        ) == self.signature.manifest

    def save(self, artifact_bytes: bytes, *, cache_info: Mapping[str, Any] | None = None) -> Path:
        if not isinstance(artifact_bytes, (bytes, bytearray, memoryview)) or not artifact_bytes:
            raise ValueError("compiler cache artifact must be non-empty bytes")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write_bytes(self.artifact_path, bytes(artifact_bytes))
        manifest = {
            "schema_version": 1,
            "engine_key": self.signature.key,
            "engine_manifest": self.signature.manifest,
            "cache_info": dict(cache_info or {}),
        }
        self._atomic_write_text(self.manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        return self.artifact_path

    def load(self, loader: Callable[[bytes], Any] | None = None) -> bool:
        if not self.is_loadable():
            return False
        if loader is None:
            try:
                import torch

                loader = getattr(torch.compiler, "load_cache_artifacts", None)
            except ImportError:
                loader = None
        if loader is None:
            return False
        try:
            loader(self.artifact_path.read_bytes())
        except Exception:
            # Cache artifacts are disposable. A runtime mismatch must trigger
            # a normal compile, not prevent the training job from starting.
            return False
        return True

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_write_text(path: Path, data: str) -> None:
        CompilerCache._atomic_write_bytes(path, data.encode("utf-8"))


def cache_environment(root: str | Path, signature: EngineSignature) -> dict[str, str]:
    directory = Path(root) / signature.key
    return {
        "TORCHINDUCTOR_CACHE_DIR": str(directory),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TRITON_CACHE_DIR": str(directory / "triton"),
    }
