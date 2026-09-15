"""Fast, CPU-only validation before a job is handed to a GPU worker."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


_PATH_FLAGS = {
    "base_weights",
    "dataset_config",
    "network_weights",
    "pretrained_model_name_or_path",
    "qwen3",
    "sample_prompts",
    "text_encoder",
    "train_data_dir",
    "vae",
}
_DIRECTORY_FLAGS = {"base_weights", "train_data_dir"}


def extract_path_arguments(argv: Sequence[str]) -> dict[str, str]:
    """Extract known path flags without importing the heavyweight trainer."""

    result: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not isinstance(token, str) or not token.startswith("--"):
            index += 1
            continue
        name, separator, value = token[2:].partition("=")
        if name not in _PATH_FLAGS:
            index += 1
            continue
        if not separator:
            if index + 1 >= len(argv) or not isinstance(argv[index + 1], str) or argv[index + 1].startswith("--"):
                index += 1
                continue
            value = argv[index + 1]
            index += 1
        if value:
            result[name] = value
        index += 1
    return result


def validate_input_paths(argv: Sequence[str]) -> list[str]:
    errors = []
    for name, raw_path in extract_path_arguments(argv).items():
        path = Path(raw_path).expanduser()
        if name in _DIRECTORY_FLAGS:
            valid = path.is_dir()
            expected = "directory"
        else:
            valid = path.is_file()
            expected = "file"
        if not valid:
            errors.append(f"--{name} must point to an existing {expected}: {raw_path}")
    return errors
