from runtime.preflight import extract_path_arguments, validate_input_paths


def test_extract_path_arguments_supports_equals_and_separate_values():
    argv = [
        "--pretrained_model_name_or_path=/models/anima.safetensors",
        "--train_data_dir",
        "/data/anima",
        "--network_dim=16",
    ]

    paths = extract_path_arguments(argv)

    assert paths["pretrained_model_name_or_path"] == "/models/anima.safetensors"
    assert paths["train_data_dir"] == "/data/anima"


def test_validate_input_paths_reports_missing_files_and_directories(tmp_path):
    errors = validate_input_paths(
        [
            "--pretrained_model_name_or_path=/missing/model.safetensors",
            "--train_data_dir",
            str(tmp_path / "missing-data"),
        ]
    )

    assert len(errors) == 2
    assert "pretrained_model_name_or_path" in errors[0]
    assert "train_data_dir" in errors[1]
