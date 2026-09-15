from runtime.compiler_cache import CompilerCache, cache_environment
from runtime.engine_key import build_engine_signature


def signature():
    return build_engine_signature(
        {
            "model_sha256": "model",
            "torch_version": "2.13.0+cu130",
            "gpu": "RTX 5060 Ti",
            "network_dim": 16,
            "checkpoint_indices": [0],
        }
    )


def test_compiler_cache_round_trip_loads_only_matching_signature(tmp_path):
    current = signature()
    cache = CompilerCache(tmp_path, current)
    cache.save(b"compiled-cache", cache_info={"graphs": 28})

    loaded = []
    assert cache.is_loadable()
    assert cache.load(loader=lambda payload: loaded.append(payload)) is True
    assert loaded == [b"compiled-cache"]
    assert cache.read_manifest()["engine_key"] == current.key
    assert cache.read_manifest()["cache_info"] == {"graphs": 28}


def test_compiler_cache_does_not_load_when_artifact_is_missing(tmp_path):
    cache = CompilerCache(tmp_path, signature())

    assert cache.is_loadable() is False
    assert cache.load(loader=lambda _: (_ for _ in ()).throw(AssertionError("must not load"))) is False


def test_cache_environment_is_deterministic_and_scoped_to_engine(tmp_path):
    current = signature()
    environment = cache_environment(tmp_path, current)

    assert environment["TORCHINDUCTOR_CACHE_DIR"].endswith(current.key)
    assert environment["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert environment["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"
    assert environment["TRITON_CACHE_DIR"].endswith(f"{current.key}/triton")
