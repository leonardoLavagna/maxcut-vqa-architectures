from qaoa_structure.runner import _models


def _config():
    return {
        "ansatze": {
            "principal": ["ry", "agnostic_xx"],
            "controls": ["rx", "ryrz"],
            "qaoa_depths": [1, 3, 5],
            "qaoa_parameter_matched": True,
        }
    }


def test_parameter_matched_qaoa_adds_only_missing_depths():
    cfg = _config()
    models_n4 = _models(cfg, 4)
    models_n6 = _models(cfg, 6)
    models_n8 = _models(cfg, 8)
    models_n10 = _models(cfg, 10)
    models_n12 = _models(cfg, 12)

    assert ("qaoa", 2) in models_n4
    assert models_n6.count(("qaoa", 3)) == 1
    assert ("qaoa", 4) in models_n8
    assert models_n10.count(("qaoa", 5)) == 1
    assert ("qaoa", 6) in models_n12


def test_parameter_matched_qaoa_requires_even_n():
    cfg = _config()
    try:
        _models(cfg, 5)
    except ValueError as exc:
        assert "even node count" in str(exc)
    else:
        raise AssertionError("Expected ValueError for odd n")
