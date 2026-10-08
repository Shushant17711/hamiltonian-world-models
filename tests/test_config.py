import pytest

from hwm.config import Config, config_hash, load_config


def test_attribute_access_and_immutability():
    cfg = Config({"model": {"hidden": 256, "layers": [1, 2]}})
    assert cfg.model.hidden == 256
    assert cfg.model.layers == (1, 2)
    assert cfg.get("missing", 3) == 3
    with pytest.raises(TypeError):
        cfg.model = 1
    with pytest.raises(AttributeError):
        _ = cfg.nope


def test_overrides_parse_yaml_values_and_create_keys():
    cfg = Config({"train": {"lr": 1e-3}}).with_overrides(["train.lr=3e-4", "train.steps=100", "env=orbit"])
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.train.steps == 100
    assert cfg.env == "orbit"
    with pytest.raises(ValueError):
        Config({}).with_overrides(["noequals"])


def test_round_trip_and_extends(tmp_path):
    (tmp_path / "base.yaml").write_text("train: {lr: 0.001, steps: 10}\nmodel: {name: mlp}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain: {steps: 20}\n")
    cfg = load_config(tmp_path / "child.yaml", ["model.name=pinn"])
    assert cfg.to_dict() == {"train": {"lr": 0.001, "steps": 20}, "model": {"name": "pinn"}}
    cfg.dump(tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml") == cfg


def test_hash_is_stable_and_order_independent():
    a = Config({"x": 1, "y": {"b": 2, "a": [1, 2]}})
    b = Config({"y": {"a": [1, 2], "b": 2}, "x": 1})
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 12
    assert config_hash(a) != config_hash(a.with_overrides(["x=2"]))


def test_yaml_files_read_exponent_floats(tmp_path):
    (tmp_path / "c.yaml").write_text("lr: 3e-4\ntol: 1.0e-13\nname: '1e3'\n")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.lr == pytest.approx(3e-4) and cfg.tol == pytest.approx(1e-13)
    assert cfg.name == "1e3"
