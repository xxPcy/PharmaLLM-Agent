import pytest

from pharma_llm.utils.config import ConfigError, get_required, load_yaml_config


def test_load_yaml_config_reads_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name_or_path: test-model\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert get_required(config, "model.name_or_path") == "test-model"


def test_get_required_raises_for_missing_key():
    with pytest.raises(ConfigError):
        get_required({}, "model.name_or_path")

