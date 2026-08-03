import json

from app.config import load_config


def test_load_config_reads_fields(tmp_path):
    cfg_file = tmp_path / "config.json"
    work = tmp_path / "ws"
    cfg_file.write_text(json.dumps({
        "base_url": "http://x", "api_key": "", "model": "qwen-plus",
        "work_dir": str(work), "port": 8000
    }), encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.model == "qwen-plus"
    assert cfg.work_dir == work.resolve()
    assert work.exists()          # 工作目录不存在会被创建
    assert cfg.max_steps == 20    # 默认值


def test_load_config_missing_workdir_defaults(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"api_key": ""}), encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.model == "qwen-plus"


def test_read_limits_are_two_separate_knobs(tmp_path):
    """拒读阈值按内存来定，返回上限按上下文预算来定，两者相差两个数量级。"""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"api_key": ""}), encoding="utf-8")
    cfg = load_config(str(cfg_file))

    assert cfg.max_file_bytes == 1_000_000     # 拒读：防内存
    assert cfg.read_output_limit == 8_000      # 截断：防超窗
    assert cfg.read_output_limit < cfg.max_file_bytes


def test_write_limit_default(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"api_key": ""}), encoding="utf-8")
    assert load_config(str(cfg_file)).max_write_chars == 64_000
