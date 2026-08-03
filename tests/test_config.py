import json

import pytest

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


def test_db_path_is_outside_work_dir(tmp_path):
    """agent.db 不能落在 work_dir 里——agent 对那里有读写权，会改到自己的记忆。"""
    cfg_file = tmp_path / "config.json"
    work = tmp_path / "ws"
    cfg_file.write_text(json.dumps({"api_key": "", "work_dir": str(work)}), encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.db_path.name == "agent.db"
    assert work.resolve() not in cfg.db_path.parents


def test_every_knob_can_be_configured(tmp_path):
    """写进 config.json 的键得真的生效。原来只读 7 个键，max_steps 之类
    改了完全没反应，也不报错——用户只会以为是自己填错了地方。"""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "api_key": "", "work_dir": str(tmp_path / "ws"),
        "max_steps": 40, "cmd_timeout": 5, "max_file_bytes": 123,
        "read_output_limit": 456, "cmd_output_limit": 789, "max_write_chars": 1011,
    }), encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.max_steps == 40
    assert cfg.cmd_timeout == 5
    assert cfg.max_file_bytes == 123
    assert cfg.read_output_limit == 456
    assert cfg.cmd_output_limit == 789
    assert cfg.max_write_chars == 1011


def test_db_inside_work_dir_is_rejected(tmp_path):
    """agent 对 work_dir 有写权，db 放进去它就能改到自己的记忆。
    config.py 的注释和 README 都警告过，但只有代码拦得住。"""
    cfg_file = tmp_path / "config.json"
    work = tmp_path / "ws"
    cfg_file.write_text(json.dumps({
        "api_key": "", "work_dir": str(work), "db_path": str(work / "sub" / "agent.db"),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="work_dir"):
        load_config(str(cfg_file))


def test_write_limit_default(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"api_key": ""}), encoding="utf-8")
    assert load_config(str(cfg_file)).max_write_chars == 64_000
