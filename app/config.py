from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    base_url: str
    api_key: str
    model: str
    work_dir: Path
    port: int = 8000
    max_steps: int = 20
    cmd_timeout: int = 30
    # 拒读阈值：超过就一个字都不读，防止把超大文件读进内存
    max_file_bytes: int = 1_000_000
    # 返回给模型的字符上限：上下文预算只有 2 万多字符，必须先截断再返回
    read_output_limit: int = 8_000
    cmd_output_limit: int = 2000
    # 单次写入的内容上限，挡畸形大文件（挡不住上下文膨胀，见 fs.write_file）
    max_write_chars: int = 64_000


def load_config(path: str = "config.json") -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    work_dir = Path(raw.get("work_dir", "./workspace")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        base_url=raw.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=raw.get("api_key", ""),
        model=raw.get("model", "qwen-plus"),
        work_dir=work_dir,
        port=int(raw.get("port", 8000)),
    )
