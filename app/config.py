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
    # agent.db 单独配，不要放进 work_dir（agent 能读写那里，会改到自己的记忆），
    # 也不要放 work_dir 的上级目录（那是别人的地盘）。默认落在程序自己的目录。
    db_path: Path = Path("agent.db")
    # dashscope：复用同一个 api_key，从被 DDG 风控的网络出口也能用，但拿不到 URL
    # duckduckgo：免 key、有 URL，但对网络出口敏感
    search_provider: str = "dashscope"
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

    @property
    def checkpoint_path(self) -> Path:
        """LangGraph 的 checkpoint 库，和 agent.db 分开放。

        两者的锁和生命周期是两回事（一个我们自己管，一个 SqliteSaver 自己管），
        混在一个文件里只会让退出顺序更难理清。跟着 db_path 走，所以「不能放进
        work_dir」那道校验对它同样有效。
        """
        return self.db_path.with_suffix(".checkpoints.sqlite")


def load_config(path: str = "config.json") -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    work_dir = Path(raw.get("work_dir", "./workspace")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(raw.get("db_path", "agent.db")).resolve()
    # 光靠注释和 README 提醒不够：db 落在 work_dir 里，agent 就能改自己的记忆。
    if db_path == work_dir or work_dir in db_path.parents:
        raise ValueError(
            f"db_path 不能放在 work_dir 里（agent 对那里有写权，会改到自己的记忆）：\n"
            f"  work_dir = {work_dir}\n  db_path  = {db_path}")

    _defaults = Config("", "", "", work_dir)          # 各上限的默认值只写在 dataclass 里

    def _int(key: str) -> int:
        return int(raw.get(key, getattr(_defaults, key)))

    return Config(
        base_url=raw.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=raw.get("api_key", ""),
        model=raw.get("model", "qwen-plus"),
        work_dir=work_dir,
        db_path=db_path,
        search_provider=raw.get("search_provider", "dashscope"),
        port=_int("port"),
        # 这些上限以前读都不读，用户改了 config.json 完全没反应也没有提示
        max_steps=_int("max_steps"),
        cmd_timeout=_int("cmd_timeout"),
        max_file_bytes=_int("max_file_bytes"),
        read_output_limit=_int("read_output_limit"),
        cmd_output_limit=_int("cmd_output_limit"),
        max_write_chars=_int("max_write_chars"),
    )
