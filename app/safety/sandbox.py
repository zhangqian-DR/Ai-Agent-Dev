from __future__ import annotations

import os
from pathlib import Path


class SandboxError(Exception):
    pass


def resolve_in_sandbox(work_dir: Path, user_path: str) -> Path:
    root = work_dir.resolve()
    target = (root / user_path).resolve()
    try:
        common = os.path.commonpath([str(root), str(target)])
    except ValueError:  # 不同盘符
        raise SandboxError(f"路径超出工作目录：{user_path}")
    if os.path.normcase(common) != os.path.normcase(str(root)):
        raise SandboxError(f"路径超出工作目录：{user_path}")
    return target
