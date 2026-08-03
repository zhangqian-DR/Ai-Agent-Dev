from pathlib import Path
from app.safety.sandbox import resolve_in_sandbox

_BINARY_EXT = {".exe", ".dll", ".png", ".jpg", ".jpeg", ".gif", ".zip",
               ".gz", ".pdf", ".mp4", ".mp3", ".ico", ".class", ".jar"}

def list_dir(work_dir: Path, path: str = ".") -> str:
    p = resolve_in_sandbox(work_dir, path)
    if not p.is_dir():
        return f"不是目录：{path}"
    items = []
    for c in sorted(p.iterdir()):
        items.append(("[D] " if c.is_dir() else "[F] ") + c.name)
    return "\n".join(items) or "(空目录)"

def read_file(work_dir: Path, path: str, max_bytes: int = 1_000_000,
              output_limit: int = 8_000) -> str:
    """两个上限管两件不同的事，不要混用：
    max_bytes    —— 拒读阈值，超过就一个字都不读，防止把超大文件读进内存。
    output_limit —— 返回给模型的字符上限。上下文预算只有 2 万多字符，
                    一个 30KB 的源文件就能吃光它，所以必须先截断再返回。
    """
    p = resolve_in_sandbox(work_dir, path)
    if not p.is_file():
        return f"文件不存在：{path}"
    if p.suffix.lower() in _BINARY_EXT:
        return f"二进制文件，无法读取：{path}"
    size = p.stat().st_size
    if size > max_bytes:
        return f"文件过大（{size} 字节，上限 {max_bytes}），拒绝读取：{path}"
    data = p.read_bytes()
    if b"\x00" in data[:8000]:
        return f"二进制文件，无法读取：{path}"
    text = data.decode("utf-8", errors="replace")
    if len(text) <= output_limit:
        return text
    lines = text.splitlines()
    note = f"（内容过长，已截断；原文共 {len(lines)} 行 / {len(text)} 字符）"
    if len(lines) > 40:
        text = "\n".join(lines[:20]) + f"\n...{note}...\n" + "\n".join(lines[-20:])
    # 压缩过的文件可能整个只有一行，按行截断无效，再加一道字符级硬上限
    if len(text) > output_limit:
        text = text[:output_limit] + f"\n...{note}"
    return text

import os, difflib
from datetime import datetime

def _backup_path(p: Path) -> Path:
    """逐代备份：`a.txt` → `a.txt.20260803-2231.bak`。
    固定叫 `a.txt.bak` 的话每次写入都覆盖上一代，agent 连改同一文件两次
    最原始的内容就永久丢失了。同秒内多次写入用 -1/-2 递增避让。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = p.with_name(f"{p.name}.{stamp}.bak")
    n = 1
    while bak.exists():
        bak = p.with_name(f"{p.name}.{stamp}-{n}.bak")
        n += 1
    return bak

def preview_write(work_dir: Path, path: str, content: str) -> str:
    p = resolve_in_sandbox(work_dir, path)
    old = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if p.is_file() else []
    new = content.splitlines(keepends=True)
    diff = difflib.unified_diff(old, new, fromfile=path, tofile=path)
    return "".join(diff) or "(无变化)"

def write_file(work_dir: Path, path: str, content: str, max_chars: int = 64_000) -> str:
    """max_chars 挡的是畸形大文件，不能指望它控制上下文：content 来自模型输出，
    等这里看到它时 token 已经花掉、那条 assistant 消息也已经进入历史。
    """
    p = resolve_in_sandbox(work_dir, path)
    if len(content) > max_chars:
        return f"内容过长（{len(content)} 字符，上限 {max_chars}），已拒绝写入，请分块写：{path}"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        _backup_path(p).write_bytes(p.read_bytes())
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, p)   # 原子替换
    return f"已写入 {path}（{len(content)} 字符）"

_IGNORE = {".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__"}

def search_in_files(work_dir: Path, pattern: str, max_hits: int = 50) -> str:
    root = work_dir.resolve()
    hits = []
    for f in root.rglob("*"):
        if any(part in _IGNORE for part in f.parts):
            continue
        if not f.is_file() or f.suffix.lower() in _BINARY_EXT:
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if pattern in line:
                    hits.append(f"{f.relative_to(root)}:{i}: {line.strip()[:120]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits) + f"\n...（已达 {max_hits} 条上限）"
        except OSError:
            continue
    return "\n".join(hits) or f"未找到：{pattern}"
