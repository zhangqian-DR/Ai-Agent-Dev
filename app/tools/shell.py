import subprocess
from pathlib import Path

def run_command(work_dir: Path, cmd: str, timeout: int = 30, output_limit: int = 2000) -> str:
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(work_dir),
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"命令执行超时（>{timeout}s），已终止：{cmd}"
    def _decode(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("cp936", errors="replace")   # Windows GBK 回退
    out = _decode(proc.stdout) + _decode(proc.stderr)
    out = f"[exit={proc.returncode}]\n{out}"
    if len(out) > output_limit:
        out = out[:output_limit] + f"\n...（输出过长，已截断，共 {len(out)} 字符）"
    return out
