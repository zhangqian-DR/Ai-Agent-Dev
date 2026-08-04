import subprocess
from pathlib import Path


def _decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("cp936", errors="replace")   # Windows GBK 回退


def run_and_check(work_dir: Path, cmd: str, timeout: int = 30,
                  output_limit: int = 2000) -> tuple[bool, str]:
    """跑一条命令，返回 ``(过没过, 输出)``。

    验收闸要的是「过没过」这个布尔值。让它去输出里认 ``[exit=0]`` 那几个字，
    等于把展示格式当成了协议——改一下措辞验收就失灵，而且不会有人发现。
    超时算**没过**：卡住的验收命令不能当成通过。
    """
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(work_dir),
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"命令执行超时（>{timeout}s），已终止：{cmd}"

    out = f"[exit={proc.returncode}]\n{_decode(proc.stdout)}{_decode(proc.stderr)}"
    if len(out) > output_limit:
        out = out[:output_limit] + f"\n...（输出过长，已截断，共 {len(out)} 字符）"
    return proc.returncode == 0, out


def run_command(work_dir: Path, cmd: str, timeout: int = 30, output_limit: int = 2000) -> str:
    """给模型用的工具：只要输出，成败由模型自己从 exit 码读。"""
    return run_and_check(work_dir, cmd, timeout, output_limit)[1]
