"""把 tests/frontend_checks.js 接进 pytest，一条 `pytest` 命令跑完前后端。

为什么不引入 jest / vitest：这个项目是纯 Python 的，双击 .bat 就能用，不该为了
测页面逻辑拖进一整套 JS 工具链。frontend_checks.js 不装任何依赖，只用 node 自带
的东西跑。**没装 node 的机器上整组跳过**，不会让 pytest 变红——前端检查是加分项，
不该成为跑不了测试的理由。

JS 每条检查输出一行 RESULT<TAB>PASS|FAIL<TAB>名字<TAB>细节，这里按行解析成一个个
用例，红了直接能看出是哪条，不用去翻一大段 stdout。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_SCRIPT = _HERE / "frontend_checks.js"
_NODE = shutil.which("node")


def _run() -> tuple[list[tuple[str, bool, str]], str]:
    proc = subprocess.run([_NODE, str(_SCRIPT)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    out = proc.stdout + proc.stderr
    results = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "RESULT" and parts[1] in ("PASS", "FAIL"):
            results.append((parts[2], parts[1] == "PASS", parts[3] if len(parts) > 3 else ""))
    return results, out


# 在收集阶段跑一次：这样每条检查能变成一个独立的 pytest 用例。
_RESULTS, _OUTPUT = _run() if _NODE else ([], "")


@pytest.mark.skipif(_NODE is None, reason="没装 node，跳过页面逻辑检查")
def test_frontend_checks_actually_ran():
    """脚本本身得能跑起来——一条检查都没解析出来时，别让整组静悄悄地全绿。"""
    assert _RESULTS, f"frontend_checks.js 没有输出任何结果：\n{_OUTPUT[-2000:]}"


@pytest.mark.skipif(not _RESULTS, reason="没装 node，跳过页面逻辑检查")
@pytest.mark.parametrize("name,ok,detail",
                         _RESULTS or [("placeholder", True, "")],
                         ids=[r[0] for r in _RESULTS] or ["placeholder"])
def test_frontend(name, ok, detail):
    assert ok, f"{name} —— 实际：{detail}"
