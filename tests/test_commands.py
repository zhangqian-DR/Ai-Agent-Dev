from app.safety.commands import is_readonly, needs_confirmation
from app.tools.shell import run_and_check


def test_run_and_check_reports_success(tmp_path):
    """验收闸要的是「过没过」这个布尔值，不是从输出里认 [exit=0] 那几个字。"""
    ok, out = run_and_check(tmp_path, "python -c \"print('hi')\"")
    assert ok is True
    assert "hi" in out


def test_run_and_check_reports_failure(tmp_path):
    ok, out = run_and_check(tmp_path, "python -c \"import sys; sys.exit(3)\"")
    assert ok is False
    assert "exit=3" in out


def test_run_and_check_timeout_is_a_failure(tmp_path):
    """验收命令卡住不能当成通过。"""
    ok, out = run_and_check(tmp_path, "python -c \"import time; time.sleep(30)\"", timeout=2)
    assert ok is False
    assert "超时" in out



def test_readonly_whitelist():
    assert is_readonly("git status")
    assert is_readonly("dir")
    assert is_readonly("python --version")

def test_non_readonly():
    assert not is_readonly("del x.txt")
    assert not is_readonly("rm -rf .")
    assert not is_readonly("pip install requests")

def test_combined_command_not_readonly():
    # 组合命令即使含白名单词也不放行
    assert not is_readonly("dir && del /f /s /q *")
    assert not is_readonly("git status | rm x")

def test_newline_is_command_separator():
    # 换行也是分隔符，否则白名单被绕过
    assert not is_readonly("dir\ndel x.txt")
    assert not is_readonly("echo hi\rrm -rf .")

def test_path_traversal_in_args_not_readonly():
    # run_command 不过沙箱，白名单命令的参数不得越界
    assert not is_readonly("type ..\\..\\secret.txt")
    assert not is_readonly("cat ../../etc/passwd")
    assert not is_readonly("type C:\\Windows\\win.ini")
    assert not is_readonly("cat /etc/passwd")

def test_version_flag_anywhere_does_not_whitelist_the_whole_command():
    # `--version` / `list` 只有作为唯一参数时才是只读操作。
    # 出现在任意位置就放行的话，`python -c "任意代码" --version` 直接绕过确认闸。
    assert not is_readonly('python -c "print(1)" --version')
    assert not is_readonly('python -c "__import__(chr(111)).system(chr(99))" list')
    assert not is_readonly("pip install requests --version")
    assert not is_readonly("pip download evil -V")
    # 真正的只读用法仍要放行
    assert is_readonly("python --version")
    assert is_readonly("pip list")
    assert is_readonly("python3 -V")

def test_env_var_expansion_not_readonly():
    # shell=True 下 cmd.exe 会把 %USERPROFILE% 展开成绝对路径，
    # 而参数检查发生在展开之前——不拦 % 的话白名单里的 type 能读走沙箱外任意文件
    assert not is_readonly("type %USERPROFILE%\\.ssh\\id_rsa")
    assert not is_readonly("cat %APPDATA%\\config")
    assert not is_readonly("echo %PATH%")

def test_windows_switches_still_readonly():
    # "/" 在 Windows 是开关字符，不能当成绝对路径误伤
    assert is_readonly("dir /b")
    assert is_readonly("dir /s /b")

def test_unknown_tool_needs_confirmation():
    """未知工具必须 fail-closed。

    今天不可利用——registry 只有 8 个工具，别的名字会被 execute 挡下。但这是
    一颗定时雷：以后加第 9 个会写盘的工具、忘了同步改这里，它就静默地自动放行，
    而这道闸是提示注入的最后一层兜底。默认必须是"要确认"。
    """
    assert needs_confirmation("delete_everything", {})
    assert needs_confirmation("some_future_write_tool", {"path": "x"})
    assert needs_confirmation("", {})


def test_auto_approved_tools_stay_auto():
    """六个只读工具仍然自动放行，别把闸收得连查目录都要点确认。"""
    for name in ("list_dir", "read_file", "search_in_files",
                 "web_search", "update_plan", "save_memory"):
        assert not needs_confirmation(name, {}), name


def test_needs_confirmation():
    assert needs_confirmation("write_file", {"path": "a", "content": "b"})
    assert needs_confirmation("run_command", {"cmd": "del x"})
    assert not needs_confirmation("run_command", {"cmd": "git status"})
    assert not needs_confirmation("read_file", {"path": "a"})
