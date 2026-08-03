from app.safety.commands import is_readonly, needs_confirmation

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

def test_windows_switches_still_readonly():
    # "/" 在 Windows 是开关字符，不能当成绝对路径误伤
    assert is_readonly("dir /b")
    assert is_readonly("dir /s /b")

def test_needs_confirmation():
    assert needs_confirmation("write_file", {"path": "a", "content": "b"})
    assert needs_confirmation("run_command", {"cmd": "del x"})
    assert not needs_confirmation("run_command", {"cmd": "git status"})
    assert not needs_confirmation("read_file", {"path": "a"})
