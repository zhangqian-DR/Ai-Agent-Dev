# 只读白名单（首 token，或 "git status/diff/log" 两段）
_SINGLE = {"dir", "ls", "type", "cat", "pwd", "echo", "whoami"}
_GIT_SUB = {"status", "diff", "log"}
# 注意：换行/回车同样是 shell 命令分隔符，必须计入，
# 否则 "dir\ndel x.txt" 会被判为只读并自动放行
_META = ["&&", "||", "|", ";", ">", "<", "&", "`", "$(", "\n", "\r"]

def _arg_escapes_sandbox(parts) -> bool:
    """run_command 不经过 resolve_in_sandbox，故在此拦掉明显越界的路径参数。
    否则白名单里的 type/cat 可以读到工作目录之外。"""
    for p in parts[1:]:
        if ".." in p:                       # 相对路径穿越
            return True
        if len(p) > 1 and p[1] == ":":      # C:\ 之类的绝对盘符路径
            return True
        # POSIX 绝对路径。注意不能只看开头的 "/"：
        # Windows 上 "/" 是开关字符（dir /b、type /?），会误伤只读命令
        if p.startswith("/") and "/" in p[1:]:
            return True
    return False

def is_readonly(cmd: str) -> bool:
    c = cmd.strip()
    if any(m in c for m in _META):   # 有 shell 元字符/换行→组合命令→不放行
        return False
    parts = c.split()
    if not parts:
        return False
    if _arg_escapes_sandbox(parts):
        return False
    head = parts[0].lower()
    if head in _SINGLE:
        return True
    if head == "git" and len(parts) >= 2 and parts[1].lower() in _GIT_SUB:
        return True
    if head in {"python", "python3", "pip", "pip3"} and any(
        a in parts for a in ("--version", "-V", "list")):
        return True
    return False

def needs_confirmation(tool_name: str, args: dict) -> bool:
    if tool_name == "write_file":
        return True
    if tool_name == "run_command":
        return not is_readonly(args.get("cmd", ""))
    return False
