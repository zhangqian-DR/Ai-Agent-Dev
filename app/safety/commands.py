# 只读白名单（首 token，或 "git status/diff/log" 两段）
_SINGLE = {"dir", "ls", "type", "cat", "pwd", "echo", "whoami"}
_GIT_SUB = {"status", "diff", "log"}
# 注意：换行/回车同样是 shell 命令分隔符，必须计入，
# 否则 "dir\ndel x.txt" 会被判为只读并自动放行。
# "%" 不是分隔符但同样要拦：shell=True 下 cmd.exe 会展开 %USERPROFILE%，
# 而下面的参数检查发生在展开之前，看到的还是没有盘符、没有 ".." 的字面量，
# 于是 "type %USERPROFILE%\.ssh\id_rsa" 会被判为只读并读走沙箱外的文件。
_META = ["&&", "||", "|", ";", ">", "<", "&", "`", "$(", "%", "\n", "\r"]
# 这几个开关本身是只读的，但只有当它是唯一参数时才算——见 is_readonly
_INSPECT_FLAGS = {"--version", "-V", "list"}

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
    # 必须是「解释器 + 唯一的查看开关」这两个 token。之前写成"开关出现在任意
    # 位置就放行"，于是 `python -c "任意代码" --version` 整条命令被判为只读，
    # 确认闸形同虚设——而那道闸正是提示注入的最后一层兜底。
    if head in {"python", "python3", "pip", "pip3"} and len(parts) == 2 \
            and parts[1] in _INSPECT_FLAGS:
        return True
    return False

def needs_confirmation(tool_name: str, args: dict) -> bool:
    if tool_name == "write_file":
        return True
    if tool_name == "run_command":
        return not is_readonly(args.get("cmd", ""))
    return False
