from __future__ import annotations

from app.tools import fs, shell
from app.tools.web import web_search

def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

class ToolSet:
    def __init__(self, cfg, store, provider):
        self.cfg, self.store, self.provider = cfg, store, provider
        self.plan: list[str] = []
        self.plan_current: int = 0        # 模型自报的当前步号，从 1 开始；0 表示还没有计划

    def schemas(self):
        s = lambda: {"type": "string"}
        return [
            _schema("list_dir", "列出工作目录下某目录的内容", {"path": s()}, []),
            _schema("read_file", "读取工作目录内的文本文件", {"path": s()}, ["path"]),
            _schema("search_in_files", "在工作目录内按关键字搜代码/文本", {"pattern": s()}, ["pattern"]),
            _schema("write_file", "写入/覆盖工作目录内的文件（会先给你确认）",
                    {"path": s(), "content": s()}, ["path", "content"]),
            _schema("run_command", "在工作目录内执行命令（危险命令会先确认）", {"cmd": s()}, ["cmd"]),
            _schema("update_plan",
                    "更新任务步骤清单，并报告当前进行到第几步。每完成一步就带着同一份 steps "
                    "重新调用一次，只把 current 加一，用户界面的计划面板据此显示进度。",
                    {"steps": {"type": "array", "items": {"type": "string"}},
                     "current": {"type": "integer",
                                 "description": "当前正在进行的步骤序号，从 1 开始"}},
                    ["steps"]),
            _schema("web_search", "联网搜索资料", {"query": s()}, ["query"]),
            _schema("save_memory", "记住关于用户的一条长期事实", {"fact": s()}, ["fact"]),
        ]

    def preview(self, name, args):
        if name == "write_file":
            return fs.preview_write(self.cfg.work_dir, args["path"], args["content"])
        if name == "run_command":
            return f"$ {args['cmd']}"
        return ""

    def execute(self, name, args) -> str:
        wd = self.cfg.work_dir
        if name == "list_dir":       return fs.list_dir(wd, args.get("path", "."))
        if name == "read_file":      return fs.read_file(wd, args["path"], self.cfg.max_file_bytes, self.cfg.read_output_limit)
        if name == "search_in_files":return fs.search_in_files(wd, args["pattern"],
                                                               max_file_bytes=self.cfg.max_file_bytes)
        if name == "write_file":     return fs.write_file(wd, args["path"], args["content"], self.cfg.max_write_chars)
        if name == "run_command":    return shell.run_command(wd, args["cmd"], self.cfg.cmd_timeout, self.cfg.cmd_output_limit)
        if name == "web_search":     return web_search(args["query"], self.provider)
        if name == "update_plan":
            self.plan = list(args["steps"])
            # 模型偶尔会报越界的步号，夹一下，免得面板算出负数下标或高亮到不存在的行
            try:
                cur = int(args.get("current") or 1)
            except (TypeError, ValueError):
                cur = 1
            self.plan_current = max(1, min(cur, len(self.plan))) if self.plan else 0
            body = "\n".join(f"{'▸' if i + 1 == self.plan_current else ' '} {i + 1}. {x}"
                             for i, x in enumerate(self.plan))
            return f"计划已更新（当前 {self.plan_current}/{len(self.plan)} 步）：\n{body}"
        if name == "save_memory":
            self.store.add_memory(args["fact"]);  return f"已记住：{args['fact']}"
        return f"未知工具：{name}"

def build_tools(cfg, store, provider) -> ToolSet:
    return ToolSet(cfg, store, provider)
