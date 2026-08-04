from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools import fs, shell
from app.tools.web import web_search


# ---------- 参数模型 ----------
# 模型看到的函数定义由这些类自动生成，不再手写 JSON schema——一处定义，
# 既给模型看，又在执行前真的做校验（漏参数会抛出来，由循环喂回去触发反思）。

class ListDirArgs(BaseModel):
    path: str = Field(".", description="相对工作目录的路径，不传就是工作目录本身")

class ReadFileArgs(BaseModel):
    path: str

class SearchArgs(BaseModel):
    pattern: str

class WriteFileArgs(BaseModel):
    path: str
    content: str

class RunCommandArgs(BaseModel):
    cmd: str

class UpdatePlanArgs(BaseModel):
    steps: List[str]
    current: int = Field(1, description="当前正在进行的步骤序号，从 1 开始")

class WebSearchArgs(BaseModel):
    query: str

class SaveMemoryArgs(BaseModel):
    fact: str
    is_negative: bool = Field(
        False, description="这条记的是「不要做某事」时设为 true，提示词里会单独标出来")


class ToolSet:
    """把 8 个工具装成 LangChain 工具对象。

    对外仍是 ``tools() / execute() / preview()`` 三个口子，循环那层照旧；
    但工具本身已经是 ``BaseTool``，阶段 3 的 ``ToolNode`` 可以直接消费。
    """

    def __init__(self, cfg, store, provider):
        self.cfg, self.store, self.provider = cfg, store, provider
        self.plan: list[str] = []
        self.plan_current: int = 0        # 模型自报的当前步号，从 1 开始；0 表示还没有计划
        self._tools = self._build()
        self._by_name = {t.name: t for t in self._tools}

    # ---------- 各工具的实现（闭包持有 cfg / store / provider） ----------

    def _build(self) -> list[BaseTool]:
        cfg = self.cfg

        def list_dir(path: str = ".") -> str:
            return fs.list_dir(cfg.work_dir, path)

        def read_file(path: str) -> str:
            return fs.read_file(cfg.work_dir, path, cfg.max_file_bytes, cfg.read_output_limit)

        def search_in_files(pattern: str) -> str:
            return fs.search_in_files(cfg.work_dir, pattern, max_file_bytes=cfg.max_file_bytes)

        def write_file(path: str, content: str) -> str:
            return fs.write_file(cfg.work_dir, path, content, cfg.max_write_chars)

        def run_command(cmd: str) -> str:
            return shell.run_command(cfg.work_dir, cmd, cfg.cmd_timeout, cfg.cmd_output_limit)

        def do_web_search(query: str) -> str:
            return web_search(query, self.provider)

        def save_memory(fact: str, is_negative: bool = False) -> str:
            added = self.store.add_memory(fact, is_negative=is_negative)
            mark = "（禁令）" if is_negative else ""
            if not added:
                # 如实说是重复，别回"已记住"让模型以为又存了一条、
                # 也免得它以为没记住而反复再试
                return f"这条已经记过了，无需重复{mark}：{fact}"
            return f"已记住{mark}：{fact}"

        def update_plan(steps: List[str], current: int = 1) -> str:
            self.plan = list(steps)
            # 模型偶尔会报越界的步号，夹一下，免得面板算出负数下标或高亮到不存在的行
            self.plan_current = max(1, min(current, len(self.plan))) if self.plan else 0
            body = "\n".join(f"{'▸' if i + 1 == self.plan_current else ' '} {i + 1}. {x}"
                             for i, x in enumerate(self.plan))
            return f"计划已更新（当前 {self.plan_current}/{len(self.plan)} 步）：\n{body}"

        def T(func, name, desc, args_schema):
            return StructuredTool.from_function(
                func=func, name=name, description=desc, args_schema=args_schema)

        return [
            T(list_dir, "list_dir", "列出工作目录下某目录的内容", ListDirArgs),
            T(read_file, "read_file", "读取工作目录内的文本文件", ReadFileArgs),
            T(search_in_files, "search_in_files",
              "在工作目录内搜代码/文本。按**字面子串**匹配，不是正则——"
              "写 def\\s+add 之类的正则一定搜不到，直接写 def add 就行。",
              SearchArgs),
            T(write_file, "write_file", "写入/覆盖工作目录内的文件（会先给你确认）", WriteFileArgs),
            T(run_command, "run_command", "在工作目录内执行命令（危险命令会先确认）", RunCommandArgs),
            T(update_plan, "update_plan",
              "更新任务步骤清单，并报告当前进行到第几步。每完成一步就带着同一份 steps "
              "重新调用一次，只把 current 加一，用户界面的计划面板据此显示进度。",
              UpdatePlanArgs),
            T(do_web_search, "web_search", "联网搜索资料", WebSearchArgs),
            T(save_memory, "save_memory",
              "记住关于用户的一条长期事实。记「不要做某事」时把 is_negative 设为 true。",
              SaveMemoryArgs),
        ]

    # ---------- 对外 ----------

    def tools(self) -> list[BaseTool]:
        return self._tools

    def execute(self, name: str, args: dict) -> str:
        tool: Optional[BaseTool] = self._by_name.get(name)
        if tool is None:
            return f"未知工具：{name}"
        return tool.invoke(args or {})

    def preview(self, name: str, args: dict) -> str:
        """只给确认卡片生成预览，没有注册成工具，模型看不到它。"""
        if name == "write_file":
            return fs.preview_write(self.cfg.work_dir, args["path"], args["content"])
        if name == "run_command":
            return f"$ {args['cmd']}"
        return ""


def build_tools(cfg, store, provider) -> ToolSet:
    return ToolSet(cfg, store, provider)
