import pytest
from langchain_core.tools import BaseTool

from app.config import Config
from app.safety.commands import AUTO_APPROVED
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import SearchProvider

ALL_NAMES = {"list_dir", "read_file", "search_in_files", "write_file",
             "run_command", "update_plan", "web_search", "save_memory"}


class FakeProvider(SearchProvider):
    def search(self, q, max_results=5):
        return [{"title": "T", "href": "h", "body": "b"}]


def _ts(tmp_path):
    cfg = Config("", "", "qwen-max", tmp_path, db_path=tmp_path.parent / f"{tmp_path.name}.db")
    return build_tools(cfg, Store(str(cfg.db_path)), FakeProvider())


def _tool(ts, name):
    return next(t for t in ts.tools() if t.name == name)


def test_tools_cover_all(tmp_path):
    assert {t.name for t in _ts(tmp_path).tools()} == ALL_NAMES


def test_tools_are_langchain_tools(tmp_path):
    """必须是 BaseTool 而不是手写的 schema dict——阶段 3 的 ToolNode 直接吃这个，
    模型侧的函数定义也由它自动生成，不必两处维护。"""
    for t in _ts(tmp_path).tools():
        assert isinstance(t, BaseTool), t


def test_auto_approved_names_all_exist(tmp_path):
    """自动放行名单里的每一项都得是真实存在的工具。

    名单在 safety/，工具在 tools/，两边各改各的迟早对不上——名字写错一个，
    那个工具就悄悄被收进了确认闸（或者更糟：改名之后名单里留了个空壳）。
    """
    names = {t.name for t in _ts(tmp_path).tools()}
    assert AUTO_APPROVED <= names, f"名单里有不存在的工具：{AUTO_APPROVED - names}"


def test_only_the_two_dangerous_tools_are_gated(tmp_path):
    """新增工具默认落进确认闸（fail-closed）。

    这条测试的作用是让「加了工具却没想过它危不危险」显式地红一次：
    新工具会让这个集合变大，逼你回来做一次决定。
    """
    gated = {t.name for t in _ts(tmp_path).tools() if t.name not in AUTO_APPROVED}
    assert gated == {"write_file", "run_command"}


def test_every_tool_keeps_its_description(tmp_path):
    """描述是提示词的一部分，模型靠它选工具，不能在改造中丢掉。"""
    for t in _ts(tmp_path).tools():
        assert t.description.strip(), t.name


# ---------- 执行 ----------

def test_execute_read(tmp_path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    assert "hi" in _ts(tmp_path).execute("read_file", {"path": "a.txt"})


def test_execute_save_memory(tmp_path):
    assert "记住" in _ts(tmp_path).execute("save_memory", {"fact": "用户用 Java"})


def test_save_memory_can_flag_negative(tmp_path):
    """模型得能声明"这是一条禁令"——靠文本里有没有"不"字去猜太脆。"""
    ts = _ts(tmp_path)
    ts.execute("save_memory", {"fact": "用 tab 缩进", "is_negative": True})
    assert ts.store.get_memories()[0]["is_negative"] is True


def test_save_memory_tells_the_model_it_was_a_duplicate(tmp_path):
    """重复记同一件事时要如实说，别回一句"已记住"让模型以为又存了一条。"""
    ts = _ts(tmp_path)
    ts.execute("save_memory", {"fact": "用户用 Java"})
    again = ts.execute("save_memory", {"fact": "用户用 Java"})

    assert "已经" in again or "重复" in again, f"没告诉模型这是重复：{again!r}"
    assert len(ts.store.get_memories()) == 1


def test_list_dir_path_is_optional(tmp_path):
    """不传 path 时默认列工作目录本身，别逼模型每次都写 '.'。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    assert "a.txt" in _ts(tmp_path).execute("list_dir", {})


def test_missing_required_argument_raises(tmp_path):
    """漏参数要抛出来，让循环那层兜住并把错误喂回模型反思——
    不能默默当成空字符串跑下去。"""
    with pytest.raises(Exception):
        _ts(tmp_path).execute("read_file", {})


def test_preview_write_diff(tmp_path):
    assert "+new" in _ts(tmp_path).preview("write_file", {"path": "a.txt", "content": "new\n"})


# ---------- 计划进度 ----------

def test_update_plan_schema_has_current(tmp_path):
    """没有 current 的话，面板只能靠数工具调用次数猜进度——而计划的步骤粒度
    和工具调用粒度天然对不上（一步「编写并保存」实际只有一次 write_file）。"""
    args = _tool(_ts(tmp_path), "update_plan").args
    assert "current" in args
    assert args["current"]["type"] == "integer"


def test_update_plan_records_current(tmp_path):
    ts = _ts(tmp_path)
    out = ts.execute("update_plan", {"steps": ["读文件", "改代码", "跑测试"], "current": 2})
    assert ts.plan == ["读文件", "改代码", "跑测试"]
    assert ts.plan_current == 2
    assert "2/3" in out, f"回给模型的确认里要带进度，便于它自己对齐：{out!r}"


def test_update_plan_current_defaults_to_first(tmp_path):
    """模型没报 current 时按第 1 步算，不要变成 0 让面板一格都不亮。"""
    ts = _ts(tmp_path)
    ts.execute("update_plan", {"steps": ["a", "b"]})
    assert ts.plan_current == 1


def test_update_plan_current_is_clamped(tmp_path):
    """模型偶尔会报越界的步号，不能让面板算出负数下标或越界高亮。"""
    ts = _ts(tmp_path)
    ts.execute("update_plan", {"steps": ["a", "b"], "current": 9})
    assert ts.plan_current == 2
    ts.execute("update_plan", {"steps": ["a", "b"], "current": -3})
    assert ts.plan_current == 1


def test_update_plan_can_be_recalled_to_advance(tmp_path):
    """推进就是带着同一份 steps 重新调一次，只改 current。"""
    ts = _ts(tmp_path)
    ts.execute("update_plan", {"steps": ["a", "b", "c"], "current": 1})
    ts.execute("update_plan", {"steps": ["a", "b", "c"], "current": 3})
    assert ts.plan_current == 3 and ts.plan == ["a", "b", "c"]
