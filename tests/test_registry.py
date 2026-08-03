from app.config import Config
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import SearchProvider


class FakeProvider(SearchProvider):
    def search(self, q, max_results=5):
        return [{"title": "T", "href": "h", "body": "b"}]


def _ts(tmp_path):
    cfg = Config("", "", "qwen-max", tmp_path, db_path=tmp_path.parent / f"{tmp_path.name}.db")
    return build_tools(cfg, Store(str(cfg.db_path)), FakeProvider())


def test_schemas_cover_all(tmp_path):
    names = {s["function"]["name"] for s in _ts(tmp_path).schemas()}
    assert names == {"list_dir", "read_file", "search_in_files", "write_file",
                     "run_command", "update_plan", "web_search", "save_memory"}


def test_execute_read(tmp_path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    assert "hi" in _ts(tmp_path).execute("read_file", {"path": "a.txt"})


def test_preview_write_diff(tmp_path):
    assert "+new" in _ts(tmp_path).preview("write_file", {"path": "a.txt", "content": "new\n"})


def test_save_memory(tmp_path):
    assert "记住" in _ts(tmp_path).execute("save_memory", {"fact": "用户用 Java"})


# ---------- 计划进度 ----------

def test_update_plan_schema_has_current(tmp_path):
    """没有 current 的话，面板只能靠数工具调用次数猜进度——而计划的步骤粒度
    和工具调用粒度天然对不上（一步「编写并保存」实际只有一次 write_file）。"""
    fn = next(s["function"] for s in _ts(tmp_path).schemas()
              if s["function"]["name"] == "update_plan")
    props = fn["parameters"]["properties"]
    assert "current" in props
    assert props["current"]["type"] == "integer"


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
