import threading
import time

from fastapi.testclient import TestClient

from app.config import Config
from app.store.db import Store
from app.tools.web import SearchProvider
from app.web.server import Session, create_app


class FakeProvider(SearchProvider):
    def search(self, q, max_results=5):
        return []


class ListDirLLM:
    def __init__(self):
        self.n = 0

    def chat(self, m, t):
        self.n += 1
        if self.n == 1:
            return {"content": None, "tool_calls": [{"id": "1", "name": "list_dir", "args": {"path": "."}}]}
        return {"content": "完成了", "tool_calls": []}


class WriteLLM:
    def __init__(self):
        self.n = 0

    def chat(self, m, t):
        self.n += 1
        if self.n == 1:
            return {"content": None, "tool_calls":
                    [{"id": "1", "name": "write_file", "args": {"path": "x.txt", "content": "hi"}}]}
        return {"content": "写完了", "tool_calls": []}


class TwoWritesLLM:
    """一轮里返回两个 write_file——模型并行工具调用时的常见形状。
    两次确认之间只隔一次写盘，比页面 1 秒的轮询周期短得多。"""

    def __init__(self):
        self.n = 0

    def chat(self, m, t):
        self.n += 1
        if self.n == 1:
            return {"content": None, "tool_calls": [
                {"id": "1", "name": "write_file", "args": {"path": "a.txt", "content": "A"}},
                {"id": "2", "name": "write_file", "args": {"path": "b.txt", "content": "B"}}]}
        return {"content": "写完了", "tool_calls": []}


class BoomLLM:
    def chat(self, m, t):
        raise RuntimeError("模型挂了")


def _db(tmp_path):
    """数据库不能放进 work_dir（agent 有写权），但也不能所有测试共用一个文件——
    tmp_path.parent 在整轮 pytest 里是共享的，会把上个用例的会话漏进来。"""
    return tmp_path.parent / f"{tmp_path.name}.db"


def _app(tmp_path, llm):
    db = _db(tmp_path)
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=db)
    return create_app(cfg, llm=llm, store=Store(str(db)), provider=FakeProvider())


def _drain(client, timeout=6):
    end = time.time() + timeout
    seen, data = [], {"pending": None}
    while time.time() < end:
        data = client.get("/poll").json()
        seen += data["events"]
        if any(e["type"] == "final" for e in seen):
            return seen, data
        if data["pending"]:
            return seen, data
        time.sleep(0.03)
    return seen, data


_NO_PREVIOUS = object()


def _wait_pending(client, after=_NO_PREVIOUS, timeout=6):
    """等一个「不是 after 那一次」的待确认操作。

    只等 pending 非空是不够的：刚 approve 完，agent 线程还没来得及把 pending
    置空，poll 拿到的仍是上一次那条。要区分两次确认，唯一可靠的依据就是 id。
    """
    end = time.time() + timeout
    while time.time() < end:
        p = client.get("/poll").json()["pending"]
        if p and p.get("id") != after:
            return p
        time.sleep(0.03)
    return None


def test_send_and_complete(tmp_path):
    client = TestClient(_app(tmp_path, ListDirLLM()))
    assert client.post("/send", json={"goal": "看看目录"}).status_code == 200
    events, _ = _drain(client)
    assert any(e["type"] == "final" for e in events)
    assert any(e["type"] == "tool" for e in events)


def test_empty_goal_rejected(tmp_path):
    client = TestClient(_app(tmp_path, ListDirLLM()))
    assert client.post("/send", json={"goal": "   "}).status_code == 400


def test_second_task_rejected_while_running(tmp_path):
    """两个 agent 同时改同一批文件会互相覆盖，必须挡住。"""
    client = TestClient(_app(tmp_path, WriteLLM()))
    client.post("/send", json={"goal": "写文件"})
    _drain(client)                                   # 停在确认上，任务仍在跑
    assert client.post("/send", json={"goal": "再来一个"}).status_code == 409
    client.post("/approve", json={"ok": False})      # 收尾，别留下卡住的线程


def test_confirmation_round_trip(tmp_path):
    client = TestClient(_app(tmp_path, WriteLLM()))
    client.post("/send", json={"goal": "写文件"})
    _, data = _drain(client)
    assert data["pending"] and data["pending"]["name"] == "write_file"
    assert "+hi" in data["pending"]["preview"]        # 确认卡片带 diff

    assert client.post("/approve", json={"ok": True}).status_code == 200
    events, _ = _drain(client)
    assert any(e["type"] == "final" for e in events)
    assert (tmp_path / "x.txt").exists()


def test_reject_leaves_file_untouched(tmp_path):
    client = TestClient(_app(tmp_path, WriteLLM()))
    client.post("/send", json={"goal": "写文件"})
    _drain(client)
    client.post("/approve", json={"ok": False})
    _drain(client)
    assert not (tmp_path / "x.txt").exists()


def test_consecutive_confirmations_get_distinct_ids(tmp_path):
    """两次确认必须能被区分开。

    页面靠 id 判断"这张确认卡我已经画过了"。没有 id 时它只能记"当前有没有画着
    一张卡"，而两次确认之间 pending 变空的窗口比轮询周期短——页面看不到那个空窗，
    就会把第二次确认当成第一次的重复丢掉，agent 线程从此永久卡在等确认上。
    """
    client = TestClient(_app(tmp_path, TwoWritesLLM()))
    client.post("/send", json={"goal": "写两个文件"})

    first = _wait_pending(client)
    assert first is not None and first["name"] == "write_file"
    assert client.post("/approve", json={"ok": True}).status_code == 200

    second = _wait_pending(client, after=first["id"])
    assert second is not None, "第二次确认没有出现"
    assert second["id"] != first["id"]
    client.post("/approve", json={"ok": True})

    events, _ = _drain(client)
    assert any(e["type"] == "final" for e in events)
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()


def test_confirm_gives_up_when_the_page_stops_polling():
    """用户关掉页面时确认闸没人应答，agent 线程不能就这么永久挂着。

    挂住的话 running 一直是 True，之后每次 /send 都 409，只能重启进程——
    而 /send 给出的提示是"刷新页面"，刷新根本救不回来。
    """
    sess = Session()
    sess.note_poll()
    t0 = time.time()
    assert sess.confirm({"name": "write_file", "preview": ""},
                        poll_gap=0.05, tick=0.01) is False
    assert time.time() - t0 < 3, "应该很快放弃，而不是一直等"


def test_confirm_keeps_waiting_while_the_page_is_alive():
    """页面还在轮询就一直等——用户可能只是在慢慢看 diff。"""
    sess = Session()
    sess.note_poll()

    def still_reading():
        for _ in range(30):
            sess.note_poll()
            time.sleep(0.01)
        sess.resolve(True)

    threading.Thread(target=still_reading, daemon=True).start()
    assert sess.confirm({"name": "write_file", "preview": ""},
                        poll_gap=0.05, tick=0.01) is True


def test_task_failure_is_persisted(tmp_path):
    """出错信息原来直接塞进事件队列、绕过了落库，页面上闪一下就没了——
    重新打开时回放的是数据库，那次失败就凭空消失，用户不知道上次为什么没结果。"""
    client = TestClient(_app(tmp_path, BoomLLM()))
    client.post("/send", json={"goal": "会炸的任务"})

    events, _ = _drain(client)
    assert any(e["type"] == "final" and "出错" in e["content"] for e in events)

    replayed = client.get("/history").json()["messages"]
    assert any("出错" in m["content"] for m in replayed), "失败没进数据库，回放时丢了"


def test_poll_marks_the_page_alive(tmp_path):
    """确认闸靠 /poll 判断页面是否还活着，这条线必须真的接上。"""
    client = TestClient(_app(tmp_path, ListDirLLM()))
    sess = client.app.state.session
    sess.last_poll = 0.0
    client.get("/poll")
    assert sess.last_poll > 0.0


def test_approve_without_pending_is_409(tmp_path):
    """没有待确认时误点确认，不能把下一次真正的确认提前放行。"""
    client = TestClient(_app(tmp_path, ListDirLLM()))
    assert client.post("/approve", json={"ok": True}).status_code == 409


def test_poll_exposes_plan_and_env(tmp_path):
    client = TestClient(_app(tmp_path, ListDirLLM()))
    d = client.get("/poll").json()
    assert d["work_dir"] == str(tmp_path)
    assert d["model"] == "qwen-plus"
    assert d["plan"] == [] and d["running"] is False


def test_default_provider_follows_config(tmp_path):
    """不显式传 provider 时必须按 config 的 search_provider 建。
    之前所有用例都传 FakeProvider，默认这条路一次都没跑过，
    结果 server.py 里硬编码的 DuckDuckGoProvider 一直没被换掉。"""
    from app.tools.web import DashScopeProvider, DuckDuckGoProvider

    cfg = Config("http://x", "k", "qwen-max", tmp_path, db_path=_db(tmp_path),
                 search_provider="dashscope")
    app = create_app(cfg, llm=ListDirLLM(), store=Store(str(_db(tmp_path))))
    assert isinstance(app.state.provider, DashScopeProvider)

    cfg2 = Config("http://x", "k", "qwen-max", tmp_path, db_path=_db(tmp_path),
                  search_provider="duckduckgo")
    app2 = create_app(cfg2, llm=ListDirLLM(), store=Store(str(_db(tmp_path))))
    assert isinstance(app2.state.provider, DuckDuckGoProvider)


def test_index_page_is_served(tmp_path):
    """静态页要真能被托管出去，否则打开浏览器是 404。"""
    client = TestClient(_app(tmp_path, ListDirLLM()))
    r = client.get("/")
    assert r.status_code == 200
    assert "win-ai-agent" in r.text
    assert "/approve" in r.text          # 页面确实接了确认接口


class PlanLLM:
    def __init__(self):
        self.n = 0

    def chat(self, m, t):
        self.n += 1
        if self.n <= 2:
            return {"content": None, "tool_calls": [{"id": str(self.n), "name": "update_plan",
                    "args": {"steps": ["一", "二", "三"], "current": self.n * 2 - 1}}]}
        return {"content": "做完了", "tool_calls": []}


def test_poll_exposes_plan_progress(tmp_path):
    """刷新页面后进度也要还原，不能只还原步骤列表。"""
    client = TestClient(_app(tmp_path, PlanLLM()))
    client.post("/send", json={"goal": "干活"})
    _drain(client)
    d = client.get("/poll").json()
    assert d["plan"] == ["一", "二", "三"]
    assert d["plan_current"] == 3


def test_history_keeps_plan_progress(tmp_path):
    store = Store(str(_db(tmp_path)))
    cfg = Config("", "", "qwen-max", tmp_path, db_path=_db(tmp_path))
    client = TestClient(create_app(cfg, llm=PlanLLM(), store=store, provider=FakeProvider()))
    client.post("/send", json={"goal": "干活"})
    _drain(client)
    plans = [m for m in client.get("/history").json()["messages"] if m["role"] == "plan"]
    assert plans, "历史里要有 plan"
    import json as _json
    last = _json.loads(plans[-1]["content"])
    assert last["steps"] == ["一", "二", "三"] and last["current"] == 3


def test_history_endpoint_replays_last_session(tmp_path):
    """关掉页面再打开要能看回上一轮做了什么，否则聊天记录等于没存。"""
    store = Store(str(tmp_path.parent / "r.db"))
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=tmp_path.parent / "r.db")
    client = TestClient(create_app(cfg, llm=ListDirLLM(), store=store, provider=FakeProvider()))
    client.post("/send", json={"goal": "看看目录"})
    _drain(client)

    h = client.get("/history").json()
    kinds = [m["role"] for m in h["messages"]]
    assert kinds[0] == "user" and h["messages"][0]["content"] == "看看目录"
    assert "tool" in kinds, "工具调用也要能回放，否则只剩对话没有过程"
    assert kinds[-1] == "final"
    assert h["title"] == "看看目录"


def test_history_records_tool_name(tmp_path):
    store = Store(str(tmp_path.parent / "r2.db"))
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=tmp_path.parent / "r2.db")
    client = TestClient(create_app(cfg, llm=ListDirLLM(), store=store, provider=FakeProvider()))
    client.post("/send", json={"goal": "看看目录"})
    _drain(client)
    tools = [m for m in client.get("/history").json()["messages"] if m["role"] == "tool"]
    assert tools and tools[0]["tool"] == "list_dir"


def test_history_empty_when_no_session(tmp_path):
    client = TestClient(_app(tmp_path, ListDirLLM()))
    h = client.get("/history").json()
    assert h["messages"] == [] and h["title"] == ""


def test_final_stored_once(tmp_path):
    """数据库里最终回答只该有一条。"""
    store = Store(str(tmp_path.parent / "r3.db"))
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=tmp_path.parent / "r3.db")
    client = TestClient(create_app(cfg, llm=ListDirLLM(), store=store, provider=FakeProvider()))
    client.post("/send", json={"goal": "看看目录"})
    _drain(client)
    finals = [m for m in store.get_messages(1) if m["content"] == "完成了"]
    assert len(finals) == 1, f"最终回答存了 {len(finals)} 条"


def test_history_is_persisted(tmp_path):
    """§7 要求会话/消息落 SQLite，否则关掉页面就什么都不剩。"""
    store = Store(str(tmp_path.parent / "h.db"))
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=tmp_path.parent / "h.db")
    client = TestClient(create_app(cfg, llm=ListDirLLM(), store=store, provider=FakeProvider()))
    client.post("/send", json={"goal": "看看目录"})
    _drain(client)
    msgs = store.get_messages(1)
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "看看目录"
    assert any(m["role"] == "final" for m in msgs)
