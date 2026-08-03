import threading
import time

from fastapi.testclient import TestClient

from app.config import Config
from app.store.db import Store
from app.tools.web import SearchProvider
from app.web.server import create_app


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


def _app(tmp_path, llm):
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=tmp_path.parent / "t.db")
    return create_app(cfg, llm=llm, store=Store(str(tmp_path.parent / "t.db")), provider=FakeProvider())


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


def test_index_page_is_served(tmp_path):
    """静态页要真能被托管出去，否则打开浏览器是 404。"""
    client = TestClient(_app(tmp_path, ListDirLLM()))
    r = client.get("/")
    assert r.status_code == 200
    assert "win-ai-agent" in r.text
    assert "/approve" in r.text          # 页面确实接了确认接口


def test_history_is_persisted(tmp_path):
    """§7 要求会话/消息落 SQLite，否则关掉页面就什么都不剩。"""
    store = Store(str(tmp_path.parent / "h.db"))
    cfg = Config("", "", "qwen-plus", tmp_path, db_path=tmp_path.parent / "h.db")
    client = TestClient(create_app(cfg, llm=ListDirLLM(), store=store, provider=FakeProvider()))
    client.post("/send", json={"goal": "看看目录"})
    _drain(client)
    msgs = store.get_messages(1)
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "看看目录"
    assert any(m["role"] == "assistant" for m in msgs)
