from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from app.agent.loop import run_agent
from app.llm.client import LLMClient
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import build_provider

_STATIC = Path(__file__).parent / "static"


class Session:
    """单会话、单任务串行。agent 在后台线程跑，页面 1 秒轮询。"""

    def __init__(self):
        self.events = queue.Queue()
        self.pending = None                  # {"name", "preview"}
        self.plan: list[str] = []
        self.plan_current = 0
        self.running = False
        self.session_id = None
        self._approved = threading.Event()
        self._ok = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """抢占运行权。已经在跑就返回 False——否则两个 agent 会同时改同一批文件。"""
        with self._lock:
            if self.running:
                return False
            self.running = True
            return True

    def confirm(self, action):               # 在 agent 线程里调用，阻塞等前端
        self.pending = action
        self._approved.clear()
        self._approved.wait()
        self.pending = None
        return self._ok

    def resolve(self, ok: bool):
        self._ok = ok
        self._approved.set()

    def abandon(self):
        """任务结束时兜底放行一次，避免线程卡死在没人回应的确认上。"""
        self._ok = False
        self._approved.set()

    def drain(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get())
        return out


def create_app(cfg, llm=None, store=None, provider=None) -> FastAPI:
    app = FastAPI(title="win-ai-agent")
    store = store or Store(str(cfg.db_path))
    provider = provider or build_provider(cfg)
    the_llm = llm or LLMClient(cfg)
    sess = Session()
    app.state.provider = provider          # 让测试能验证默认这条路建对了没

    def _emit(ev: dict):
        t = ev.get("type")
        # 计划面板要能在刷新后恢复，所以 plan 单独存一份而不是只走事件流
        if t == "plan":
            sess.plan = list(ev.get("steps") or [])
            sess.plan_current = int(ev.get("current") or 0)
        sess.events.put(ev)

        # 落库，供下次打开页面回放。step 是瞬时进度，不入库。
        if sess.session_id is None or t == "step":
            return
        if t == "plan":
            store.add_message(sess.session_id, "plan", json.dumps(
                {"steps": ev.get("steps") or [], "current": ev.get("current") or 0},
                ensure_ascii=False))
        elif t == "tool":
            store.add_message(sess.session_id, "tool", ev.get("result", ""),
                              tool_calls=ev.get("name", ""))
        else:                                    # assistant / final
            store.add_message(sess.session_id, t, ev.get("content", ""))

    def _work(goal: str):
        try:
            sess.session_id = store.create_session(goal[:40] or "未命名会话")
            store.add_message(sess.session_id, "user", goal)
            tools = build_tools(cfg, store, provider)
            run_agent(goal, llm=the_llm, tools=tools, cfg=cfg,
                      emit=_emit, confirm=sess.confirm,
                      memories=store.get_memories())
        except Exception as e:
            sess.events.put({"type": "final", "content": f"出错：{type(e).__name__}: {e}"})
        finally:
            sess.pending = None
            sess.running = False

    @app.post("/send")
    def send(body: dict):
        goal = (body or {}).get("goal", "").strip()
        if not goal:
            return JSONResponse({"ok": False, "error": "目标不能为空"}, status_code=400)
        if not sess.start():
            return JSONResponse({"ok": False, "error": "上一个任务还在跑，等它结束或刷新页面"},
                                status_code=409)
        sess.plan = []
        sess.plan_current = 0
        threading.Thread(target=_work, args=(goal,), daemon=True).start()
        return {"ok": True}

    @app.get("/poll")
    def poll():
        return {
            "events": sess.drain(),
            "pending": sess.pending,
            "running": sess.running,
            "plan": sess.plan,
            "plan_current": sess.plan_current,
            "work_dir": str(cfg.work_dir),
            "model": cfg.model,
            "max_steps": cfg.max_steps,
        }

    @app.post("/approve")
    def approve(body: dict):
        if sess.pending is None:
            return JSONResponse({"ok": False, "error": "当前没有待确认的操作"}, status_code=409)
        sess.resolve(bool((body or {}).get("ok")))
        return {"ok": True}

    @app.get("/history")
    def history():
        """最近一次会话的完整记录，供页面重新打开时回放。"""
        s = store.latest_session()
        if not s:
            return {"title": "", "created_at": "", "messages": []}
        msgs = [{"role": m["role"], "content": m["content"], "tool": m["tool_calls"] or ""}
                for m in store.get_messages(s["id"])]
        return {"title": s["title"], "created_at": s.get("created_at", ""), "messages": msgs}

    @app.get("/memories")
    def memories():
        return {"memories": store.get_memories()}

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    return app
