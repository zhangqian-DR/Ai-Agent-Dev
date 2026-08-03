from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import asynccontextmanager
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
        self.pending = None                  # {"id", "name", "preview"}
        self._seq = 0                        # 确认闸的递增编号，见 confirm()
        self.plan: list[str] = []
        self.plan_current = 0
        self.running = False
        self.session_id = None
        self.last_poll = time.time()         # 页面还活着的唯一凭据，见 confirm()
        self._approved = threading.Event()
        self._ok = False
        self._lock = threading.Lock()

    def note_poll(self):
        self.last_poll = time.time()

    def start(self) -> bool:
        """抢占运行权。已经在跑就返回 False——否则两个 agent 会同时改同一批文件。"""
        with self._lock:
            if self.running:
                return False
            self.running = True
            return True

    def confirm(self, action, poll_gap=30.0, tick=1.0):
        """在 agent 线程里调用，阻塞等前端点确认。

        不能无限等：用户直接关掉页面时没人会来应答，线程就永久挂在这里，
        running 一直是 True，之后每次 /send 都 409，只能重启进程。页面每秒
        轮询一次，所以"超过 poll_gap 没有轮询"就是页面没了，按拒绝处理收场——
        拒绝会当成工具结果喂回模型，任务能自己跑完并释放运行权。
        """
        # 每次确认给一个递增 id：页面靠它认出"这是新的一次确认"。少了 id 它只能
        # 记"当前画着一张卡没有"，而模型一轮里连发两个 write_file 时，两次确认
        # 之间 pending 变空的窗口只有一次写盘那么长，页面 1 秒轮询根本看不到，
        # 于是第二张卡永远不画，agent 线程永久卡在下面的 wait 上。
        #
        # clear 必须在挂 pending 之前：反过来的话，前端在这两行之间抢答一次，
        # resolve 的 set 会被随后的 clear 抹掉，同样是永久阻塞。
        self._approved.clear()
        self._seq += 1
        self.pending = dict(action, id=self._seq)
        try:
            while not self._approved.wait(tick):
                if time.time() - self.last_poll > poll_gap:
                    self._ok = False         # 页面失联，当作用户拒绝
                    break
        finally:
            self.pending = None
        return self._ok

    def resolve(self, ok: bool):
        self._ok = ok
        self._approved.set()

    def drain(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get())
        return out


def create_app(cfg, llm=None, store=None, provider=None) -> FastAPI:
    store = store or Store(str(cfg.db_path))

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # sqlite 连接不关的话，Windows 上 agent.db 会一直被占着，删不掉也移不走。
        # agent 跑在 daemon 线程里，进程退出时它可能还握着这个 store——但那条线
        # 走到底也只是写失败，比让文件一直被占住强。
        store.close()

    app = FastAPI(title="win-ai-agent", lifespan=lifespan)
    provider = provider or build_provider(cfg)
    the_llm = llm or LLMClient(cfg)
    sess = Session()
    app.state.provider = provider          # 让测试能验证默认这条路建对了没
    app.state.session = sess

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
            # status 让回放能还原「这步被拒绝了」，不必再去看 content 的开头几个字
            store.add_message(sess.session_id, "tool", ev.get("result", ""),
                              tool_calls=ev.get("name", ""),
                              status="" if ev.get("ok", True) else "rejected")
        else:                                    # assistant / final
            store.add_message(sess.session_id, t, ev.get("content", ""),
                              status="" if ev.get("ok", True) else "error")

    def _work(goal: str):
        try:
            sess.session_id = store.create_session(goal[:40] or "未命名会话")
            store.add_message(sess.session_id, "user", goal)
            tools = build_tools(cfg, store, provider)
            run_agent(goal, llm=the_llm, tools=tools, cfg=cfg,
                      emit=_emit, confirm=sess.confirm,
                      memories=store.get_memories())
        except Exception as e:
            # 走 _emit 而不是直接塞队列：不落库的话页面上闪一下就没了，
            # 下次打开回放的是数据库，用户根本不知道上次为什么没有结果。
            _emit({"type": "final", "content": f"出错：{type(e).__name__}: {e}", "ok": False})
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
        sess.note_poll()                   # 确认闸靠这个判断页面还在不在
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
        msgs = [{"role": m["role"], "content": m["content"], "tool": m["tool_calls"] or "",
                 "status": m["status"] or ""}
                for m in store.get_messages(s["id"])]
        return {"title": s["title"], "created_at": s.get("created_at", ""), "messages": msgs}

    @app.get("/memories")
    def memories():
        return {"memories": store.get_memories()}

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    return app
