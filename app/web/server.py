from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.loop import AgentRunner
from app.agent.router import route
from app.llm.client import LLMClient
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import build_provider

_STATIC = Path(__file__).parent / "static"


class Session:
    """单会话、单任务串行。agent 在后台线程跑，页面 1 秒轮询。

    确认闸不再阻塞线程：图撞上闸就**返回**，把待确认的操作挂在 ``pending`` 上，
    线程随之结束；页面回答之后另起一个线程从 checkpoint 恢复。所以「等待确认」
    期间是没有线程在跑的——这也意味着用户关掉页面不会再挂住任何东西，
    重新打开还能接着答（之前那套 30 秒失联超时因此不再需要）。

    - ``running``：任务进行中，**含等待确认**。/send 靠它挡住并发任务。
    - ``driving``：当前真的有线程在跑图。防止连点确认把同一个 thread 跑两遍。
    """

    def __init__(self):
        self.events = queue.Queue()
        self.pending = None                  # {"id", "actions": [{"name","preview"}...]}
        self._seq = 0                        # 确认闸的递增编号，页面靠它判重
        self.plan: list[str] = []
        self.plan_current = 0
        self.running = False
        self.driving = False
        self.session_id = None
        self.path = ""                       # 这一轮分诊到哪条路
        self.model = ""                      # 以及实际用的哪档模型
        self.runner = None                   # AgentRunner，start 与 resume 共用同一个
        self._lock = threading.Lock()

    def start(self) -> bool:
        """抢占运行权。已经在跑就返回 False——否则两个 agent 会同时改同一批文件。"""
        with self._lock:
            if self.running:
                return False
            self.running = True
            return True

    def take_wheel(self) -> bool:
        """抢占「正在跑图」这件事，防止连点确认导致同一个 thread 被跑两遍。"""
        with self._lock:
            if self.driving:
                return False
            self.driving = True
            return True

    def set_pending(self, payload: dict):
        self._seq += 1
        self.pending = dict(payload, id=self._seq)

    def drain(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get())
        return out


def create_app(cfg, llm=None, store=None, provider=None) -> FastAPI:
    store = store or Store(str(cfg.db_path))
    # checkpoint 单独一个库文件：它和 agent.db 的锁、生命周期是两回事，混在一起
    # 只会让退出顺序更难理清。SqliteSaver 是上下文管理器，这里手动进入、
    # 在 lifespan 收尾时退出——测试里 TestClient 不带 with 时不跑 lifespan，
    # 所以不能把「进入」也放进去，否则 saver 是空的。
    checkpoint_cm = SqliteSaver.from_conn_string(str(cfg.checkpoint_path))
    checkpointer = checkpoint_cm.__enter__()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # sqlite 连接不关的话，Windows 上 agent.db 会一直被占着，删不掉也移不走。
        # agent 跑在 daemon 线程里，进程退出时它可能还握着这个 store——但那条线
        # 走到底也只是写失败，比让文件一直被占住强。
        store.close()
        checkpoint_cm.__exit__(None, None, None)

    app = FastAPI(title="win-ai-agent", lifespan=lifespan)
    provider = provider or build_provider(cfg)
    sess = Session()

    # 三条路各自一档模型。注入了假模型时三条路共用它——否则每加一条路
    # 就得改一遍所有假件。不配分层的话三档本来也是同一个模型。
    if llm is not None:
        llms = {p: llm for p in ("direct", "fast", "slow")}
    else:
        llms = {}
        for p in ("direct", "fast", "slow"):
            tier = replace(cfg, model=cfg.model_for(p))
            llms[p] = LLMClient(tier)

    app.state.provider = provider          # 让测试能验证默认这条路建对了没
    app.state.session = sess
    app.state.checkpointer = checkpointer
    app.state.llms = llms

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

    def _finish(thread_id: str):
        sess.pending = None
        sess.running = False
        # checkpoint 只用来跨越确认闸，任务结束就没有消费者了——对话记录本来就在
        # agent.db 里。不删的话这个库只增不减，而且没人会去看。
        # 真要做「关掉程序明天接着批」，改的就是这一行。
        try:
            checkpointer.delete_thread(thread_id)
        except Exception:
            pass                            # 清理失败不该影响任务本身的收场

    def _drive(step, thread_id: str):
        """在后台线程里把图推进到下一个停点：要么任务结束，要么撞上确认闸。

        撞上闸时线程就结束了——等待确认期间没有任何线程挂着。
        """
        def body():
            try:
                r = step()
                if r["done"]:
                    _finish(thread_id)
                else:
                    sess.set_pending(r["pending"])
            except Exception as e:
                # 走 _emit 而不是直接塞队列：不落库的话页面上闪一下就没了，
                # 下次打开回放的是数据库，用户根本不知道上次为什么没有结果。
                _emit({"type": "final", "content": f"出错：{type(e).__name__}: {e}", "ok": False})
                _finish(thread_id)
            finally:
                sess.driving = False

        threading.Thread(target=body, daemon=True).start()

    @app.post("/send")
    def send(body: dict):
        goal = (body or {}).get("goal", "").strip()
        if not goal:
            return JSONResponse({"ok": False, "error": "目标不能为空"}, status_code=400)
        if not sess.start():
            return JSONResponse({"ok": False, "error": "上一个任务还在跑，等它结束或回答待确认的操作"},
                                status_code=409)
        sess.plan = []
        sess.plan_current = 0
        sess.pending = None
        sess.take_wheel()
        # 分诊：纯关键词，零 LLM 成本。目前 fast 与 slow 走的是同一套 ReAct，
        # 区别只在用哪档模型——规划+反思那条路是后面的阶段。
        sess.path = route(goal)
        sess.model = cfg.model_for(sess.path)
        sess.session_id = store.create_session(goal[:40] or "未命名会话")
        store.add_message(sess.session_id, "user", goal)
        # ToolSet 与 runner 整轮共用：plan/plan_current 挂在 ToolSet 上，
        # 每次恢复都新建的话计划面板会在确认之后凭空清空。
        sess.runner = AgentRunner(llm=llms[sess.path], tools=build_tools(cfg, store, provider),
                                  cfg=cfg, emit=_emit, checkpointer=checkpointer)
        thread_id = str(sess.session_id)
        # 只注入最近 N 条：全部记忆是每轮都拼进 system prompt 的，不设上限会一直膨胀。
        # /memories 面板仍然显示全部——限的是注入，不是数据。
        memories = store.get_memories(limit=cfg.max_memories)
        _drive(lambda: sess.runner.start(goal, memories, thread_id), thread_id)
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
            # 分诊结果要看得见——路由判错了不暴露的话，用户只会觉得"它今天有点笨"
            "path": sess.path,
            "model": sess.model or cfg.model,
            "max_steps": cfg.max_steps,
        }

    @app.post("/approve")
    def approve(body: dict):
        if sess.pending is None:
            return JSONResponse({"ok": False, "error": "当前没有待确认的操作"}, status_code=409)
        if not sess.take_wheel():          # 连点两次确认不能把同一个 thread 跑两遍
            return JSONResponse({"ok": False, "error": "正在处理上一次确认"}, status_code=409)
        ok = bool((body or {}).get("ok"))
        sess.pending = None
        thread_id = str(sess.session_id)
        _drive(lambda: sess.runner.resume(ok, thread_id), thread_id)
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
        # 面板显示全部；注入提示词的那份另有上限，见 /send
        return {"memories": store.get_memories()}

    @app.delete("/memories/{memory_id}")
    def forget(memory_id: int):
        """删掉一条记忆。

        agent 没有删除工具，这是**唯一**的删除入口——没有它，模型记错一次就
        永久错下去，用户只能去改库。
        """
        if not store.delete_memory(memory_id):
            return JSONResponse({"ok": False, "error": "没有这条记忆"}, status_code=404)
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    return app
