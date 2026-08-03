from __future__ import annotations

import sqlite3
import threading

class Store:
    """SQLite 单文件存储（会话、消息、长期记忆）。

    连接用 ``check_same_thread=False`` 跨线程共享——agent 在后台线程里写，
    web 层在另一个线程读。所以每次用都得过同一把锁：没有锁的话，主线程
    ``close()`` 撞上后台线程正在写，sqlite 会在 C 层踩空，进程带
    0xC0000005 退出，**不是**抛一个能捕获的异常。

    关掉之后的调用一律丢弃（写）或返回空（读），不抛异常。close() 只发生在
    退出的时候，而 agent 是守护线程，未必正好停在两步之间；那几条消息已经
    没有意义了，但不能让它抛异常把线程的收尾流程炸掉。
    """

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._closed = False
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
          role TEXT, content TEXT, tool_calls TEXT, tool_results TEXT,
          status TEXT DEFAULT '',
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS memory(
          id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        # 老库的 messages 表是没有 status 列的，而 CREATE TABLE IF NOT EXISTS
        # 只管建表、不管加列，所以这里显式补一次。DEFAULT '' 让已有记录也拿到
        # 空串而不是 NULL——页面对这批老记录仍走"看开头几个字"的兜底。
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(messages)")}
        if "status" not in cols:
            self.conn.execute("ALTER TABLE messages ADD COLUMN status TEXT DEFAULT ''")
        self.conn.commit()

    def close(self):
        """Windows 上不关连接就一直占着 db 文件，删不掉也移不走。
        拿着锁关，等在途的读写做完；重复调用安全。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.conn.close()

    def create_session(self, title: str) -> int:
        with self._lock:
            if self._closed:
                return 0
            cur = self.conn.execute("INSERT INTO sessions(title) VALUES(?)", (title,))
            self.conn.commit()
            return cur.lastrowid

    def latest_session(self) -> dict:
        """最近一次会话。页面重新打开时用它恢复上一轮的聊天记录。"""
        with self._lock:
            if self._closed:
                return {}
            r = self.conn.execute(
                "SELECT id,title,created_at FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
            return dict(r) if r else {}

    def add_message(self, session_id, role, content, tool_calls="", tool_results="", status=""):
        with self._lock:
            if self._closed:
                return
            self.conn.execute(
                "INSERT INTO messages(session_id,role,content,tool_calls,tool_results,status)"
                " VALUES(?,?,?,?,?,?)",
                (session_id, role, content, tool_calls, tool_results, status))
            self.conn.commit()

    def get_messages(self, session_id) -> list[dict]:
        with self._lock:
            if self._closed:
                return []
            rows = self.conn.execute(
                "SELECT role,content,tool_calls,tool_results,status FROM messages"
                " WHERE session_id=? ORDER BY id",
                (session_id,)).fetchall()
            return [dict(r) for r in rows]

    def add_memory(self, fact: str):
        with self._lock:
            if self._closed:
                return
            self.conn.execute("INSERT INTO memory(fact) VALUES(?)", (fact,))
            self.conn.commit()

    def get_memories(self) -> list[str]:
        with self._lock:
            if self._closed:
                return []
            return [r["fact"] for r in
                    self.conn.execute("SELECT fact FROM memory ORDER BY id").fetchall()]
