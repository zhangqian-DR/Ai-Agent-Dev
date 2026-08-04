from __future__ import annotations

import re
import sqlite3
import threading


def _normalize(text: str) -> str:
    """去重用的规范化：折叠空白 + 忽略大小写。不做同义词判断——
    那需要 embedding，而本地单用户几十条的规模不值当。"""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


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
          is_negative INTEGER DEFAULT 0,
          norm TEXT DEFAULT '',
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        # 老库的 messages 表是没有 status 列的，而 CREATE TABLE IF NOT EXISTS
        # 只管建表、不管加列，所以这里显式补一次。DEFAULT '' 让已有记录也拿到
        # 空串而不是 NULL——页面对这批老记录仍走"看开头几个字"的兜底。
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(messages)")}
        if "status" not in cols:
            self.conn.execute("ALTER TABLE messages ADD COLUMN status TEXT DEFAULT ''")
        mcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(memory)")}
        if "is_negative" not in mcols:
            self.conn.execute("ALTER TABLE memory ADD COLUMN is_negative INTEGER DEFAULT 0")
        if "norm" not in mcols:
            # 去重用的规范化文本。老记录留空串——它们之间不做回填去重，
            # 只保证从现在起不再往里塞重复的。
            self.conn.execute("ALTER TABLE memory ADD COLUMN norm TEXT DEFAULT ''")
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

    def add_memory(self, fact: str, is_negative: bool = False) -> bool:
        """记一条长期事实。已经记过就返回 False，不重复插入。

        模型会反复记同一件事，不去重的话同一条会在提示词里出现好几遍，而且
        永远收敛不了。极性算进 key：「用 tab」和「不要用 tab」是两条不同的事实。
        """
        with self._lock:
            if self._closed:
                return False
            key = _normalize(fact)
            dup = self.conn.execute(
                "SELECT 1 FROM memory WHERE norm=? AND is_negative=? LIMIT 1",
                (key, int(is_negative))).fetchone()
            if dup:
                return False
            self.conn.execute(
                "INSERT INTO memory(fact, is_negative, norm) VALUES(?,?,?)",
                (fact, int(is_negative), key))
            self.conn.commit()
            return True

    def get_memories(self, limit: int = 0) -> list[dict]:
        """按记录顺序返回 ``{"fact", "is_negative"}``。

        ``limit`` 只限制**返回多少**（取最近的 N 条，仍按时间正序），库里一条
        都不删——全部记忆是无条件拼进每一次 system prompt 的，得有个上限；
        但删用户的东西应该由用户决定，不该是注入预算顺手做的事。
        """
        with self._lock:
            if self._closed:
                return []
            rows = self.conn.execute(
                "SELECT id, fact, is_negative FROM memory ORDER BY id").fetchall()
            if limit and len(rows) > limit:
                rows = rows[-limit:]
            return [{"id": r["id"], "fact": r["fact"],
                     "is_negative": bool(r["is_negative"])} for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        """删掉一条记忆，不存在返回 False。

        是**硬删**，不是软删：这个动作只由用户显式触发（agent 没有删除工具），
        用户说删就是真想删掉。记错一条之后能删、能重记，是这条路径的全部意义。
        """
        with self._lock:
            if self._closed:
                return False
            cur = self.conn.execute("DELETE FROM memory WHERE id=?", (memory_id,))
            self.conn.commit()
            return cur.rowcount > 0
