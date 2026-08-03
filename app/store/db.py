from __future__ import annotations

import sqlite3

class Store:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
          role TEXT, content TEXT, tool_calls TEXT, tool_results TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS memory(
          id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        self.conn.commit()

    def create_session(self, title: str) -> int:
        cur = self.conn.execute("INSERT INTO sessions(title) VALUES(?)", (title,))
        self.conn.commit()
        return cur.lastrowid

    def add_message(self, session_id, role, content, tool_calls="", tool_results=""):
        self.conn.execute(
            "INSERT INTO messages(session_id,role,content,tool_calls,tool_results) VALUES(?,?,?,?,?)",
            (session_id, role, content, tool_calls, tool_results))
        self.conn.commit()

    def get_messages(self, session_id) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role,content,tool_calls,tool_results FROM messages WHERE session_id=? ORDER BY id",
            (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_memory(self, fact: str):
        self.conn.execute("INSERT INTO memory(fact) VALUES(?)", (fact,))
        self.conn.commit()

    def get_memories(self) -> list[str]:
        return [r["fact"] for r in self.conn.execute("SELECT fact FROM memory ORDER BY id").fetchall()]
