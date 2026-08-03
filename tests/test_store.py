import sqlite3
import threading
import time

import pytest

from app.store.db import Store


def test_close_releases_the_file(tmp_path):
    """Windows 上连接不关就一直占着 db 文件，删不掉也移不走——
    pytest 清理临时目录时会 PermissionError。"""
    db = tmp_path / "a.db"
    s = Store(str(db))
    s.add_memory("记一笔")
    s.close()

    db.unlink()                      # 还占用着的话这里就 PermissionError
    assert not db.exists()


def test_close_is_idempotent(tmp_path):
    """关两次不能炸——lifespan 收尾和调用方各关一次是很正常的事。"""
    s = Store(str(tmp_path / "b.db"))
    s.close()
    s.close()


def test_use_after_close_does_not_blow_up(tmp_path):
    """关掉之后还可能来写：agent 是守护线程，退出时它未必正好停在两步之间。
    这时候丢掉就是了——库都要关了，那几条消息没有意义，
    但不能让它抛异常把线程的收尾流程炸掉。"""
    s = Store(str(tmp_path / "d.db"))
    sid = s.create_session("t")
    s.close()

    s.add_message(sid, "tool", "关了之后来的")
    s.add_memory("关了之后来的")
    assert s.get_memories() == []
    assert s.get_messages(sid) == []
    assert s.latest_session() == {}


def test_status_round_trips(tmp_path):
    """页面靠 status 区分「出错的最终回答」和「被拒绝的工具调用」，
    不能再拿 content 的开头几个字去猜。"""
    s = Store(str(tmp_path / "s.db"))
    sid = s.create_session("t")
    s.add_message(sid, "final", "炸了", status="error")
    s.add_message(sid, "final", "好了")

    rows = s.get_messages(sid)
    assert rows[0]["status"] == "error"
    assert rows[1]["status"] == ""


def test_status_column_is_added_to_an_old_db(tmp_path):
    """老库的 messages 表没有 status 列，而 CREATE TABLE IF NOT EXISTS 加不了列——
    得显式补一次，且不能碰已有记录。"""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
    CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
      role TEXT, content TEXT, tool_calls TEXT, tool_results TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP);""")
    conn.execute("INSERT INTO messages(session_id,role,content) VALUES(1,'final','出错：旧记录')")
    conn.commit()
    conn.close()

    s = Store(str(db))
    rows = s.get_messages(1)
    assert len(rows) == 1
    assert rows[0]["content"] == "出错：旧记录"
    assert rows[0]["status"] == ""      # 老记录没有标志，前端对它仍走字符串兜底


def test_close_is_safe_while_another_thread_is_writing(tmp_path):
    """连接是 check_same_thread=False 跨线程共享的，却没有锁。

    agent 跑在后台守护线程里，Ctrl+C 时 lifespan 会在它还在写的当口关连接——
    没有锁的话 sqlite 直接在 C 层踩空，进程带 0xC0000005 退出，**不是**抛异常。
    所以这条用例失败时的表现是整个 pytest 进程没了，不是一行红字。
    """
    s = Store(str(tmp_path / "e.db"))
    sid = s.create_session("t")
    stop = threading.Event()
    errors = []

    def writer():
        while not stop.is_set():
            try:
                s.add_message(sid, "tool", "一直写")
            except Exception as e:      # noqa: BLE001 —— 什么都不该抛
                errors.append(repr(e))
                return

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    time.sleep(0.05)                    # 让它真的写起来，别在空转时就关了
    s.close()
    stop.set()
    t.join(timeout=5)

    assert not errors, errors
