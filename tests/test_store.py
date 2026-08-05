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


# ---------- 记忆 ----------

def test_memory_round_trips_with_polarity(tmp_path):
    """正向和负向要分得开——「要用 tabs」和「不要用 tabs」在提示词里长得一样，
    模型很容易把那个「不」读漏。"""
    s = Store(str(tmp_path / "m1.db"))
    s.add_memory("用户用 Java")
    s.add_memory("不要用 tab 缩进", is_negative=True)

    got = s.get_memories()
    assert [m["fact"] for m in got] == ["用户用 Java", "不要用 tab 缩进"]
    assert [m["is_negative"] for m in got] == [False, True]


def test_duplicate_memory_is_not_stored_twice(tmp_path):
    """模型会反复记同一件事。不去重的话同一条会在提示词里出现好几遍，
    而且永远没有办法收敛。"""
    s = Store(str(tmp_path / "m2.db"))
    assert s.add_memory("用户用 Java") is True
    assert s.add_memory("  用户用   Java  ") is False, "空白差异不算新记忆"
    assert s.add_memory("用户用 JAVA") is False, "大小写差异不算新记忆"

    assert len(s.get_memories()) == 1


def test_same_text_with_different_polarity_is_a_new_memory(tmp_path):
    """极性相反就是两条不同的事实，不能当重复吞掉。"""
    s = Store(str(tmp_path / "m3.db"))
    assert s.add_memory("用 tab 缩进") is True
    assert s.add_memory("用 tab 缩进", is_negative=True) is True
    assert len(s.get_memories()) == 2


def test_memory_injection_is_capped(tmp_path):
    """全部记忆是无条件拼进每一次 system prompt 的，不设上限就会一直膨胀。
    上限只作用于「注入多少」，库里一条都不删——删用户的东西得由用户决定。"""
    s = Store(str(tmp_path / "m4.db"))
    for i in range(60):
        s.add_memory(f"事实 {i}")

    latest = s.get_memories(limit=50)
    assert len(latest) == 50
    assert latest[-1]["fact"] == "事实 59", "要保留最近的"
    assert latest[0]["fact"] == "事实 10"
    assert len(s.get_memories()) == 60, "库里一条都不该少"


def test_memories_carry_an_id(tmp_path):
    """要能删就得能指名道姓，光有文本不行——同一句话可能记过又删过。"""
    s = Store(str(tmp_path / "m5.db"))
    s.add_memory("用户用 Java")
    assert isinstance(s.get_memories()[0]["id"], int)


def test_delete_memory(tmp_path):
    s = Store(str(tmp_path / "m6.db"))
    s.add_memory("记错的一条")
    s.add_memory("对的一条")
    wrong = s.get_memories()[0]["id"]

    assert s.delete_memory(wrong) is True
    assert [m["fact"] for m in s.get_memories()] == ["对的一条"]


def test_delete_unknown_memory_returns_false(tmp_path):
    s = Store(str(tmp_path / "m7.db"))
    assert s.delete_memory(9999) is False


def test_deleted_memory_can_be_added_again(tmp_path):
    """这才是真正的用户故事：记错了 → 删掉 → 重新记对的。
    去重不能把「删过之后再记」也一并挡住。"""
    s = Store(str(tmp_path / "m8.db"))
    s.add_memory("用户用 Python")
    s.delete_memory(s.get_memories()[0]["id"])

    assert s.add_memory("用户用 Python") is True
    assert len(s.get_memories()) == 1


def test_memory_columns_added_to_an_old_db(tmp_path):
    """老库的 memory 表只有 id/fact/created_at，得补列且不碰已有记录。"""
    db = tmp_path / "old_mem.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
    CREATE TABLE memory(id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP);""")
    conn.execute("INSERT INTO memory(fact) VALUES('祖传的一条记忆')")
    conn.commit()
    conn.close()

    s = Store(str(db))
    got = s.get_memories()
    assert [m["fact"] for m in got] == ["祖传的一条记忆"]
    assert got[0]["is_negative"] is False


# ---------- 分诊依据 ----------

def test_session_records_how_it_was_routed(tmp_path):
    """判错了得留下痕迹。不然用户只会觉得「它今天有点笨」，而我们连
    「兜底那层被触发过多少次」都答不上来。"""
    s = Store(str(tmp_path / "r1.db"))
    sid = s.create_session("分析一下", path="slow", route_reason="analyze_wide")

    got = s.latest_session()
    assert got["id"] == sid
    assert got["path"] == "slow" and got["route_reason"] == "analyze_wide"


def test_route_stats_counts_by_reason(tmp_path):
    """跑一段时间之后要能直接问：各条规则各判了多少次。"""
    s = Store(str(tmp_path / "r2.db"))
    for reason in ("chitchat", "fallback", "fallback", "analyze_wide"):
        s.create_session("x", path="fast", route_reason=reason)

    assert s.route_stats() == {"fallback": 2, "analyze_wide": 1, "chitchat": 1}


def test_route_columns_added_to_an_old_db(tmp_path):
    """老库的 sessions 表没有这两列，补列且不碰已有记录。"""
    db = tmp_path / "old_sess.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
    CREATE TABLE sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP);""")
    conn.execute("INSERT INTO sessions(title) VALUES('祖传会话')")
    conn.commit()
    conn.close()

    s = Store(str(db))
    got = s.latest_session()
    assert got["title"] == "祖传会话"
    assert got["path"] == "" and got["route_reason"] == ""


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
