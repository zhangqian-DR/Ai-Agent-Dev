from app.agent.prompt import system_prompt


def test_negative_memories_are_marked_differently():
    """「要用 tab」和「不要用 tab」在提示词里必须长得不一样。

    原来所有记忆都是同一种 `- xxx` 的结构，那个「不」字全靠模型自己读到，
    读漏一次就会做出正好相反的事。
    """
    p = system_prompt("C:/ws", [
        {"fact": "用户主要写 Java", "is_negative": False},
        {"fact": "用 tab 缩进", "is_negative": True},
    ])

    neg = [ln for ln in p.splitlines() if "tab" in ln][0]
    pos = [ln for ln in p.splitlines() if "Java" in ln][0]
    assert "禁止" in neg, f"负向记忆没有打标：{neg!r}"
    assert "禁止" not in pos, f"正向记忆被误标：{pos!r}"


def test_memories_without_polarity_default_to_positive():
    """老记录没有极性字段，当正向处理，别整片变成「禁止」。"""
    p = system_prompt("C:/ws", [{"fact": "祖传的一条记忆"}])
    line = [ln for ln in p.splitlines() if "祖传" in ln][0]
    assert "禁止" not in line


def test_no_memories_says_so():
    assert "（暂无）" in system_prompt("C:/ws", [])


def test_work_dir_is_in_the_prompt():
    assert "C:/ws" in system_prompt("C:/ws", [])
