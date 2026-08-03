from app.tools.fs import list_dir, read_file

def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = list_dir(tmp_path, ".")
    assert "a.txt" in out and "sub" in out

def test_read_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    assert "hello" in read_file(tmp_path, "a.txt")

def test_read_binary_rejected(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    assert "二进制" in read_file(tmp_path, "b.bin")

def test_read_binary_extension_rejected(tmp_path):
    (tmp_path / "c.png").write_text("not really a png", encoding="utf-8")
    assert "二进制" in read_file(tmp_path, "c.png")

def test_read_small_file_untouched(tmp_path):
    # 用 write_bytes：Windows 上 write_text 会把 \n 转成 \r\n
    (tmp_path / "s.txt").write_bytes(b"just a line\n")
    assert read_file(tmp_path, "s.txt") == "just a line\n"

# max_bytes 与 output_limit 是两回事：前者防内存，后者防超窗

def test_read_output_capped_to_context_budget(tmp_path):
    """文件远小于 max_bytes(1MB) 却远大于上下文预算时必须截断后再返回。
    否则读一个 30KB 的源文件就吃掉整个预算，agent 随即失忆或超窗报错。"""
    big = "\n".join(f"line {i}" for i in range(4000))   # 约 34KB，未超 1MB
    (tmp_path / "big.py").write_text(big, encoding="utf-8")
    out = read_file(tmp_path, "big.py")                 # 全用默认值
    assert "截断" in out
    assert len(out) < 10000, f"返回了 {len(out)} 字符，未受 output_limit 约束"

def test_read_minified_single_line_capped(tmp_path):
    """按行取前 20 + 后 20 行对压缩文件无效——整个文件就一行，
    必须还有一道字符级硬上限兜底。"""
    (tmp_path / "app.min.js").write_text("var x=1;" * 8000, encoding="utf-8")
    out = read_file(tmp_path, "app.min.js", output_limit=2000)
    assert len(out) < 2500, f"单行文件未被截断，返回了 {len(out)} 字符"

def test_read_respects_custom_output_limit(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(str(i) for i in range(1000)), encoding="utf-8")
    out = read_file(tmp_path, "big.txt", output_limit=100)
    assert "截断" in out
    assert len(out) < 400

def test_read_oversized_file_is_rejected_not_truncated(tmp_path):
    """超过 max_bytes 的文件直接拒读——这是内存保护，不是截断。"""
    (tmp_path / "huge.log").write_text("x" * 5000, encoding="utf-8")
    out = read_file(tmp_path, "huge.log", max_bytes=1000)
    assert "过大" in out and "拒绝" in out
    assert "xxx" not in out          # 内容一个字都不该出现

from app.tools.fs import preview_write, write_file

def test_preview_write_shows_diff(tmp_path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    d = preview_write(tmp_path, "a.txt", "new\n")
    assert "-old" in d and "+new" in d

def test_write_file_atomic_and_backup(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("old\n", encoding="utf-8")
    write_file(tmp_path, "a.txt", "new\n")
    assert f.read_text(encoding="utf-8") == "new\n"
    assert (tmp_path / "a.txt.bak").read_text(encoding="utf-8") == "old\n"

def test_write_new_file_no_backup(tmp_path):
    write_file(tmp_path, "n.txt", "hi")
    assert (tmp_path / "n.txt").read_text(encoding="utf-8") == "hi"
    assert not (tmp_path / "n.txt.bak").exists()

def test_write_leaves_no_tmp_file(tmp_path):
    write_file(tmp_path, "a.txt", "hi")
    assert not list(tmp_path.glob("*.tmp"))

def test_write_rejects_oversized_content(tmp_path):
    """单次写入的内容上限。注意它挡不住上下文膨胀（content 来自模型输出，
    进历史时 token 已经花掉），作用是给模型明确的分块反馈并挡住畸形大文件。"""
    out = write_file(tmp_path, "big.txt", "x" * 5000, max_chars=1000)
    assert "过长" in out and "分块" in out
    assert not (tmp_path / "big.txt").exists()   # 拒绝后不能落盘

def test_registry_style_positional_call(tmp_path):
    """Task 11 的 registry 按位置传这几个参数，签名顺序不能改。"""
    (tmp_path / "a.txt").write_bytes(b"hi\n")
    assert read_file(tmp_path, "a.txt", 1_000_000, 8_000) == "hi\n"
    assert "已写入" in write_file(tmp_path, "b.txt", "x", 64_000)

from app.tools.fs import search_in_files

def test_search_finds_and_ignores(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    ign = tmp_path / "node_modules"; ign.mkdir()
    (ign / "b.py").write_text("def foo(): pass\n", encoding="utf-8")
    out = search_in_files(tmp_path, "foo")
    assert "a.py" in out
    assert "node_modules" not in out   # 被忽略
