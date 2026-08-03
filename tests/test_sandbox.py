import pytest

from app.safety.sandbox import SandboxError, resolve_in_sandbox


def test_inside_ok(tmp_path):
    (tmp_path / "a").mkdir()
    p = resolve_in_sandbox(tmp_path, "a/b.txt")
    assert str(p).startswith(str(tmp_path.resolve()))


def test_dotdot_escape_blocked(tmp_path):
    with pytest.raises(SandboxError):
        resolve_in_sandbox(tmp_path, "../secret.txt")


def test_absolute_outside_blocked(tmp_path):
    with pytest.raises(SandboxError):
        resolve_in_sandbox(tmp_path, "/etc/passwd")


def test_sibling_prefix_not_confused(tmp_path):
    # work_dir=tmp/proj 不应放行 tmp/proj2
    proj = tmp_path / "proj"
    proj.mkdir()
    (tmp_path / "proj2").mkdir()
    with pytest.raises(SandboxError):
        resolve_in_sandbox(proj, "../proj2/x.txt")
