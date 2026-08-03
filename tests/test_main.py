import socket

from main import find_free_port


def test_returns_requested_port_when_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert find_free_port(free) == free


def test_skips_a_port_that_is_bound():
    """connect_ex 探测法在这里会误判：端口被 bind 但没 accept 时连不上，
    却依然 bind 不了，uvicorn 起来照样报错。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    taken = s.getsockname()[1]
    try:
        assert find_free_port(taken) != taken
    finally:
        s.close()


def test_gives_up_with_clear_message(monkeypatch):
    import main

    class AlwaysBusy:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def bind(self, addr): raise OSError("busy")

    monkeypatch.setattr(main.socket, "socket", lambda *a, **k: AlwaysBusy())
    try:
        find_free_port(9000, tries=3)
        assert False, "应该抛 SystemExit"
    except SystemExit as e:
        assert "被占用" in str(e)
