import pytest

from app.store import db


@pytest.fixture(autouse=True)
def close_stores(monkeypatch):
    """每个用例结束后收掉它建的所有 sqlite 连接。

    Windows 上连接不关就一直占着 db 文件，pytest 清理旧的临时目录时会
    PermissionError。测试里有十来处 ``Store(...)``，与其挨个改调用点，
    不如在这里统一登记再统一关——``close()`` 可以重复调用，
    用例自己关过的也不受影响。
    """
    opened = []
    original = db.Store.__init__

    def tracked(self, db_path):
        original(self, db_path)
        opened.append(self)

    monkeypatch.setattr(db.Store, "__init__", tracked)
    yield
    for store in opened:
        store.close()
