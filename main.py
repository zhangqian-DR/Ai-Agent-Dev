import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from app.config import load_config
from app.web.server import create_app


def find_free_port(start: int, tries: int = 50) -> int:
    """真正去 bind 一下再决定。

    只用 connect_ex 探测是不够的：端口可能被别的进程以独占方式绑定、或处于
    TIME_WAIT，此时"连不上"但也 bind 不了，uvicorn 起来照样报错。
    """
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"{start} 起往后 {tries} 个端口都被占用了，改 config.json 里的 port 试试。")


def main():
    cfg_path = Path(__file__).parent / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"找不到配置文件：{cfg_path}\n从 config.example.json 复制一份再填 api_key。")

    cfg = load_config(str(cfg_path))
    port = find_free_port(cfg.port)
    url = f"http://127.0.0.1:{port}"

    print(f"win-ai-agent 已启动：{url}")
    print(f"  工作目录：{cfg.work_dir}")
    print(f"  数据库  ：{cfg.db_path}")
    print(f"  模型    ：{cfg.model}")
    if not cfg.api_key:
        print("  ⚠️  api_key 是空的，发消息会报错。填好 config.json 再重启。")
    if port != cfg.port:
        print(f"  ⚠️  端口 {cfg.port} 被占用，已改用 {port}。")
    print("  按 Ctrl+C 退出。")

    threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    try:
        uvicorn.run(create_app(cfg), host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)


if __name__ == "__main__":
    main()
