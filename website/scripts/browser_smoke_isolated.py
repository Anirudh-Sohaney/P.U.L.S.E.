"""Run the built browser smoke against disposable local API and frontend processes."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


WEBSITE = Path(__file__).resolve().parents[1]
FRONTEND = WEBSITE / "frontend"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait_ready(url: str, process: subprocess.Popen, log: Path) -> None:
    for _ in range(90):
        if process.poll() is not None:
            raise RuntimeError(f"Service exited before {url} was ready:\n{log.read_text()[-3000:]}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Service did not become ready at {url}:\n{log.read_text()[-3000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-years", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--simulate-rate-limit", action="store_true")
    args = parser.parse_args()
    if not (FRONTEND / "dist/index.html").is_file():
        raise RuntimeError("Build the frontend with npm run build before this smoke test")

    api_port, frontend_port = _free_port(), _free_port()
    while frontend_port == api_port:
        frontend_port = _free_port()
    base_url = f"http://127.0.0.1:{frontend_port}"
    with tempfile.TemporaryDirectory(prefix="pulse-isolated-browser-") as directory:
        temporary = Path(directory)
        environment = os.environ.copy()
        environment.update({
            "DATA_PATH": str(temporary / "data"),
            "ENVIRONMENT": "development",
            "API_HOST": "127.0.0.1",
            "API_PORT": str(api_port),
            "CORS_ORIGINS": json.dumps([base_url]),
            "VITE_API_PROXY_TARGET": f"http://127.0.0.1:{api_port}",
        })
        subprocess.run([sys.executable, "-m", "scripts.import_signal_catalog"],
                       cwd=WEBSITE, env=environment, check=True,
                       stdout=subprocess.DEVNULL)
        api_log = temporary / "api.log"
        frontend_log = temporary / "frontend.log"
        processes = []
        try:
            with api_log.open("w") as api_output, frontend_log.open("w") as frontend_output:
                api = subprocess.Popen([sys.executable, "-m", "backend.main"],
                                       cwd=WEBSITE, env=environment,
                                       stdout=api_output, stderr=subprocess.STDOUT)
                processes.append(api)
                _wait_ready(f"http://127.0.0.1:{api_port}/ready", api, api_log)
                frontend = subprocess.Popen([
                    str(FRONTEND / "node_modules/.bin/vite"), "preview", "--host",
                    "127.0.0.1", "--port", str(frontend_port), "--strictPort"],
                    cwd=FRONTEND, env=environment,
                    stdout=frontend_output, stderr=subprocess.STDOUT)
                processes.append(frontend)
                _wait_ready(base_url + "/", frontend, frontend_log)
                command = [sys.executable, "-m", "scripts.browser_smoke",
                           "--base-url", base_url, "--signal-years", str(args.signal_years)]
                if args.simulate_rate_limit:
                    command.append("--simulate-rate-limit")
                subprocess.run(command, cwd=WEBSITE, env=environment, check=True,
                               timeout=420)
        except Exception:
            for log in (api_log, frontend_log):
                if log.is_file():
                    print(f"--- {log.name} ---\n{log.read_text()[-3000:]}", file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                process.terminate()
            for process in reversed(processes):
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
