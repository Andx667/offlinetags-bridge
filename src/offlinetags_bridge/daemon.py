"""Local HTTP daemon that keeps the models loaded between images.

Binds to 127.0.0.1 on a free port and writes the port plus a random token to
the state file. Every request must carry that token, so other local programs
and web pages cannot make the daemon read or write files.

    GET  /health    -> {"status": "ok", ...}
    POST /process   {"input": path, "output": path, "annotate": bool, "jpeg_quality": int}
    POST /shutdown
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .paths import log_file, state_dir, state_file

TOKEN_HEADER = "X-OfflineTags-Token"
MAX_BODY = 1 << 20

log = logging.getLogger("offlinetags_bridge")


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, pipeline, token: str) -> None:
        super().__init__(address, Handler)
        self.pipeline = pipeline
        self.token = token
        self.last_activity = time.monotonic()

    def process(self, request: dict) -> dict:
        # imported here so tests can run the server with a stub pipeline and no OpenCV
        from .pipeline import read_image, write_image

        started = time.perf_counter()
        img = read_image(request["input"])
        result = self.pipeline.process(img, annotate=request.get("annotate", True))
        response = result.to_dict()
        response["output"] = None
        if result.processed is not None and request.get("output"):
            write_image(request["output"], result.processed, int(request.get("jpeg_quality", 92)))
            response["output"] = request["output"]
        response["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return response


class Handler(BaseHTTPRequestHandler):
    server: BridgeServer

    def log_message(self, fmt, *args) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        self.server.last_activity = time.monotonic()
        if hmac.compare_digest(self.headers.get(TOKEN_HEADER, ""), self.server.token):
            return True
        self._reply(403, {"error": "missing or wrong token"})
        return False

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path == "/health":
            self._reply(200, {"status": "ok", "version": __version__, "pid": os.getpid()})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path == "/shutdown":
            self._reply(200, {"status": "stopping"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != "/process":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                raise ValueError("bad request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request.get("input"), str):
                raise ValueError("'input' must be a file path")
        except (ValueError, AttributeError) as exc:
            self._reply(400, {"error": str(exc)})
            return
        try:
            response = self.server.process(request)
        except Exception as exc:
            log.exception("processing %s failed", request.get("input"))
            self._reply(500, {"error": str(exc)})
            return
        log.info("%s -> %s (%d ms)", request["input"], response["action"], response["elapsed_ms"])
        self._reply(200, response)
        self.server.last_activity = time.monotonic()


def _idle_watchdog(server: BridgeServer, idle_seconds: float) -> None:
    while True:
        time.sleep(min(30.0, idle_seconds))
        if time.monotonic() - server.last_activity > idle_seconds:
            log.info("idle for %.0f s, stopping", idle_seconds)
            server.shutdown()
            return


def serve(port: int = 0, idle_minutes: float = 30.0, pipeline=None) -> None:
    """Load the models, publish the state file and serve until stopped or idle."""
    if pipeline is None:
        from .pipeline import Pipeline

        started = time.perf_counter()
        pipeline = Pipeline()
        log.info("models loaded from %s in %.1f s", pipeline.model_dir, time.perf_counter() - started)

    server = BridgeServer(("127.0.0.1", port), pipeline, secrets.token_urlsafe(32))
    state = {"port": server.server_address[1], "token": server.token, "pid": os.getpid()}
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # written only once the models are loaded, so a state file means "ready"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)
    log.info("listening on 127.0.0.1:%d (pid %d)", state["port"], state["pid"])

    if idle_minutes > 0:
        threading.Thread(target=_idle_watchdog, args=(server, idle_minutes * 60), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            # leave the file alone if another daemon has replaced us
            if json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
                path.unlink()
        except (OSError, ValueError):
            pass
        log.info("stopped")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="offlinetags_bridge.daemon", description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=0, help="port to bind (default: any free port)")
    parser.add_argument("--idle-minutes", type=float, default=30.0,
                        help="stop after this long without requests; 0 disables (default: 30)")
    args = parser.parse_args(argv)

    state_dir().mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file(), encoding="utf-8")],
    )
    try:
        serve(args.port, args.idle_minutes)
    except Exception:
        log.exception("daemon failed")
        raise


if __name__ == "__main__":
    main()
