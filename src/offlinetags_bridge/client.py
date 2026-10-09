"""Command line client for the daemon. Standard library only, so it starts fast.

``offlinetags-bridge`` is the console command. ``offlinetags-irfanview`` is the
windowless variant meant to be registered as an external editor: it takes file
names, reports through message boxes and opens the result in IrfanView.
"""

from __future__ import annotations

import argparse
import glob
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .paths import log_file, state_file

TOKEN_HEADER = "X-OfflineTags-Token"
OUTPUT_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXIT_OK, EXIT_ERROR, EXIT_DISCARDED = 0, 1, 2


class BridgeError(Exception):
    pass


def _read_state() -> dict | None:
    try:
        return json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _call(state: dict, method: str, path: str, payload: dict | None = None, timeout: float = 300) -> dict:
    # http.client ignores proxy environment variables, which urllib would apply to localhost
    conn = http.client.HTTPConnection("127.0.0.1", state["port"], timeout=timeout)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {TOKEN_HEADER: state["token"], "Content-Type": "application/json"}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = json.loads(response.read() or b"{}")
    finally:
        conn.close()
    if response.status != 200:
        raise BridgeError(data.get("error", f"daemon answered HTTP {response.status}"))
    return data


def running_daemon() -> dict | None:
    """Return the state of a daemon that answers its health check, else None."""
    state = _read_state()
    if not state:
        return None
    try:
        _call(state, "GET", "/health", timeout=2)
    except (OSError, ValueError, BridgeError, http.client.HTTPException):
        return None
    return state


def _spawn_daemon() -> subprocess.Popen:
    python = Path(sys.executable)
    windowless = python.with_name("pythonw.exe")
    if windowless.exists():
        python = windowless
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        [str(python), "-m", "offlinetags_bridge.daemon"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, close_fds=True, start_new_session=sys.platform != "win32",
    )


def _log_tail(lines: int = 3) -> str:
    try:
        return "\n".join(log_file().read_text(encoding="utf-8").splitlines()[-lines:])
    except OSError:
        return ""


def ensure_daemon(timeout: float = 90) -> dict:
    """Return the state of a running daemon, starting one if needed."""
    state = running_daemon()
    if state:
        return state
    process = _spawn_daemon()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.2)
        # no pid comparison: a venv's python.exe is a launcher, the daemon is its child
        state = running_daemon()
        if state:
            return state
        if process.poll() is not None:
            raise BridgeError(f"The daemon stopped while starting.\n{_log_tail()}\nLog: {log_file()}")
    raise BridgeError(f"The daemon did not become ready within {timeout:.0f} s. Log: {log_file()}")


def default_output(source: Path) -> Path:
    """photo.jpg -> photo.offlinetags.jpg, next to the original."""
    suffix = source.suffix if source.suffix.lower() in OUTPUT_FORMATS else ".jpg"
    return source.with_name(f"{source.stem}.offlinetags{suffix}")


def process_file(source: Path, output: Path | None = None, annotate: bool = True) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise BridgeError(f"File not found: {source}")
    output = (output or default_output(source)).resolve()
    return _call(ensure_daemon(), "POST", "/process",
                 {"input": str(source), "output": str(output), "annotate": annotate})


def describe(result: dict) -> str:
    tags = ", ".join(sorted({d["tag_type"] for d in result["detections"]})) or "none"
    action = result["action"]
    if action == "discarded":
        return f"'No Photo' tag found - do not share this photo. (tags: {tags})"
    if action == "clean":
        return f"No action needed, nothing written. (tags: {tags})"
    return f"{action}: {result['output']} (tags: {tags})"


# ── console command ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offlinetags-bridge", description="OfflineTags local daemon client")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="detect privacy tags in images and write the processed copies")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, help="output file (single input only)")
    p.add_argument("--in-place", action="store_true", help="overwrite the input file")
    p.add_argument("--no-annotate", action="store_true",
                   help="only blur; do not draw boxes for Tag Me / Upload Me")
    p.add_argument("--json", action="store_true", help="print the daemon's JSON answer per file")

    sub.add_parser("start", help="start the daemon in the background")
    sub.add_parser("status", help="show whether the daemon is running")
    sub.add_parser("stop", help="stop the daemon")
    args = parser.parse_args(argv)

    try:
        if args.command == "process":
            return _cmd_process(args)
        if args.command == "start":
            state = ensure_daemon()
            print(f"running on 127.0.0.1:{state['port']} (pid {state['pid']})")
        elif args.command == "status":
            state = running_daemon()
            print(f"running on 127.0.0.1:{state['port']} (pid {state['pid']})" if state else "not running")
        elif args.command == "stop":
            state = running_daemon()
            if state:
                _call(state, "POST", "/shutdown")
            print("stopped" if state else "not running")
    except (BridgeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def _expand(files: list[Path]) -> list[Path]:
    """Expand wildcards ourselves: PowerShell and cmd pass them through unexpanded."""
    expanded: list[Path] = []
    for item in files:
        if glob.has_magic(str(item)) and not item.exists():
            expanded.extend(Path(match) for match in sorted(glob.glob(str(item))))
        else:
            expanded.append(item)
    return expanded


def _cmd_process(args) -> int:
    if args.output and (len(args.files) > 1 or args.in_place):
        raise BridgeError("--output needs exactly one input file and excludes --in-place")
    code = EXIT_OK
    for source in _expand(args.files):
        output = source if args.in_place else args.output
        try:
            result = process_file(source, output, annotate=not args.no_annotate)
        except (BridgeError, OSError) as exc:
            print(f"{source}: error: {exc}", file=sys.stderr)
            code = EXIT_ERROR
            continue
        print(json.dumps({"input": str(source), **result}) if args.json else f"{source}: {describe(result)}")
        if result["action"] == "discarded" and code == EXIT_OK:
            code = EXIT_DISCARDED
    return code


# ── IrfanView external editor ────────────────────────────────────────────────

MB_OK, MB_ICONERROR, MB_ICONWARNING, MB_ICONINFORMATION = 0x0, 0x10, 0x30, 0x40


def _message_box(text: str, icon: int = MB_ICONINFORMATION) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "OfflineTags", MB_OK | icon)
    else:
        print(text)


def find_irfanview() -> str | None:
    env = os.environ.get("OFFLINETAGS_BRIDGE_VIEWER")
    if env:
        return env
    for base, exe in (("ProgramFiles", "i_view64.exe"), ("ProgramFiles(x86)", "i_view32.exe")):
        candidate = Path(os.environ.get(base, "")) / "IrfanView" / exe
        if candidate.is_file():
            return str(candidate)
    return shutil.which("i_view64") or shutil.which("i_view32")


def _open_in_viewer(path: str) -> None:
    viewer = find_irfanview()
    if viewer:
        subprocess.Popen([viewer, path])
    elif sys.platform == "win32":
        os.startfile(path)


def gui_main(argv: list[str] | None = None) -> int:
    """Entry point for IrfanView: file names in, message boxes and a viewer window out."""
    files = [Path(a) for a in (sys.argv[1:] if argv is None else argv)]
    if not files:
        _message_box("No file given.\n\nRegister this program as an external editor in IrfanView "
                     "(Properties > Miscellaneous) and start it with Shift+1.", MB_ICONWARNING)
        return EXIT_ERROR

    code = EXIT_OK
    notes: list[str] = []
    icon = MB_ICONINFORMATION
    for source in files:
        try:
            result = process_file(source)
        except (BridgeError, OSError) as exc:
            notes.append(f"{source.name}: {exc}")
            icon, code = MB_ICONERROR, EXIT_ERROR
            continue
        if result["output"]:
            _open_in_viewer(result["output"])
        elif result["action"] == "discarded":
            notes.append(f"{source.name}: 'No Photo' tag found - do not share this photo.")
            if icon != MB_ICONERROR:
                icon, code = MB_ICONWARNING, EXIT_DISCARDED
        else:
            notes.append(f"{source.name}: no privacy tags that need action.")
    if notes:
        _message_box("\n".join(notes), icon)
    return code


if __name__ == "__main__":
    sys.exit(main())
