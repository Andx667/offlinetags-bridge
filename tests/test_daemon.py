import http.client
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from offlinetags_bridge import client
from offlinetags_bridge.daemon import TOKEN_HEADER, BridgeServer
from offlinetags_bridge.pipeline import Result, read_image, write_image


class StubPipeline:
    """Stands in for the real pipeline so these tests need no model files."""

    def __init__(self, action="blurred"):
        self.action = action

    def process(self, img, annotate=True):
        processed = 255 - img if self.action in ("blurred", "annotated") else None
        return Result(self.action, [], [[0, 0, 4, 4]] if processed is not None else [], processed)


@pytest.fixture
def server():
    srv = BridgeServer(("127.0.0.1", 0), StubPipeline(), "secret")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _state(server, token="secret"):
    return {"port": server.server_address[1], "token": token}


def _photo(tmp_path: Path, name="phöto.png") -> Path:
    path = tmp_path / name
    write_image(path, np.full((8, 8, 3), 10, dtype=np.uint8))
    return path


def test_health(server):
    assert client._call(_state(server), "GET", "/health")["status"] == "ok"


def test_requests_without_the_token_are_rejected(server, tmp_path):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request("POST", "/process", body=json.dumps({"input": str(_photo(tmp_path))}),
                 headers={TOKEN_HEADER: "wrong"})
    assert conn.getresponse().status == 403
    conn.close()


def test_process_writes_output_for_non_ascii_path(server, tmp_path):
    source, output = _photo(tmp_path), tmp_path / "öut.png"
    result = client._call(_state(server), "POST", "/process",
                          {"input": str(source), "output": str(output)})
    assert result["action"] == "blurred"
    assert result["output"] == str(output)
    assert result["blur_regions"] == [[0, 0, 4, 4]]
    assert int(read_image(output)[0, 0, 0]) == 245


def test_discarded_writes_nothing(server, tmp_path):
    server.pipeline = StubPipeline("discarded")
    output = tmp_path / "out.png"
    result = client._call(_state(server), "POST", "/process",
                          {"input": str(_photo(tmp_path)), "output": str(output)})
    assert result["action"] == "discarded" and result["output"] is None
    assert not output.exists()


def test_unreadable_input_is_an_error(server, tmp_path):
    with pytest.raises(client.BridgeError):
        client._call(_state(server), "POST", "/process", {"input": str(tmp_path / "missing.jpg")})


def test_bad_request(server):
    with pytest.raises(client.BridgeError):
        client._call(_state(server), "POST", "/process", {"output": "x"})


def test_default_output_name():
    assert client.default_output(Path("a/photo.JPG")) == Path("a/photo.offlinetags.JPG")
    assert client.default_output(Path("a/raw.cr2")) == Path("a/raw.offlinetags.jpg")


def test_gui_entry_opens_result_and_reports_the_rest(server, tmp_path, monkeypatch):
    opened, boxes = [], []
    monkeypatch.setattr(client, "ensure_daemon", lambda: _state(server))
    monkeypatch.setattr(client, "_open_in_viewer", opened.append)
    monkeypatch.setattr(client, "_message_box", lambda text, icon=0: boxes.append((text, icon)))

    source = _photo(tmp_path)
    assert client.gui_main([str(source)]) == client.EXIT_OK
    assert opened == [str(client.default_output(source))] and boxes == []

    server.pipeline = StubPipeline("discarded")
    assert client.gui_main([str(source)]) == client.EXIT_DISCARDED
    assert "No Photo" in boxes[0][0] and boxes[0][1] == client.MB_ICONWARNING

    assert client.gui_main([str(tmp_path / "missing.jpg")]) == client.EXIT_ERROR
    assert boxes[-1][1] == client.MB_ICONERROR
