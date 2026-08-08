from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen

import chess
import numpy as np

from chess_ai.human.server import GameSession, create_server


class FirstMoveSearch:
    def search(self, board):
        move = next(iter(board.legal_moves))
        return SimpleNamespace(move=move, wdl=np.array([0.2, 0.6, 0.2]), simulations=4, elapsed_s=0.01, pv=[move])


def request_json(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return response.status, json.load(response)


def test_game_session_plays_human_and_engine_moves():
    session = GameSession(FirstMoveSearch())
    state = session.play("e2e4")
    assert state["human_color"] == "white"
    assert state["turn"] == "white"
    assert state["last_engine_move"]["move"] == "g8h6"
    assert len(session.board.move_stack) == 2


def test_http_server_supports_state_new_game_and_move():
    server = create_server(GameSession(FirstMoveSearch()), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/") as response:
            assert response.status == 200
            assert b"ChessModel-XPU" in response.read()
        status, state = request_json(url, "/api/state")
        assert status == 200 and state["human_color"] == "white"
        status, state = request_json(url, "/api/new", {"human_color": "black"})
        assert status == 200 and state["last_engine_move"]["move"] == "g1h3"
        status, state = request_json(url, "/api/move", {"move": "g8h6"})
        assert status == 200 and len(state["legal_moves"]) > 0
    finally:
        server.shutdown(); server.server_close(); thread.join()
