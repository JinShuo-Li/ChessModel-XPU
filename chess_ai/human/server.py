from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

from chess_ai.uci.engine import build_engine


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ChessModel-XPU</title>
  <style>
    body{font:16px system-ui;max-width:640px;margin:32px auto;padding:0 16px;background:#171717;color:#eee}
    pre{font:30px/1.3 monospace;background:#262626;padding:18px;border-radius:8px;overflow:auto}
    input,button{font:inherit;padding:9px;margin:4px 2px} #status{min-height:24px;color:#b8d8ff}
  </style>
</head>
<body>
  <h1>ChessModel-XPU</h1>
  <button onclick="newGame('white')">我执白</button><button onclick="newGame('black')">我执黑</button>
  <pre id="board">loading…</pre><div id="status"></div>
  <form onsubmit="move(event)"><input id="move" placeholder="输入 UCI，例如 e2e4" autocomplete="off"><button>走棋</button></form>
  <script>
    const pieces={K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
    function draw(fen){let rows=fen.split(' ')[0].split('/');return rows.map((r,i)=>`${8-i} `+[...r].map(c=>/\d/.test(c)?'· '.repeat(+c):pieces[c]+' ').join('')).join('\n')+'\n  a b c d e f g h'}
    function show(s){document.querySelector('#board').textContent=draw(s.fen);document.querySelector('#status').textContent=s.game_over?`结束：${s.result}`:`轮到${s.turn==='white'?'白方':'黑方'}；你执${s.human_color==='white'?'白':'黑'}${s.last_engine_move?'；模型：'+s.last_engine_move.move:''}`}
    async function api(path,body){let r=await fetch(path,{method:body?'POST':'GET',headers:{'content-type':'application/json'},body:body?JSON.stringify(body):undefined});let j=await r.json();if(!r.ok)throw Error(j.error);return j}
    async function refresh(){try{show(await api('/api/state'))}catch(e){document.querySelector('#status').textContent=e.message}}
    async function newGame(c){try{show(await api('/api/new',{human_color:c}))}catch(e){alert(e.message)}}
    async function move(e){e.preventDefault();let x=document.querySelector('#move');try{show(await api('/api/move',{move:x.value.trim()}));x.value=''}catch(err){alert(err.message)}}
    refresh();
  </script>
</body></html>"""


class GameSession:
    def __init__(self, search, human_color: str = "white"):
        self.search = search
        self.lock = threading.RLock()
        self.board = chess.Board()
        self.human_color = chess.WHITE
        self.last_engine_move = None
        self.new_game(human_color)

    @staticmethod
    def _parse_color(color: str) -> chess.Color:
        if color not in ("white", "black"):
            raise ValueError("human_color must be 'white' or 'black'")
        return chess.WHITE if color == "white" else chess.BLACK

    def new_game(self, human_color: str = "white") -> dict:
        with self.lock:
            self.board = chess.Board()
            self.human_color = self._parse_color(human_color)
            self.last_engine_move = None
            if self.board.turn != self.human_color:
                self._play_engine_move()
            return self.state()

    def play(self, move_uci: str) -> dict:
        with self.lock:
            if self.board.is_game_over(claim_draw=True):
                raise ValueError("the game is already over")
            if self.board.turn != self.human_color:
                raise ValueError("it is not the human player's turn")
            try:
                move = self.board.parse_uci(move_uci)
            except ValueError as exc:
                raise ValueError(f"illegal UCI move: {move_uci}") from exc
            self.board.push(move)
            self.last_engine_move = None
            if not self.board.is_game_over(claim_draw=True):
                self._play_engine_move()
            return self.state()

    def _play_engine_move(self) -> None:
        result = self.search.search(self.board)
        self.board.push(result.move)
        self.last_engine_move = {
            "move": result.move.uci(),
            "wdl": result.wdl.tolist(),
            "simulations": result.simulations,
            "elapsed_s": result.elapsed_s,
            "pv": [move.uci() for move in result.pv],
        }

    def state(self) -> dict:
        with self.lock:
            game_over = self.board.is_game_over(claim_draw=True)
            return {
                "fen": self.board.fen(),
                "board": str(self.board),
                "turn": "white" if self.board.turn else "black",
                "human_color": "white" if self.human_color else "black",
                "game_over": game_over,
                "result": self.board.result(claim_draw=True) if game_over else "*",
                "legal_moves": [] if game_over else [move.uci() for move in self.board.legal_moves],
                "last_engine_move": self.last_engine_move,
            }


def _handler(session: GameSession):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(200, session.state())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("request body is too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/api/new":
                    result = session.new_game(payload.get("human_color", "white"))
                elif self.path == "/api/move":
                    result = session.play(payload.get("move", ""))
                else:
                    self._json(404, {"error": "not found"}); return
                self._json(200, result)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format, *args):
            return

    return Handler


def create_server(session: GameSession, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler(session))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local browser game backed by a selected checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--leaf-batch-size", type=int, default=64)
    args = parser.parse_args()
    search = build_engine(args.checkpoint, args.device, args.simulations, args.leaf_batch_size)
    server = create_server(GameSession(search), args.host, args.port)
    print(f"ChessModel-XPU human game: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
