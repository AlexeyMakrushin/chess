"""
Self-play game recorder that saves PGNs with a timestamp to games/.

Usage:
  .venv/bin/python record_game.py --mode simple --max-fullmoves 200
  .venv/bin/python record_game.py --mode custom --max-fullmoves 200
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import chess
import chess.pgn

import config
from engine import ChessEngine


def ensure_games_dir() -> pathlib.Path:
    games_dir = pathlib.Path("games")
    games_dir.mkdir(parents=True, exist_ok=True)
    return games_dir


def play_self(engine: ChessEngine, max_fullmoves: int) -> chess.pgn.Game:
    board = chess.Board()
    while not board.is_game_over() and board.fullmove_number <= max_fullmoves:
        move = engine.get_best_move(board)
        if move is None:
            break
        board.push(move)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Self-play"
    game.headers["White"] = "m-chess"
    game.headers["Black"] = "m-chess"
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-play and save PGN to games/.")
    parser.add_argument("--mode", choices=["simple", "custom"], default="simple", help="Evaluator mode.")
    parser.add_argument("--max-fullmoves", type=int, default=200, help="Stop after N fullmoves.")
    args = parser.parse_args()

    # Set mode dynamically
    config.EVALUATION_MODE = "CUSTOM_NN" if args.mode == "custom" else "SIMPLE"
    games_dir = ensure_games_dir()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    engine = ChessEngine()
    game = play_self(engine, max_fullmoves=args.max_fullmoves)
    filename = games_dir / f"selfplay_{args.mode}_{timestamp}.pgn"
    with open(filename, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    print(f"Saved PGN to {filename}")


if __name__ == "__main__":
    main()
