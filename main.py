#!/usr/bin/env python3
"""
UCI interface for the chess engine.
"""

import sys
import chess
from engine import ChessEngine
import config


def parse_position(command: str) -> chess.Board:
    parts = command.split()
    board = chess.Board()

    if "startpos" in parts:
        board = chess.Board()
        moves_index = parts.index("moves") if "moves" in parts else None
    elif "fen" in parts:
        fen_start = parts.index("fen") + 1
        moves_index = parts.index("moves") if "moves" in parts else None
        fen_end = moves_index if moves_index else len(parts)
        fen = " ".join(parts[fen_start:fen_end])
        board = chess.Board(fen)
    else:
        return board

    if moves_index:
        for move_uci in parts[moves_index + 1 :]:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
            except ValueError:
                continue

    return board


def uci_loop() -> None:
    board = chess.Board()
    engine = ChessEngine()

    for line in sys.stdin:
        command = line.strip()

        if command == "uci":
            print("id name ChessNN")
            print("id author Contributors")
            print(f"option name EvaluationMode type string default {config.EVALUATION_MODE}")
            print(f"option name MinimaxDepth type spin default {config.MINIMAX_DEPTH} min 1 max 6")
            print("uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            board = chess.Board()
        elif command.startswith("position"):
            board = parse_position(command)
        elif command.startswith("go"):
            best_move = engine.get_best_move(board)
            if best_move:
                print(f"bestmove {best_move.uci()}", flush=True)
            else:
                print("bestmove 0000", flush=True)
        elif command == "quit":
            break
        else:
            sys.stdout.flush()


if __name__ == "__main__":
    uci_loop()
