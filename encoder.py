"""
Board/FEN encoding utilities.
"""

import numpy as np
import chess


def encode_board(board: chess.Board) -> np.ndarray:
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    tensor = np.zeros(768, dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        piece_idx = piece_types.index(piece.piece_type)
        color_offset = 0 if piece.color == chess.WHITE else 384
        index = color_offset + piece_idx * 64 + square
        tensor[index] = 1.0

    return tensor


def encode_fen(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    return encode_board(board)
