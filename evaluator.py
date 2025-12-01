"""
Evaluation strategies for chess positions.
"""

import chess
import config
from encoder import encode_board


class SimpleEvaluator:
    """Material-only evaluation."""

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    # Simple piece-square tables (middle-game-ish, centipawn-ish but kept small).
    PST = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 5, 5, 5, 5, 5, 5, 5,
            1, 1, 2, 3, 3, 2, 1, 1,
            0, 0, 0, 2, 2, 0, 0, 0,
            0, 0, 0, 3, 3, 0, 0, 0,
            1, -1, -2, 0, 0, -2, -1, 1,
            1, 2, 2, -2, -2, 2, 2, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        chess.KNIGHT: [
            -5, -4, -2, -2, -2, -2, -4, -5,
            -4, 0, 0, 0, 0, 0, 0, -4,
            -2, 0, 1, 2, 2, 1, 0, -2,
            -2, 1, 2, 3, 3, 2, 1, -2,
            -2, 0, 2, 3, 3, 2, 0, -2,
            -2, 1, 2, 2, 2, 2, 1, -2,
            -4, 0, 1, 0, 0, 1, 0, -4,
            -5, -4, -2, -2, -2, -2, -4, -5,
        ],
        chess.BISHOP: [
            -2, -1, -1, -1, -1, -1, -1, -2,
            -1, 1, 0, 0, 0, 0, 1, -1,
            -1, 0, 2, 1, 1, 2, 0, -1,
            -1, 1, 1, 2, 2, 1, 1, -1,
            -1, 1, 1, 2, 2, 1, 1, -1,
            -1, 0, 2, 1, 1, 2, 0, -1,
            -1, 1, 0, 0, 0, 0, 1, -1,
            -2, -1, -1, -1, -1, -1, -1, -2,
        ],
        chess.ROOK: [
            0, 0, 0, 1, 1, 0, 0, 0,
            -1, 0, 0, 0, 0, 0, 0, -1,
            -1, 0, 0, 0, 0, 0, 0, -1,
            -1, 0, 0, 0, 0, 0, 0, -1,
            -1, 0, 0, 0, 0, 0, 0, -1,
            -1, 0, 0, 0, 0, 0, 0, -1,
            1, 2, 2, 2, 2, 2, 2, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        ],
        chess.QUEEN: [
            -2, -1, -1, -0, -0, -1, -1, -2,
            -1, 0, 0, 0, 0, 0, 0, -1,
            -1, 0, 1, 1, 1, 1, 0, -1,
            -0, 0, 1, 1, 1, 1, 0, -0,
            -0, 0, 1, 1, 1, 1, 0, -0,
            -1, 0, 1, 1, 1, 1, 0, -1,
            -1, 0, 0, 0, 0, 0, 0, -1,
            -2, -1, -1, -0, -0, -1, -1, -2,
        ],
        chess.KING: [
            -3, -4, -4, -5, -5, -4, -4, -3,
            -3, -4, -4, -5, -5, -4, -4, -3,
            -3, -4, -4, -5, -5, -4, -4, -3,
            -3, -4, -4, -5, -5, -4, -4, -3,
            -2, -3, -3, -4, -4, -3, -3, -2,
            -1, -2, -2, -2, -2, -2, -2, -1,
            2, 2, 0, 0, 0, 0, 2, 2,
            2, 3, 1, 0, 0, 1, 3, 2,
        ],
    }

    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            value = self.PIECE_VALUES[piece.piece_type]
            # Add piece-square bonus
            pst = self.PST[piece.piece_type]
            idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            value += pst[idx] / 10.0  # keep impact small
            score += value if piece.color == chess.WHITE else -value
        return score


class CustomNNEvaluator:
    """Custom MLP evaluator backed by a trained PyTorch model."""

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for CUSTOM_NN mode.") from exc

        self.torch = torch
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        from model import ChessEvaluatorMLP

        try:
            checkpoint = torch.load(config.CUSTOM_MODEL_PATH, map_location=self.device)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Model file not found at {config.CUSTOM_MODEL_PATH}. Train it via train.py first."
            ) from exc
        self.model = ChessEvaluatorMLP().to(self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def evaluate(self, board: chess.Board) -> float:
        tensor = encode_board(board)
        x = self.torch.tensor(tensor, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            out = self.model(x)
        # Map tanh output [-1,1] to a rough pawn-scale for search stability.
        return float(out.item() * 4.0)


class EvaluatorFactory:
    """Factory to return evaluator for configured mode."""

    @staticmethod
    def create():
        if config.EVALUATION_MODE == "SIMPLE":
            return SimpleEvaluator()
        if config.EVALUATION_MODE == "CUSTOM_NN":
            return CustomNNEvaluator()
        raise ValueError(f"Unknown evaluation mode: {config.EVALUATION_MODE}")
