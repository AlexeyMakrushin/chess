"""
Search logic for the chess engine (minimax with optional alpha-beta pruning).
"""

import random
import chess
import config
from evaluator import EvaluatorFactory


class ChessEngine:
    """Engine that selects a move using minimax or alpha-beta."""

    def __init__(self) -> None:
        self.evaluator = EvaluatorFactory.create()

    def get_best_move(self, board: chess.Board) -> chess.Move | None:
        if board.is_game_over():
            return None

        maximizing = board.turn == chess.WHITE
        if config.USE_ALPHA_BETA:
            best_move, ranked = self._alpha_beta_search(board, maximizing, return_ranked=True)
        else:
            best_move, ranked = self._minimax_search(board, maximizing, return_ranked=True)

        if (
            board.fullmove_number <= config.RANDOMIZE_OPENINGS_UNTIL
            and random.random() < config.RANDOM_MOVE_CHANCE
            and ranked
        ):
            best_value = ranked[0][1]
            scale = max(0.0, 1 - abs(best_value) / config.RANDOM_DECAY_THRESHOLD)
            chance = config.RANDOM_MOVE_CHANCE * scale
            if random.random() < chance:
                top_k = ranked[: config.RANDOM_TOP_K]
                return random.choice(top_k)[0]

        return best_move

    def _minimax_search(
        self, board: chess.Board, maximizing: bool, return_ranked: bool = False
    ) -> chess.Move | tuple[chess.Move | None, list[tuple[chess.Move, float]]]:
        best_move = None
        best_value = -float("inf") if maximizing else float("inf")
        ranked: list[tuple[chess.Move, float]] = []

        for move in board.legal_moves:
            board.push(move)
            value = self._minimax(board, config.MINIMAX_DEPTH - 1, maximizing=not maximizing)
            board.pop()
            ranked.append((move, value))

            if maximizing and value > best_value:
                best_value, best_move = value, move
            if not maximizing and value < best_value:
                best_value, best_move = value, move

        ranked.sort(key=lambda mv: mv[1], reverse=maximizing)
        return (best_move, ranked) if return_ranked else best_move

    def _minimax(self, board: chess.Board, depth: int, maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluator.evaluate(board)

        if maximizing:
            value = -float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self._minimax(board, depth - 1, maximizing=False))
                board.pop()
            return value

        value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = min(value, self._minimax(board, depth - 1, maximizing=True))
            board.pop()
        return value

    def _alpha_beta_search(
        self, board: chess.Board, maximizing: bool, return_ranked: bool = False
    ) -> chess.Move | tuple[chess.Move | None, list[tuple[chess.Move, float]]]:
        best_move = None
        alpha, beta = -float("inf"), float("inf")
        best_value = -float("inf") if maximizing else float("inf")
        ranked: list[tuple[chess.Move, float]] = []

        for move in board.legal_moves:
            board.push(move)
            value = self._alpha_beta(board, config.MINIMAX_DEPTH - 1, alpha, beta, maximizing=not maximizing)
            board.pop()

            ranked.append((move, value))

            if maximizing and value > best_value:
                best_value, best_move = value, move
                alpha = max(alpha, value)
            elif not maximizing and value < best_value:
                best_value, best_move = value, move
                beta = min(beta, value)

        ranked.sort(key=lambda mv: mv[1], reverse=maximizing)
        return (best_move, ranked) if return_ranked else best_move

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluator.evaluate(board)

        if maximizing:
            value = -float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self._alpha_beta(board, depth - 1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = min(value, self._alpha_beta(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
