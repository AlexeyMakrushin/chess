"""
Dataset preparation: download Lichess evaluations, encode boards, and save to npz.
"""

import numpy as np
import config


def _lazy_imports():
    try:
        import chess  # noqa: F401
        from datasets import load_dataset  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ImportError as exc:
        raise ImportError("Run `uv pip install python-chess datasets tqdm` to prepare data.") from exc


def normalize_score(cp: int | None, mate: int | None) -> float:
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    if cp is None:
        return 0.0
    return float(np.tanh(cp / 400.0))


def prepare_dataset() -> tuple[np.ndarray, np.ndarray]:
    _lazy_imports()
    import chess
    from datasets import load_dataset
    from tqdm import tqdm
    from encoder import encode_board

    print(f"Loading {config.DATASET_SIZE} positions from HuggingFace...")
    dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    X, y = [], []
    for i, sample in enumerate(tqdm(dataset, total=config.DATASET_SIZE)):
        if i >= config.DATASET_SIZE:
            break
        fen = sample.get("fen")
        cp = sample.get("cp")
        mate = sample.get("mate")
        if not fen:
            continue
        try:
            board = chess.Board(fen)
            tensor = encode_board(board)
            score = normalize_score(cp, mate)
            X.append(tensor)
            y.append(score)
        except Exception:
            continue

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    output_path = f"{config.DATA_DIR}/training_data.npz"
    np.savez_compressed(output_path, X=X_arr, y=y_arr)
    print(f"Prepared {len(X_arr)} positions; saved to {output_path}")
    return X_arr, y_arr


if __name__ == "__main__":
    prepare_dataset()
