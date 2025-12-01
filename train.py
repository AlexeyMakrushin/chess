"""
Training loop for the custom MLP evaluator.
"""

import numpy as np
import config


def _lazy_imports():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import torch.optim as optim  # noqa: F401
        from torch.utils.data import TensorDataset, DataLoader  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ImportError as exc:
        raise ImportError("Run `uv pip install torch tqdm` to train the model.") from exc


def load_data():
    data_path = f"{config.DATA_DIR}/training_data.npz"
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    split_idx = int(len(X) * config.TRAIN_SPLIT)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def train():
    _lazy_imports()
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    from model import ChessEvaluatorMLP

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_train, y_train, X_val, y_val = load_data()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.BATCH_SIZE, shuffle=True)
    model = ChessEvaluatorMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val = float("inf")
    for epoch in range(config.EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device))
            val_loss = criterion(val_preds, y_val_t.to(device)).item()

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict()}, config.CUSTOM_MODEL_PATH)
            print(f"Saved state_dict to {config.CUSTOM_MODEL_PATH}")
        scheduler.step()

    return model


if __name__ == "__main__":
    train()
