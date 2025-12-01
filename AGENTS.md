# Repository Guidelines

## Project Structure & Modules
- Core entrypoint: `main.py` (UCI loop) with configuration in `config.py`; search depth set in `MINIMAX_DEPTH`.
- Search logic: `engine.py` (minimax/alpha-beta + opening randomization).
- Evaluation: `evaluator.py` (s-chess with PST, m-chess with MLP).
- ML stack: `encoder.py`, `model.py`, `prepare_data.py`, `train.py`.
- Binaries: `bin/s-engine` (s-chess) and `bin/m-engine` (m-chess).
- Data/artifacts: `data/training_data.npz`, `models/m-chess.pth`, PGNs in `games/`. Keep large artifacts out of Git if regenerating.

## Environment, Build, and Run
- Requirements: macOS Sonoma+, Python 3.10+, Homebrew. Use `uv` for dependency management only; do not use conda.
- Setup: `uv venv && source .venv/bin/activate`, then `uv pip install python-chess torch numpy datasets tqdm`.
- Run s-chess: `bin/s-engine`; run m-chess: `bin/m-engine`.
- Prepare data: `.venv/bin/python prepare_data.py` (size from `config.py`).
- Train model: `.venv/bin/python train.py` (writes `models/m-chess.pth`).
- Playtests: smoke in terminal (`uci → isready → position → go`) and Banksia GUI (commands above).

## Coding Style & Naming
- Python: 4-space indentation, `snake_case` for variables/functions, `CamelCase` for classes.
- Filenames/modules in kebab-case or lowercase as already present (`main.py`, `config.py`).
- Keep functions small; prefer pure helpers where possible. Add docstrings for public functions/classes and concise inline comments for non-obvious logic.
- Avoid hard-coding absolute paths; read from `config.py`.
- Wrap external I/O or engine invocations in `try/except` with clear error messages.

## Testing Guidelines
- Smoke test every mode after changes: set `EVALUATION_MODE` or use `bin/s-engine` / `bin/m-engine`, then `uci → isready → position → go`.
- For regression checks, run smoke tests and quick Banksia games to ensure нет нелегальных ходов/сбоев.
- When touching training or data prep, spot-check tensor shapes/output ranges; avoid committing regenerated data/models.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (`Add custom NN evaluator`, `Fix UCI position parsing`). Group related edits; exclude bulky artifacts (`data/`, `models/`).
- PRs: include a brief summary, testing notes (commands run), and any relevant config values (depth, evaluation mode). Attach logs or PGN snippets if you tweaked search or evaluation behavior.
