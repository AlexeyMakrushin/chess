# Chess Neural Network Engine

Два UCI-режима на Python:
- **s-chess (SIMPLE)** — материальная оценка + простые piece-square таблицы (PST) и дебютная рандомизация среди топ-ходов.
- **m-chess (CUSTOM_NN)** — MLP (768→256→64→1, ~213k параметров) обученная на 50k позициях Lichess.

## Установка
```bash
# 1) Homebrew (если нет)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 2) uv и окружение
brew install uv
uv venv
source .venv/bin/activate
# 3) Зависимости
uv pip install python-chess torch numpy datasets tqdm
```

## Структура (ключевое)
```
main.py           # UCI интерфейс
config.py         # Параметры поиска/обучения/рандома
engine.py         # Minimax/alpha-beta + дебютная рандомизация
evaluator.py      # s-chess (PST) и m-chess (MLP)
encoder.py        # FEN/board -> вектор 768
model.py          # Архитектура MLP
prepare_data.py   # Скачивание/подготовка Lichess датасета
train.py          # Обучение MLP, сохраняет models/m-chess.pth
record_game.py    # Самоигра с PGN в games/
bin/s-engine      # UCI-лаунчер s-chess
bin/m-engine      # UCI-лаунчер m-chess
data/             # training_data.npz
models/           # m-chess.pth
games/            # PGN партии
```

## Запуск движка
- s-chess: `bin/s-engine`
- m-chess: `bin/m-engine`
- Smoke-тест UCI: `printf 'uci\nisready\nposition startpos\ngo\nquit\n' | bin/s-engine`

## Подготовка данных и обучение (m-chess)
```bash
.venv/bin/python prepare_data.py   # data/training_data.npz
.venv/bin/python train.py          # models/m-chess.pth (state_dict)
```

## Banksia GUI
- Add Engine → Protocol: UCI.
- Command (s-chess): `/Users/alexeymakrushin/Services/chess/bin/s-engine`
- Command (m-chess): `/Users/alexeymakrushin/Services/chess/bin/m-engine`
- Working folder: `/Users/alexeymakrushin/Services/chess`
- Test → должно ответить `uciok`, можно играть на доске.

## PGN
- Самоигра с таймстампом:  
  `.venv/bin/python record_game.py --mode simple --max-fullmoves 100`  
  `.venv/bin/python record_game.py --mode custom --max-fullmoves 100`
- Смотреть: импорт `games/*.pgn` на lichess.org (Tools → Import game) или любой PGN-вьюер.

## Конфиг (config.py, главное)
- `EVALUATION_MODE` (по умолчанию SIMPLE; можно через env).
- `CUSTOM_MODEL_PATH = models/m-chess.pth`.
- Поиск: `MINIMAX_DEPTH=4`, `USE_ALPHA_BETA=True`.
- Рандом в дебюте: `RANDOM_MOVE_CHANCE=0.2`, `RANDOMIZE_OPENINGS_UNTIL=3`, `RANDOM_TOP_K=10`, затухает при высокой оценке.
- Обучение: `DATASET_SIZE=50000`, `BATCH_SIZE=64`, `EPOCHS=15`, `LEARNING_RATE=0.001`.

## Задача и решение
Цель: воспроизводимый UCI-движок, готовый для локальной игры (Banksia) и экспериментов с обучаемой оценкой. Решение: минималистичный Python-стек (python-chess + свой поиск), два режима (s-chess, m-chess), скрипты подготовки/обучения и готовые лаунчеры для GUI/CLI.
