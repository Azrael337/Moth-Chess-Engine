# Moth-Chess-Engine

# ChessCPP Engine

[Brief 1-2 sentence description of your engine]

A UCI-compliant chess engine written in C++ featuring:
- Alpha-Beta Negamax with fail-soft
- Iterative deepening + aspiration windows
- Quiescence search
- Transposition table (Zobrist hashing)
- Null-move pruning & Late Move Reduction
- Tapered evaluation (MG/EG blending)

## 📊 Estimated Strength

[Your estimated rating, e.g., "~2200-2400 Elo"]

## 🚀 Features

List your key features here (from your code):
- Move ordering: hash move, MVV-LVA, killer moves, history heuristic
- Check extensions
- Dynamic time management
- Bishop pair bonus, passed pawn bonuses
- Isolated & doubled pawn penalties
- King safety: pawn shield + open-file penalty

## 🔧 Building from Source

### Prerequisites
- C++17 compiler (g++ or clang)
- CMake 3.16+ (optional)
- [chess.hpp](https://github.com/Disservin/chess.hpp) library

### Quick Start

```bash
git clone https://github.com/yourusername/chesscpp-engine.git
cd chesscpp-engine
mkdir build && cd build
cmake ..
cmake --build .
