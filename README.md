<p align="center">
  <img src="assets/Moth_banner.png" alt="Moth chess engine">
</p>

# Moth Chess Engine

> If Stockfish's NNUE is like a human brain, then Moth's NNUE is comparable to a stegosaurus's brain, containing only around ~400k parameters.

Moth is a lightweight UCI chess engine written in C++ with a focus on efficient search and a compact NNUE evaluation.

## Evaluation

* **NNUE:** 768 × 2 → 256 → 1
* **Parameters:** ~400k
* **Network:** Acherontia
* Incremental NNUE evaluation
* Tapered material evaluation

## Search

* Alpha-Beta Negamax with fail-soft
* Principal Variation Search (PVS)
* Iterative Deepening
* Transposition Table
* Late Move Reductions (LMR)
* Null Move Pruning
* Reverse Futility Pruning
* Futility Pruning
* ProbCut
* Singular Extensions
* SEE for capture ordering and pruning
* Killer Move Heuristic
* History Heuristic
* Continuation History
* Countermove Heuristic
* Aspiration Windows
* Quiescence Search
* Delta Pruning
* Multi-threaded search

## Play Strength

Not yet officially tested.

Estimated strength is currently around 2000–2200 Elo.

More testing will be done using engine-vs-engine games.

## Disservin Chess Library

Moth uses the
[Disservin chess-library](https://github.com/Disservin/chess-library)
for board representation, legal move generation, and move handling.

## Building

Moth is written in C++ and requires a C++ compiler supporting modern C++ standards.

### Requirements

* C++ compiler
* CMake / Make (if applicable)

## UCI

Moth supports the Universal Chess Interface (UCI) protocol and can be used with chess GUIs such as Cute Chess.

## License

See [LICENSE](LICENSE).
