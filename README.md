# Moth chess engine 
> If Stockfish's nnue is like human brain then Moth's nnue is comparable to stegosaurus's brain, containing only around ~400k parameters.

## Evaluation
* **Architecture** 716x2 -> 512 -> 64 -> 1
* Around 400k parameters

## Search
* Alpha-Beta Negamax with fail-soft
* Reverse Futility Pruning
* Futility Pruning at depth 1
* Delta Pruning in qsearch
* SEE for capture ordering & bad-capture pruning in qsearch
* LMR
* Iterative Deepening
* PVS

## Play strength
Not yet confirm but around 2000-2200 elo

## Disservin chess-library
This chess engine uses https://github.com/Disservin/chess-library for move detections and backened.

