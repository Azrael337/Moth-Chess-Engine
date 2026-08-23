// Moth-001 A UCI-compatible chess engine using the Disservin chess-library (chess.hpp)
//
// Main file containing search and uci

#include "chess.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace chess;
#include "nnue.h"

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr int INF        = 1'000'000;
constexpr int MATE_SCORE = 900'000;
constexpr int DRAW_SCORE = 0;
constexpr int MAX_DEPTH  = 64;
constexpr int MAX_PLY    = 128;

// ─────────────────────────────────────────────
//  LMR Reduction Table
// ─────────────────────────────────────────────
static int LMR_TABLE[MAX_DEPTH][64]; // [depth][move_index]
void init_lmr() {
    for (int d = 0; d < MAX_DEPTH; ++d)
        for (int m = 0; m < 64; ++m) {
            if (d == 0 || m == 0) { LMR_TABLE[d][m] = 0; continue; }
            LMR_TABLE[d][m] = (int)(0.75 + std::log(d) * std::log(m) / 2.25);
        }
}

// ─────────────────────────────────────────────
//  Transposition Table
// ─────────────────────────────────────────────
enum TTFlag : uint8_t { TT_NONE = 0, TT_EXACT, TT_ALPHA, TT_BETA };
static uint8_t tt_generation = 0;

struct TTEntry {
    uint64_t key   = 0;
    int      score = 0;
    Move     best  = Move::NO_MOVE;
    uint8_t  depth = 0;
    uint8_t  gen   = 0;
    TTFlag   flag  = TT_NONE;
    void clear() { key = 0; score = 0; best = Move::NO_MOVE; depth = 0; gen = 0; flag = TT_NONE; }
};

constexpr size_t TT_SIZE = 1 << 22;
static TTEntry tt[TT_SIZE];

inline size_t tt_index(uint64_t key) { return key & (TT_SIZE - 1); }

void tt_clear() { for (auto& e : tt) e.clear(); tt_generation = 0; }

TTEntry* tt_probe(uint64_t key) {
    TTEntry* e = &tt[tt_index(key)];
    if (e->key != key || e->flag == TT_NONE) return nullptr;
    return e;
}

void tt_store(uint64_t key, int score, Move best, int depth, TTFlag flag) {
    TTEntry* e = &tt[tt_index(key)];
    bool stale  = (e->gen != tt_generation);
    bool deeper = (depth >= (int)e->depth);
    if (e->flag == TT_NONE || stale || deeper) {
        e->key   = key;
        e->score = score;
        e->best  = best;
        e->depth = (uint8_t)std::min(depth, 255);
        e->gen   = tt_generation;
        e->flag  = flag;
    }
}

// ─────────────────────────────────────────────
//  Killer Moves & History Heuristic
// ─────────────────────────────────────────────
static Move killers[MAX_PLY][2];
static int  history[2][64][64]; // [color][from][to]

void clear_history() {
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
}

void decay_history() {
    for (int c = 0; c < 2; ++c)
        for (int f = 0; f < 64; ++f)
            for (int t = 0; t < 64; ++t)
                history[c][f][t] /= 2;
}

// ─────────────────────────────────────────────
//  Search statistics
// ─────────────────────────────────────────────
struct SearchInfo {
    std::atomic<bool> stop{false};
    uint64_t nodes    = 0;
    int      max_time = 0;
    std::chrono::steady_clock::time_point start_time;
    bool time_up() const {
        if (max_time <= 0) return false;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start_time).count();
        return elapsed >= max_time;
    }
};
static SearchInfo info;

// ─────────────────────────────────────────────
//  NNUE state
// ─────────────────────────────────────────────
static nnue::Network g_nnue;
static std::string   g_nnue_path = "moth.nnue";

// ─────────────────────────────────────────────
//  SEE (Static Exchange Evaluation)
// ─────────────────────────────────────────────
static const int SEE_VAL[7] = { 100, 320, 330, 500, 900, 20000, 0 };

inline Color see_flip(Color c) { return (c == Color::WHITE) ? Color::BLACK : Color::WHITE; }

inline Bitboard attackers_to_square(const Board& board, Square sq, Bitboard occ) {
    Bitboard attackers(0ULL);
    attackers |= attacks::pawn(Color::BLACK, sq) & board.pieces(PieceType::PAWN, Color::WHITE);
    attackers |= attacks::pawn(Color::WHITE, sq) & board.pieces(PieceType::PAWN, Color::BLACK);
    attackers |= attacks::knight(sq) & (board.pieces(PieceType::KNIGHT, Color::WHITE) |
                                         board.pieces(PieceType::KNIGHT, Color::BLACK));
    Bitboard bishops_queens = board.pieces(PieceType::BISHOP, Color::WHITE) |
                               board.pieces(PieceType::BISHOP, Color::BLACK) |
                               board.pieces(PieceType::QUEEN,  Color::WHITE) |
                               board.pieces(PieceType::QUEEN,  Color::BLACK);
    attackers |= attacks::bishop(sq, occ) & bishops_queens;
    Bitboard rooks_queens = board.pieces(PieceType::ROOK, Color::WHITE) |
                             board.pieces(PieceType::ROOK, Color::BLACK) |
                             board.pieces(PieceType::QUEEN, Color::WHITE) |
                             board.pieces(PieceType::QUEEN, Color::BLACK);
    attackers |= attacks::rook(sq, occ) & rooks_queens;
    attackers |= attacks::king(sq) & (board.pieces(PieceType::KING, Color::WHITE) |
                                       board.pieces(PieceType::KING, Color::BLACK));
    return attackers & occ;
}

bool see_ge(const Board& board, Move move, int threshold = 0) {
    if (move.typeOf() == Move::CASTLING) return true;

    Square from = move.from();
    Square to   = move.to();

    Piece movedPiece = board.at(from);
    if (movedPiece == Piece::NONE) return true;

    PieceType attackerPT = movedPiece.type();
    PieceType capturedPT = (move.typeOf() == Move::ENPASSANT)
                                ? PieceType::PAWN
                                : board.at<PieceType>(to);

    int balance = ((capturedPT == PieceType::NONE) ? 0 : SEE_VAL[(int)capturedPT]) - threshold;
    if (balance < 0) return false;

    PieceType nextVictim = attackerPT;
    if (move.typeOf() == Move::PROMOTION) {
        balance += SEE_VAL[(int)move.promotionType()] - SEE_VAL[(int)PieceType::PAWN];
        nextVictim = move.promotionType();
    }

    balance = SEE_VAL[(int)nextVictim] - balance;
    if (balance <= 0) return true;

    uint64_t occRaw = board.occ().getBits();
    occRaw &= ~(1ULL << from.index());
    occRaw |= (1ULL << to.index());
    if (move.typeOf() == Move::ENPASSANT) {
        int capSq = to.index() + ((board.sideToMove() == Color::WHITE) ? -8 : 8);
        occRaw &= ~(1ULL << capSq);
    }
    Bitboard occ(occRaw);

    Color side = see_flip(board.sideToMove());
    Bitboard attackers = attackers_to_square(board, to, occ);

    bool relativeStm = true;
    while (true) {
        Bitboard sideAttackers = attackers & board.us(side);
        if (!sideAttackers) break;

        PieceType pt = PieceType::NONE;
        Square    attSq(0);
        for (PieceType cand : {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                                PieceType::ROOK, PieceType::QUEEN, PieceType::KING}) {
            Bitboard cbb = sideAttackers & board.pieces(cand, side);
            if (cbb) { pt = cand; attSq = Square(cbb.lsb()); break; }
        }
        if (pt == PieceType::NONE) break;

        occ &= ~Bitboard(1ULL << attSq.index());
        attackers = attackers_to_square(board, to, occ);

        relativeStm = !relativeStm;
        balance = -balance - 1 - SEE_VAL[(int)nextVictim];
        nextVictim = pt;

        if (balance >= 0) {
            if (nextVictim == PieceType::KING && (attackers & board.us(see_flip(side))))
                relativeStm = !relativeStm;
            break;
        }
        side = see_flip(side);
    }
    return relativeStm;
}

// ─────────────────────────────────────────────
//  Evaluation
// ─────────────────────────────────────────────
int evaluate(const Board& board) {
    // If NNUE is loaded, use it.
    if (g_nnue.loaded) {
        return g_nnue.evaluate(board);
    }
    
    // Fallback: If no NNUE, return 0 (or a very small random noise if you prefer)
    // This ensures the engine doesn't crash if the file is missing, 
    // but it will play randomly/blindly without weights.
    return 0;
}

// ─────────────────────────────────────────────
//  Move Ordering
// ─────────────────────────────────────────────
// Simplified MVV-LVA since we don't have PIECE_VAL_MG anymore in global scope for ordering
// We can define local constants or just use SEE_VAL which is similar enough for ordering
inline int mvv_lva(PieceType attacker, PieceType victim) {
    return SEE_VAL[(int)victim] * 10 - SEE_VAL[(int)attacker];
}

int score_move(const Board& board, const Move& move, const Move& hash_move, int ply) {
    if (move == hash_move) return 2'000'000;
    bool is_capture = board.at(move.to()) != Piece::NONE ||
                      move.typeOf() == Move::ENPASSANT;
    if (is_capture) {
        PieceType attacker = board.at<PieceType>(move.from());
        PieceType victim   = (move.typeOf() == Move::ENPASSANT)
                                 ? PieceType::PAWN
                                 : board.at<PieceType>(move.to());
        bool winning = see_ge(board, move, 0);
        int  base    = winning ? 1'500'000 : 500'000;
        return base + mvv_lva(attacker, victim);
    }
    if (move.typeOf() == Move::PROMOTION)
        return 1'200'000 + SEE_VAL[(int)move.promotionType()];
    if (move == killers[ply][0]) return 900'000;
    if (move == killers[ply][1]) return 850'000;
    int col  = (int)board.sideToMove();
    int from = move.from().index();
    int to   = move.to().index();
    return history[col][from][to];
}

void order_moves(const Board& board, Movelist& moves, const Move& hash_move, int ply) {
    std::vector<int> scores(moves.size());
    for (int i = 0; i < (int)moves.size(); ++i)
        scores[i] = score_move(board, moves[i], hash_move, ply);
    for (int i = 1; i < (int)moves.size(); ++i) {
        Move tmp_m = moves[i]; int tmp_s = scores[i]; int j = i - 1;
        while (j >= 0 && scores[j] < tmp_s) {
            moves[j + 1] = moves[j]; scores[j + 1] = scores[j]; --j;
        }
        moves[j + 1] = tmp_m; scores[j + 1] = tmp_s;
    }
}

// ─────────────────────────────────────────────
//  Quiescence Search
// ─────────────────────────────────────────────
int qsearch(Board& board, int alpha, int beta, int ply) {
    ++info.nodes;
    if (info.stop || info.time_up()) { info.stop = true; return 0; }
    int stand_pat = evaluate(board);
    if (stand_pat >= beta) return beta;
    // Delta pruning: if even a queen capture can't raise alpha, skip
    constexpr int DELTA = 900 + 100; 
    if (stand_pat < alpha - DELTA) return alpha;
    if (stand_pat > alpha) alpha = stand_pat;
    Movelist moves;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);
    for (int i = 0; i < (int)moves.size(); ++i) {
        int best_score = -INF, best_idx = i;
        for (int j = i; j < (int)moves.size(); ++j) {
            PieceType att = board.at<PieceType>(moves[j].from());
            PieceType vic = (moves[j].typeOf() == Move::ENPASSANT)
                                ? PieceType::PAWN
                                : board.at<PieceType>(moves[j].to());
            int s = mvv_lva(att, vic);
            if (s > best_score) { best_score = s; best_idx = j; }
        }
        std::swap(moves[i], moves[best_idx]);
        if (!see_ge(board, moves[i], 0)) continue;
        board.makeMove(moves[i]);
        int score = -qsearch(board, -beta, -alpha, ply + 1);
        board.unmakeMove(moves[i]);
        if (info.stop) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─────────────────────────────────────────────
//  Alpha-Beta Negamax Search
// ─────────────────────────────────────────────
int negamax(Board& board, int depth, int alpha, int beta, int ply, bool null_ok) {
    ++info.nodes;
    if (info.stop || info.time_up()) { info.stop = true; return 0; }
    if (board.isRepetition()) return DRAW_SCORE;
    if (board.isHalfMoveDraw()) {
        auto [reason, result] = board.getHalfMoveDrawType();
        if (reason == GameResultReason::CHECKMATE) return -(MATE_SCORE - ply);
        return DRAW_SCORE;
    }
    bool in_check = board.inCheck();
    if (in_check) ++depth; // check extension
    if (depth <= 0) return qsearch(board, alpha, beta, ply);
    bool is_pv   = (beta - alpha > 1);
    bool is_root = (ply == 0);
    
    // TT probe
    uint64_t key       = board.hash();
    TTEntry* tt_entry  = tt_probe(key);
    Move     hash_move = Move::NO_MOVE;
    if (tt_entry) {
        hash_move = tt_entry->best;
        if (!is_pv && tt_entry->gen == tt_generation && (int)tt_entry->depth >= depth) {
            int s = tt_entry->score;
            if (tt_entry->flag == TT_EXACT) return s;
            if (tt_entry->flag == TT_ALPHA && s <= alpha) return alpha;
            if (tt_entry->flag == TT_BETA  && s >= beta)  return beta;
        }
    }

    int  node_static_eval    = 0;
    bool static_eval_computed = false;
    auto get_static_eval = [&]() -> int {
        if (!static_eval_computed) {
            node_static_eval    = evaluate(board);
            static_eval_computed = true;
        }
        return node_static_eval;
    };

    // ── Reverse Futility Pruning (static null move) ─────────────────────────
    if (!is_pv && !in_check && depth <= 6) {
        int margin = 80 * depth;
        int static_eval = get_static_eval();
        if (static_eval - margin >= beta) return static_eval - margin;
    }

    // ── Null-Move Pruning ────────────────────────────────────────────────────
    bool has_pieces = board.pieces(PieceType::KNIGHT, board.sideToMove()).count() > 0 ||
                      board.pieces(PieceType::BISHOP, board.sideToMove()).count() > 0 ||
                      board.pieces(PieceType::ROOK,   board.sideToMove()).count() > 0 ||
                      board.pieces(PieceType::QUEEN,  board.sideToMove()).count() > 0;
    if (!in_check && null_ok && depth >= 3 && !is_pv && has_pieces) {
        int R = 3 + depth / 6;
        board.makeNullMove();
        int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
        board.unmakeNullMove();
        if (null_score >= beta) return beta;
    }

    // ── Internal Iterative Deepening ─────────────────────────────────────────
    if (is_pv && depth >= 4 && hash_move == Move::NO_MOVE) {
        negamax(board, depth - 2, alpha, beta, ply, false);
        TTEntry* e2 = tt_probe(key);
        if (e2) hash_move = e2->best;
    }

    // Generate and order moves
    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.empty())
        return in_check ? -(MATE_SCORE - ply) : DRAW_SCORE;
    order_moves(board, moves, hash_move, ply);

    int  best_score = -INF;
    Move best_move  = moves[0];
    int  orig_alpha = alpha;
    int  moves_searched = 0;

    for (int i = 0; i < (int)moves.size(); ++i) {
        bool is_capture  = board.at(moves[i].to()) != Piece::NONE ||
                           moves[i].typeOf() == Move::ENPASSANT;
        bool is_promo    = moves[i].typeOf() == Move::PROMOTION;
        bool is_quiet    = !is_capture && !is_promo;

        // ── Futility Pruning (depth 1-2) ────────────────────────────────────
        if (!is_root && !in_check && is_quiet && depth <= 2 && best_score > -MATE_SCORE) {
            int futility_margin = 100 + 80 * depth;
            int static_eval = get_static_eval();
            if (static_eval + futility_margin <= alpha) continue;
        }

        board.makeMove(moves[i]);
        ++moves_searched;
        int score;
        if (i == 0) {
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, true);
        } else {
            // ── Late Move Reduction ─────────────────────────────────────────
            int reduction = 0;
            bool gives_check = board.inCheck();
            if (!in_check && !gives_check && is_quiet && depth >= 3 && i >= 4) {
                reduction = LMR_TABLE[std::min(depth, MAX_DEPTH - 1)][std::min(i, 63)];
                if (is_pv) reduction = std::max(0, reduction - 1);
            }
            score = -negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && reduction > 0)
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta)
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, true);
        }
        board.unmakeMove(moves[i]);
        if (info.stop) return 0;
        if (score > best_score) {
            best_score = score;
            best_move  = moves[i];
        }
        if (score > alpha) {
            alpha = score;
            if (is_quiet) {
                int col  = (int)board.sideToMove();
                int from = moves[i].from().index();
                int to   = moves[i].to().index();
                history[col][from][to] += depth * depth;
                if (history[col][from][to] > 50'000) decay_history();
            }
        }
        if (alpha >= beta) {
            if (is_quiet && ply < MAX_PLY) {
                if (moves[i] != killers[ply][0]) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = moves[i];
                }
            }
            tt_store(key, beta, best_move, depth, TT_BETA);
            return beta;
        }
    }
    TTFlag flag = (alpha > orig_alpha) ? TT_EXACT : TT_ALPHA;
    tt_store(key, best_score, best_move, depth, flag);
    return best_score;
}

// ─────────────────────────────────────────────
//  Iterative Deepening with Aspiration Windows
// ─────────────────────────────────────────────
Move search(Board& board, int max_depth, int time_ms) {
    info.nodes      = 0;
    info.stop       = false;
    info.max_time   = time_ms;
    info.start_time = std::chrono::steady_clock::now();
    tt_generation++;
    clear_history();
    Move best_move = Move::NO_MOVE;
    {
        Movelist ml; movegen::legalmoves(ml, board);
        if (ml.empty()) return Move::NO_MOVE;
        best_move = ml[0];
    }
    auto elapsed_ms = [&]() {
        return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - info.start_time).count();
    };
    int prev_score       = 0;
    int ms_before_depth  = 0;
    for (int depth = 1; depth <= max_depth; ++depth) {
        int score;
        if (depth >= 5) {
            int delta = 25;
            int alpha = prev_score - delta;
            int beta  = prev_score + delta;
            while (true) {
                score = negamax(board, depth, alpha, beta, 0, true);
                if (info.stop) goto search_done;
                if      (score <= alpha) { alpha -= delta; delta *= 2; }
                else if (score >= beta)  { beta  += delta; delta *= 2; }
                else break;
                if (alpha <= -INF / 2) alpha = -INF;
                if (beta  >=  INF / 2) beta  =  INF;
                if (alpha == -INF && beta == INF) {
                    score = negamax(board, depth, -INF, INF, 0, true);
                    if (info.stop) goto search_done;
                    break;
                }
            }
        } else {
            score = negamax(board, depth, -INF, INF, 0, true);
            if (info.stop) goto search_done;
        }
        prev_score = score;
        {
            TTEntry* e = tt_probe(board.hash());
            if (e && e->best != Move::NO_MOVE) best_move = e->best;
        }
        {
            int ms       = elapsed_ms();
            int ms_depth = ms - ms_before_depth;
            int nps      = (ms > 0) ? (int)(info.nodes * 1000ULL / ms) : 0;
            std::cout << "info depth " << depth
                      << " score cp "  << score
                      << " nodes "     << info.nodes
                      << " nps "       << nps
                      << " time "      << ms
                      << " pv "        << uci::moveToUci(best_move)
                      << "\n";
            std::cout.flush();
            if (time_ms > 0) {
                int time_left     = time_ms - ms;
                long long pred_ms = (long long)std::max(ms_depth, 1) * 3;
                bool used_enough  = ms >= time_ms / 5;
                if (time_left <= 0 || (used_enough && pred_ms > time_left)) break;
            }
            ms_before_depth = ms;
        }
    }
search_done:
    return best_move;
}

// ─────────────────────────────────────────────
//  UCI Protocol
// ─────────────────────────────────────────────
void uci_loop() {
    Board board;
    board.setFen(constants::STARTPOS);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "uci") {
            std::cout << "id name Moth-nnue-001\n";
            std::cout << "id author Ranak Chongtham\n";
            std::cout << "option name EvalFile type string default " << g_nnue_path << "\n";
            std::cout << "uciok\n";
        } else if (token == "isready") {
            std::cout << "readyok\n";
        } else if (token == "ucinewgame") {
            board.setFen(constants::STARTPOS);
            tt_clear();
            clear_history();
        } else if (token == "setoption") {
            std::string word, name, value;
            iss >> word; // "name"
            while (iss >> word && word != "value") {
                if (!name.empty()) name += " ";
                name += word;
            }
            std::getline(iss, value);
            if (!value.empty() && value.front() == ' ') value.erase(0, 1);
            if (name == "EvalFile" && !value.empty()) {
                g_nnue_path = value;
                if (g_nnue.load(g_nnue_path)) {
                    std::cout << "info string loaded NNUE from " << g_nnue_path << "\n";
                } else {
                    std::cout << "info string failed to load NNUE from " << g_nnue_path
                               << ", using blind eval (0)\n";
                }
            }
        } else if (token == "position") {
            std::string pos_token;
            iss >> pos_token;
            if (pos_token == "startpos") {
                board.setFen(constants::STARTPOS);
                iss >> pos_token;
            } else if (pos_token == "fen") {
                std::string fen;
                for (int i = 0; i < 6; ++i) {
                    std::string f; iss >> f;
                    if (i > 0) fen += " ";
                    fen += f;
                }
                board.setFen(fen);
                iss >> pos_token;
            }
            while (iss >> pos_token) {
                Move m = uci::uciToMove(board, pos_token);
                if (m == Move::NO_MOVE) break;
                board.makeMove(m);
            }
        } else if (token == "go") {
            int  depth    = -1;
            int  movetime = 0;
            int  wtime = 0, btime = 0, winc = 0, binc = 0;
            bool infinite = false;
            while (iss >> token) {
                if      (token == "depth")    iss >> depth;
                else if (token == "movetime") iss >> movetime;
                else if (token == "wtime")    iss >> wtime;
                else if (token == "btime")    iss >> btime;
                else if (token == "winc")     iss >> winc;
                else if (token == "binc")     iss >> binc;
                else if (token == "infinite") infinite = true;
            }
            if (movetime == 0 && !infinite) {
                int myTime = (board.sideToMove() == Color::WHITE) ? wtime : btime;
                int myInc  = (board.sideToMove() == Color::WHITE) ? winc  : binc;
                if (myTime > 0) movetime = myTime / 20 + myInc / 2;
            }
            if (infinite)          { depth = MAX_DEPTH; movetime = 0; }
            else if (movetime > 0) { if (depth < 0) depth = MAX_DEPTH; }
            else                   { if (depth < 0) depth = 8; }
            Move best = search(board, depth, movetime);
            std::cout << "bestmove " << uci::moveToUci(best) << "\n";
        } else if (token == "stop") {
            info.stop = true;
        } else if (token == "quit") {
            break;
        } else if (token == "d") {
            std::cout << board << "\n";
            std::cout << "FEN: "  << board.getFen() << "\n";
            std::cout << "Hash: " << board.hash()   << "\n";
            std::cout << "Eval (side to move): " << evaluate(board)
                       << (g_nnue.loaded ? "  [NNUE]\n" : "  [blind/zero]\n");
        } else if (token == "perft") {
            int pd = 5; iss >> pd;
            std::function<uint64_t(Board&, int)> perft = [&](Board& b, int d) -> uint64_t {
                Movelist ml; movegen::legalmoves(ml, b);
                if (d == 1) return ml.size();
                uint64_t n = 0;
                for (int i = 0; i < (int)ml.size(); ++i) {
                    b.makeMove(ml[i]); n += perft(b, d - 1); b.unmakeMove(ml[i]);
                }
                return n;
            };
            auto t0 = std::chrono::steady_clock::now();
            uint64_t n = perft(board, pd);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
            std::cout << "Perft(" << pd << ") = " << n << "  time " << ms << "ms\n";
        }
        std::cout.flush();
    }
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    init_lmr();
    tt_clear();
    clear_history();
    
    if (g_nnue.load(g_nnue_path)) {
        std::cerr << "info string loaded NNUE from " << g_nnue_path << "\n";
    } else {
        std::cerr << "info string no NNUE loaded (looked for " << g_nnue_path
                   << "), using blind eval. Use 'setoption name EvalFile value <path>' to load one.\n";
    }
    uci_loop();
    return 0;
}
