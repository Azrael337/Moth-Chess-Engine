#include "chess.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace chess;
#include "nnue.h"

constexpr int INF         = 1'000'000;
constexpr int MATE_SCORE  = 900'000;
constexpr int DRAW_SCORE  = 0;
constexpr int MAX_DEPTH   = 64;
constexpr int MAX_PLY     = 128;
constexpr int MAX_THREADS = 8;
constexpr int MATE_BOUND  = MATE_SCORE - MAX_PLY;
constexpr int NO_EVAL     = -2'000'000;
constexpr int CORR_HIST_SIZE = 16384;

static const int PIECE_VAL_MG[7] = { 100, 320, 330, 500, 900, 20000, 0 };
static const int PIECE_VAL_EG[7] = { 120, 300, 320, 550, 950, 20000, 0 };

static int LMR_TABLE[MAX_DEPTH][64];
void init_lmr() {
    for (int d = 0; d < MAX_DEPTH; ++d)
        for (int m = 0; m < 64; ++m) {
            if (d <= 0 || m <= 0) { LMR_TABLE[d][m] = 0; continue; }
            LMR_TABLE[d][m] = (int)(0.5 + std::log((double)d) * std::log((double)m) / 2.25);
        }
}

// ─────────────────────────────────────────────
//  Transposition Table
// ─────────────────────────────────────────────
enum TTFlag : uint8_t { TT_NONE = 0, TT_EXACT, TT_ALPHA, TT_BETA };

struct TTEntry {
    uint32_t key32;
    int32_t  score;
    int32_t  static_eval;
    uint32_t move_data[1];
    uint8_t  depth;
    uint8_t  gen_flag;
    void clear() { std::memset(this, 0, sizeof(TTEntry)); }
    Move get_best_move() const {
        Move m; std::memcpy(&m, move_data, sizeof(Move)); return m;
    }
    TTFlag get_flag() const { return static_cast<TTFlag>(gen_flag & 0x0F); }
    uint8_t get_gen() const { return gen_flag >> 4; }
};

static TTEntry* tt = nullptr;
static size_t   g_tt_size = 1 << 21;
static size_t   g_tt_mask = g_tt_size - 1;
static uint8_t  tt_generation = 0;

inline int score_to_tt(int s, int ply) {
    if (s >=  MATE_BOUND) return s + ply;
    if (s <= -MATE_BOUND) return s - ply;
    return s;
}
inline int score_from_tt(int s, int ply) {
    if (s >=  MATE_BOUND) return s - ply;
    if (s <= -MATE_BOUND) return s + ply;
    return s;
}

void tt_clear() {
    if (!tt) return;
    std::memset(tt, 0, g_tt_size * sizeof(TTEntry));
    tt_generation = 0;
}
void tt_resize(size_t mb) {
    delete[] tt;
    size_t target = std::max<size_t>(mb, 1) * 1024 * 1024 / sizeof(TTEntry);
    size_t s = 1;
    while (s * 2 <= target) s <<= 1;
    g_tt_size = s; g_tt_mask = s - 1;
    tt = new TTEntry[g_tt_size];
    tt_clear();
}
TTEntry* tt_probe(uint64_t key) {
    TTEntry* e = &tt[key & g_tt_mask];
    if (e->key32 != static_cast<uint32_t>(key >> 32)) return nullptr;
    if (e->get_flag() == TT_NONE) return nullptr;
    return e;
}
void tt_store(uint64_t key, int score, int static_eval, Move best, int depth, TTFlag flag, int ply) {
    TTEntry* e = &tt[key & g_tt_mask];
    uint32_t key32 = static_cast<uint32_t>(key >> 32);
    bool replace = (e->get_flag() == TT_NONE)
                || (e->get_gen() != tt_generation)
                || (depth + 3 >= (int)e->depth);
    if (replace) {
        e->key32 = key32;
        e->score = score_to_tt(score, ply);
        e->static_eval = static_eval;
        std::memcpy(e->move_data, &best, sizeof(Move));
        e->depth = static_cast<uint8_t>(std::min(depth, 255));
        e->gen_flag = (uint8_t)((tt_generation << 4) | static_cast<uint8_t>(flag));
    }
}

// ─────────────────────────────────────────────
//  NEW: shared best move across threads.
//  Deepest completed iteration wins (CAS on depth). Ties -> first publisher.
// ─────────────────────────────────────────────
static_assert(sizeof(Move) == sizeof(uint32_t), "Move must be 4 bytes for lock-free sharing");
static std::atomic<uint32_t> g_shared_best{0};
static std::atomic<int>      g_shared_depth{0};

// ─────────────────────────────────────────────
//  Thread State — CHANGED: 4-ply continuation history + move stack
// ─────────────────────────────────────────────
struct ThreadState {
    Move killers[MAX_PLY][2];
    Move counterMoves[6][64];
    int  history[2][64][64];
    int  contHist[4][6][64][6][64];      // CHANGED: [plies-back-1][prev_pt][prev_to][pt][to]
    int  capHist[7][64][7];
    int  pawn_corrHist[CORR_HIST_SIZE];
    int  eval_stack[512];
    PieceType mv_pt[MAX_PLY + 2];        // NEW: piece moved to arrive at ply
    int       mv_to[MAX_PLY + 2];        // NEW: target square of that move (-1 = none)
    nnue::Accumulator accStack[MAX_PLY + 64];
    int  accDepth;
    uint64_t nodes;
    int  seldepth;
    int  thread_id;

    void clear_history() {
        std::memset(killers,       0, sizeof(killers));
        std::memset(counterMoves,  0, sizeof(counterMoves));
        std::memset(history,       0, sizeof(history));
        std::memset(contHist,      0, sizeof(contHist));
        std::memset(capHist,       0, sizeof(capHist));
        std::memset(pawn_corrHist, 0, sizeof(pawn_corrHist));
    }
};

struct SearchInfo {
    std::atomic<bool> stop{false};
    int      max_time = 0;
    std::chrono::steady_clock::time_point start_time;
    bool time_up() const {
        if (max_time <= 0) return false;
        auto e = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start_time).count();
        return e >= max_time;
    }
};

static SearchInfo info;
static ThreadState* thread_states = nullptr;
static int num_threads = 1;

static nnue::Network g_nnue;
static std::string   g_nnue_path = "acherontia.nnue";

// ─────────────────────────────────────────────
//  Correction History
// ─────────────────────────────────────────────
inline uint64_t pawn_corr_key(const Board& b) {
    return (b.pieces(PieceType::PAWN, Color::WHITE).getBits() * 0x9E3779B97F4A7C15ULL)
         ^ (b.pieces(PieceType::PAWN, Color::BLACK).getBits() * 0xC2B2AE3D27D4EB4FULL);
}
inline void update_pawn_corr(ThreadState& ts, const Board& b, int diff, int depth) {
    int idx = (int)(pawn_corr_key(b) & (CORR_HIST_SIZE - 1));
    int w = std::min(depth, 16);
    int target = std::max(-64, std::min(64, diff / 4));
    ts.pawn_corrHist[idx] = (ts.pawn_corrHist[idx] * (32 - w) + target * w) / 32;
}
inline int get_pawn_corr(ThreadState& ts, const Board& b) {
    return ts.pawn_corrHist[pawn_corr_key(b) & (CORR_HIST_SIZE - 1)];
}

// ─────────────────────────────────────────────
//  NNUE / move wrappers
// ─────────────────────────────────────────────
inline void do_move(Board& b, const Move& m, ThreadState& ts) {
    if (g_nnue.loaded) {
        if (ts.accDepth >= MAX_PLY + 60) { b.makeMove(m); return; }
        ts.accDepth++;
        ts.accStack[ts.accDepth] = ts.accStack[ts.accDepth - 1];
        if (m.typeOf() == Move::CASTLING) { b.makeMove(m); g_nnue.refresh(b, ts.accStack[ts.accDepth]); }
        else { g_nnue.update_for_move(ts.accStack[ts.accDepth], b, m); b.makeMove(m); }
        return;
    }
    b.makeMove(m);
}
inline void undo_move(Board& b, const Move& m, ThreadState& ts) { b.unmakeMove(m); if (g_nnue.loaded) ts.accDepth--; }
inline void do_null_move(Board& b, ThreadState& ts) {
    if (g_nnue.loaded) {
        ts.accDepth++;
        ts.accStack[ts.accDepth] = ts.accStack[ts.accDepth - 1];
        g_nnue.update_for_null_move(ts.accStack[ts.accDepth], b);
        b.makeNullMove(); return;
    }
    b.makeNullMove();
}
inline void undo_null_move(Board& b, ThreadState& ts) { b.unmakeNullMove(); if (g_nnue.loaded) ts.accDepth--; }

// ─────────────────────────────────────────────
//  Eval glue
// ─────────────────────────────────────────────
inline int game_phase(const Board& b) {
    constexpr int MAX_PHASE = 24;
    int p = MAX_PHASE;
    for (Color c : {Color::WHITE, Color::BLACK}) {
        p -= (int)b.pieces(PieceType::KNIGHT, c).count() * 1;
        p -= (int)b.pieces(PieceType::BISHOP, c).count() * 1;
        p -= (int)b.pieces(PieceType::ROOK,   c).count() * 2;
        p -= (int)b.pieces(PieceType::QUEEN,  c).count() * 4;
    }
    p = std::max(0, p);
    return (p * 256 + MAX_PHASE / 2) / MAX_PHASE;
}
static const int SEE_VAL[7] = { 100, 320, 330, 500, 900, 20000, 0 };
inline Color see_flip(Color c) { return c == Color::WHITE ? Color::BLACK : Color::WHITE; }

int eval_material_simple(const Board& b) {
    int ph = game_phase(b);
    int mg = 0, eg = 0;
    for (Color c : {Color::WHITE, Color::BLACK}) {
        int s = (c == Color::WHITE) ? 1 : -1;
        for (PieceType pt : {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                              PieceType::ROOK, PieceType::QUEEN}) {
            int cnt = (int)b.pieces(pt, c).count();
            mg += s * cnt * PIECE_VAL_MG[(int)pt];
            eg += s * cnt * PIECE_VAL_EG[(int)pt];
        }
    }
    int sc = (mg * (256 - ph) + eg * ph) / 256;
    return (b.sideToMove() == Color::WHITE) ? sc : -sc;
}
int evaluate(const Board& b) {
    if (g_nnue.loaded) return g_nnue.evaluate(b) + eval_material_simple(b);
    return eval_material_simple(b);
}
inline int evaluate_tree(const Board& b, ThreadState& ts) {
    if (g_nnue.loaded)
        return g_nnue.evaluate_from_accumulator(ts.accStack[ts.accDepth], b.sideToMove())
               + eval_material_simple(b);
    return eval_material_simple(b);
}

// ─────────────────────────────────────────────
//  SEE
// ─────────────────────────────────────────────
bool see_ge(const Board& board, Move move, int threshold = 0) {
    if (move.typeOf() == Move::CASTLING) return true;
    Square from = move.from(), to = move.to();
    Piece movedPiece = board.at(from);
    if (movedPiece == Piece::NONE) return true;
    PieceType attackerPT = movedPiece.type();
    PieceType capturedPT = (move.typeOf() == Move::ENPASSANT) ? PieceType::PAWN : board.at<PieceType>(to);
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
    occRaw |=  (1ULL << to.index());
    if (move.typeOf() == Move::ENPASSANT) {
        int capSq = to.index() + ((board.sideToMove() == Color::WHITE) ? -8 : 8);
        occRaw &= ~(1ULL << capSq);
    }
    Bitboard occ(occRaw);
    Color side = see_flip(board.sideToMove());
    Bitboard bq = board.pieces(PieceType::BISHOP, Color::WHITE) | board.pieces(PieceType::BISHOP, Color::BLACK)
                | board.pieces(PieceType::QUEEN,  Color::WHITE) | board.pieces(PieceType::QUEEN,  Color::BLACK);
    Bitboard rq = board.pieces(PieceType::ROOK,   Color::WHITE) | board.pieces(PieceType::ROOK,   Color::BLACK)
                | board.pieces(PieceType::QUEEN,  Color::WHITE) | board.pieces(PieceType::QUEEN,  Color::BLACK);
    auto get_atk = [&](Bitboard o) {
        Bitboard a(0ULL);
        a |= attacks::pawn(Color::BLACK, to) & board.pieces(PieceType::PAWN, Color::WHITE);
        a |= attacks::pawn(Color::WHITE, to) & board.pieces(PieceType::PAWN, Color::BLACK);
        a |= attacks::knight(to) & (board.pieces(PieceType::KNIGHT, Color::WHITE) | board.pieces(PieceType::KNIGHT, Color::BLACK));
        a |= attacks::bishop(to, o) & bq;
        a |= attacks::rook(to, o)   & rq;
        a |= attacks::king(to) & (board.pieces(PieceType::KING, Color::WHITE) | board.pieces(PieceType::KING, Color::BLACK));
        return a & o;
    };
    Bitboard atk = get_atk(occ);
    bool relStm = true;
    while (true) {
        Bitboard sa = atk & board.us(side);
        if (!sa) break;
        PieceType pt = PieceType::NONE;
        Square    asq(0);
        for (PieceType c : {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN, PieceType::KING}) {
            Bitboard cbb = sa & board.pieces(c, side);
            if (cbb) { pt = c; asq = Square(cbb.lsb()); break; }
        }
        if (pt == PieceType::NONE) break;
        occ &= ~Bitboard(1ULL << asq.index());
        atk = get_atk(occ);
        relStm = !relStm;
        balance = -balance - 1 - SEE_VAL[(int)nextVictim];
        nextVictim = pt;
        if (balance >= 0) {
            if (nextVictim == PieceType::KING && (atk & board.us(see_flip(side))))
                relStm = !relStm;
            break;
        }
        side = see_flip(side);
    }
    return relStm;
}

// ─────────────────────────────────────────────
//  History helpers
// ─────────────────────────────────────────────
constexpr int HIST_LIMIT = 16384;
inline void add_hist(int& h, int bonus) { h += bonus - h * std::abs(bonus) / HIST_LIMIT; }
inline int history_bonus(int d) { return std::min(8 * d * d, 8192); }
inline int mvv_lva(PieceType a, PieceType v) { return PIECE_VAL_MG[(int)v] * 10 - PIECE_VAL_MG[(int)a]; }

// NEW: continuation-history access via the per-ply move stack.
inline int cont_hist_at(const ThreadState& ts, int ply, int back, int moved_pt, int to) {
    int p = ply - back;
    if (p < 0) return 0;
    if (ts.mv_pt[p] == PieceType::NONE || ts.mv_to[p] < 0) return 0;
    return ts.contHist[back - 1][(int)ts.mv_pt[p]][ts.mv_to[p]][moved_pt][to];
}
// NEW: apply bonus/malus across all available continuation-history plies.
inline void update_cont_hist(ThreadState& ts, int ply, int moved_pt, int to, int bonus) {
    for (int back = 1; back <= 4; ++back) {
        int p = ply - back;
        if (p < 0) break;
        if (ts.mv_pt[p] == PieceType::NONE || ts.mv_to[p] < 0) break;
        add_hist(ts.contHist[back - 1][(int)ts.mv_pt[p]][ts.mv_to[p]][moved_pt][to], bonus);
    }
}

// CHANGED: prev-move params removed — read from the move stack instead.
int score_move(const Board& board, const Move& move, const Move& hash_move, int ply, ThreadState& ts) {
    if (move == hash_move) return 100'000'000;
    bool is_cap = board.at(move.to()) != Piece::NONE || move.typeOf() == Move::ENPASSANT;
    PieceType att = board.at<PieceType>(move.from());
    if (is_cap) {
        PieceType vic = (move.typeOf() == Move::ENPASSANT) ? PieceType::PAWN : board.at<PieceType>(move.to());
        int base = see_ge(board, move, 0) ? 60'000'000 : 30'000'000;
        return base + ts.capHist[(int)att][move.to().index()][(int)vic] + mvv_lva(att, vic);
    }
    if (move.typeOf() == Move::PROMOTION) return 50'000'000 + PIECE_VAL_MG[(int)move.promotionType()];
    if (ply < MAX_PLY) {
        if (move == ts.killers[ply][0]) return 45'000'000;
        if (move == ts.killers[ply][1]) return 44'900'000;
    }
    if (ply >= 1 && ts.mv_pt[ply - 1] != PieceType::NONE && ts.mv_to[ply - 1] >= 0 &&
        move == ts.counterMoves[(int)ts.mv_pt[ply - 1]][ts.mv_to[ply - 1]]) return 43'000'000;

    int c  = (int)board.sideToMove();
    int mp = (int)att;
    int qh = ts.history[c][move.from().index()][move.to().index()];
    for (int back = 1; back <= 4; ++back)
        qh += cont_hist_at(ts, ply, back, mp, move.to().index());
    return qh;
}
void order_moves(const Board& board, Movelist& moves, const Move& hm, int ply, ThreadState& ts) {
    std::array<int, 256> sc;
    int n = (int)moves.size();
    for (int i = 0; i < n; ++i)
        sc[i] = score_move(board, moves[i], hm, ply, ts);
    for (int i = 1; i < n; ++i) {
        Move tm = moves[i]; int tv = sc[i]; int j = i - 1;
        while (j >= 0 && sc[j] < tv) { moves[j+1] = moves[j]; sc[j+1] = sc[j]; --j; }
        moves[j+1] = tm; sc[j+1] = tv;
    }
}

// ─────────────────────────────────────────────
//  Quiescence — CHANGED: TT move searched first
// ─────────────────────────────────────────────
int qsearch(Board& b, int alpha, int beta, int ply, ThreadState& ts) {
    ts.nodes++;
    if (info.stop.load(std::memory_order_relaxed) ||
        ((ts.nodes & 2047) == 0 && info.time_up())) {
        info.stop.store(true, std::memory_order_relaxed);
        return 0;
    }
    if (ply > ts.seldepth) ts.seldepth = ply;
    if (ply >= MAX_PLY - 1) return evaluate_tree(b, ts);

    uint64_t key = b.hash();
    __builtin_prefetch(&tt[key & g_tt_mask]);
    bool is_pv = (beta - alpha > 1);

    Move qtt = Move::NO_MOVE;                       // NEW: TT move for ordering
    TTEntry* e = tt_probe(key);
    if (e) {
        qtt = e->get_best_move();
        if (!is_pv) {
            int s = score_from_tt(e->score, ply);
            TTFlag f = e->get_flag();
            if (f == TT_EXACT) return s;
            if (f == TT_ALPHA && s <= alpha) return alpha;
            if (f == TT_BETA  && s >= beta)  return beta;
        }
    }

    bool in_check = b.inCheck();
    int stand_pat = in_check ? -INF : evaluate_tree(b, ts);

    if (!in_check) {
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
        if (!is_pv && stand_pat < alpha - 1000) return alpha;
    }

    Movelist moves;
    if (in_check) movegen::legalmoves(moves, b);
    else          movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, b);

    if (in_check && moves.empty()) return -(MATE_SCORE - ply);

    int n = (int)moves.size();
    std::array<int, 256> sc;
    for (int i = 0; i < n; ++i) {
        if (qtt != Move::NO_MOVE && moves[i] == qtt) { sc[i] = 100'000'000; continue; }   // NEW
        PieceType a = b.at<PieceType>(moves[i].from());
        PieceType v = (moves[i].typeOf() == Move::ENPASSANT) ? PieceType::PAWN : b.at<PieceType>(moves[i].to());
        sc[i] = ts.capHist[(int)a][moves[i].to().index()][(int)v] + mvv_lva(a, v);
    }

    int best_score = stand_pat;
    Move best_move = Move::NO_MOVE;
    int orig_alpha = alpha;

    for (int i = 0; i < n; ++i) {
        int bi = i;
        for (int j = i + 1; j < n; ++j) if (sc[j] > sc[bi]) bi = j;
        if (bi != i) { std::swap(moves[i], moves[bi]); std::swap(sc[i], sc[bi]); }

        if (!in_check) {
            PieceType a = b.at<PieceType>(moves[i].from());
            PieceType v = (moves[i].typeOf() == Move::ENPASSANT) ? PieceType::PAWN : b.at<PieceType>(moves[i].to());
            if (PIECE_VAL_MG[(int)v] < PIECE_VAL_MG[(int)a] && !see_ge(b, moves[i], 0)) continue;
        }

        do_move(b, moves[i], ts);
        int s = -qsearch(b, -beta, -alpha, ply + 1, ts);
        undo_move(b, moves[i], ts);
        if (info.stop.load(std::memory_order_relaxed)) return 0;

        if (s > best_score) {
            best_score = s;
            best_move  = moves[i];
            if (s > alpha) alpha = s;
            if (alpha >= beta) break;
        }
    }

    if (!info.stop.load(std::memory_order_relaxed)) {
        TTFlag f = (alpha > orig_alpha) ? TT_EXACT : TT_ALPHA;
        tt_store(key, best_score, in_check ? 0 : stand_pat, best_move, 0, f, ply);
    }
    return best_score;
}

// ─────────────────────────────────────────────
//  Main search — CHANGED: prev-move params removed (move stack instead),
//  check extension removed, futility refined
// ─────────────────────────────────────────────
int negamax(Board& b, int depth, int alpha, int beta, int ply, bool null_ok, bool cut_node,
            ThreadState& ts, Move excluded = Move::NO_MOVE) {
    ts.nodes++;
    if (info.stop.load(std::memory_order_relaxed) ||
        ((ts.nodes & 2047) == 0 && info.time_up())) {
        info.stop.store(true, std::memory_order_relaxed);
        return 0;
    }
    if (ply >= MAX_PLY - 1)
        return b.inCheck() ? evaluate_tree(b, ts) : qsearch(b, alpha, beta, ply, ts);

    if (b.isRepetition()) return DRAW_SCORE;
    if (b.isHalfMoveDraw()) {
        auto [reason, result] = b.getHalfMoveDrawType();
        if (reason == GameResultReason::CHECKMATE) return -(MATE_SCORE - ply);
        return DRAW_SCORE;
    }

    bool in_check = b.inCheck();
    // CHANGED: unconditional check extension REMOVED.
    // Rationale: extension explosions in forcing lines cost more than they gain;
    // qsearch now handles in-check evasions, so leaf mates are not missed.
    if (depth <= 0) return qsearch(b, alpha, beta, ply, ts);

    bool is_pv   = (beta - alpha > 1);
    bool is_root = (ply == 0);

    uint64_t key = b.hash();
    __builtin_prefetch(&tt[key & g_tt_mask]);

    Move   hm = Move::NO_MOVE;
    int    tt_depth = 0, tt_score = NO_EVAL, tt_eval = NO_EVAL;
    TTFlag tt_flag = TT_NONE;

    if (excluded == Move::NO_MOVE) {
        TTEntry* e = tt_probe(key);
        if (e) {
            hm = e->get_best_move();
            tt_depth = e->depth;
            tt_score = score_from_tt(e->score, ply);
            tt_eval  = e->static_eval;
            tt_flag  = e->get_flag();
            if (!is_pv && tt_depth >= depth) {
                if (tt_flag == TT_EXACT) return tt_score;
                if (tt_flag == TT_ALPHA && tt_score <= alpha) return alpha;
                if (tt_flag == TT_BETA  && tt_score >= beta)  return beta;
            }
        }
    }

    int raw_eval = in_check ? 0 : (tt_eval != NO_EVAL ? tt_eval : evaluate_tree(b, ts));
    int corr = in_check ? 0 : get_pawn_corr(ts, b) / 4;
    int static_eval = in_check ? 0 : raw_eval + corr;

    ts.eval_stack[ply] = in_check ? NO_EVAL : static_eval;
    bool improving = !in_check && ply >= 2 &&
                     ts.eval_stack[ply - 2] != NO_EVAL && static_eval > ts.eval_stack[ply - 2];

    if (!is_pv && !in_check && depth <= 5) {
        int m = 300 + 200 * depth;
        if (static_eval + m < alpha) {
            int r = qsearch(b, alpha, alpha + 1, ply, ts);
            if (r <= alpha) return r;
        }
    }
    if (!is_pv && !in_check && depth <= 7) {
        int m = 75 * depth;
        if (static_eval - m >= beta) return static_eval - m;
    }
    if (excluded == Move::NO_MOVE && hm == Move::NO_MOVE && depth >= 4) --depth;  // IIR

    // ProbCut
    if (!is_pv && !in_check && excluded == Move::NO_MOVE && depth >= 5 &&
        std::abs(beta) < MATE_BOUND) {
        int pb = beta + 180;
        if (static_eval + 40 * depth >= pb) {
            Movelist caps;
            movegen::legalmoves<movegen::MoveGenType::CAPTURE>(caps, b);
            for (const Move& m : caps) {
                if (!see_ge(b, m, 150)) continue;
                do_move(b, m, ts);
                ts.mv_pt[ply] = PieceType::NONE;  // ProbCut children get no conthist context
                ts.mv_to[ply] = -1;
                int s = -negamax(b, depth - 4, -pb, -pb + 1, ply + 1, false, true, ts);
                undo_move(b, m, ts);
                if (info.stop.load(std::memory_order_relaxed)) return 0;
                if (s >= pb) return pb;
            }
        }
    }

    // Null move pruning (move-stack slot neutralized across the null search)
    bool has_pieces = (b.us(b.sideToMove()) & ~b.pieces(PieceType::PAWN, b.sideToMove())).count() > 1;
    if (!in_check && null_ok && !is_pv && depth >= 3 && has_pieces) {
        int R = 3 + depth / 4 + std::min((static_eval - beta) / 200, 3) + (cut_node ? 1 : 0);
        R = std::min(R, depth - 1);
        PieceType saved_pt = ts.mv_pt[ply]; int saved_to = ts.mv_to[ply];
        ts.mv_pt[ply] = PieceType::NONE; ts.mv_to[ply] = -1;
        do_null_move(b, ts);
        int ns = -negamax(b, depth - 1 - R, -beta, -beta + 1, ply + 1, false, false, ts);
        undo_null_move(b, ts);
        ts.mv_pt[ply] = saved_pt; ts.mv_to[ply] = saved_to;
        if (info.stop.load(std::memory_order_relaxed)) return 0;
        if (ns >= beta) {
            if (ns >= MATE_BOUND) return beta;
            return ns;
        }
    }

    // Singular + double extensions
    int extension = 0;
    if (!is_root && !in_check && excluded == Move::NO_MOVE &&
        hm != Move::NO_MOVE && depth >= 8 &&
        tt_depth >= depth - 3 && tt_flag != TT_ALPHA &&
        tt_score >= beta && std::abs(tt_score) < MATE_BOUND) {
        int sb = tt_score - 2 * depth;
        int sd = (depth - 1) / 2;
        int s = negamax(b, sd, sb - 1, sb, ply, false, false, ts, hm);
        if (info.stop.load(std::memory_order_relaxed)) return 0;
        if (s < sb) {
            extension = 1;
            if (depth >= 12 && s < sb - 20) extension = 2;
        } else if (sb >= beta) return sb;
    }

    Movelist moves;
    movegen::legalmoves(moves, b);
    if (moves.empty())
        return in_check ? -(MATE_SCORE - ply) : DRAW_SCORE;

    // OPTIONAL (SPRT this): modern conditional check extension — replaces the
    // removed unconditional one; extends only nearly-forced evasions (<=2 replies).
    // If enabled, change nd below to: depth - 1 + base_ext + ((moves[i]==hm) ? extension : 0)
    // int base_ext = (in_check && (int)moves.size() <= 2) ? 1 : 0;
    int base_ext = 0;

    order_moves(b, moves, hm, ply, ts);

    // CHANGED: conthist sum over up to 4 plies via the move stack.
    // NOTE: col/mpt are captured BEFORE do_move by each caller.
    auto qhist_of = [&](const Move& m, int mpt, int col) {
        int h = ts.history[col][m.from().index()][m.to().index()];
        for (int back = 1; back <= 4; ++back)
            h += cont_hist_at(ts, ply, back, mpt, m.to().index());
        return h;
    };

    int best_score = -INF;
    Move best_move = moves[0];
    int orig_alpha = alpha;
    int quiets_tried = 0;
    Move quiet_list[64];
    int n_quiet = 0;

    for (int i = 0; i < (int)moves.size(); ++i) {
        bool is_cap   = b.at(moves[i].to()) != Piece::NONE || moves[i].typeOf() == Move::ENPASSANT;
        bool is_promo = moves[i].typeOf() == Move::PROMOTION;
        bool is_quiet = !is_cap && !is_promo;

        if (!is_root && best_score > -MATE_BOUND) {
            if (!is_pv && !in_check && is_quiet && depth <= 8) {
                int lmp = improving ? (2 + depth * depth / 2) : (1 + depth * depth / 3);
                if (quiets_tried >= lmp) continue;
                if (depth <= 5 &&
                    qhist_of(moves[i], (int)b.at<PieceType>(moves[i].from()), (int)b.sideToMove()) < -3000 * depth) continue;
            }
            // CHANGED: futility — deeper limit (4 -> 6), improving-aware margin. SPRT-tune.
            if (!in_check && is_quiet && depth <= 6) {
                int fp_margin = 110 + 85 * depth - (improving ? 40 : 0);
                if (static_eval + fp_margin <= alpha) continue;
            }
            if (!is_pv && !in_check && depth <= 8) {
                if (is_quiet && !see_ge(b, moves[i], -20 * depth * depth)) continue;
                if (is_cap   && !see_ge(b, moves[i], -90 * depth)) continue;
            }
        }

        int mc = (int)b.sideToMove();
        int mf = moves[i].from().index();
        int mt = moves[i].to().index();
        PieceType mp = b.at<PieceType>(moves[i].from());

        if (is_quiet) {
            if (n_quiet < 64) quiet_list[n_quiet++] = moves[i];
            ++quiets_tried;
        }

        do_move(b, moves[i], ts);
        // NEW: record this move on the stack for the child's conthist lookups
        ts.mv_pt[ply] = mp;
        ts.mv_to[ply] = mt;
        bool gives_check = b.inCheck();

        int nd = depth - 1 + base_ext + ((moves[i] == hm) ? extension : 0);
        int score;

        if (i == 0) {
            score = -negamax(b, nd, -beta, -alpha, ply + 1, true, false, ts);
        } else {
            int r = 0;
            if (is_quiet && depth >= 3 && !in_check) {
                r = LMR_TABLE[std::min(depth, MAX_DEPTH - 1)][std::min(i, 63)];
                r += cut_node;
                if (is_pv) r = std::max(0, r - 1);
                if (gives_check) r -= 1;
                if (ply < MAX_PLY && (moves[i] == ts.killers[ply][0] || moves[i] == ts.killers[ply][1])) r -= 1;
                int qh = qhist_of(moves[i], (int)mp, mc);
                if (qh > 8000) r -= 1;
                else if (qh < -3000) r += 1;
                r = std::max(0, std::min(r, depth - 2));
            }
            if (r > 0)
                score = -negamax(b, nd - r, -alpha - 1, -alpha, ply + 1, true, true, ts);
            else
                score = -negamax(b, nd, -alpha - 1, -alpha, ply + 1, true, true, ts);
            if (score > alpha && r > 0)
                score = -negamax(b, nd, -alpha - 1, -alpha, ply + 1, true, true, ts);
            if (score > alpha && score < beta)
                score = -negamax(b, nd, -beta, -alpha, ply + 1, true, false, ts);
        }
        undo_move(b, moves[i], ts);
        if (info.stop.load(std::memory_order_relaxed)) return 0;

        if (score > best_score) { best_score = score; best_move = moves[i]; }
        if (score > alpha) {
            alpha = score;
            int bonus = history_bonus(depth);
            if (is_quiet) {
                add_hist(ts.history[mc][mf][mt], bonus);
                update_cont_hist(ts, ply, (int)mp, mt, bonus);          // CHANGED: 4 plies
            } else if (is_cap) {
                PieceType v = (moves[i].typeOf() == Move::ENPASSANT) ? PieceType::PAWN : b.at<PieceType>(mt);
                add_hist(ts.capHist[(int)mp][mt][(int)v], bonus);
            }
        }
        if (alpha >= beta) {
            if (ply >= 1 && ts.mv_pt[ply - 1] != PieceType::NONE && ts.mv_to[ply - 1] >= 0)
                ts.counterMoves[(int)ts.mv_pt[ply - 1]][ts.mv_to[ply - 1]] = moves[i];   // CHANGED
            if (is_quiet && ply < MAX_PLY) {
                if (moves[i] != ts.killers[ply][0]) {
                    ts.killers[ply][1] = ts.killers[ply][0];
                    ts.killers[ply][0] = moves[i];
                }
                int malus = -history_bonus(depth);
                for (int k = 0; k < n_quiet; ++k) {
                    if (quiet_list[k] == moves[i]) continue;
                    int f2 = quiet_list[k].from().index();
                    int t2 = quiet_list[k].to().index();
                    PieceType pt2 = b.at<PieceType>(quiet_list[k].from());
                    add_hist(ts.history[mc][f2][t2], malus);
                    update_cont_hist(ts, ply, (int)pt2, t2, malus);      // CHANGED: 4 plies
                }
            } else if (is_cap) {
                PieceType v = (moves[i].typeOf() == Move::ENPASSANT) ? PieceType::PAWN : b.at<PieceType>(mt);
                add_hist(ts.capHist[(int)mp][mt][(int)v], -history_bonus(depth));
            }
            if (excluded == Move::NO_MOVE)
                tt_store(key, beta, raw_eval, best_move, depth, TT_BETA, ply);
            return beta;
        }
    }

    if (excluded == Move::NO_MOVE) {
        TTFlag f = (alpha > orig_alpha) ? TT_EXACT : TT_ALPHA;
        tt_store(key, best_score, raw_eval, best_move, depth, f, ply);
        if (!in_check && depth >= 4 && std::abs(best_score) < MATE_BOUND)
            update_pawn_corr(ts, b, best_score - static_eval, depth);
    }
    return best_score;
}

// ─────────────────────────────────────────────
//  Output helpers
// ─────────────────────────────────────────────
std::string score_str(int score) {
    if (score >=  MATE_BOUND) return "score mate "  + std::to_string((MATE_SCORE - score + 1) / 2);
    if (score <= -MATE_BOUND) return "score mate -" + std::to_string((MATE_SCORE + score + 1) / 2);
    return "score cp " + std::to_string(score);
}
std::string pv_line(const Board& b, int max_len) {
    std::string s;
    Board b2 = b;
    Movelist ml;
    int len = std::min(max_len + 6, 20);
    for (int i = 0; i < len; ++i) {
        TTEntry* e = tt_probe(b2.hash());
        if (!e) break;
        Move m = e->get_best_move();
        if (m == Move::NO_MOVE) break;
        movegen::legalmoves(ml, b2);
        bool ok = false;
        for (const auto& mm : ml) if (mm == m) { ok = true; break; }
        if (!ok) break;
        s += " " + uci::moveToUci(m);
        b2.makeMove(m);
        if (b2.isRepetition()) break;
    }
    return s;
}
int tt_hashfull() {
    int used = 0;
    size_t stride = g_tt_size / 1000; if (!stride) stride = 1;
    for (int i = 0; i < 1000; ++i) {
        TTEntry& e = tt[i * stride];
        if (e.get_flag() != TT_NONE && e.get_gen() == tt_generation) ++used;
    }
    return 1000 - used;
}

// ─────────────────────────────────────────────
//  Thread Worker — CHANGED: move-stack init, atomic publish, new signatures
// ─────────────────────────────────────────────
void thread_worker(Board board, int thread_id, int max_depth, Move* best_move_out) {
    ThreadState& ts = thread_states[thread_id];
    ts.nodes = 0;
    ts.seldepth = 0;
    ts.thread_id = thread_id;
    ts.clear_history();
    std::fill(std::begin(ts.eval_stack), std::end(ts.eval_stack), NO_EVAL);
    std::fill(std::begin(ts.mv_pt), std::end(ts.mv_pt), PieceType::NONE);   // NEW
    std::fill(std::begin(ts.mv_to), std::end(ts.mv_to), -1);                // NEW
    if (g_nnue.loaded) {
        ts.accDepth = 0;
        g_nnue.refresh(board, ts.accStack[ts.accDepth]);
    }

    Move best_move = Move::NO_MOVE;
    {
        Movelist ml; movegen::legalmoves(ml, board);
        if (ml.empty()) { if (thread_id == 0) *best_move_out = Move::NO_MOVE; return; }
        best_move = ml[0];
    }

    int prev_score = 0;
    int prev_prev_score = 0;
    int ms_before_depth = 0;
    Move prev_best_move = Move::NO_MOVE;
    int best_stable = 0;
    int score_stable = 0;

    auto elapsed_ms = [&]() {
        return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - info.start_time).count();
    };

    for (int depth = 1; depth <= max_depth; ++depth) {
        int ms = elapsed_ms();
        if (info.max_time > 0 && ms >= info.max_time) {
            info.stop.store(true, std::memory_order_relaxed);
            break;
        }

        int score;
        bool completed = false;

        bool use_asp = depth >= 5 && std::abs(prev_score) < MATE_BOUND;
        if (use_asp) {
            int delta = 15;
            int a  = prev_score - delta;
            int bt = prev_score + delta;
            while (true) {
                score = negamax(board, depth, a, bt, 0, true, false, ts);
                if (info.stop.load(std::memory_order_relaxed)) break;
                if      (score <= a)   { a  -= delta; delta *= 2; }
                else if (score >= bt)  { bt += delta; delta *= 2; }
                else { completed = true; break; }
                if (a < -INF / 2) a = -INF;
                if (bt >  INF / 2) bt =  INF;
                if (a == -INF && bt == INF) {
                    score = negamax(board, depth, -INF, INF, 0, true, false, ts);
                    if (info.stop.load(std::memory_order_relaxed)) break;
                    completed = true;
                    break;
                }
            }
        } else {
            score = negamax(board, depth, -INF, INF, 0, true, false, ts);
            if (!info.stop.load(std::memory_order_relaxed)) completed = true;
        }

        if (!completed) {
            if (thread_id == 0 && depth > 1) *best_move_out = best_move;
            break;
        }

        prev_prev_score = prev_score;
        prev_score = score;

        {
            TTEntry* e = tt_probe(board.hash());
            if (e) {
                Move m = e->get_best_move();
                if (m != Move::NO_MOVE) {
                    Movelist ml; movegen::legalmoves(ml, board);
                    for (const auto& mm : ml) if (mm == m) { best_move = m; break; }
                }
            }
        }

        // NEW: publish to shared best move if this is the deepest result so far
        {
            int prev_d = g_shared_depth.load(std::memory_order_relaxed);
            if (depth > prev_d &&
                g_shared_depth.compare_exchange_strong(prev_d, depth)) {
                uint32_t raw = 0;
                std::memcpy(&raw, &best_move, sizeof(raw));
                g_shared_best.store(raw, std::memory_order_relaxed);
            }
        }

        if (best_move == prev_best_move) ++best_stable; else { best_stable = 0; prev_best_move = best_move; }
        if (std::abs(score - prev_prev_score) < 15) ++score_stable; else score_stable = 0;

        if (thread_id == 0) {
            int ms_now = elapsed_ms();
            int ms_depth = ms_now - ms_before_depth;
            uint64_t total = 0;
            for (int i = 0; i < num_threads; ++i) total += thread_states[i].nodes;
            int nps = (ms_now > 0) ? (int)(total * 1000ULL / ms_now) : 0;

            std::cout << "info depth " << depth
                      << " seldepth "  << ts.seldepth
                      << " "           << score_str(score)
                      << " nodes "     << total
                      << " nps "       << nps
                      << " hashfull "  << tt_hashfull()
                      << " time "      << ms_now
                      << " pv"         << pv_line(board, depth)
                      << "\n";
            std::cout.flush();

            if (info.max_time > 0) {
                double factor = 1.0;
                if (best_stable >= 6)      factor = 0.4;
                else if (best_stable >= 4) factor = 0.55;
                else if (best_stable >= 2) factor = 0.75;
                if (score_stable >= 3) factor *= 0.8;
                if (score_stable >= 5) factor *= 0.8;
                int target_ms = (int)(info.max_time * factor);
                int used_min  = info.max_time / 20;
                if (ms_now >= std::max(target_ms, used_min)) {
                    info.stop.store(true, std::memory_order_relaxed);
                    break;
                }
                long long pred = (long long)std::max(ms_depth, 1) * 4;
                if (ms_now + pred > info.max_time) {
                    if (ms_now >= info.max_time / 8) {
                        info.stop.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }
            ms_before_depth = ms_now;
        }
    }

    if (thread_id == 0) {
        if (best_move == Move::NO_MOVE) {
            Movelist ml; movegen::legalmoves(ml, board);
            if (!ml.empty()) best_move = ml[0];
        }
        *best_move_out = best_move;
    }
}

// CHANGED: prefers the shared atomic best move (validated legal), falls back to thread 0's.
Move search(Board& board, int max_depth, int time_ms) {
    info.stop.store(false, std::memory_order_relaxed);
    info.max_time = time_ms;
    info.start_time = std::chrono::steady_clock::now();
    tt_generation = (uint8_t)((tt_generation + 1) & 0x0F);
    g_shared_best.store(0, std::memory_order_relaxed);
    g_shared_depth.store(0, std::memory_order_relaxed);

    Move best_move = Move::NO_MOVE;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(thread_worker, board, i, max_depth, &best_move);
    for (auto& t : threads) t.join();

    uint32_t raw = g_shared_best.load(std::memory_order_relaxed);
    if (raw) {
        Move m;
        std::memcpy(&m, &raw, sizeof(m));
        Movelist ml; movegen::legalmoves(ml, board);
        for (const auto& mm : ml) if (mm == m) return m;   // legality guard vs torn/rare bad entry
    }
    return best_move;
}

// ─────────────────────────────────────────────
//  UCI
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
            std::cout << "id name Moth\n";
            std::cout << "id author Ranak Chongtham\n";
            std::cout << "option name EvalFile type string default " << g_nnue_path << "\n";
            std::cout << "option name Hash type spin default 256 min 1 max 4096\n";
            std::cout << "option name Threads type spin default 1 min 1 max " << MAX_THREADS << "\n";
            std::cout << "uciok\n";
        } else if (token == "isready") {
            std::cout << "readyok\n";
        } else if (token == "ucinewgame") {
            board.setFen(constants::STARTPOS);
            tt_clear();
            for (int i = 0; i < MAX_THREADS; ++i) thread_states[i].clear_history();
        } else if (token == "setoption") {
            std::string word, name, value;
            iss >> word;
            while (iss >> word && word != "value") {
                if (!name.empty()) name += " ";
                name += word;
            }
            std::getline(iss, value);
            if (!value.empty() && value.front() == ' ') value.erase(0, 1);
            if (name == "EvalFile" && !value.empty()) {
                g_nnue_path = value;
                if (g_nnue.load(g_nnue_path))
                    std::cout << "info string loaded NNUE from " << g_nnue_path << "\n";
                else
                    std::cout << "info string failed to load NNUE from " << g_nnue_path << "\n";
            } else if (name == "Threads") {
                num_threads = std::max(1, std::min(MAX_THREADS, std::stoi(value)));
            } else if (name == "Hash") {
                tt_resize((size_t)std::max(1, std::stoi(value)));
            }
        } else if (token == "position") {
            std::string p;
            iss >> p;
            if (p == "startpos") { board.setFen(constants::STARTPOS); iss >> p; }
            else if (p == "fen") {
                std::string fen;
                for (int i = 0; i < 6; ++i) { std::string f; iss >> f; if (i > 0) fen += " "; fen += f; }
                board.setFen(fen);
                iss >> p;
            }
            while (iss >> p) {
                Move m = uci::uciToMove(board, p);
                if (m == Move::NO_MOVE) break;
                board.makeMove(m);
            }
        } else if (token == "go") {
            int depth = -1, movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0;
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
                int myT = (board.sideToMove() == Color::WHITE) ? wtime : btime;
                int myI = (board.sideToMove() == Color::WHITE) ? winc  : binc;
                if (myT > 0) movetime = myT / 20 + myI / 2;
            }
            if (infinite)          { depth = MAX_DEPTH; movetime = 0; }
            else if (movetime > 0) { if (depth < 0) depth = MAX_DEPTH; }
            else                   { if (depth < 0) depth = 8; }

            Move best = search(board, depth, movetime);
            std::cout << "bestmove " << uci::moveToUci(best) << "\n";
        } else if (token == "stop") {
            info.stop.store(true, std::memory_order_relaxed);
        } else if (token == "quit") {
            break;
        } else if (token == "d") {
            std::cout << board << "\n";
            std::cout << "FEN: "  << board.getFen() << "\n";
            std::cout << "Hash: " << board.hash() << "\n";
            std::cout << "Eval: " << evaluate(board) << "\n";
        }
        std::cout.flush();
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    init_lmr();
    tt_resize(256);
    thread_states = new ThreadState[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; ++i) thread_states[i].clear_history();

    if (g_nnue.load(g_nnue_path))
        std::cerr << "info string loaded NNUE from " << g_nnue_path << "\n";
    else
        std::cerr << "info string no NNUE loaded. Falling back to material eval.\n";

    uci_loop();
    delete[] tt;
    delete[] thread_states;
    return 0;
}
