// nnue.h
// main evaluation code
#pragma once

#include "chess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue {

constexpr int INPUT_DIM = 716;
constexpr int L1_DIM    = 512;
constexpr int L2_DIM    = 64;

inline chess::Color flip_color(chess::Color c) {
    return (c == chess::Color::WHITE) ? chess::Color::BLACK : chess::Color::WHITE;
}

// ---------------------------------------------------------------------
// Low-level vector kernels. Each has an AVX2(+FMA) fast path and a plain
// scalar fallback, so correctness never depends on compiler flags.
// ---------------------------------------------------------------------

// acc[j] += row[j] for j in [0, n)
inline void add_row(float* __restrict acc, const float* __restrict row, int n) {
#if defined(__AVX2__)
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 a = _mm256_loadu_ps(acc + j);
        __m256 r = _mm256_loadu_ps(row + j);
        _mm256_storeu_ps(acc + j, _mm256_add_ps(a, r));
    }
    for (; j < n; ++j) acc[j] += row[j];
#else
    for (int j = 0; j < n; ++j) acc[j] += row[j];
#endif
}

// acc[k] += v * row[k] for k in [0, n)
inline void add_scaled_row(float* __restrict acc, const float* __restrict row, float v, int n) {
#if defined(__AVX2__)
    __m256 vv = _mm256_set1_ps(v);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m256 a = _mm256_loadu_ps(acc + k);
        __m256 r = _mm256_loadu_ps(row + k);
#if defined(__FMA__)
        a = _mm256_fmadd_ps(r, vv, a);
#else
        a = _mm256_add_ps(a, _mm256_mul_ps(r, vv));
#endif
        _mm256_storeu_ps(acc + k, a);
    }
    for (; k < n; ++k) acc[k] += v * row[k];
#else
    for (int k = 0; k < n; ++k) acc[k] += v * row[k];
#endif
}

// a[j] = clamp(a[j], 0, 1) for j in [0, n)  — the trainer's clipped ReLU
inline void clip01(float* __restrict a, int n) {
#if defined(__AVX2__)
    __m256 zero = _mm256_setzero_ps();
    __m256 one  = _mm256_set1_ps(1.0f);
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 v = _mm256_loadu_ps(a + j);
        v = _mm256_max_ps(v, zero);
        v = _mm256_min_ps(v, one);
        _mm256_storeu_ps(a + j, v);
    }
    for (; j < n; ++j) a[j] = std::min(std::max(a[j], 0.0f), 1.0f);
#else
    for (int j = 0; j < n; ++j) a[j] = std::min(std::max(a[j], 0.0f), 1.0f);
#endif
}

// Kept for reference / debugging / tooling — NOT used by evaluate() below,
// which extracts+accumulates in one bitboard pass instead of materializing
// this list. Semantics match the original 1:1.
inline void extract_features(const chess::Board& board, std::vector<int>& out,
                              chess::Color perspective) {
    using namespace chess;
    out.clear();
    out.reserve(34);

    for (int sqi = 0; sqi < 64; ++sqi) {
        Square sq(sqi);
        Piece p = board.at(sq);
        if (p == Piece::NONE) continue;
        PieceType pt = p.type();
        if (pt == PieceType::KING) continue;

        int p_type      = (int)pt;
        Color pc        = p.color();
        int feat_sq     = (perspective == Color::WHITE) ? sqi : (sqi ^ 56);
        int is_enemy    = (pc != perspective) ? 1 : 0;
        int channel     = is_enemy * 5 + p_type;
        out.push_back(channel * 64 + feat_sq);
    }

    Square ksq   = board.kingSq(perspective);
    int    kfeat = (perspective == Color::WHITE) ? ksq.index() : (ksq.index() ^ 56);
    out.push_back(640 + kfeat);

    std::string cr = board.getCastleString();
    bool wk = cr.find('K') != std::string::npos;
    bool wq = cr.find('Q') != std::string::npos;
    bool bk = cr.find('k') != std::string::npos;
    bool bq = cr.find('q') != std::string::npos;

    if (perspective == Color::WHITE) {
        if (wk) out.push_back(704);
        if (wq) out.push_back(705);
        if (bk) out.push_back(706);
        if (bq) out.push_back(707);
    } else {
        if (bk) out.push_back(704);
        if (bq) out.push_back(705);
        if (wk) out.push_back(706);
        if (wq) out.push_back(707);
    }

    Square ep = board.enpassantSq();
    if (ep.index() < 64) {
        int epFile = ep.index() % 8;
        out.push_back(708 + epFile);
    }
}

struct Network {
    // Feature-major: W1[feature * L1_DIM + neuron] — lets us accumulate by
    // summing whole rows for each active (sparse) input feature.
    std::vector<float> W1;
    std::vector<float> b1;
    // Input-major: W2[input_neuron * L2_DIM + out_neuron] — the concatenated
    // 1024-wide layer is dense, so this is a plain matrix multiply, done
    // as a row-major accumulation (see accumulate_l2 below) for cache
    // locality and to exploit clipped-ReLU sparsity.
    std::vector<float> W2;
    std::vector<float> b2;
    std::vector<float> W3; // [L2_DIM]
    float b3   = 0.0f;
    bool loaded = false;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { loaded = false; return false; }

        char magic[8] = {0};
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "MOTHNNU1", 8) != 0) { loaded = false; return false; }

        int32_t dims[3] = {0, 0, 0};
        f.read(reinterpret_cast<char*>(dims), sizeof(dims));
        if (!f || dims[0] != INPUT_DIM || dims[1] != L1_DIM || dims[2] != L2_DIM) {
            loaded = false;
            return false;
        }

        W1.resize((size_t)INPUT_DIM * L1_DIM);
        b1.resize(L1_DIM);
        W2.resize((size_t)2 * L1_DIM * L2_DIM);
        b2.resize(L2_DIM);
        W3.resize(L2_DIM);

        f.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(W3.data()), W3.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(&b3), sizeof(float));

        loaded = (bool)f || f.eof();
        return loaded;
    }

    // Builds one perspective's L1 accumulator directly from the board,
    // without ever materializing a feature-index list. `wk/wq/bk/bq` and
    // `epFile` are computed once by the caller and shared between both
    // perspectives (they don't depend on which perspective we're building).
    inline void accumulate_perspective(const chess::Board& board, chess::Color perspective,
                                        bool wk, bool wq, bool bk, bool bq, int epFile,
                                        std::array<float, L1_DIM>& acc) const {
        using namespace chess;

        std::copy(b1.begin(), b1.end(), acc.begin());

        static constexpr PieceType PTS[5] = {
            PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
            PieceType::ROOK, PieceType::QUEEN
        };

        const bool whitePersp = (perspective == Color::WHITE);

        for (int p_type = 0; p_type < 5; ++p_type) {
            for (int side = 0; side < 2; ++side) {
                Color pc = (side == 0) ? Color::WHITE : Color::BLACK;
                Bitboard bb = board.pieces(PTS[p_type], pc);
                if (bb.empty()) continue;

                int is_enemy = (pc != perspective) ? 1 : 0;
                int channel  = is_enemy * 5 + p_type;
                const float* channelBase = &W1[(size_t)channel * 64 * L1_DIM];

                while (bb) {
                    Square sq = bb.pop(); // extracts + clears the LSB
                    int sqi = sq.index();
                    int feat_sq = whitePersp ? sqi : (sqi ^ 56);
                    add_row(acc.data(), channelBase + (size_t)feat_sq * L1_DIM, L1_DIM);
                }
            }
        }

        // own king
        {
            Square ksq   = board.kingSq(perspective);
            int    kfeat = whitePersp ? ksq.index() : (ksq.index() ^ 56);
            add_row(acc.data(), &W1[(size_t)(640 + kfeat) * L1_DIM], L1_DIM);
        }

        // castling rights, own/enemy reordered per perspective (same
        // semantics as the original: slots are own-KS, own-QS, enemy-KS,
        // enemy-QS)
        bool ownKS, ownQS, enemyKS, enemyQS;
        if (whitePersp) { ownKS = wk; ownQS = wq; enemyKS = bk; enemyQS = bq; }
        else            { ownKS = bk; ownQS = bq; enemyKS = wk; enemyQS = wq; }

        if (ownKS)   add_row(acc.data(), &W1[(size_t)704 * L1_DIM], L1_DIM);
        if (ownQS)   add_row(acc.data(), &W1[(size_t)705 * L1_DIM], L1_DIM);
        if (enemyKS) add_row(acc.data(), &W1[(size_t)706 * L1_DIM], L1_DIM);
        if (enemyQS) add_row(acc.data(), &W1[(size_t)707 * L1_DIM], L1_DIM);

        // en passant (perspective-independent index)
        if (epFile >= 0) {
            add_row(acc.data(), &W1[(size_t)(708 + epFile) * L1_DIM], L1_DIM);
        }

        clip01(acc.data(), L1_DIM);
    }

    // Returns the evaluation from the side-to-move's perspective, in the
    // same centipawn-ish scale the training targets used (raw cp, not
    // normalized) — matches how the rest of the engine's evaluate() works.
    int evaluate(const chess::Board& board) const {
        using namespace chess;

        // Shared, board-global info computed once (was previously
        // recomputed inside each perspective's extraction).
        std::string cr = board.getCastleString();
        bool wk = cr.find('K') != std::string::npos;
        bool wq = cr.find('Q') != std::string::npos;
        bool bk = cr.find('k') != std::string::npos;
        bool bq = cr.find('q') != std::string::npos;

        Square ep = board.enpassantSq();
        int epFile = (ep.index() < 64) ? (ep.index() % 8) : -1;

        alignas(32) std::array<float, L1_DIM> wAcc{}, bAcc{};
        accumulate_perspective(board, Color::WHITE, wk, wq, bk, bq, epFile, wAcc);
        accumulate_perspective(board, Color::BLACK, wk, wq, bk, bq, epFile, bAcc);

        alignas(32) std::array<float, 2 * L1_DIM> concat{};
        bool stmWhite = (board.sideToMove() == Color::WHITE);
        if (stmWhite) {
            std::copy(wAcc.begin(), wAcc.end(), concat.begin());
            std::copy(bAcc.begin(), bAcc.end(), concat.begin() + L1_DIM);
        } else {
            std::copy(bAcc.begin(), bAcc.end(), concat.begin());
            std::copy(wAcc.begin(), wAcc.end(), concat.begin() + L1_DIM);
        }

        // L2: for-m/for-k order gives contiguous 64-float row reads
        // (vs. the strided k-major/m-minor order), and skips rows whose
        // input activation is exactly zero (clipped ReLU produces true
        // zeros often).
        alignas(32) std::array<float, L2_DIM> hidden;
        std::copy(b2.begin(), b2.end(), hidden.begin());

        for (int m = 0; m < 2 * L1_DIM; ++m) {
            float v = concat[m];
            if (v == 0.0f) continue;
            add_scaled_row(hidden.data(), &W2[(size_t)m * L2_DIM], v, L2_DIM);
        }
        clip01(hidden.data(), L2_DIM);

        float out = b3;
        for (int k = 0; k < L2_DIM; ++k) out += hidden[k] * W3[k];

        return (int)std::lround(out);
    }
};

} // namespace nnue
