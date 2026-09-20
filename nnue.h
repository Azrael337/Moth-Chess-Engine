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

constexpr int INPUT_DIM = 768;          // 12 channels (6 piece types x own/enemy) x 64 squares
constexpr int L1_DIM    = 256;
constexpr int CONCAT    = 2 * L1_DIM;   // 512 = [stm_acc | nstm_acc]

// ── int16 feature rows widened into int32 accumulators (incremental hot path) ──
inline void add_row_i16(int32_t* __restrict acc, const int16_t* __restrict row, int n) {
#if defined(__AVX2__)
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m128i r16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
        __m256i r32 = _mm256_cvtepi16_epi32(r16);
        __m256i a32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + j));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j), _mm256_add_epi32(a32, r32));
    }
    for (; j < n; ++j) acc[j] += (int32_t)row[j];
#else
    for (int j = 0; j < n; ++j) acc[j] += (int32_t)row[j];
#endif
}
inline void sub_row_i16(int32_t* __restrict acc, const int16_t* __restrict row, int n) {
#if defined(__AVX2__)
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m128i r16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
        __m256i r32 = _mm256_cvtepi16_epi32(r16);
        __m256i a32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + j));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j), _mm256_sub_epi32(a32, r32));
    }
    for (; j < n; ++j) acc[j] -= (int32_t)row[j];
#else
    for (int j = 0; j < n; ++j) acc[j] -= (int32_t)row[j];
#endif
}

struct Accumulator {
    alignas(32) std::array<int32_t, L1_DIM> white{};
    alignas(32) std::array<int32_t, L1_DIM> black{};
};

struct Network {
    // Feature transformer: int16 weights, bias pre-folded into accumulator space
    std::vector<int16_t> W1;        // [INPUT_DIM][L1_DIM], input-major
    std::vector<int32_t> b1;        // [L1_DIM], in W1 integer scale
    float W1_scale     = 1.0f;
    float W1_inv_scale = 1.0f;

    // Output layer: flat [CONCAT], dequantized to float at load
    std::vector<float> W2;          // [CONCAT]
    float b3   = 0.0f;
    bool loaded = false;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { loaded = false; return false; }

        char magic[8] = {0};
        f.read(magic, 8);
        if (!f || std::memcmp(magic, "MOTHTIN1", 8) != 0) { loaded = false; return false; }

        int32_t dims[4] = {0,0,0,0};
        f.read(reinterpret_cast<char*>(dims), sizeof(dims));
        if (!f || dims[0] != INPUT_DIM || dims[1] != L1_DIM ||
            dims[2] != CONCAT  || dims[3] != 1) { loaded = false; return false; }

        f.read(reinterpret_cast<char*>(&W1_scale), sizeof(float));
        W1.resize((size_t)INPUT_DIM * L1_DIM);
        f.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(int16_t));
        W1_inv_scale = (W1_scale != 0.0f) ? 1.0f / W1_scale : 1.0f;

        std::vector<float> b1f(L1_DIM);
        f.read(reinterpret_cast<char*>(b1f.data()), b1f.size() * sizeof(float));
        b1.resize(L1_DIM);
        for (int j = 0; j < L1_DIM; ++j)
            b1[j] = (int32_t)std::lround((double)b1f[j] * (double)W1_scale);

        float w2s = 1.0f;
        f.read(reinterpret_cast<char*>(&w2s), sizeof(float));
        std::vector<int16_t> w2q(CONCAT);
        f.read(reinterpret_cast<char*>(w2q.data()), w2q.size() * sizeof(int16_t));
        float inv2 = (w2s != 0.0f) ? 1.0f / w2s : 1.0f;
        W2.resize(CONCAT);
        for (int i = 0; i < CONCAT; ++i) W2[i] = (float)w2q[i] * inv2;

        f.read(reinterpret_cast<char*>(&b3), sizeof(float));
        loaded = (bool)f || f.eof();
        return loaded;
    }

    // Channel layout — must match the trainer exactly:
    //   own view:  channel = (piece is enemy ? 6 : 0) + piece_type
    //   white view squares unflipped; black view squares ^ 56
    inline void toggle_feature(Accumulator& acc, int pt, chess::Color c,
                               int sq, bool adding) const {
        const int chW = ((c == chess::Color::BLACK) ? 6 : 0) + pt;
        const int16_t* rowW = &W1[(size_t)(chW * 64 + sq) * L1_DIM];

        const int chB = ((c == chess::Color::WHITE) ? 6 : 0) + pt;
        const int16_t* rowB = &W1[(size_t)(chB * 64 + (sq ^ 56)) * L1_DIM];

        if (adding) {
            add_row_i16(acc.white.data(), rowW, L1_DIM);
            add_row_i16(acc.black.data(), rowB, L1_DIM);
        } else {
            sub_row_i16(acc.white.data(), rowW, L1_DIM);
            sub_row_i16(acc.black.data(), rowB, L1_DIM);
        }
    }

    void refresh(const chess::Board& board, Accumulator& acc) const {
        using namespace chess;
        std::copy(b1.begin(), b1.end(), acc.white.begin());
        std::copy(b1.begin(), b1.end(), acc.black.begin());
        Bitboard occ = board.occ();
        while (occ) {
            Square sq = Square(occ.pop());
            Piece p = board.at(sq);
            if (p == Piece::NONE) continue;
            toggle_feature(acc, (int)p.type(), p.color(), sq.index(), true);
        }
    }

    // Board is in PRE-move state. Kings are ordinary features here — no special
    // handling. Castling never reaches this (moth.cpp refreshes on castling).
    void update_for_move(Accumulator& acc, const chess::Board& board,
                         const chess::Move& move) const {
        using namespace chess;
        Square from = move.from(), to = move.to();
        Piece moved = board.at(from);
        if (moved == Piece::NONE) return;

        toggle_feature(acc, (int)moved.type(), moved.color(), from.index(), false);

        if (move.typeOf() == Move::ENPASSANT) {
            int capSq = to.index() + ((moved.color() == Color::WHITE) ? -8 : 8);
            toggle_feature(acc, (int)PieceType::PAWN, ~moved.color(), capSq, false);
        } else {
            Piece cap = board.at(to);
            if (cap != Piece::NONE)
                toggle_feature(acc, (int)cap.type(), cap.color(), to.index(), false);
        }

        int destPT = (move.typeOf() == Move::PROMOTION)
                         ? (int)move.promotionType() : (int)moved.type();
        toggle_feature(acc, destPT, moved.color(), to.index(), true);
    }

    // No castling/ep inputs in the 768 layout -> no-op
    void update_for_null_move(Accumulator& acc, const chess::Board& board) const {
        (void)acc; (void)board;
    }

    // Output = single dot: concat([own|nstm] clipped) . W2 + b3
    int evaluate_from_accumulator(const Accumulator& acc, chess::Color stm) const {
        const int32_t* own   = (stm == chess::Color::WHITE) ? acc.white.data() : acc.black.data();
        const int32_t* enemy = (stm == chess::Color::WHITE) ? acc.black.data() : acc.white.data();
#if defined(__AVX2__)
        const __m256 inv  = _mm256_set1_ps(W1_inv_scale);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 one  = _mm256_set1_ps(1.0f);
        __m256 sum = _mm256_setzero_ps();

        auto feed = [&](const int32_t* a, const float* w) {
            for (int j = 0; j < L1_DIM; j += 8) {
                __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
                __m256 v  = _mm256_mul_ps(_mm256_cvtepi32_ps(q), inv);
                v = _mm256_max_ps(v, zero);
                v = _mm256_min_ps(v, one);
#if defined(__FMA__)
                sum = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + j), sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(v, _mm256_loadu_ps(w + j)));
#endif
            }
        };
        feed(own,   W2.data());
        feed(enemy, W2.data() + L1_DIM);

        alignas(32) float red[8];
        _mm256_store_ps(red, sum);
        float out = b3;
        for (int k = 0; k < 8; ++k) out += red[k];
        return (int)std::lround(out);
#else
        float out = b3;
        auto feed = [&](const int32_t* a, const float* w) {
            for (int j = 0; j < L1_DIM; ++j) {
                float v = (float)a[j] * W1_inv_scale;
                if (v <= 0.0f) continue;
                if (v > 1.0f)  v = 1.0f;
                out += v * w[j];
            }
        };
        feed(own,   W2.data());
        feed(enemy, W2.data() + L1_DIM);
        return (int)std::lround(out);
#endif
    }

    int evaluate(const chess::Board& board) const {
        Accumulator acc;
        refresh(board, acc);
        return evaluate_from_accumulator(acc, board.sideToMove());
    }
};

} // namespace nnue
