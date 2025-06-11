#pragma once
#include <stdint.h>
#include <cuda.h>
#include "des_tables.cuh"

typedef struct {
    uint32_t bits[64];
} bs_block;

__device__ void apply_permutation(const bs_block& block, const int* table, bs_block& result) {
    for (int i = 0; i < 64; ++i) {
        result.bits[i] = block.bits[table[i] - 1];
    }
}

__device__ void xor_bitsliced(uint32_t* out, const uint32_t* in1, const uint32_t* in2, int len = 48) {
    for (int i = 0; i < len; ++i) {
        out[i] = in1[i] ^ in2[i];
    }
}

__device__ void expand_bitsliced(const uint32_t* R, uint32_t* E_out) {
    for (int i = 0; i < 48; ++i) {
        int idx = E[i] - 1;
        E_out[i] = R[idx];
    }
}

__device__ void sbox_substitution_bitsliced(const uint32_t* in, uint32_t* out) {
    #pragma unroll 2
    for (int s = 0; s < 8; ++s) {
        uint32_t b0 = in[s * 6 + 0];
        uint32_t b5 = in[s * 6 + 5];

        // row bits
        uint32_t row0 = (~b0) & (~b5);
        uint32_t row1 = (~b0) & b5;
        uint32_t row2 = b0 & (~b5);
        uint32_t row3 = b0 & b5;

        for (int bit = 0; bit < 4; ++bit) {
            uint32_t outbit = 0;

            // Precache b1-b4 for this S-box
            uint32_t b1 = in[s * 6 + 1];
            uint32_t b2 = in[s * 6 + 2];
            uint32_t b3 = in[s * 6 + 3];
            uint32_t b4 = in[s * 6 + 4];

            // Precompute masks for columns 0..15
            // We'll loop over columns, but with minimal unroll
            #pragma unroll 1
            for (int r = 0; r < 4; ++r) {
                uint32_t rmatch = (r == 0) ? row0 : (r == 1) ? row1 : (r == 2) ? row2 : row3;
                #pragma unroll 1
                for (int c = 0; c < 16; ++c) {
                    // Check if output bit is set in SBOX[s][r][c]
                    uint8_t val = (SBOX[s][r][c] >> (3 - bit)) & 1;
                    if (val == 0) continue;

                    // Compute mask: col bits = b1 b2 b3 b4
                    uint32_t mask = ~0u;
                    mask &= ((c & 8) ? b1 : ~b1);
                    mask &= ((c & 4) ? b2 : ~b2);
                    mask &= ((c & 2) ? b3 : ~b3);
                    mask &= ((c & 1) ? b4 : ~b4);

                    outbit |= mask & rmatch;
                }
            }
            out[s * 4 + bit] = outbit;
        }
    }
}

__device__ void permute_p_bitsliced(const uint32_t* in, uint32_t* out) {
    for (int i = 0; i < 32; ++i) {
        int idx = P[i] - 1;
        out[i] = in[idx];
    }
}

__device__ void des_round_bitsliced(uint32_t* L, uint32_t* R, const uint32_t* subkey) {
    uint32_t ER[48];
    expand_bitsliced(R, ER);

    uint32_t ERK[48];
    xor_bitsliced(ERK, ER, subkey);

    uint32_t SBOX[32];
    sbox_substitution_bitsliced(ERK, SBOX);

    uint32_t f_out[32];
    permute_p_bitsliced(SBOX, f_out);

    uint32_t newR[32];
    xor_bitsliced(newR, L, f_out, 32);

    for (int i = 0; i < 32; ++i) {
        L[i] = R[i];
        R[i] = newR[i];
    }
}

__device__ void des_encrypt(bs_block& block, const uint32_t subkeys_bs[16][48]) {
   bs_block temp;
    apply_permutation(block, IP, temp);

    uint32_t L[32], R[32];
    for (int i = 0; i < 32; ++i) {
        L[i] = temp.bits[i];
        R[i] = temp.bits[32 + i];
    }

    for (int round = 0; round < 16; ++round) {
        des_round_bitsliced(L, R, subkeys_bs[round]);
    }

    for (int i = 0; i < 32; ++i) {
        temp.bits[i] = R[i];
        temp.bits[32 + i] = L[i];
    }

    apply_permutation(temp, FP, block);
}

