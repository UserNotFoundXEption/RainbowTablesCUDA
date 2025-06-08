#pragma once
#include <stdint.h>
#include <cuda.h>
#include "des_tables.cuh"

__device__ uint64_t apply_permutation(uint64_t block, const int* table) {
    uint64_t res = 0;
    #pragma unroll 63
    for (int i = 0; i < 64; i++) {
        int pos = table[i] - 1;
        res |= ((block >> (63 - pos)) & 1ULL) << (63 - i);
    }
    return res;
}

__device__  uint64_t expand(uint32_t R) {
    uint64_t ER = 0;
    #pragma unroll
    for(int i = 0; i < 48; i++) {
        int bitpos = 32 - E[i]; // DES indeksuje od 1
        ER |= ((uint64_t)((R >> bitpos) & 1U)) << (47 - i);
    }
    return ER;
}

__device__  uint32_t sbox_substitution(uint64_t ER) {
    uint32_t output = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // Wyciągamy 6-bit fragment
        int shift = 42 - (i * 6);
        uint8_t chunk = (ER >> shift) & 0x3F;

        int row = ((chunk & 0x20) >> 4) | (chunk & 0x01);  // bit 1 i 6
        int col = (chunk >> 1) & 0x0F;                     // bity 2–5
        uint8_t s_out = SBOX[i][row][col];

        output |= ((uint32_t)s_out) << (28 - 4 * i);
    }

    uint32_t permuted = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int bit = (output >> (32 - P[i])) & 0x01;
        permuted |= bit << (31 - i);
    }

    return permuted;
}

__device__  void des_round(uint32_t& L, uint32_t& R, uint64_t subkey) {
    uint64_t ER = expand(R);
    ER ^= subkey;
    uint32_t f = sbox_substitution(ER);
    uint32_t temp = R;
    R = L ^ f;
    L = temp;
}

__device__  uint64_t des_encrypt(uint64_t block, const uint64_t* __restrict__ subkeys) {
    block = apply_permutation(block, IP);
    uint32_t L = (block >> 32) & 0xFFFFFFFF;
    uint32_t R = block & 0xFFFFFFFF;
    for (int i = 0; i < 16; i++)
        des_round(L, R, subkeys[i]);
    block = ((uint64_t)R << 32) | L;
    block = apply_permutation(block, FP);
    return block;
}
