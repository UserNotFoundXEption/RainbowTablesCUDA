#pragma once
#include <stdint.h>
#include <cuda.h>
#include "des_tables.cuh"

__device__ static void apply_permutation(uint64_t& block, const int* table, int n) {
    uint64_t res = 0;
    for (int i = 0; i < n; i++) {
        int pos = table[i] - 1;
        res |= ((block >> (63 - pos)) & 1ULL) << (63 - i);
    }
    block = res;
}

__device__ static void initial_perm(uint64_t& block) { 
    apply_permutation(block, IP, 64);
}

__device__ static void final_perm(uint64_t& block) {
    apply_permutation(block, FP, 64);
}

__device__ static void expand(uint32_t R, uint64_t& ER) {
    ER = 0;
    for(int i = 0; i < 48; i++) {
        int bitpos = 32 - E[i]; // DES indeksuje od 1
        ER |= ((uint64_t)((R >> bitpos) & 1)) << (47 - i);
    }
}

__device__ static uint32_t sbox_substitution(uint64_t ER) {
    uint32_t output = 0;
    for (int i = 0; i < 8; i++) {
        // Wyciągamy 6-bit fragment
        int shift = 42 - (i * 6);
        uint8_t chunk = (ER >> shift) & 0x3F;

        int row = ((chunk & 0x20) >> 4) | (chunk & 0x01);  // bit 1 i 6
        int col = (chunk >> 1) & 0x0F;                     // bity 2–5
        uint8_t s_out = SBOX[i][row][col];

        output |= ((uint32_t)s_out) << (28 - 4 * i);
    }

    // Permutacja P (32 bitów)
    uint32_t permuted = 0;
    for (int i = 0; i < 32; i++) {
        int bit = (output >> (32 - P[i])) & 0x01;
        permuted |= bit << (31 - i);
    }

    return permuted;
}

__device__ static void des_round(uint32_t& L, uint32_t& R, uint64_t subkey) {
    uint64_t ER; expand(R, ER);
    ER ^= subkey;
    uint32_t f = sbox_substitution(ER);
    uint32_t temp = R;
    R = L ^ f;
    L = temp;
}

__device__ static void generate_subkeys(uint64_t key, uint64_t* subkeys) {
    uint64_t perm_key = 0;
    for (int i = 0; i < 56; i++)
        perm_key |= ((key >> (64 - PC1[i])) & 1ULL) << (55 - i);

    uint32_t C = (perm_key >> 28) & 0x0FFFFFFF;
    uint32_t D = perm_key & 0x0FFFFFFF;

    for (int i = 0; i < 16; i++) {
        C = ((C << SHIFTS[i]) | (C >> (28 - SHIFTS[i]))) & 0x0FFFFFFF;
        D = ((D << SHIFTS[i]) | (D >> (28 - SHIFTS[i]))) & 0x0FFFFFFF;
        uint64_t CD = (((uint64_t)C) << 28) | D;
        uint64_t subkey = 0;
        for (int j = 0; j < 48; j++)
            subkey |= ((CD >> (56 - PC2[j])) & 1ULL) << (47 - j);
        subkeys[i] = subkey;
    }
}

__device__ static uint64_t des_encrypt(uint64_t block, uint64_t* subkeys) {
    apply_permutation(block, IP, 64);
    uint32_t L = (block >> 32) & 0xFFFFFFFF;
    uint32_t R = block & 0xFFFFFFFF;
    for (int i = 0; i < 16; i++)
        des_round(L, R, subkeys[i]);
    uint64_t preout = ((uint64_t)R << 32) | L;
    apply_permutation(preout, FP, 64);
    return preout;
}
