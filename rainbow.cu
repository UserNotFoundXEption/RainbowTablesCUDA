#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <stdint.h>
#include "des_tables.h"

#define PW_LEN 8
#define CHAIN_LEN 10000
#define THREADS_PER_BLOCK 512
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"

__device__ void apply_permutation(uint64_t& block, const int* table, int n) {
    uint64_t res = 0;
    for (int i = 0; i < n; i++) {
        int pos = table[i] - 1;
        res |= ((block >> (63 - pos)) & 1ULL) << (63 - i);
    }
    block = res;
}

__device__ void initial_perm(uint64_t& block) { 
    apply_permutation(block, IP, 64);
}

__device__ void final_perm(uint64_t& block) {
    apply_permutation(block, FP, 64);
}

__device__ void expand(uint32_t R, uint64_t& ER) {
    ER = 0;
    for(int i = 0; i < 48; i++) {
        int bitpos = 32 - E[i]; // DES indeksuje od 1
        ER |= ((uint64_t)((R >> bitpos) & 1)) << (47 - i);
    }
}

__device__ uint32_t sbox_substitution(uint64_t ER) {
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

__device__ void des_round(uint32_t& L, uint32_t& R, uint64_t subkey) {
    uint64_t ER; expand(R, ER);
    ER ^= subkey;
    uint32_t f = sbox_substitution(ER);
    uint32_t temp = R;
    R = L ^ f;
    L = temp;
}

__device__ void generate_subkeys(uint64_t key, uint64_t* subkeys) {
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

__device__ uint64_t des_encrypt(uint64_t block, uint64_t* subkeys) {
    apply_permutation(block, IP, 64);
    uint32_t L = (block >> 32) & 0xFFFFFFFF;
    uint32_t R = block & 0xFFFFFFFF;
    for (int i = 0; i < 16; i++)
        des_round(L, R, subkeys[i]);
    uint64_t preout = ((uint64_t)R << 32) | L;
    apply_permutation(preout, FP, 64);
    return preout;
}

__device__ uint64_t text_to_block(const char* pw) {
    uint64_t block = 0;
    for (int i = 0; i < PW_LEN; i++)
        block |= ((uint64_t)pw[i]) << ((5 - i) * 8);
    return block;
}

__device__ void block_to_pw(uint64_t block, char* out) {
    for (int i = 0; i < PW_LEN; i++)
        out[i] = (block >> ((5 - i) * 8)) & 0xFF;
    out[PW_LEN] = '\0';
}

__device__ void reduce(uint64_t block, int round, char* out) {
    for (int i = 0; i < PW_LEN; i++) {
        uint8_t byte = (block >> (8 * (i % 8))) & 0xFF;
        int idx = (byte + round + i) % 26;
        out[i] = ALPHABET[idx];
    }
    out[PW_LEN] = '\0';
}

__global__ void kernel(char* out, int total_chains, uint64_t key) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total_chains) return;

    char pw[PW_LEN + 1], red[PW_LEN + 1];
    uint64_t subkeys[16];
    generate_subkeys(key, subkeys);

    // Start password = "aaaaaa" + id (w ascii)
    for (int i = 0; i < PW_LEN; i++) pw[i] = 'a';
    int n = id;
    for (int i = PW_LEN - 1; i >= 0 && n > 0; i--) {
        pw[i] += n % 26;
        n /= 26;
    }
    pw[PW_LEN] = '\0';

    char start[PW_LEN + 1];
    memcpy(start, pw, PW_LEN + 1);

    for (int i = 0; i < CHAIN_LEN; i++) {
        uint64_t blk = text_to_block(pw);
        blk = des_encrypt(blk, subkeys);
        reduce(blk, i, red);
        memcpy(pw, red, PW_LEN + 1);
	pw[PW_LEN] = '\0';
    }

    int idx = id * 2 * PW_LEN;
    memcpy(&out[idx], start, PW_LEN);
    memcpy(&out[idx + PW_LEN], pw, PW_LEN);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Użycie: %s <liczba_łańcuchów> <klucz_hex>\n", argv[0]);
        return 1;
    }
    
    int total_chains = atoi(argv[1]);
    uint64_t key = strtoull(argv[2], NULL, 16);

    int threads = (total_chains + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t size = total_chains * 2 * PW_LEN;
    char* d_out;
    cudaMalloc(&d_out, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<threads, THREADS_PER_BLOCK>>>(d_out, total_chains, key);

    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU dzilal przez %f sekund.\n", ms / 1000.0);

    char* h_out = new char[size];
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    FILE* f = fopen("rainbow_des.txt", "w");
    for (int i = 0; i < total_chains; i++) {
        for (int j = 0; j < PW_LEN; j++) fputc(h_out[i * 2 * PW_LEN + j], f);
		fputc(':', f);
		for (int j = 0; j < PW_LEN; j++) fputc(h_out[i * 2 * PW_LEN + PW_LEN + j], f);
		fputc('\n', f);
    }
    fclose(f);
    delete[] h_out;
    cudaFree(d_out);
    return 0;
}
