#include <cstdio>
#include <cstring>
#include <stdint.h> 
#include "des.cuh"
#include "des_tables.cuh"

#define PW_LEN 8
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"

__device__ __constant__ uint64_t subkeys[16];

int PC1_host[56];
int SHIFTS_host[16];
int PC2_host[48];

__device__ uint64_t reduce(uint64_t block, int round) {
    uint64_t result = 0;
    for (int i = 0; i < PW_LEN; i++) {
        uint8_t byte = (block >> (8 * (i % 8))) & 0xFF;
        int idx = (byte + round + i) % 26;
        result |= ((uint64_t)('a' + idx)) << ((PW_LEN - 1 - i) * 8);
    }
    return result;
}

__global__ void kernel(uint64_t* out, int total_chains, int chain_len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total_chains) return;

    uint64_t pw = 0;
    for (int i = 0; i < PW_LEN; i++) {
        pw |= ((uint64_t)'a') << ((PW_LEN - 1 - i) * 8);
    }
    int n = id;
    for (int i = PW_LEN - 1; i >= 0 && n > 0; i--) {
        int shift = (PW_LEN - 1 - i) * 8;
        uint8_t c = ((pw >> shift) & 0xFF) + (n % 26);
        pw = (pw & ~(0xFFULL << shift)) | ((uint64_t)c << shift);
        n /= 26;
    }

    uint64_t start = pw;
    for (int i = 0; i < chain_len; i++) {
        pw = des_encrypt(pw, subkeys);
        pw = reduce(pw, i);
    }

    out[id * 2] = start;
    out[id * 2 + 1] = pw;
}

void generate_subkeys(uint64_t key, uint64_t* subkeys) {
    uint64_t perm_key = 0;
    for (int i = 0; i < 56; i++)
        perm_key |= ((key >> (64 - PC1_host[i])) & 1ULL) << (55 - i);

    uint32_t C = (perm_key >> 28) & 0x0FFFFFFF;
    uint32_t D = perm_key & 0x0FFFFFFF;

    for (int i = 0; i < 16; i++) {
        C = ((C << SHIFTS_host[i]) | (C >> (28 - SHIFTS_host[i]))) & 0x0FFFFFFF;
        D = ((D << SHIFTS_host[i]) | (D >> (28 - SHIFTS_host[i]))) & 0x0FFFFFFF;
        uint64_t CD = (((uint64_t)C) << 28) | D;
        uint64_t subkey = 0;
        for (int j = 0; j < 48; j++)
            subkey |= ((CD >> (56 - PC2_host[j])) & 1ULL) << (47 - j);
        subkeys[i] = subkey;
    }
}


int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Użycie: %s <liczba_łańcuchów> <dlugosc_lancucha> <klucz_hex> <watki_na_blok>\n", argv[0]);
        return 1;
    }
    
    cudaMemcpyToSymbol(PC1, PC1_host, sizeof(PC1_host));
    cudaMemcpyToSymbol(SHIFTS, SHIFTS_host, sizeof(SHIFTS_host));
    cudaMemcpyToSymbol(PC2, PC2_host, sizeof(PC2_host));
    
    int total_chains = atoi(argv[1]);
    int chain_len = atoi(argv[2]);
    uint64_t key = strtoull(argv[3], NULL, 16);
    int threads_per_block = atoi(argv[4]);
    
    uint64_t h_subkeys[16];
    generate_subkeys(key, h_subkeys);
    cudaMemcpyToSymbol(subkeys, h_subkeys, sizeof(uint64_t) * 16);

    int blocks = (total_chains + threads_per_block - 1) / threads_per_block;
    size_t size = total_chains * 2 * PW_LEN;
    uint64_t* d_out;
    cudaMalloc(&d_out, sizeof(uint64_t) * total_chains * 2);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<blocks, threads_per_block>>>(d_out, total_chains, chain_len);

    cudaDeviceSynchronize();    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }

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
