#include <cstdio>
#include <cstring>
#include <stdint.h> 
#include "des_sliced.cuh"
#include "des_tables.cuh"

#define PW_LEN 8
#define CHAIN_LEN 10000
#define THREADS_PER_BLOCK 64
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"

__device__ __constant__ uint32_t subkeys[16][48];

int PC1_host[56];
int SHIFTS_host[16];
int PC2_host[48];

__device__ void bs_pack(const uint64_t* in, bs_block& block) {
    for (int bit = 0; bit < 64; ++bit) {
        block.bits[bit] = 0;
        for (int i = 0; i < 32; ++i) {
            block.bits[bit] |= ((in[i] >> (63 - bit)) & 1ULL) << i;
        }
    }
}

__device__ void bs_unpack(const bs_block& block, uint64_t* out) {
    for (int i = 0; i < 32; ++i) {
        out[i] = 0;
        for (int bit = 0; bit < 64; ++bit) {
            out[i] |= ((uint64_t)((block.bits[bit] >> i) & 1U)) << (63 - bit);
        }
    }
}

__device__ void reduce(bs_block& block, int round) {
    for (int i = 0; i < PW_LEN; ++i) {
        int byte_start = i * 8;
        for (int b = 0; b < 8; ++b) {
            int bit_index = byte_start + b;
            uint32_t bitplane = block.bits[bit_index];
            block.bits[bit_index] = ((round + i) & 0x1) ? ~bitplane : bitplane;
        }
    }
}

__global__ void kernel(uint64_t* out, int total_chains) {
    int base_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (base_id * 32 >= total_chains) return;

    uint64_t pw[32];
    for (int j = 0; j < 32; ++j) {
        int id = base_id * 32 + j;
        uint64_t p = 0;
        for (int i = 0; i < PW_LEN; i++)
            p |= ((uint64_t)'a') << ((PW_LEN - 1 - i) * 8);
        int n = id;
        for (int i = PW_LEN - 1; i >= 0 && n > 0; i--) {
            int shift = (PW_LEN - 1 - i) * 8;
            uint8_t c = ((p >> shift) & 0xFF) + (n % 26);
            p = (p & ~(0xFFULL << shift)) | ((uint64_t)c << shift);
            n /= 26;
        }
        pw[j] = p;
    }

    bs_block block;
    bs_pack(pw, block);
    for (int i = 0; i < CHAIN_LEN; ++i) {
        des_encrypt(block, subkeys);
        reduce(block, i);
    }

    uint64_t result[32];
    bs_unpack(block, result);
    for (int j = 0; j < 32 && (base_id * 32 + j) < total_chains; ++j) {
        out[(base_id * 32 + j) * 2] = pw[j];
        out[(base_id * 32 + j) * 2 + 1] = result[j];
    }
}

void printOccupancy() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int numBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        kernel,
        THREADS_PER_BLOCK,
        0
    );

    int activeThreadsPerSM = numBlocksPerSM * THREADS_PER_BLOCK;
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    float occupancy = 100.0f * activeThreadsPerSM / maxThreadsPerSM;
    printf("Occupancy: %.2f%%\nMax: %d\nActive: %d\n", occupancy, maxThreadsPerSM, activeThreadsPerSM);
}

void generate_subkeys(uint64_t key, uint32_t subkeys[16][48]) {
    uint64_t perm_key = 0;
    for (int i = 0; i < 56; i++)
        perm_key |= ((key >> (64 - PC1_host[i])) & 1ULL) << (55 - i);

    uint32_t C = (perm_key >> 28) & 0x0FFFFFFF;
    uint32_t D = perm_key & 0x0FFFFFFF;

    for (int i = 0; i < 16; i++) {
        C = ((C << SHIFTS_host[i]) | (C >> (28 - SHIFTS_host[i]))) & 0x0FFFFFFF;
        D = ((D << SHIFTS_host[i]) | (D >> (28 - SHIFTS_host[i]))) & 0x0FFFFFFF;
        uint64_t CD = (((uint64_t)C) << 28) | D;
        for (int j = 0; j < 48; j++) {
            uint8_t bit = (CD >> (56 - PC2_host[j])) & 1ULL;
            subkeys[i][j] = bit ? 0xFFFFFFFF : 0x00000000;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Użycie: %s <liczba_łańcuchów> <klucz_hex>\n", argv[0]);
        return 1;
    }

    cudaMemcpyToSymbol(PC1, PC1_host, sizeof(PC1_host));
    cudaMemcpyToSymbol(SHIFTS, SHIFTS_host, sizeof(SHIFTS_host));
    cudaMemcpyToSymbol(PC2, PC2_host, sizeof(PC2_host));

    int total_chains = atoi(argv[1]);
    uint64_t key = strtoull(argv[2], NULL, 16);

    uint32_t h_subkeys[16][48];
    generate_subkeys(key, h_subkeys);
    cudaMemcpyToSymbol(subkeys, h_subkeys, sizeof(h_subkeys));

    int threads_total = (total_chains + 31) / 32;
    int blocks = (threads_total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    size_t size = total_chains * 2 * sizeof(uint64_t);
    uint64_t* d_out;
    cudaMalloc(&d_out, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<blocks, THREADS_PER_BLOCK>>>(d_out, total_chains);

    cudaDeviceSynchronize();    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU dzia\u0142a\u0142 przez %f sekund.\n", ms / 1000.0);
    printOccupancy();

    char* h_out = new char[total_chains * 2 * PW_LEN];
    cudaMemcpy(h_out, d_out, total_chains * 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

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
