#include <cstdio>
#include <cstring>
#include <stdint.h> 
#include "des.cuh"
#include "des_tables.cuh"

#define PW_LEN 8
#define CHAIN_LEN 10000
#define THREADS_PER_BLOCK 256
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"

__device__ __constant__ uint64_t subkeys[16];

int PC1_host[56];
int SHIFTS_host[16];
int PC2_host[48];

__device__ uint64_t text_to_block(const char* pw) {
    uint64_t block = 0;
    for (int i = 0; i < PW_LEN; i++)
        block |= ((uint64_t)pw[i]) << ((PW_LEN - 1 - i) * 8);
    return block;
}

__device__ void block_to_pw(uint64_t block, char* out) {
    for (int i = 0; i < PW_LEN; i++)
        out[i] = (block >> ((PW_LEN - 1 - i) * 8)) & 0xFF;
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
    if (argc < 3) {
        printf("Użycie: %s <liczba_łańcuchów> <klucz_hex>\n", argv[0]);
        return 1;
    }
    
    cudaMemcpyToSymbol(PC1, PC1_host, sizeof(PC1_host));
    cudaMemcpyToSymbol(SHIFTS, SHIFTS_host, sizeof(SHIFTS_host));
    cudaMemcpyToSymbol(PC2, PC2_host, sizeof(PC2_host));
    
    int total_chains = atoi(argv[1]);
    uint64_t key = strtoull(argv[2], NULL, 16);
    
    uint64_t h_subkeys[16];
    generate_subkeys(key, h_subkeys);
    cudaMemcpyToSymbol(subkeys, h_subkeys, sizeof(uint64_t) * 16);

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
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU dzilal przez %f sekund.\n", ms / 1000.0);
    printOccupancy();

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
