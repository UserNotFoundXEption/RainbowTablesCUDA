#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <new> 
#include <getopt.h>
#include "des.cuh"
#include "des_tables.cuh"

#define PW_LEN 8
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"

__device__ __constant__ uint64_t subkeys[16];

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

__global__ void kernel_sequential(uint64_t* out, int total_chains, int chain_len) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int id = 0; id < total_chains; ++id) {
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
    }
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
    bool sequential = false;

    int opt;
    while ((opt = getopt(argc, argv, "s")) != -1) {
        if (opt == 's') sequential = true;
    }

    if (argc - optind < 4) {
        printf("Użycie: %s [-s] <liczba_łańcuchów> <dlugosc_lancucha> <klucz_hex> <watki_na_blok>\n", argv[0]);
        return 1;
    }

    int total_chains = atoi(argv[optind]);
    if(total_chains <= 0) {
	printf("Nieprawidłowa liczba łańcuchów. Wybierz liczbę większą niż 0.\n");
	return 2;
    }

    int chain_len = atoi(argv[optind + 1]);
    if(chain_len <= 0) {
        printf("Nieprawidłowa długość łańcuchów. Wybierz liczbę większą niż 0.\n");
        return 3;
    }
 
    uint64_t key = 0;
    if (sscanf(argv[optind + 2], "%lx", &key) != 1) {
        fprintf(stderr, "Nieprawidłowy klucz w formacie szesnastkowym.\n");
        return 4;
    }
    
    int threads_per_block = sequential ? 1 : atoi(argv[optind + 3]);
    if(threads_per_block > 256 || threads_per_block <= 0) {
        printf("Nieprawidłowa liczba wątków na blok. Wybierz liczbę z przedziału <1, 256>.\n");
	return 5;
    }

    uint64_t h_subkeys[16];
    generate_subkeys(key, h_subkeys);
    cudaError_t err;
    if ((err = cudaMemcpyToSymbol(subkeys, h_subkeys, sizeof(h_subkeys))) != cudaSuccess) {
        fprintf(stderr, "Błąd kopiowania kluczy do GPU: %s\n", cudaGetErrorString(err));
        return 6;
    }

    int blocks = sequential ? 1 : (total_chains + threads_per_block - 1) / threads_per_block;
    size_t size_bytes = sizeof(uint64_t) * total_chains * 2;
    
    uint64_t* d_out = nullptr;
    if ((err = cudaMalloc(&d_out, size_bytes)) != cudaSuccess) {
        fprintf(stderr, "Błąd alokacji pamięci GPU: %s\n", cudaGetErrorString(err));
        return 7;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (sequential) {
        kernel_sequential<<<1, 1>>>(d_out, total_chains, chain_len);
    } else {
        kernel<<<blocks, threads_per_block>>>(d_out, total_chains, chain_len);
    }

    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        fprintf(stderr, "Błąd synchronizacji GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 8;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s działał przez %.4f sekund.\n", sequential ? "Tryb sekwencyjny (GPU, 1 wątek)" : "GPU", ms / 1000.0);

    char* h_out = new(std::nothrow) char[size_bytes];
    if (!h_out) {
        fprintf(stderr, "Błąd alokacji pamięci na host.\n");
        cudaFree(d_out);
        return 9;
    }

    if ((err = cudaMemcpy(h_out, d_out, size_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
        fprintf(stderr, "Błąd kopiowania wyników z GPU: %s\n", cudaGetErrorString(err));
        delete[] h_out;
        cudaFree(d_out);
        return 10;
    }
    
    FILE* f = fopen("output/rainbow_des.txt", "w");
    if (!f) {
        fprintf(stderr, "Nie można otworzyć pliku do zapisu.\n");
        delete[] h_out;
        cudaFree(d_out);
        return 11;
    }
    
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

