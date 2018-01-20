#include "cclass.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <assert.h>

bool use_gpu = false;
bool do_train = false;

FILE* dataFile = NULL;

void Data::setDataFile(const char* filename, std::string mode) {
    if (dataFile) fclose(dataFile);
    if (filename) dataFile = fopen(filename, mode.c_str());
    else dataFile = NULL;
}

static uint64_t fast_rand(uint64_t& x) { // xorshift64star
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x *= 2685821657736338717ULL;
    return x;
}

static double generateNormalDistribution(uint64_t& seed) {
    double r1 = fast_rand(seed) * 5.421010862427522e-20;
    double r2 = fast_rand(seed) * (2.168404344971009e-19 * M_PI_2);
    return sin(r2) * sqrt(-2.0 * log(r1));
}

uint64_t gseed = 1;
void Data::setSeed(uint64_t s) { gseed = s; }
void Data::init(bool read, float gauss_std) {
    data = (float*)malloc(size*sizeof(float));
    if (read && dataFile) assert(fread(data, sizeof(float), size, dataFile) == size);
    else if (gauss_std == 0) memset(data, 0, size*sizeof(float));
    else {
        #pragma omp parallel
        {
            #ifdef _OPENMP
            uint64_t seed = gseed + omp_get_thread_num() + 1;
            #pragma omp for
            #else
            uint64_t seed = gseed + 1;
            #endif
            for (size_t i = 0; i < size; ++i)
                data[i] = gauss_std * generateNormalDistribution(seed);
        }
        fast_rand(gseed);
    }
    #ifdef USE_CUDA
    if (use_gpu) {
        float* hd = data;
        cudaError_t err = cudaMalloc(&data, size*sizeof(float));
        if (err != cudaSuccess) { printf("cudaMalloc -> %s\n", cudaGetErrorString(err)); exit(1); }
        err = cudaMemcpy(data, hd, size*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("cudaMemcpy -> %s\n", cudaGetErrorString(err)); exit(1); }
        cudaDeviceSynchronize();
        ::free(hd);
    }
    #endif
}

void Data::free(bool write) {
    float *hd = data;
    #ifdef USE_CUDA
    if (use_gpu) {
        cudaError_t err;
        if (write && dataFile) {
            hd = (float*)malloc(size*sizeof(float));
            err = cudaMemcpy(hd, data, size*sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { printf("cudaMemcpy -> %s\n", cudaGetErrorString(err)); exit(1); }
        } else hd = NULL;
        err = cudaFree(data);
        if (err != cudaSuccess) { printf("cudaFree -> %s\n", cudaGetErrorString(err)); exit(1); }
    }
    #endif
    if (write && dataFile && hd) fwrite(hd, size, sizeof(float), dataFile);
    if (hd) ::free(hd);
    data = NULL;
}

