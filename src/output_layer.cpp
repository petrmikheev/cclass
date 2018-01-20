#include "cclass.h"
#include <cstdio>
#include <cstdlib>

OutputLayer::OutputLayer(Data data_in, int* labels) {
    this->data_in = data_in;
    this->answer = new int[data_in.batch];
    h_data = data_in.data;
    if (use_gpu) h_data = new float[data_in.size];
    this->labels = labels;
}

OutputLayer::~OutputLayer() {
    delete [] answer;
    if (use_gpu) delete [] h_data;
}

void OutputLayer::forward() {
    #ifdef USE_CUDA
    if (use_gpu) {
        cudaError_t err = cudaMemcpy(h_data, data_in.data, data_in.size*sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { printf("cudaMemcpy -> %s\n", cudaGetErrorString(err)); exit(1); }
    }
    #endif
    float l = 0, acc = 0;
    size_t isize = data_in.w*data_in.h*data_in.c;
    #pragma omp parallel for reduction(+:l,acc)
    for (size_t i = 0; i < data_in.batch; ++i) {
        float* __restrict__ o = h_data + i*isize;
        int ans = 0;
        float mv = 0;
        float sum = 0;
        /*printf("res\n");
        for (size_t j = 0; j < isize; ++j) printf("%f ", o[j]);
        printf("\n");*/
        for (size_t j = 0; j < isize; ++j) sum += (o[j] = exp(o[j]));
        for (size_t j = 0; j < isize; ++j) {
            float s = (o[j] /= sum);
            if (j==0 || s > mv) { ans = j; mv = s; }
            float g = labels[i]==j ? s-1 : s;
            l += g*g;
            if (do_train) o[j] = 2*g*s*(1-s);
        }
        answer[i] = ans;
        if (labels[i] == ans) acc++;
    }
    loss = l;
    accuracy = acc / data_in.batch;
}

void OutputLayer::backward() {
    if (use_gpu) {
        #ifdef USE_CUDA
        cudaError_t err = cudaMemcpy(data_in.data, h_data, data_in.size*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("cudaMemcpy -> %s\n", cudaGetErrorString(err)); exit(1); }
        cudaDeviceSynchronize();
        #endif
    }
}

