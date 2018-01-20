#include "cclass.h"
#include <stdexcept>

FullyConnectedLayer::FullyConnectedLayer(Data data_in, Activation act, size_t num_outputs, float init_gauss) {
    this->data_in = data_in;
    this->act = act;
    data_out = Data(1, 1, num_outputs, data_in.batch);
    data_out.init();
    grad = weight = Data(num_outputs, data_in.w*data_in.h*data_in.c+1, 1, 1);
    weight.init(true, init_gauss);
    grad.init();
}

FullyConnectedLayer::~FullyConnectedLayer() {
    data_out.free();
    weight.free(true);
    grad.free();
}

#ifdef USE_CUDA
__global__ void FullyConnectedLayer_forward(Data data_in, Data weight, Data data_out) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int c = blockIdx.x*blockDim.x + tx;
    int b = blockIdx.y*blockDim.y + ty;
    int isize = weight.h-1;
    __shared__ float in[32][32];
    __shared__ float w[32][32];
    float * w_offset = weight.data + ty*weight.w + c;
    float ans = weight.data[weight.size-data_out.c+c];
    for (int bi = 0; bi < isize; bi+=blockDim.x) {
        in[ty][tx] = bi+tx < isize ? data_in.data[b*isize + bi+tx] : 0;
        w[ty][tx] = bi+ty < isize ? w_offset[bi*weight.w] : 0;
        __syncthreads();
        float s = 0;
        #pragma unroll
        for (int k=0; k<32; ++k) s += in[ty][k] * w[k][tx];
        ans += s;
        __syncthreads();
    }
    if (c < data_out.c && b < data_out.batch) data_out.data[b * data_out.c + c] = ans;
}

__global__ void FullyConnectedLayer_backward_grad(Data data_in, Data grad, Data data_out, float moment) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int c = blockIdx.x*blockDim.x + tx;
    int i = blockIdx.y*blockDim.y + ty;
    int batch = data_in.batch;
    int isize = grad.h-1;
    __shared__ float in[32][32];
    __shared__ float out[32][32];
    float ans = 0;
    for (int bb = 0; bb < batch; bb+=blockDim.x) {
        in[ty][tx] = bb+tx < batch ? data_in.data[(bb+tx)*isize + i] : 0;
        if (bb+ty == isize) in[ty][tx] = 1;
        out[ty][tx] = bb+ty < batch ? data_out.data[(bb+ty)*data_out.c + c] : 0;
        __syncthreads();
        float s = 0;
        #pragma unroll
        for (int k=0; k<32; ++k) s += in[ty][k] * out[k][tx];
        ans += s;
        __syncthreads();
    }
    if (i < grad.h && c < data_out.c) grad.data[i*data_out.c+c] = grad.data[i*data_out.c+c] * moment + ans;
}

__global__ void FullyConnectedLayer_backward_data(Data data_in, Data weight, Data data_out, Activation act) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x*blockDim.x + tx;
    int b = blockIdx.y*blockDim.y + ty;
    int isize = weight.h-1;
    __shared__ float out[32][32];
    __shared__ float w[32][32];
    float ans = 0;
    for (int bc = 0; bc < data_out.c; bc+=blockDim.x) {
        out[ty][tx] = bc+tx < data_out.c ? data_out.data[b*data_out.c + bc+tx] : 0;
        w[ty][tx] = bc+ty < data_out.c ? weight.data[i*data_out.c + bc+ty] : 0;
        __syncthreads();
        float s = 0;
        #pragma unroll
        for (int k=0; k<32; ++k) s += out[ty][k] * w[k][tx];
        ans += s;
        __syncthreads();
    }
    if (i < isize && b < data_out.batch) {
        float v = data_in.data[b*isize + i];
        if (act.type == Activation::SIGMOID) v *= 1-v;
        if (act.type == Activation::RELU) v = v > 0 ? 1 : act.relu;
        if (act.type == Activation::NOTHING) v = 1;
        data_in.data[b*isize + i] = ans * v;
    }
}
#endif

void FullyConnectedLayer::forward() {
    size_t isize = data_in.w*data_in.h*data_in.c;
    if (use_gpu) {
        #ifdef USE_CUDA
        gpu_forward_activate(act);
        dim3 threads(32, 32);
        dim3 grid((data_out.c + threads.x-1) / threads.x, (data_out.batch + threads.y-1) / threads.y);
        FullyConnectedLayer_forward<<<grid, threads>>>(data_in, weight, data_out);
        cudaDeviceSynchronize();
        return;
        #else
        throw std::logic_error("Builded without CUDA support");
        #endif
    }
    #pragma omp parallel for
    for (size_t b = 0; b < data_in.batch; ++b) {
        float* in = data_in.data + b*isize;
        float* __restrict__ out = data_out.data + b*data_out.c;
        for (size_t c = 0; c < data_out.c; ++c)
            out[c] = weight.data[isize*data_out.c + c];
        for (size_t i = 0; i < isize; ++i) {
            float v = in[i] = act.activate(in[i]);
            float* w = weight.data + i*data_out.c;
            for (size_t c = 0; c < data_out.c; ++c) {
                out[c] += w[c] * v;
            }
        }
    }
}

void FullyConnectedLayer::backward() {
    if (use_gpu) {
        #ifdef USE_CUDA
        dim3 threads(32, 32);
        dim3 grid((grad.w + threads.x-1) / threads.x, (grad.h + threads.y-1) / threads.y);
        FullyConnectedLayer_backward_grad<<<grid, threads>>>(data_in, grad, data_out, moment);
        grid = dim3((grad.h-1 + threads.x-1) / threads.x, (data_in.batch + threads.y-1) / threads.y);
        cudaDeviceSynchronize();
        FullyConnectedLayer_backward_data<<<grid, threads>>>(data_in, weight, data_out, act);
        cudaDeviceSynchronize();
        gpu_backward_update(weight, grad);
        return;
        #else
        throw std::logic_error("Builded without CUDA support");
        #endif
    }
    #pragma omp parallel for
    for (size_t i = 0; i < grad.size; ++i) GRAD_CLEAN(grad.data[i]);
    size_t isize = data_in.w*data_in.h*data_in.c;
    for (size_t b = 0; b < data_in.batch; ++b) {
        float* __restrict__ o = data_out.data + b*data_out.c;
        #pragma omp parallel for
        for (size_t i = 0; i < isize; ++i) {
            float d = 0;
            float s = data_in.data[b*isize + i];
            float *w = weight.data + i*data_out.c;
            float * __restrict__ g = grad.data + i*data_out.c;
            for (size_t j = 0; j < data_out.c; ++j) {
                d += w[j] * o[j];
                g[j] += s * o[j];
            }
            data_in.data[b*isize + i] = d * act.derivative(s);
        }
        float * __restrict__ g = grad.data + isize*data_out.c;
        for (size_t j = 0; j < data_out.c; ++j) g[j] += o[j];
    }
    #pragma omp parallel for
    for (size_t i = 0; i < grad.size; ++i) GRAD_APPLY(weight.data[i], grad.data[i]);
}

