#include "cclass.h"

#ifdef USE_CUDA
__global__ void device_forward_activate(Data data_in, Activation act) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= data_in.size) return;
    float v = data_in.data[id];
    if (act.type == Activation::SIGMOID) v = 1.0 / (1.0 + exp(-v));
    if (act.type == Activation::RELU) v = v > 0 ? v : v*act.relu;
    data_in.data[id] = v;
}

__global__ void device_backward_update(Data weight, Data grad, float step) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < grad.size) GRAD_APPLY(weight.data[id], grad.data[id]);
}

void Layer::gpu_forward_activate(Activation act) {
    if (act.type == Activation::NOTHING) return;
    const int bsize = 256;
    int count = (data_in.size + bsize-1) / bsize;
    device_forward_activate<<<count, bsize>>>(data_in, act);
    cudaDeviceSynchronize();
}

void Layer::gpu_backward_update(Data weight, Data grad) {
    const int bsize = 256;
    int count = (grad.size + bsize-1) / bsize;
    device_backward_update<<<count, bsize>>>(weight, grad, step);
    cudaDeviceSynchronize();
}
#endif

