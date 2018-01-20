#include "cclass.h"
#include <omp.h>
#include <stdexcept>

ConvolutionLayer::ConvolutionLayer(Data data_in, Activation act, size_t size, size_t stride, size_t count, float init_gauss) {
    this->data_in = data_in;
    this->act = act;
    this->stride = stride;
    data_out = Data((data_in.w-size) / stride + 1, (data_in.h-size) / stride + 1, count, data_in.batch);
    data_out.init();
    grad = weight = Data(size, size, data_in.c+1, count);
    weight.init(true, init_gauss);
    grad.init();
    data_grad = Data(data_in.w, data_in.h, omp_get_max_threads(), 1);
    data_grad.init();
    pool = false;
}

ConvolutionLayer::~ConvolutionLayer() {
    data_out.free();
    weight.free(true);
    grad.free();
    data_grad.free();
}

#ifdef USE_CUDA
template<int ksize>
__global__ void ConvolutionLayer_forward(Data data_in, Data weight, Data data_out, int stride, int bz) {
    extern __shared__ float data[];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    bz += blockIdx.z;
    const int oc = bz % data_out.c;
    const int b = bz / data_out.c;
    float res = weight.data[(oc*weight.c+data_in.c) * (ksize*ksize)];
    
    int fx = blockIdx.x * blockDim.x;
    int fy = blockIdx.y * blockDim.y;
    int sx = data_out.w - fx;
    int sy = data_out.h - fy;
    sx = sx < blockDim.x ? sx : blockDim.x;
    sy = sy < blockDim.y ? sy : blockDim.y;
    fx *= stride; fy *= stride;
    
    const int cx = data_in.w;//(sx-1) * stride + ksize;
    const int count = cx * ((sy-1) * stride + ksize);
    float* w = data + count;
    
    /*if (t==0) {
        printf("ksize = %d\nsx = %d\nsy = %d\nstride=%d\ncx=%d\ncount=%d\n", ksize, sx, sy, stride, cx, count);
    }*/
    float* in = data_in.data + b*data_in.c*(data_in.w*data_in.h) + fy*data_in.w + fx;
    float* wr = weight.data + oc*weight.c*(ksize*ksize);
    for (int c = 0; c < data_in.c; ++c) {
        for (int i=ty*blockDim.x+tx; i<ksize*ksize; i+=blockDim.x*blockDim.y) w[i] = wr[i];
        for (int i=ty*blockDim.x+tx; i<count; i+=blockDim.x*blockDim.y) data[i] = in[i];
        __syncthreads();
        in += data_in.w*data_in.h;
        float csum = 0;
        float* row = data + (ty*cx + tx)*stride;
        #pragma unroll
        for (int dy = 0; dy < ksize; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < ksize; ++dx)
                csum += row[dx] * w[dy*ksize + dx];
            row += cx;
        }
        res += csum;
        wr += ksize*ksize;
        __syncthreads();
    }
    
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    if (tx < sx && ty < sy) data_out.data[(bz*data_out.h + y) * data_out.w + x] = res;
}
#endif

void ConvolutionLayer::forward() {
    if (use_gpu) {
        #if USE_CUDA
        gpu_forward_activate(act);
        int sx = data_out.w < 32 ? data_out.w : 32;
        int sy = data_out.h < 32 ? data_out.h : 32;
        int shared_size = (weight.h*weight.w + data_in.w * ((sy-1)*stride + weight.h)) * sizeof(float);
        dim3 threads(sx, sy);
        dim3 grid((data_out.w + sx-1) / sx, (data_out.h + sy-1) / sy, data_out.c*data_out.batch);
        int maxz = grid.z;
        int zstep = 16384 / (grid.x*grid.y);
        for (int zoffset = 0; zoffset < maxz; zoffset += zstep) {
            grid.z = zoffset+zstep < maxz ? zstep : maxz - zoffset;
            switch (weight.w) {
                case 2: ConvolutionLayer_forward<2><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 3: ConvolutionLayer_forward<3><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 4: ConvolutionLayer_forward<4><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 5: ConvolutionLayer_forward<5><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 6: ConvolutionLayer_forward<6><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 7: ConvolutionLayer_forward<7><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 8: ConvolutionLayer_forward<8><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 9: ConvolutionLayer_forward<9><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 10: ConvolutionLayer_forward<10><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
                case 11: ConvolutionLayer_forward<11><<<grid, threads, shared_size>>>(data_in, weight, data_out, stride, zoffset); break;
            }
            
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("ConvolutionLayer::forward() -> %s\n", cudaGetErrorString(err)); exit(1); }
        return;
        #else
        throw std::logic_error("Builded without CUDA support");
        #endif
    }
    #pragma omp parallel for
    for (size_t i = 0; i < data_in.size; ++i) data_in.data[i] = act.activate(data_in.data[i]);
    for (size_t b = 0; b < data_in.batch; ++b) {
        #pragma omp parallel for
        for (size_t oc = 0; oc < data_out.c; ++oc) {
            float* out = data_out.data + (b*data_out.c + oc) * data_out.w*data_out.h;
            float bias = weight.data[(oc*weight.c+data_in.c) * weight.w * weight.h];
            for (size_t j = 0; j < data_out.w*data_out.h; ++j) out[j] = bias;
            for (size_t c = 0; c < data_in.c; ++c) {
                float* in = data_in.data + (b*data_in.c + c) * data_in.w*data_in.h;
                
                /*printf("in:\n");
                for (int y=0; y<data_in.h; ++y) {
                    for (int x=0; x<data_in.w; ++x) printf("%f ", in[y*data_in.w+x]);
                    printf("\n");
                }
                printf("w:\n");
                for (int y=0; y<weight.h; ++y) {
                    float* rw = weight.data + ((oc*weight.c+c)*weight.h + y) * weight.w;
                    for (int x=0; x<weight.w; ++x) printf("%f ", rw[x]);
                    printf("\n");
                }*/
                
                
                for (size_t oy = 0; oy < data_out.h; ++oy) {
                    float* __restrict__ ro = out + oy * data_out.w;
                    for (size_t dy = 0; dy < weight.h; ++dy) {
                        float* rw = weight.data + ((oc*weight.c+c)*weight.h + dy) * weight.w;
                        float* rin = in + (oy * stride + dy) * data_in.w;
                        for (size_t ox = 0; ox < data_out.w; ++ox) {
                            float* i = rin + ox * stride;
                            float res = 0;
                            for (size_t dx = 0; dx < weight.w; ++dx) res += i[dx] * rw[dx];
                            ro[ox] += res;
                        }
                    }
                }
                
                
                /*printf("out:\n");
                for (int y=0; y<data_out.h; ++y) {
                    for (int x=0; x<data_out.w; ++x) printf("%f ", out[y*data_out.w+x]);
                    printf("\n");
                }*/
                
            }
        }
    }
}

void ConvolutionLayer::backward() {
    if (use_gpu) {
        throw std::logic_error("Unimplemented"); // TODO
    }
    const size_t in_wh = data_in.w*data_in.h;
    #pragma omp parallel for
    for (size_t i = 0; i < grad.size; ++i) GRAD_CLEAN(grad.data[i]);
    size_t threads = omp_get_max_threads();
    for (size_t b = 0; b < data_in.batch; ++b) {
        for (size_t c = 0; c < data_in.c; ++c) {
            float* in = data_in.data + (b*data_in.c + c) * in_wh;
            memset(data_grad.data, 0, sizeof(float)*data_grad.size);
            #pragma omp parallel for
            for (size_t oc = 0; oc < data_out.c; ++oc) {
                float* out = data_out.data + (b*data_out.c + oc) * data_out.w*data_out.h;
                float grad_bias = 0;
                float* grad_in = data_grad.data + omp_get_thread_num() * in_wh;
                for (size_t oy = 0; oy < data_out.h; ++oy) {
                    float* __restrict__ ro = out + oy * data_out.w;
                    for (size_t dy = 0; dy < weight.h; ++dy) {
                        float* __restrict__ rw = weight.data + ((oc*weight.c+c)*weight.h + dy) * weight.w;
                        float* __restrict__ rg = grad.data + ((oc*weight.c+c)*weight.h + dy) * weight.w;
                        float* __restrict__ rin = in + (oy * stride + dy) * data_in.w;
                        float* __restrict__ rdg = grad_in + (oy * stride + dy) * data_in.w;
                        for (size_t ox = 0; ox < data_out.w; ++ox) {
                            float* __restrict__ i = rin + ox * stride;
                            float* __restrict__ dg = rdg + ox * stride;
                            float o = ro[ox];
                            if (dy == 0) grad_bias += o;
                            for (size_t dx = 0; dx < weight.w; ++dx) {
                                dg[dx] += rw[dx] * o;
                                rg[dx] += i[dx] * o;
                            }
                        }
                    }
                }
                grad.data[(oc*weight.c+data_in.c) * weight.w * weight.h] += grad_bias;
            }
            for (int t=threads-1; t>0; --t) {
                float* src = data_grad.data + t * in_wh;
                float* __restrict__ dst = data_grad.data - in_wh;
                for (size_t i = 0; i < in_wh; ++i) dst[i] += src[i];
            }
            for (size_t i = 0; i < in_wh; ++i) in[i] = data_grad.data[i] * act.derivative(in[i]);
        }
    }
    #pragma omp parallel for
    for (size_t i = 0; i < grad.size; ++i) GRAD_APPLY(weight.data[i], grad.data[i]);
}

