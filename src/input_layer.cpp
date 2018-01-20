#include <assert.h>
#include <cstdlib>
#include "cclass.h"
#include "BMP.h"

InputLayer::InputLayer(Data data_out) {
    data_out.init();
    this->data_out = data_out;
    this->labels = new int[data_out.batch];
    h_data = data_out.data;
    if (use_gpu) h_data = new float[data_out.size];
}

InputLayer::~InputLayer() {
    delete [] labels;
    if (use_gpu) delete [] h_data;
    data_out.free();
}

void InputLayer::forward() {
    if (bmp_file) {
        std::string fname = std::string(bmp_file);
        BMP bmp(fname);
        int offset_x = (data_out.w - bmp.width) / 2;
        int offset_y = (data_out.h - bmp.height) / 2;
        for (int y = 0; y < bmp.height; ++y) {
            int gy = bmp.height-y-1+offset_y;
            bmp.readRow();
            if (gy<0 || gy>=data_out.h) continue;
            for (int x = 0; x < bmp.width; ++x) {
                int gx = x+offset_x;
                if (gx<0 || gx>=data_out.w) continue;
                BMP::Pixel& p = bmp.row_data[x];
                h_data[gy*data_out.w + gx] = (float)p.r / 255.0;
                if (data_out.c == 3) {
                    h_data[(gy+data_out.h)*data_out.w + gx] = (float)p.g / 255.0;
                    h_data[(gy+data_out.h*2)*data_out.w + gx] = (float)p.b / 255.0;
                }
            }
        }
        bmp.close();
        labels[0] = 0;
    } else if (lmdb) {
        size_t isize = data_out.w*data_out.h*data_out.c;
        for (size_t i = 0; i < data_out.batch; ++i) {
            labels[i] = 0;
            int w, h, c;
            unsigned char* img = lmdb->getImageData(&w, &h, &c, labels+i);
            int offset_x = ((int)data_out.w - w) / 2;
            int offset_y = ((int)data_out.h - h) / 2;
            assert(c == data_out.c);
            for (int cn = 0; cn < c; ++cn) {
                unsigned char* src = img + w*h*cn;
                float* dst = h_data + i*isize + cn*data_out.w*data_out.h;
                #pragma omp parallel for
                for (int y = 0; y < h; ++y) {
                    int gy = y + offset_y;
                    if (gy<0 || gy>=data_out.h) continue;
                    unsigned char* row_src = src + y*w;
                    float* row_dst = dst + gy*data_out.w + offset_x;
                    for (int x = std::max(0, -offset_x); x < std::min(w, (int)data_out.w-offset_x); ++x)
                        row_dst[x] = (float)row_src[x] * (1.0/255);
                }
            }
        }
    } else {
        printf("InputLayer::forward() -> no data\n");
        exit(1);
    }
    #ifdef USE_CUDA
    if (use_gpu) {
        cudaError_t err = cudaMemcpy(data_out.data, h_data, data_out.size*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("[InputLayer] cudaMemcpy -> %s\n", cudaGetErrorString(err)); exit(1); }
        cudaDeviceSynchronize();
    }
    #endif
}

void InputLayer::backward() {}

