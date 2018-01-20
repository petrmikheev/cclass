#ifndef CCLASS_H
#define CCLASS_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <string>
#include <cstring>
#include <cstdio>
#include <stdint.h>
#include "lmdb_reader.h"

extern bool use_gpu;
extern bool do_train;
extern float step;
extern float moment;

#define GRAD_CLEAN(g) g *= moment
#define GRAD_APPLY(w, g) w -= step*g

struct Data {
    static void setDataFile(const char* filename, std::string mode = "r");
    static void setSeed(uint64_t);

    unsigned int batch, w, h, c;
    size_t size;
    float* data;
    inline Data() { this->data = NULL; }
    inline Data(unsigned int w, unsigned int h, unsigned int c=1, unsigned int batch=1) {
        this->w = w;
        this->h = h;
        this->c = c;
        this->batch = batch;
        this->size = w*h*c*batch;
        this->data = NULL;
    }
    void init(bool read=false, float gauss_std=0);
    void free(bool write=false);
};
struct Activation {
    enum ActType {NOTHING, RELU, SIGMOID};
    ActType type;
    float relu;
    inline Activation(ActType t = NOTHING, float r = 0) { type = t; relu = r; }
    float inline activate(float v) {
        if (type == SIGMOID) return 1.0 / (1.0 + exp(-v));
        if (type == RELU) return v > 0 ? v : v*relu;
        return v;
    }
    float inline derivative(float v) {
        if (type == SIGMOID) return v*(1-v);
        if (type == RELU) return v > 0 ? 1 : relu;
        return 1;
    }
};

class Layer {
    public:
        inline Layer() { time_forward = time_backward = 0; }
        inline virtual ~Layer() {};
        virtual void forward() = 0;
        virtual void backward() = 0;
        Data data_in, data_out;
        double time_forward, time_backward;
    protected:
        void gpu_forward_activate(Activation act);
        void gpu_backward_update(Data weight, Data grad);
};
class OutputLayer : public Layer {
    public:
        OutputLayer(Data data_in, int* labels);
        ~OutputLayer();
        void forward();
        void backward();
        float loss, accuracy;
        int* answer;
    private:
        float* h_data;
        int* labels;
};
class InputLayer : public Layer {
    public:
        InputLayer(Data data_out);
        ~InputLayer();
        void forward();
        void backward();
        LMDB* lmdb;
        char* bmp_file;
        int* labels;
    private:
        float* h_data;
};
class ConvolutionLayer : public Layer {
    public:
        ConvolutionLayer(Data data_in, Activation act, size_t size, size_t stride, size_t count, float init_gauss);
        ~ConvolutionLayer();
        void forward();
        void backward();
        Data weight, grad, data_grad;
        int stride;
        bool pool;
        Activation act;
};
class FullyConnectedLayer : public Layer {
    public:
        FullyConnectedLayer(Data data_in, Activation act, size_t num_outputs, float init_gauss);
        ~FullyConnectedLayer();
        void forward();
        void backward();
        Data weight, grad;
        Activation act;
};

#endif // CCLASS_H
