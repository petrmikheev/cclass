#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdint.h>
#include <ctime>
#include <vector>
#include <assert.h>
#include "cclass.h"
#include <omp.h>

using namespace std;

static enum {UNKNOWN, TRAIN, TEST, RUN} mode = UNKNOWN;
size_t batch=1, iter=0;
vector<Layer*> layers;
InputLayer* inputLayer = NULL;
OutputLayer* outputLayer;

void createNet(char* filename) {
    FILE* f = fopen(filename, "r");
    char buf[256];
    char lname[256];
    Activation act;
    while (fgets(buf, 256, f)) {
        lname[0] = '#';
        if (!sscanf(buf, "%s", lname)) continue;
        if (lname[0] == '#') continue;
        if (strcmp(lname, "INPUT") && !inputLayer) {
            printf("INPUT layer expected\n");
            exit(1);
        }
        if (strcmp(lname, "INPUT") == 0 && !inputLayer) {
            int w, h, c;
            assert(sscanf(buf, "%s %d %d %d", lname, &w, &h, &c) == 4);
            inputLayer = new InputLayer(Data(w, h, c, batch));
            layers.push_back(inputLayer);
        } else if (strcmp(lname, "FULL") == 0) {
            int c;
            float std;
            assert(sscanf(buf, "%s %d %f", lname, &c, &std) == 3);
            layers.push_back(new FullyConnectedLayer(layers.back()->data_out, act, c, std));
            act = Activation();
        } else if (strcmp(lname, "CONV") == 0) {
            int size, stride, count;
            float std;
            assert(sscanf(buf, "%s %d %d %d %f", lname, &size, &stride, &count, &std) == 5);
            layers.push_back(new ConvolutionLayer(layers.back()->data_out, act, size, stride, count, std));
            act = Activation();
        } else if (strcmp(lname, "RELU") == 0) {
            float relu = 0;
            if (sscanf(buf, "%s %f", lname, &relu) < 2) relu = 0;
            act = Activation(Activation::RELU, relu);
        } else if (strcmp(lname, "SIGMOID") == 0) act = Activation(Activation::SIGMOID);
        else {
            printf("Unknown layer: %s\n", lname);
            exit(1);
        }
    }
    assert(inputLayer);
    for (int i = 0; i < layers.size(); ++i) printf("L %d %d %d\n", layers[i]->data_out.w, layers[i]->data_out.h, layers[i]->data_out.c);
    outputLayer = new OutputLayer(layers.back()->data_out, inputLayer->labels);
    layers.push_back(outputLayer);
}

float step = 1.0;
float moment = 0.0;

int main(int argc, char** argv) {
    Data::setSeed(time(NULL));
    char* net_filename = NULL;
    char* lmdb_filename = NULL;
    int arg = 1;
    while (arg < argc) {
        if (strcmp(argv[arg], "-help") == 0 || strcmp(argv[arg], "-h") == 0 || strcmp(argv[arg], "--help") == 0) {
            printf("Using: cclass [-cpu/-gpu] [-batch <NUM>] [-step <NUM>] [-moment <NUM>] [-iter <NUM>] [-seed <NUM>] train <net> <LMDB_DIR>\n");
            printf("       cclass [-cpu/-gpu] [-batch <NUM>] test <net> <LMDB_DIR>\n");
            printf("       cclass [-cpu/-gpu] run <net>        -- it reads path to BMP image (24bit) from stdin\n");
            return 0;
        }
        if (strcmp(argv[arg], "-cpu") == 0) use_gpu = false;
        else if (strcmp(argv[arg], "-gpu") == 0) {
            use_gpu = true;
            #ifndef USE_CUDA
            printf("Builded without CUDA support\n");
            exit(1);
            #endif
        }
        else if (strcmp(argv[arg], "-seed") == 0 && arg+1 < argc) {
            uint64_t seed;
            sscanf(argv[++arg], "%zu", &seed);
            Data::setSeed(seed);
        } else if (strcmp(argv[arg], "-batch") == 0 && arg+1 < argc) sscanf(argv[++arg], "%zu", &batch);
        else if (strcmp(argv[arg], "-iter") == 0 && arg+1 < argc) sscanf(argv[++arg], "%zu", &iter);
        else if (strcmp(argv[arg], "-step") == 0 && arg+1 < argc) sscanf(argv[++arg], "%f", &step);
        else if (strcmp(argv[arg], "-moment") == 0 && arg+1 < argc) sscanf(argv[++arg], "%f", &moment);
        else if (strcmp(argv[arg], "train") == 0 && arg+2 < argc) {
            mode = TRAIN; do_train = true;
            net_filename = argv[++arg];
            lmdb_filename = argv[++arg];
        }
        else if (strcmp(argv[arg], "test") == 0 && arg+2 < argc) {
            mode = TEST;
            net_filename = argv[++arg];
            lmdb_filename = argv[++arg];
        }
        else if (strcmp(argv[arg], "run") == 0 && arg+1 < argc) {
            mode = RUN;
            net_filename = argv[++arg];
        }
        else { printf("Incorrect option\nTry 'cclass --help'\n"); return 1; }
        arg++;
    }
    if (mode == UNKNOWN) { printf("Command expected\nTry 'cclass --help'\n"); return 1; }
    string net_binary = string(net_filename) + ".bin";
    if (net_filename) Data::setDataFile(net_binary.c_str());
    createNet(net_filename);
    step /= batch;
    if (mode == RUN) {
        char buf[256];
        inputLayer->bmp_file = buf;
        while (true) {
            if (!fgets(buf, 256, stdin)) break;
            int l = strlen(buf);
            if (buf[l-1] == '\n') buf[l-1] = 0;
            for (int i = 0; i < layers.size(); ++i) layers[i]->forward();
            printf("%d   ( ", outputLayer->answer[0]);
            for (int i = 0; i < outputLayer->data_in.c; ++i) printf("%.2f ", outputLayer->data_in.data[i]);
            printf(")\n");
        }
    } else {
        LMDB lmdb(lmdb_filename);
        inputLayer->lmdb = &lmdb;
        if (iter == 0) iter = (lmdb.getSize() + batch-1) / batch;
        float acc_sum = 0;
        for (size_t j = 0; j < iter; ++j) {
            for (int i = 0; i < layers.size(); ++i) {
                double t = omp_get_wtime();
                layers[i]->forward();
                layers[i]->time_forward += omp_get_wtime() - t;
            }
            if (mode == TRAIN)
                for (int i = layers.size()-1; i >= 0; --i) {
                    double t = omp_get_wtime();
                    layers[i]->backward();
                    layers[i]->time_backward += omp_get_wtime() - t;
                }
            printf("#%zu  loss=%f  accuracy=%f\n", j, outputLayer->loss, outputLayer->accuracy);
            acc_sum += outputLayer->accuracy;
        }
        if (mode == TEST) printf("Total accuracy: %f\n", acc_sum / iter);
        for (int i = 0; i < layers.size(); ++i)
            printf("TIME [%02d] forward %lf   backward %lf\n", i, layers[i]->time_forward, layers[i]->time_backward);
    }
    Data::setDataFile(mode == TRAIN ? net_binary.c_str() : NULL, "w");
    for (int i = 0; i < layers.size(); ++i) delete layers[i];
    Data::setDataFile(NULL);
    return 0;
}

