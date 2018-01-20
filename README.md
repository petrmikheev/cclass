# cclass

It is an implementation of convolutional neural networks developed from scratch.
It was developed to gain experience in neural networks and CUDA.

## Format of a network structure

Network structure should be specified in a text format.
Each line corresponds to one layer.

Types of layers:

* `INPUT W H C` - an input layer; W x H - width and height; C - a number of channels
* `CONV size stride count std` - a convolution layer; count - a number of convolutions; std - standart deviation for initial values of weights
* `FULL count std` - a fully-connected layer
* `RELU coef` - RELU layer
* `SIGMOID` - SIGMOID layer

An example:
```
INPUT 28 28 1
CONV 5 2 20 3
CONV 5 2 50 1
FULL 500 0.01
RELU 0.1
FULL 10 0.01
```

## Input data format

Input data should be in LMDB database in the same format as for caffe (i.e. packed with Protocol buffers).
So it is recommended to download and use caffe examples.

## Building

Requirements:

* g++, make
* liblmdb
* nvcc (optional; to build with CUDA support)

Building steps:
```bash
cd src
./configure
make
```

## Content

* src - sources
* cclass - the main binary
* view_lmdb - a utility to extract one image from lmdb database
* net - structure of a neural network for MNIST
* net.bin - a trained network for MNIST
* test.bmp - an example image

## Options

```
Using: cclass [-cpu/-gpu] [-batch <NUM>] [-step <NUM>] [-moment <NUM>] [-iter <NUM>] [-seed <NUM>] train <net> <LMDB_DIR>
       cclass [-cpu/-gpu] [-batch <NUM>] test <net> <LMDB_DIR>
       cclass [-cpu/-gpu] run <net>        -- it reads path to BMP image (24bit) from stdin
       
       view_lmdb <LMDB_DIR> <id> <output>.bmp - extract the image with number <id> from the LMDB database
```

## Examples

Test the trained network on a single image: `echo test.bmp | ./cclass run net`

Test on the `mnist_train_lmdb` and `mnist_test_lmdb` databases:
```
$ ./cclass -batch 5000 test net <PATH_TO_CAFFE>/examples/mnist/mnist_train_lmdb
L 28 28 1
L 12 12 20
L 4 4 50
L 1 1 500
L 1 1 10
#0  loss=243.298752  accuracy=0.969800
#1  loss=267.474823  accuracy=0.968000
#2  loss=305.515839  accuracy=0.964200
#3  loss=257.702301  accuracy=0.967800
#4  loss=305.810333  accuracy=0.962000
#5  loss=295.774200  accuracy=0.964800
#6  loss=288.317444  accuracy=0.965800
#7  loss=293.627106  accuracy=0.964200
#8  loss=282.595978  accuracy=0.965800
#9  loss=346.519592  accuracy=0.958800
#10  loss=310.164520  accuracy=0.963600
#11  loss=225.422485  accuracy=0.973200
Total accuracy: 0.965667
TIME [00] forward 0.129446   backward 0.000000
TIME [01] forward 1.212619   backward 0.000000
TIME [02] forward 6.029575   backward 0.000000
TIME [03] forward 0.980933   backward 0.000000
TIME [04] forward 0.086596   backward 0.000000
TIME [05] forward 0.008588   backward 0.000000

$ ./cclass -batch 5000 test net <PATH_TO_CAFFE>/examples/mnist/mnist_test_lmdb
L 28 28 1
L 12 12 20
L 4 4 50
L 1 1 500
L 1 1 10
#0  loss=522.917236  accuracy=0.929600
#1  loss=259.557159  accuracy=0.966000
Total accuracy: 0.947800
TIME [00] forward 0.034327   backward 0.000000
TIME [01] forward 0.204181   backward 0.000000
TIME [02] forward 0.995707   backward 0.000000
TIME [03] forward 0.166055   backward 0.000000
TIME [04] forward 0.014212   backward 0.000000
TIME [05] forward 0.001507   backward 0.000000
```

Test on the `mnist_test_lmdb` database (GPU):  
`./cclass -gpu -batch 5000 test net <PATH_TO_CAFFE>/examples/mnist/mnist_test_lmdb`

Train from the beginning:
```bash
cp net new_net # let's create a copy
./cclass -batch 5000 -step 0.0001 -moment 0.9 train new_net <PATH_TO_CAFFE>/examples/mnist/mnist_train_lmdb
./cclass -batch 5000 -step 0.000025 -moment 0.9 train new_net <PATH_TO_CAFFE>/examples/mnist/mnist_train_lmdb
```

Output will be like this:
```
L 28 28 1
L 12 12 20
L 4 4 50
L 1 1 500
L 1 1 10
#0  loss=7862.435059  accuracy=0.124800
#1  loss=7747.129395  accuracy=0.118600
#2  loss=7218.831055  accuracy=0.132400
#3  loss=6835.105957  accuracy=0.146400
#4  loss=6838.940918  accuracy=0.132600
#5  loss=7049.811523  accuracy=0.117200
#6  loss=6863.768066  accuracy=0.141600
#7  loss=6573.500488  accuracy=0.171400
#8  loss=6171.474609  accuracy=0.208600
#9  loss=5972.420410  accuracy=0.240400
#10  loss=5856.879395  accuracy=0.257400
#11  loss=5331.467285  accuracy=0.311000
TIME [02] forward 6.418426   backward 19.648632
TIME [03] forward 1.026834   backward 5.319392
TIME [04] forward 0.090634   backward 0.204555
TIME [05] forward 0.009265   backward 0.000002

L 28 28 1
L 12 12 20
L 4 4 50
L 1 1 500
L 1 1 10
#0  loss=4708.574707  accuracy=0.383800
#1  loss=4517.254395  accuracy=0.414200
#2  loss=4575.213867  accuracy=0.404800
#3  loss=4440.705078  accuracy=0.418400
#4  loss=4457.536621  accuracy=0.413400
#5  loss=4365.724609  accuracy=0.432200
#6  loss=4489.228027  accuracy=0.415400
#7  loss=4337.015625  accuracy=0.433600
#8  loss=4319.756348  accuracy=0.435000
#9  loss=4307.382812  accuracy=0.438400
#10  loss=4258.408691  accuracy=0.446800
#11  loss=4122.996094  accuracy=0.465000
TIME [00] forward 0.153965   backward 0.000005
TIME [01] forward 1.350927   backward 3.587744
TIME [02] forward 6.758774   backward 20.904354
TIME [03] forward 1.090129   backward 5.691336
TIME [04] forward 0.096885   backward 0.216543
TIME [05] forward 0.009696   backward 0.000003
```
