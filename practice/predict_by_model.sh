cd $CAFFE_HOME
./build/tools/caffe.bin test \
    -model examples/mnist/lenet_train_test.prototxt \
    -weights examples/mnist/lenet_iter_10000.caffemodel \
    -iterations 100
