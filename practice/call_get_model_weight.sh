#!/bin/bash
#sh examples/mnist/train_lenet.sh >&1|tee practice/logs/log_train_lenet
#nohup sh examples/mnist/train_lenet.sh >&1|tee practice/logs/log_train_lenet
python get_model_weight.py \
    'examples/mnist/lenet_train_test.prototxt' \
    'examples/mnist/lenet_iter_10000.caffemodel' \
    'practice/mnist_lenet_params.txt'
