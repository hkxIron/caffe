#!/usr/bin/env sh
set -e

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
#${CAFFE_HOME}/build/tools/caffe train --solver=../../examples/mnist/lenet_solver.prototxt $@
# 使用预训练好的mnist模型进行训练
./build/tools/caffe train --solver=./examples/mnist/lenet_solver_hkx.prototxt \
    --weights ./examples/mnist/lenet_iter_10000.caffemodel \
    $@
