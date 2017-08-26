#!/usr/bin/env sh
set -e

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
#${CAFFE_HOME}/build/tools/caffe train --solver=../../examples/mnist/lenet_solver.prototxt $@
./build/tools/caffe train --solver=./examples/mnist/lenet_solver.prototxt \
    --snapshot examples/mnist/lenet_iter_5000.solverstate\
    $@
