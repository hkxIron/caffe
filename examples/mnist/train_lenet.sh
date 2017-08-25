#!/usr/bin/env sh
set -e

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
#${CAFFE_HOME}/build/tools/caffe train --solver=../../examples/mnist/lenet_solver.prototxt $@
${CAFFE_HOME}/build/tools/caffe train --solver=${CAFFE_HOME}/examples/mnist/lenet_solver.prototxt $@
