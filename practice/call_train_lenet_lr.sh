#!/bin/bash
cd $CAFFE_HOME
#sh examples/mnist/train_lenet.sh >&1|tee practice/logs/log_train_lenet
#nohup sh examples/mnist/train_lenet.sh >&1|tee practice/logs/log_train_lenet
nohup ${CAFFE_HOME}/build/tools/caffe train --solver=${CAFFE_HOME}/examples/mnist/lenet_lr_solver.prototxt >&1|tee practice/logs/log_train_lenet_lr.log
