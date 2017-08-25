#!/bin/bash
cd $CAFFE_HOME
sh examples/mnist/train_lenet.sh >&1|tee practice/logs/log_train_lenet
