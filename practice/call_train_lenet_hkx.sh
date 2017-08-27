#!/bin/bash
cd $CAFFE_HOME
#sh examples/mnist/train_lenet.sh >&1|tee practice/logs/log_train_lenet
#使用预训练好的模型进行微调
nohup sh examples/mnist/train_lenet_hkx.sh >&1|tee practice/logs/log_train_lenet_hkx
