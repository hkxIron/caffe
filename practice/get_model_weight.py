#!/usr/bin/env python
#coding:utf-8

# 引入“咖啡”
import caffe
import os,sys
import numpy as np

# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold='nan')
CAFFE_HOME='/home/hkx/caffe/'
# deploy文件
#MODEL_PROTO_FILE = 'caffe_deploy.prototxt'
if len(sys.argv)<4:
    print("No enough arguments!"+sys.argv[0]+" args num:"+str(len(sys.argv)) \
            +" Arguments are:model_proto_file pretrain_model_file out_params_file")
    sys.exit(-1)
MODEL_PROTO_FILE =sys.argv[1]  #'examples/mnist/lenet_train_test.prototxt'
# 预先训练好的caffe模型
PRETRAIN_MODEL_FILE =sys.argv[2] # 'examples/mnist/lenet_iter_10000.caffemodel'
# 保存参数的文件
out_params_file =sys.argv[3] #'practice/mnist_lenet_params.txt'

os.chdir(CAFFE_HOME)
print("current dir:"+os.getcwd()+"\n")
out_file = open(out_params_file, 'w')
print("model file:"+MODEL_PROTO_FILE
        +"\nweight file:"+PRETRAIN_MODEL_FILE
        +"\nout weight file:"+out_params_file+"\n")
# 让caffe以测试模式读取网络参数
try:
    net = caffe.Net(MODEL_PROTO_FILE, PRETRAIN_MODEL_FILE, caffe.TEST)
except Exception as e:
    print("[Load model error]:"+str(e))
print("Load model success!\n")
print("layer with weight count:"+str(len(net.params.keys())))
# 遍历每一层
for param_name in net.params.keys():
    # 权重参数
    weight = net.params[param_name][0].data
    # 偏置参数
    bias = net.params[param_name][1].data
    layer_str="layer_name:"+param_name+"\n"
    print(layer_str)
    #print weight 
    # 该层在prototxt文件中对应“top”的名称
    out_file.write(layer_str)

    # 写权重参数
    out_file.write('weight shape:'+str(weight.shape)+' weight data:\n')
    # 权重参数是多维数组，为了方便输出，转为单列数组
    weight.shape = (-1, 1)

    for w in weight:
        out_file.write('%ff, ' % w)

    # 写偏置参数
    out_file.write('\nbias shape:'+str(bias.shape) +' bias data:\n')
    # 偏置参数是多维数组，为了方便输出，转为单列数组
    bias.shape = (-1, 1)
    for b in bias:
        out_file.write('%ff, ' % b)
    out_file.write('\n\n')

out_file.close()
print "write model success to file:"+out_params_file
