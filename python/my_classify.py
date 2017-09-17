# coding:utf-8
import caffe
import numpy as np
# ===================配置========================
# caffemodel文件
MODEL_FILE = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# deploy文件，参考/caffe/models/bvlc_alexnet/deploy.prototxt
DEPLOY_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
#DEPLOY_FILE = '../models/deploy.prototxt'
# 测试图片存放文件夹
# TEST_ROOT = 'datas/'
# caffe.set_mode_gpu()
caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, MODEL_FILE, caffe.TEST)
# ===================数据预处理========================
# 'data'对应于deploy文件：
# input: "data"
# input_dim: 1
# input_dim: 3
# input_dim: 32
# input_dim: 96
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# python读取的图片文件格式为H×W×K，需转化为K×H×W
transformer.set_transpose('data', (2, 0, 1))
# python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，
# 所以需要一个转换
transformer.set_raw_scale('data', 255)
# caffe中图片是BGR格式，而原始格式是RGB，所以要转化
transformer.set_channel_swap('data', (2, 1, 0))
# 将输入图片格式转化为合适格式（与deploy文件相同）
print 'data shape:', net.blobs['data'].data.shape
net.blobs['data'].reshape(50,3, 227, 227)
#net.blobs['data'].reshape(1,3, 227, 227)
#net.blobs['data'].reshape(1, 3, 32, 96)

# 读取图片
# 详见/caffe/python/caffe/io.py
img = caffe.io.load_image('../examples/images/cat.jpg')
print "original img dims:",img.shape
#img = caffe.io.load_image('temp.jpg')
# 读取的图片文件格式为H×W×K，需转化
# 数据输入、预处理
net.blobs['data'].data[...] = transformer.preprocess('data', img)
# 前向迭代，即分类
out = net.forward()
# 输出结果为各个可能分类的概率分布
predicts = out['prob'][0] # first images
# 上述'prob'来源于deploy文件：
# layer {
#   name: "prob"
#   type: "Softmax"
#   bottom: "ip2"
#   top: "prob"
# }

# 最可能分类
predict = predicts.argmax()
print "predict:",predict

# load ImageNet labels
labels_file = '../data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[predict]
