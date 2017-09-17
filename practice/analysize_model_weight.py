#!/usr/bin/env python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os,sys
caffe_root = '/home/hkx/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe

sep = "=" * 50
def get_labels():
    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    return labels

def get_transformer(net):
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    print "mu shape:",mu.shape
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    print "data shape:", net.blobs['data'].data.shape
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #将图片设置为模型的大小
    # python读取的图片文件格式为H×W×K，需转化为K×H×W，set_transpose
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    # python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR, caffe中图片是BGR格式，而原始格式是RGB，所以要转化
    print "data shape:", net.blobs['data'].data.shape
    return transformer

def vis_square(data,title=""):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # 注意：最后一位是通道
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    # print "data shape:",data.shape
    n = int(np.ceil(np.sqrt(data.shape[0])))  # 比如若有96个图片，那么设置10×20个格子
    
    padding = (((0, n ** 2 - data.shape[0]),   # before, after edge for each axis
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    #print "paddding:",padding
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).\
    transpose((0, 2, 1, 3) 
              + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    if len(title)>0:plt.title(title)
    print "original data shape:",data.shape
    if len(data.shape) == 3:
        if data.shape[2]==1:
            print "WARN:repeat channel to 3!"
            data = np.repeat(data,3,axis=2)
        elif data.shape[2] > 3 :
            print "WARN:show the first 3 channels!"
            data =data[:,:,0:3]
    print "show data shape:",data.shape
    plt.imshow(data); plt.axis('off')
    plt.show()

def get_net(m_def,m_weights):
    caffe.set_mode_cpu()
    model_def = caffe_root +m_def
    model_weights = caffe_root + m_weights

    #model_def = caffe_root + 'examples/mnist/lenet_train_test.prototxt' #并不是这个net,如果用此net会报错
    #model_def = caffe_root + 'examples/mnist/lenet.prototxt' #注意这里的net并不是训练时的net，而是发布时的
    #model_weights = caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout), 终于明白了它的意思
    return net

def show_layer_output_and_params(net):
    print sep+"\nEach layer output shape:"
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    """
    for param_name in net.params.keys():
        # 权重参数  
        weight = net.params[param_name][0].data
        # 偏置参数  
        bias = net.params[param_name][1].data
        print param_name+" weight:"+str(weight.shape)+ " bias:"+ str(bias.shape)
    """

    print sep+"\nEach params shape:"
    for layer_name, param in net.params.iteritems():
        print layer_name + '\t'+" weight:" + str(param[0].data.shape), ' bias:',str(param[1].data.shape)

def show_some_layer_param(net):
    layer_name="conv1"
    folder = "params/"
    filters = net.params[layer_name][0].data
    file_name = folder+layer_name+"_weight.npy"
    np.save(file_name,filters)
    print("save to "+file_name)
    filter_bias = net.params[layer_name][1].data
    print "filters weights shape:",filters.shape
    vis_square(filters.transpose(0, 2, 3, 1),layer_name+" weights")  # 显示时，需要将rgb通道放到最后，以适应imshow函数,即 NCWH->NWHC

def show_pred(net,image):
    output_prob = net.blobs['prob'].data[0]
    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]
    plt.imshow(image)
    plt.show()
    labels = get_labels()
    print 'probabilities and labels:'
    print zip(output_prob[top_inds], labels[top_inds])

def show_features(net):
    # 打印featutes
    layer_name="conv1"
    print "conv1 data shape:",net.blobs[layer_name].data.shape
    feat = net.blobs[layer_name].data[0, :36] #显示部分feat
    print "feat shape:",feat.shape
    vis_square(feat,layer_name+"_feature")
    # 显示统计信息
    layer_name="fc6"
    feat = net.blobs[layer_name].data[0]
    plt.subplot(2, 1, 1)
    plt.title("feat")
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    plt.title("feat hist")
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
    plt.show()

def show_output_prob(net):
    # 查看输出的概率分布
    layer_name="prob"
    feat = net.blobs[layer_name].data[0]
    print "blob prob size:", net.blobs[layer_name].data.shape
    plt.figure(figsize=(15, 3))
    plt.plot(feat.flat)
    plt.title(layer_name)
    plt.show()


def predict_image(net):
    # for each layer, show the output shape
    # the parameters are a list of [weights, biases]
    my_image_url = "../examples/image.jpg"  # paste your URL here
    # transform it and copy it into the net
    image = caffe.io.load_image(my_image_url)
    transformer = get_transformer(net)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    # perform classification
    net.forward()
    # obtain the output probabilities
    show_pred(net,image)
    show_features(net)
    show_output_prob(net)

def main():
    net = get_net('models/bvlc_reference_caffenet/deploy.prototxt','models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    show_layer_output_and_params(net)
    show_some_layer_param(net)
    predict_image(net)

main()
