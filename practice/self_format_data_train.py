#coding=utf-8
"""
设置自定义格式的数据训练caffe
"""
import caffe
import numpy as np
from caffe import layers as L
from caffe.proto import caffe_pb2

# 这里使用 NetSpec 生成 prototxt 文件
# 你也可以手工去写 prototxt 文件
def createNet(shape1, shape2):
    blobShape1 = caffe_pb2.BlobShape()
    blobShape1.dim.extend(shape1)
    blobShape2 = caffe_pb2.BlobShape()
    blobShape2.dim.extend(shape2)

    n = caffe.NetSpec() # 创建net
    n.data1, n.data2 = L.Input(shape=[blobShape1, blobShape2], ntop=2)
    
    prototxt = 'temp.prototxt'
    with open(prototxt, 'w') as f:
        f.write(str(n.to_proto()))
    net = caffe.Net(prototxt, caffe.TEST)
    #os.remove(prototxt)
    return net

def forward(net, data1, data2):
    net.blobs['data1'].data[...] = data1
    net.blobs['data2'].data[...] = data2
    net.forward()
    return net.blobs['data1'].data, net.blobs['data2'].data

def caffeInputTest(data1, data2):
    net = createNet(data1.shape, data1.shape)
    return forward(net, data1, data2)

if __name__ == '__main__':
    x1 = np.random.rand(3, 4, 5)
    x2 = np.random.rand(3, 4, 5)
    print "x1: ", x1
    print "x2: ", x2
    y1, y2 = caffeInputTest(x1, x2)
    print(np.allclose(x1, y1)) # 判断 a, b中的所有元素否相等
    print(np.allclose(x2, y2))
