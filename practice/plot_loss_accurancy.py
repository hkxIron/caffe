#coding:utf-8
#加载必要的库
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
#%%matplotlib inline
import sys,os
"""
pycaffe加载模型时通常会输出加载模型的日志，影响我们查看自己的日志，因此需要移除Caffe加载模型时的日志
Caffe使用的日志是GLOG，其日志级别如下：
0 - debug
1 - info (still a LOT of outputs)
2 - warnings
3 - errors
注意：由于在导入Caffe时Caffe会加载GLOG，因此os.environ['GLOG_minloglevel'] = '2'需要在import caffe之前。
"""
os.environ['GLOG_minloglevel'] = '2'
import caffe
#设置当前目录
caffe_root = '/home/hkx/caffe/' 
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
print "current dir:"+os.getcwd()
#caffe.set_device(0)
#caffe.set_mode_gpu()
#solver = caffe.SGDSolver('examples/cifar10/cifar10_quick_solver.prototxt')
# set the solver prototxt
solver = caffe.SGDSolver('examples/mnist/lenet_solver.prototxt')

#%%time
niter =4000
test_interval = 200
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')
    
    if it % test_interval == 0:#每隔一段时间打印一次
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration:', it, ' testing...',' accuracy:',acc
        test_acc[it // test_interval] = acc

print test_acc
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
show()
