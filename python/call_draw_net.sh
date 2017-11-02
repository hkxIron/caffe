python draw_net.py ../examples/mnist/lenet_train_test.prototxt model_imgs/lenet_train_test.png
<<EOF
python draw_net.py ../models/bvlc_alexnet/train_val.prototxt model_imgs/alexnet.png
python draw_net.py ../models/bvlc_googlenet/train_val.prototxt model_imgs/googlenet.png
python draw_net.py ../models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt model_imgs/rcnn.png
python draw_net.py ../examples/mnist/mnist_autoencoder.prototxt model_imgs/mnist_autoencoder.png
python draw_net.py ../examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt model_imgs/cifar10_full_sigmoid_bn.png
python draw_net.py /home/hkx/windata/mnist/lenet_fusion_train_val.prototxt model_imgs/mnist_fusion.png
python draw_net.py /home/hkx/windata/shopsearch/ss_lstm_test_word2vec_4.prototxt model_imgs/shopsearch_lstm_word2vec.png
EOF
