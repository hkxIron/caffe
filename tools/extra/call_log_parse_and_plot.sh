log_name="log_train_lenet.log"
#test accuracy vs iters
<<EOF
./plot_training_log.py.example \
    0\
    log_parse/test_acc_vs_iters.png\
    $CAFFE_HOME/practice/logs/$log_name

#test accuracy vs seconds
./plot_training_log.py.example \
    1\
    log_parse/test_acc_vs_seconds.png\
    $CAFFE_HOME/practice/logs/$log_name


#test loss vs iters
./plot_training_log.py.example \
    2\
    log_parse/test_loss_vs_iters.png\
    $CAFFE_HOME/practice/logs/$log_name

#test loss vs seconds
./plot_training_log.py.example \
    3\
    log_parse/test_loss_vs_seconds.png\
    $CAFFE_HOME/practice/logs/$log_name

EOF
#train learning rate vs iters
./plot_training_log.py.example \
    4\
    log_parse/train_lr_rate_vs_iters.png\
    $CAFFE_HOME/practice/logs/$log_name

<<EOF
#train loss vs iters
./plot_training_log.py.example \
    6\
    log_parse/train_loss_vs_iters.png\
    $CAFFE_HOME/practice/logs/$log_name
EOF

<<EOF
EOF
rm $log_name.test
rm $log_name.train
