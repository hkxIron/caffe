g++ -o blob_demo_diff blob_demo_diff.cpp  -I $CAFFE_ROOT/include/ -D CPU_ONLY -I $CAFFE_ROOT/.build_release/src/ -L $CAFFE_ROOT/build/lib/ -lcaffe -lglog -lboost_system
