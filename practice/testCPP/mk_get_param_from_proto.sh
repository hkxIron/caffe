g++ -o test get_param_from_proto.cpp ${CAFFE_HOME}/build/src/caffe/proto/caffe.pb.cc -I${CAFFE_HOME}/build/src/caffe/proto/ -I/usr/include/ -L/usr/lib -lprotobuf
