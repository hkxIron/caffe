#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.
set -e
#DATA_PREFIX=/home/hkx/windata/cifar-10-binary/cifar-10-batches-bin
#EXAMPLE=$DATA_PREFIX/examples/cifar10
#DATA=$DATA_PREFIX/data/cifar10
DBTYPE=lmdb
EXAMPLE=examples/cifar10
DATA=data/cifar10

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

./build/examples/cifar10/convert_cifar_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

./build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
