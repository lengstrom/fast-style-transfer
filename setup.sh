#! /bin/bash

mkdir data
cd data
wget -c -t 0 http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
mkdir bin
wget -c -t 0 http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
