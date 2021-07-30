#!/usr/bin/env bash

trtexec \
--onnx=faster-rcnn_r50.onnx \
--explicitBatch \
--minShapes=1x3x384x384 \
--optShapes=8x3x384x384 \
--maxShapes=48x3x384x384 \
--saveEngine=faster-rcnn_r50.trt \
--plugins=/home/ubuntu/source_code/trt_projects/LGT/build/LGTPlugin/libFasterRCNNKernels.so \
--verbose

