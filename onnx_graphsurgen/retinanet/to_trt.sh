#!/usr/bin/env bash

trtexec \
--onnx=retinanet_r101_nms.onnx \
--explicitBatch \
--minShapes=1x3x800x800 \
--optShapes=8x3x800x800 \
--maxShapes=48x3x800x800 \
--saveEngine=retinanet_r101_nms.trt \
--verbose
