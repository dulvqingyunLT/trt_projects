# trt_projects 
./FasterRCNN convert  first stage onnx file to trt engine, construct rcnn and load weights, add two customizes plugin: 
DynamicDelta2Bbox_TRT, DynamicPyramidROIAlign_TRT.

./RetinaNet construct retinanet from onnx file, nms is included int the onnx file.
./RetinaNet_quant construc retinaent from onnx file withou nms.
./LGTPlugin contains definition of customized plugins.
./utis/calibrator.hpp definition of customized calibration of int8 quantization.
./utils/file_utils functions of reading files from disk.
./utils/post_process.h  detection results process such as  iou, nms, delta2bbox
./utils/pre_process.h  image pre_process like normalization resize with/without padding. 
