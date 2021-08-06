#!/usr/bin/env python3


from os import name
from numpy.core.fromnumeric import transpose
import onnx_graphsurgeon as gs
import numpy as np
import onnx


# def loadWeights(weight_file):
#     weight_map = dict()
#     count =0
#     with open(weight_file, 'r') as f:
#         count = f.readline()



def main():

    graph = gs.import_onnx(onnx.load("/home/ubuntu/source_code/detection/mmdetection/onnx_file/faster_rcnn_stageone_0723.onnx"))
    tensors_stage_one = graph.tensors()
    rois = tensors_stage_one['rois']
    roi_scores = tensors_stage_one['roi_scores']
    fp0 = tensors_stage_one['fp0']
    fp1 = tensors_stage_one['fp1']
    fp2 = tensors_stage_one['fp2']
    fp3 = tensors_stage_one['fp3']

    input = tensors_stage_one['input']


    
    # build rpn nms node
    rpn_keepTopK = 1000
    rpn_num_detections = gs.Variable(name='rpn_num_detections', dtype=np.int32, shape=(gs.Tensor.DYNAMIC, 1))
    rpn_nmsed_boxes = gs.Variable(name='rpn_nmsed_boxes', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, rpn_keepTopK, 4))
    rpn_nmsed_scores = gs.Variable(name='rpn_nmsed_scores', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, rpn_keepTopK))
    rpn_nmsed_classes = gs.Variable(name='rpn_nmsed_classes', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, rpn_keepTopK))

    rpn_batch_nms_node = gs.Node(op='BatchedNMSDynamic_TRT', inputs=[rois, roi_scores],
                outputs=[rpn_num_detections,rpn_nmsed_boxes,rpn_nmsed_scores,rpn_nmsed_classes],
                attrs={
                    'shareLocation':False,
                    'backgroundLabelId':-1, 
                    'numClasses':1,
                    'topK':3600,
                    'keepTopK':rpn_keepTopK,
                    'scoreThreshold':0.0,
                    'iouThreshold':0.7,
                    'isNormalized':False,
                    'clipBoxes':False,
                    'scoreBits':16,
                    }
                )
    graph.nodes.append(rpn_batch_nms_node)



    # build roi_align node
    # input_image_size = input.shape[-1]
    roi_align_out = gs.Variable(name='roi_align_out', dtype=np.float32, shape=(-1,rpn_keepTopK,256,7,7))
    roi_align_node = gs.Node(op='DynamicPyramidROIAlign_TRT', inputs=[rpn_nmsed_boxes,fp0,fp1,fp2,fp3],
                outputs=[roi_align_out],
                attrs={
                    'pooled_size':7,
                    'input_size_h':input.shape[-2],
                    'input_size_w':input.shape[-1],
                }

            )
    graph.nodes.append(roi_align_node)
    


    # build rcnn that's second stage
    rcnn_weight_map = np.load('./onnx_file/rcnn.npy', allow_pickle=True).item()

    tmp_share_fc_0_weight = rcnn_weight_map['bbox_head.shared_fcs.0.weight'] #(1024,12544)
    tmp_share_fc_0_weight_t = np.transpose(tmp_share_fc_0_weight)
    share_fc_0_weight = gs.Constant(name='share_fc_0_weight', values=tmp_share_fc_0_weight_t)
    share_fc_0_bias = gs.Constant(name='share_fc_0_bias', values=rcnn_weight_map['bbox_head.shared_fcs.0.bias']) #(1024,)

    tmp_share_fc_1_weight = rcnn_weight_map['bbox_head.shared_fcs.1.weight'] #(1024,1024)
    tmp_share_fc_1_weight_t = np.transpose(tmp_share_fc_1_weight)
    share_fc_1_weight = gs.Constant(name='share_fc_1_weight', values=tmp_share_fc_1_weight_t)
    share_fc_1_bias = gs.Constant(name='share_fc_1_bias', values=rcnn_weight_map['bbox_head.shared_fcs.1.bias'])  #(1024,) 

    tmp_fc_cls_weight = rcnn_weight_map['bbox_head.fc_cls.weight'] #(81,1024)
    tmp_fc_cls_weight_t = np.transpose(tmp_fc_cls_weight)
    fc_cls_weight = gs.Constant(name='fc_cls_weight', values=tmp_fc_cls_weight_t)
    fc_cls_bias = gs.Constant(name='fc_cls_bias', values=rcnn_weight_map['bbox_head.fc_cls.bias'])   #(81,)


    tmp_fc_reg_weight = rcnn_weight_map['bbox_head.fc_reg.weight'] #(320,1024)
    tmp_fc_reg_weight_t = np.transpose(tmp_fc_reg_weight)
    fc_reg_weight = gs.Constant(name='fc_reg_weight', values=tmp_fc_reg_weight_t)
    fc_reg_bias = gs.Constant(name='fc_reg_bias', values=rcnn_weight_map['bbox_head.fc_reg.bias'])  #(320,)

    cls_num = tmp_fc_reg_weight.shape[0]//4

    # reshape_0_toshape = gs.Constant(name='reshape_0_toshape', values=np.array([-1, rpn_keepTopK, tmp_share_fc_0_weight.shape[1]], dtype=np.int64))
    # reshape_0_out = gs.Variable(name='reshape_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_0_weight.shape[1])) #(-1, 1000, 12544)
    # reshape_node_0 = gs.Node(op='Reshape', inputs=[roi_align_out, reshape_0_toshape], outputs=[reshape_0_out])
    # graph.nodes.append(reshape_node_0)
    flatten_0_out = gs.Variable(name='flatten_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_0_weight.shape[1])) #(-1, 1000, 12544)
    flatten_node_0 = gs.Node(op='Flatten', inputs=[roi_align_out], outputs=[flatten_0_out], attrs={'axis':2})
    graph.nodes.append(flatten_node_0)
    
    # graph.outputs = [flatten_0_out]
    # onnx.save(gs.export_onnx(graph), "./tools/onnx_graphsurgen/faster-rcnn/faster-rcnn_r50.onnx")
    # return

  
    share_fc_0_out = gs.Variable(name='share_fc_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_0_weight.shape[0]))
    share_fc_0_node = gs.Node(op='MatMul', inputs=[flatten_0_out, share_fc_0_weight], outputs=[share_fc_0_out])
    graph.nodes.append(share_fc_0_node)
    share_fc_0_bias_out = gs.Variable(name='share_fc_0_bias_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_0_weight.shape[0]))
    share_fc_0_bias_node = gs.Node(op='Add', inputs=[share_fc_0_out, share_fc_0_bias], outputs=[share_fc_0_bias_out])
    graph.nodes.append(share_fc_0_bias_node)
    share_fc_relu_0_out = gs.Variable(name='share_fc_relu_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_0_weight.shape[0]))
    share_fc_relu_0_node = gs.Node(op='Relu', inputs=[share_fc_0_bias_out], outputs=[share_fc_relu_0_out])
    graph.nodes.append(share_fc_relu_0_node)


    share_fc_1_out = gs.Variable(name='share_fc_1_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_1_weight.shape[0]))
    share_fc_1_node = gs.Node(op='MatMul', inputs=[share_fc_relu_0_out, share_fc_1_weight], outputs=[share_fc_1_out])
    graph.nodes.append(share_fc_1_node)
    share_fc_1_bias_out = gs.Variable(name='share_fc_1_bias_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_1_weight.shape[0]))
    share_fc_1_bias_node = gs.Node(op='Add', inputs=[share_fc_1_out, share_fc_1_bias], outputs=[share_fc_1_bias_out])
    graph.nodes.append(share_fc_1_bias_node)
    share_fc_relu_1_out = gs.Variable(name='share_fc_relu_1_out', dtype=np.float32, shape=(-1, rpn_keepTopK, tmp_share_fc_1_weight.shape[0]))
    share_fc_relu_1_node = gs.Node(op='Relu', inputs=[share_fc_1_bias_out], outputs=[share_fc_relu_1_out])
    graph.nodes.append(share_fc_relu_1_node)


    fc_cls_out = gs.Variable(name='fc_cls_out', dtype=np.float32, shape=(-1, rpn_keepTopK, cls_num+1))
    fc_cls_node = gs.Node(op='MatMul', inputs=[share_fc_relu_1_out, fc_cls_weight], outputs=[fc_cls_out])
    graph.nodes.append(fc_cls_node)
    fc_cls_bias_out = gs.Variable(name='fc_cls_bias_out', dtype=np.float32, shape=(-1,rpn_keepTopK,cls_num+1))
    fc_cls_bias_node = gs.Node(op='Add', inputs=[fc_cls_out, fc_cls_bias], outputs=[fc_cls_bias_out])
    graph.nodes.append(fc_cls_bias_node)


    reshape_cls_toshape = gs.Constant(name='reshape_cls_toshape', values=np.array([-1, rpn_keepTopK, cls_num+1], dtype=np.int64))
    reshape_cls_out = gs.Variable(name='reshape_cls_out', dtype=np.float32, shape=(-1, rpn_keepTopK, cls_num+1)) 
    reshape_cls_noe = gs.Node(op='Reshape', inputs=[fc_cls_bias_out,reshape_cls_toshape], outputs=[reshape_cls_out])
    graph.nodes.append(reshape_cls_noe) 


    softmax_0_out = gs.Variable(name='softmax_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, cls_num+1))
    softmax_0_node = gs.Node(op='Softmax', inputs=[reshape_cls_out], outputs=[softmax_0_out], attrs={'axis':2})
    graph.nodes.append(softmax_0_node)




    slice_0_out = gs.Variable(name='slice_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, cls_num)) #(-1, 1000, 80)
    # slice_0_node = gs.Node(op='DynamicSliceBackground_TRT', inputs=[softmax_0_out], outputs=[slice_0_out])

    slic_0_axes = gs.Constant(name='slic_0_axes', values=np.array([2], dtype=np.int64))
    slic_0_starts = gs.Constant(name='slic_0_starts', values=np.array([0], dtype=np.int64))
    slic_0_ends = gs.Constant(name='slic_0_ends', values=np.array([cls_num], dtype=np.int64))
    slice_0_node = gs.Node(op='Slice', inputs=[softmax_0_out, slic_0_starts, slic_0_ends, slic_0_axes], outputs=[slice_0_out])
    graph.nodes.append(slice_0_node)


    fc_reg_out = gs.Variable(name='fc_reg_out', dtype=np.float32, shape=(-1, rpn_keepTopK, fc_reg_bias.shape[0]))
    fc_reg_node = gs.Node(op='MatMul', inputs=[share_fc_relu_1_out, fc_reg_weight], outputs=[fc_reg_out])
    graph.nodes.append(fc_reg_node)
    fc_reg_bias_out = gs.Variable(name='fc_reg_bias_out', dtype=np.float32, shape=(-1, rpn_keepTopK, fc_reg_bias.shape[0]))
    fc_reg_bias_node = gs.Node(op='Add', inputs=[fc_reg_out, fc_reg_bias], outputs=[fc_reg_bias_out])
    graph.nodes.append(fc_reg_bias_node)


    reshape_1_toshape = gs.Constant(name='reshape_1_toshape', values=np.array([-1, rpn_keepTopK, cls_num, 4], dtype=np.int64))
    reshape_1_out = gs.Variable(name='reshape_1_out', dtype=np.float32, shape=(-1, rpn_keepTopK, cls_num, 4)) #[-1,1000,80,4]
    reshape_1_noe = gs.Node(op='Reshape', inputs=[fc_reg_bias_out,reshape_1_toshape], outputs=[reshape_1_out])
    graph.nodes.append(reshape_1_noe) 





    # decode:delta2bbox
    decode_0_out = gs.Variable(name='decode_0_out', dtype=np.float32, shape=(-1, rpn_keepTopK, cls_num, 4))
    decode_0_node = gs.Node(op='DynamicDelta2Bbox_TRT', inputs=[rpn_nmsed_boxes, reshape_1_out], outputs=[decode_0_out],
                            attrs={
                                    'input_h':input.shape[-2],
                                    'input_w':input.shape[-1]
                                }
                            )
    graph.nodes.append(decode_0_node)


    # detection output nms
    bbox_keepTopK = 100
    bbox_num_detections = gs.Variable(name='num_detections', dtype=np.int32, shape=(gs.Tensor.DYNAMIC, 1))
    bbox_nmsed_boxes = gs.Variable(name='nmsed_boxes', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, bbox_keepTopK, 4))
    bbox_nmsed_scores = gs.Variable(name='nmsed_scores', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, bbox_keepTopK))
    bbox_nmsed_classes = gs.Variable(name='nmsed_classes', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, bbox_keepTopK))
    bbox_batch_nms_node = gs.Node(op='BatchedNMSDynamic_TRT', inputs=[decode_0_out, slice_0_out],
                outputs=[bbox_num_detections,bbox_nmsed_boxes,bbox_nmsed_scores,bbox_nmsed_classes],
                attrs={
                    'shareLocation':False,
                    'backgroundLabelId':-1, 
                    'numClasses':cls_num,
                    'topK':rpn_keepTopK,
                    'keepTopK':bbox_keepTopK,
                    'scoreThreshold':0.3,
                    'iouThreshold':0.5,
                    'isNormalized':False,
                    'clipBoxes':False,
                    'scoreBits':16,
                    }
                )    
    graph.nodes.append(bbox_batch_nms_node)


    graph.outputs = [bbox_num_detections,bbox_nmsed_boxes,bbox_nmsed_scores,bbox_nmsed_classes]

    # graph.cleanup()

    onnx.save(gs.export_onnx(graph), "./tools/onnx_graphsurgen/faster-rcnn/faster-rcnn_r50.onnx")


if __name__ == '__main__':
    main()




