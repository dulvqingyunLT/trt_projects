#!/usr/bin/env python3


import onnx_graphsurgeon as gs
import numpy as np
import onnx


def main():

    graph = gs.import_onnx(onnx.load("/home/ubuntu/source_code/detection/mmdetection/onnx_file/retinanet_101.onnx"))

    tensors = graph.tensors()
    origin_output_score = tensors["scores"]
    origin_output_bbox = tensors["boxes"]

    # nms = None

    # for node in graph.nodes:
    #     if node.op == 'NonMaxSuppression':
    #         nms = node

    # boxes = nms.inputs[0]

    keepTopK = 300

    num_detectinos = gs.Variable(name='num_detection', dtype=np.int32, shape=(gs.Tensor.DYNAMIC, 1))
    nmsed_boxes = gs.Variable(name='nmsed_boxes', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, keepTopK, 4))
    nmsed_scores = gs.Variable(name='nmsed_scores', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, keepTopK))
    nmsed_classes = gs.Variable(name='nmsed_classes', dtype=np.float32, shape=(gs.Tensor.DYNAMIC, keepTopK))


    batch_nms = gs.Node(op='BatchedNMSDynamic_TRT', inputs=[origin_output_bbox, origin_output_score],
                outputs=[num_detectinos,nmsed_boxes,nmsed_scores,nmsed_classes],
                attrs={
                    'shareLocation':True,
                    'backgroundLabelId':-1, 
                    'numClasses':80,
                    'topK':1000,
                    'keepTopK':keepTopK,
                    'scoreThreshold':0.3,
                    'iouThreshold':0.5,
                    'isNormalized':False,
                    'clipBoxes':True,
                    'scoreBits':16,
                    }
                )

    # def insertnode(graph, node):
    #     floattensor = node.inputs[1]
    #     cast_int32 = gs.Variable(name='cast_int32'+node.name,dtype=np.int32)
    #     cast_node = gs.Node(op='Cast',inputs=[floattensor],outputs=[cast_int32],attrs={"to":onnx.TensorProto.INT32})
    #     node.inputs[1] = cast_int32
    #     graph.nodes.append(cast_node)

    # for node in graph.nodes:
    #     if node.name in['Add_449','Add_587','Add_725']:
    #         insertnode(graph,node)


    graph.nodes.append(batch_nms)
    graph.outputs = [num_detectinos,nmsed_boxes,nmsed_scores,nmsed_classes]
    graph.cleanup()

    onnx.save(gs.export_onnx(graph), "./tools/onnx_graphsurgen/retinanet/retinanet_r101_nms.onnx")


if __name__ == '__main__':
    main()




