/* implentation of faster_rcnn 
 *
 *
 */

#ifndef FASTERRCNN_FASTERRCNN_H
#define FASTERRCNN_FASTERRCNN_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

#include <sstream>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "opencv2/opencv.hpp"

#include "../LGTPlugin/plugins/DynamicPyramidROIAlignPlugin.h"
#include "../LGTPlugin/plugins/DynamicDelta2BboxPlugin.h"
#include "../LGTPlugin/plugins/DynamicSliceBackgroundPlugin.h"

namespace LGT
{
    


static constexpr int poolSize = 7;
static constexpr int featureMapChannels = 256;
static constexpr int inputSize = 384;

static constexpr    bool rpn_shareLocation = false;
static constexpr    int rpn_backgroundLabelId = -1;
static constexpr    int rpn_numClasses = 1;
static constexpr    int rpn_topK = 1000;
static constexpr    int rpn_keepTopK = 600;
static constexpr    float rpn_scoreThreshold = 0.0;
static constexpr    float rpn_iouThreshold = 0.7;
static constexpr    bool rpn_isNormalized = false;
static constexpr    bool rpn_clipBoxes = false;
static constexpr    int rpn_scoreBits = 16;


static constexpr    bool bbox_shareLocation = false;
static constexpr    int bbox_backgroundLabelId = -1;
static constexpr    int bbox_numClasses = 80;
static constexpr    int bbox_topK = 300;
static constexpr    int bbox_keepTopK = 100;
static constexpr    float bbox_scoreThreshold = 0.3;
static constexpr    float bbox_iouThreshold = 0.5;
static constexpr    bool bbox_isNormalized = false;
static constexpr    bool bbox_clipBoxes = false;
static constexpr    int bbox_scoreBits = 16;




static constexpr float target_mean[4] = {0, 0, 0, 0};
static constexpr float target_std[4] = {0.1, 0.1, 0.2, 0.2};

static constexpr float img_mean[3]={103.53, 116.28, 123.675,};// BGR order
static constexpr float img_std[3]={57.375, 57.12, 58.395 };// BGR order

// static  std::vector<std::string> imageList = {"000000000285.jpg"};//测试用
static  std::vector<std::string> imageList = {"000000397133.jpg", "000000037777.jpg"};//测试用
// static  std::vector<std::string> imageList = {"000000037777.jpg"};//测试用

const std::vector<std::string> classes{
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

const std::string gSampleName = "TensorRT.FasterRCNN";


struct FasterRCNNParams : public samplesCommon::OnnxSampleParams
{
    int outputClsSize; //!< The number of output classes
    int nmsMaxOut;     //!< The maximum number of detection post-NMS
    std::string rcnnWeightsFile;
    // int roiCount; // proposal numbers
    bool saveEngine{false};
    bool profile{false};
};

struct NMSParameters
{
    bool shareLocation;
    int32_t backgroundLabelId, numClasses, topK, keepTopK;
    float scoreThreshold, iouThreshold;
    bool isNormalized;
};

class FasterRCNN
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    FasterRCNN(const FasterRCNNParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    FasterRCNNParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    static const int kIMG_CHANNELS = 3;
    static const int kIMG_H = 384;
    static const int kIMG_W = 384;
    std::vector<samplesCommon::PPM<kIMG_CHANNELS, kIMG_H, kIMG_W>> mPPMs; //!< PPMs of test images
    std::vector<float>mImgInfo;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    bool constructNetwork(SampleUniquePtr<nvonnxparser::IParser>& parser,
        SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    std::vector<ITensor*>  roiAlignBuild(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
    std::vector<ITensor*> rcnnBuild(SampleUniquePtr<nvinfer1::INetworkDefinition>& network, ITensor* roiAlignTensor);
    void decodeRCNNOut(SampleUniquePtr<nvinfer1::INetworkDefinition>& network,std::vector<ITensor*> rcnn_results, ITensor* rois);

    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections, handles post-processing of bounding boxes and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Performs inverse bounding box transform and clipping
    //!
    void bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes, const float* imInfo,
        const int N, const int nmsMaxOut, const int numCls);

    //!
    //! \brief Performs non maximum suppression on final bounding boxes
    //!
    std::vector<int> nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
        const int classNum, const int numClasses, const float nmsThreshold);
};

}

#endif