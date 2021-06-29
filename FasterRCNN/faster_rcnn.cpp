
#include "faster_rcnn.h"
#include "../utils/pre_process.h"
#include "../utils/trt_utils.h"


namespace LGT
{


bool FasterRCNN::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }
    constructNetwork(parser, builder, network, config);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    // assert(network->getNbOutputs() == 4);
    // auto mOutput0Dims = network->getOutput(0)->getDimensions();
    // assert(mOutput0Dims.nbDims == 4);
    // auto mOutput1Dims = network->getOutput(1)->getDimensions();
    // assert(mOutput1Dims.nbDims == 3);

    // serialize engine
    {

       std::ofstream p("Faster_RCNN.plan", std::ios::binary);
       if (!p)
       {
           return false;
       }

       nvinfer1::IHostMemory* ptr = mEngine->serialize();

       assert(ptr);

       p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());

       ptr->destroy();

       p.close();
    }

    return true;
}

std::vector<ITensor*>  FasterRCNN::roiAlignBuild(SampleUniquePtr<nvinfer1::INetworkDefinition>& network){
    
    ITensor* rois = network->getOutput(0); //[1,*,1,4]
    ITensor* roi_scores = network->getOutput(1); //[1,*,1]
    ITensor* fp0 = network->getOutput(2);  //[1,256,*,*]
    ITensor* fp1 = network->getOutput(3);
    ITensor* fp2 = network->getOutput(4);
    ITensor* fp3 = network->getOutput(5);
    
    // add rpn nms

    std::vector<PluginField> batchedRPNNMSPlugin_attr;

    batchedRPNNMSPlugin_attr.emplace_back(PluginField("shareLocation", &rpn_shareLocation, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("backgroundLabelId", &rpn_backgroundLabelId, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("numClasses", &rpn_numClasses, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("topK", &rpn_topK, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("keepTopK", &rpn_keepTopK, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("scoreThreshold", &rpn_scoreThreshold, PluginFieldType::kFLOAT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("iouThreshold", &rpn_iouThreshold, PluginFieldType::kFLOAT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("isNormalized", &rpn_isNormalized, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("clipBoxes", &rpn_clipBoxes, PluginFieldType::kINT32, 1));
    batchedRPNNMSPlugin_attr.emplace_back(PluginField("scoreBits", &rpn_scoreBits, PluginFieldType::kINT32, 1));
    
    PluginFieldCollection *pluginFC = new PluginFieldCollection();
    pluginFC->nbFields = batchedRPNNMSPlugin_attr.size();
    pluginFC->fields = batchedRPNNMSPlugin_attr.data();

    auto creator = getPluginRegistry()->getPluginCreator("BatchedNMSDynamic_TRT", "1");
    auto pluginObj = creator->createPlugin("BatchedPRNNMS",pluginFC); //自己给该layer取的名字
    ITensor* inputRPNNMSTensors[] = {rois, roi_scores};

    IPluginV2Layer* RPN_NMS_layer = network->addPluginV2(inputRPNNMSTensors, 2, *pluginObj);

    // ITensor *nmsed_rpns = RPN_NMS_layer->getOutput(1);

    // RPN_NMS_layer->getOutput(1)->setName("nmsed_rpns");
    // network->markOutput(*RPN_NMS_layer->getOutput(1));


    // add roiAlign

    REGISTER_TENSORRT_PLUGIN(DynamicPyramidROIAlignPluginCreator);
    creator = getPluginRegistry()->getPluginCreator("DynamicPyramidROIAlign_TRT", "1");
    std::vector<PluginField> roialign_attr;

    roialign_attr.emplace_back(PluginField("pooled_size", &poolSize, PluginFieldType::kINT32, 1));
    roialign_attr.emplace_back(PluginField("input_size", &inputSize, PluginFieldType::kINT32, 1));
    pluginFC = new PluginFieldCollection();
    pluginFC->nbFields = roialign_attr.size();
    pluginFC->fields = roialign_attr.data();

    pluginObj = creator->createPlugin("CustomizedPyramidROIAlign",pluginFC);

    ITensor* inputROIAlignTensors[] = {RPN_NMS_layer->getOutput(1), fp0, fp1, fp2, fp3};

    IPluginV2Layer* PyramidROIAlign_layer = network->addPluginV2(inputROIAlignTensors, 5, *pluginObj);
    auto dim_output_roiAlign = PyramidROIAlign_layer->getOutput(0)->getDimensions();


    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));

    // RPN_NMS_layer->getOutput(0)->setName("nmsed_nums_out");
    // network->markOutput(*RPN_NMS_layer->getOutput(0));

    // RPN_NMS_layer->getOutput(1)->setName("nmsed_rpns");
    // network->markOutput(*RPN_NMS_layer->getOutput(1));

    // PyramidROIAlign_layer->getOutput(0)->setName("roiAlign");
    // network->markOutput(*PyramidROIAlign_layer->getOutput(0));

    std::vector<ITensor*> instances;
    instances.push_back(RPN_NMS_layer->getOutput(1));
    instances.push_back(PyramidROIAlign_layer->getOutput(0));

    return instances;

}
std::vector<ITensor*> FasterRCNN:: rcnnBuild(SampleUniquePtr<nvinfer1::INetworkDefinition>& network, ITensor* roiAlignTensor)
 {
   // auto dim_output_rois = rois->getDimensions();
    // mParams.roiCount = dim_output_rois.d[1];
    auto rcnnWeightFile = locateFile(mParams.rcnnWeightsFile, mParams.dataDirs);
    std::map<std::string, Weights> weightMap = LGT::spanner::loadWeights(rcnnWeightFile);

    // auto* reshapeLayer_0 = network->addShuffle(*PyramidROIAlign_layer->getOutput(0));

    // reshapeLayer_0->setReshapeDimensions(

    //        Dims4{dim_output_roiAlign.d[0]*dim_output_roiAlign.d[1], dim_output_roiAlign.d[2], dim_output_roiAlign.d[3],  dim_output_roiAlign.d[4]});
    
    IFullyConnectedLayer* share_fc_0_layer = network->addFullyConnected(*roiAlignTensor,

                                                                       1024,

                                                                       weightMap["bbox_head.shared_fcs.0.weight"],

                                                                       weightMap["bbox_head.shared_fcs.0.bias"]);

    auto dim_output_fc_0 = share_fc_0_layer->getOutput(0)->getDimensions();

    IActivationLayer* relu_0_layer = network->addActivation(*share_fc_0_layer->getOutput(0), ActivationType::kRELU);

    IFullyConnectedLayer* share_fc_1_layer = network->addFullyConnected(*relu_0_layer->getOutput(0),

                                                                       1024,

                                                                       weightMap["bbox_head.shared_fcs.1.weight"],

                                                                       weightMap["bbox_head.shared_fcs.1.bias"]);

    IActivationLayer* relu_1_layer = network->addActivation(*share_fc_1_layer->getOutput(0), ActivationType::kRELU);

    IFullyConnectedLayer* cls_layer = network->addFullyConnected(*relu_1_layer->getOutput(0),

                                                                         bbox_numClasses+1,

                                                                       weightMap["bbox_head.fc_cls.weight"],

                                                                       weightMap["bbox_head.fc_cls.bias"]);

    //output_scores (-1,1000,81,1,1)
    auto dim_output_cls_layer = cls_layer->getOutput(0)->getDimensions();
    auto* reshapeLayer_scores = network->addShuffle(*cls_layer->getOutput(0));
    reshapeLayer_scores->setReshapeDimensions(
        Dims3{-1, mParams.roiCount, dim_output_cls_layer.d[2]});      //(-1, 1000, 81)    
    

    ISoftMaxLayer* softmax_layer = network->addSoftMax(*reshapeLayer_scores->getOutput(0));
    softmax_layer->setAxes(2<<1);
    auto dim_output_softmax_layer = softmax_layer->getOutput(0)->getDimensions();


    auto score_slice_layer_0 = network->addSlice(*softmax_layer->getOutput(0), Dims3{ 0, 0, 0},
        Dims3{ mParams.batchSize, mParams.roiCount, dim_output_softmax_layer.d[2]-1}, Dims3{ 1, 1, 1});


    //output_bboxes (-1,1000,320,1,1)
    IFullyConnectedLayer* reg_layer = network->addFullyConnected(*relu_1_layer->getOutput(0),

                                                                         (bbox_numClasses)*4,

                                                                         weightMap["bbox_head.fc_reg.weight"],

                                                                         weightMap["bbox_head.fc_reg.bias"]);

    auto dim_output_bbox = reg_layer->getOutput(0)->getDimensions();
    
    auto* reshapeLayer_delta = network->addShuffle(*reg_layer->getOutput(0));

    reshapeLayer_delta->setReshapeDimensions(
           Dims4{-1, mParams.roiCount, dim_output_bbox.d[2]/4, 4});
           
    auto dim_output_reshapeLayer_delta = reshapeLayer_delta->getOutput(0)->getDimensions();
    

    network->unmarkOutput(*network->getOutput(1));
    network->unmarkOutput(*network->getOutput(1));
    network->unmarkOutput(*network->getOutput(1));
    network->unmarkOutput(*network->getOutput(1));
    network->unmarkOutput(*network->getOutput(1));

    // reshapeLayer_delta->getOutput(0)->setName("delta_out");
    // network->markOutput(*reshapeLayer_delta->getOutput(0));

    // softmax_layer->getOutput(0)->setName("softmax_out");
    // network->markOutput(*softmax_layer->getOutput(0));

    // score_slice_layer_0->getOutput(0)->setName("slice_cls_out_0");
    // network->markOutput(*score_slice_layer_0->getOutput(0));



    std::vector<ITensor*> instances;
    instances.push_back(reshapeLayer_delta->getOutput(0));
    instances.push_back(score_slice_layer_0->getOutput(0)); //用这个时，只有两个结果
    // instances.push_back(softmax_layer->getOutput(0));
    
    return instances;

 }

void FasterRCNN:: decodeRCNNOut(SampleUniquePtr<nvinfer1::INetworkDefinition>& network,std::vector<ITensor*> rcnn_results, ITensor* nmsed_rpns)
{
    
   // add detla2bbox plugin
    REGISTER_TENSORRT_PLUGIN(DynamicDelta2BboxPluginCreator);
 
    auto delta2bbox_creator = getPluginRegistry()->getPluginCreator("DynamicDelta2Bbox_TRT", "1");
    std::vector<PluginField> delta2bbox_attr;
    delta2bbox_attr.emplace_back(PluginField("input_h", &inputSize, PluginFieldType::kINT32, 1));
    delta2bbox_attr.emplace_back(PluginField("input_w", &inputSize, PluginFieldType::kINT32, 1));


    auto delta2bbox_pluginFC = new PluginFieldCollection();
    
    delta2bbox_pluginFC->nbFields = delta2bbox_attr.size();
    delta2bbox_pluginFC->fields = delta2bbox_attr.data();

    auto delta2bbox_pluginObj = delta2bbox_creator->createPlugin("CustomizedDelta2Bbox",delta2bbox_pluginFC);

    ITensor* inputDelta2BboxTensors[] = {nmsed_rpns, rcnn_results[0]};

    IPluginV2Layer* Delta2Bbox_layer = network->addPluginV2(inputDelta2BboxTensors, 2, *delta2bbox_pluginObj);
    auto dim_output_delta2bbox = Delta2Bbox_layer->getOutput(0)->getDimensions(); 

    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));
    // network->unmarkOutput(*network->getOutput(1));
    // reshapeLayer_delta->getOutput(0)->setName("delta_out");
    // network->markOutput(*reshapeLayer_delta->getOutput(0));
    // Delta2Bbox_layer->getOutput(0)->setName("bbox_out");
    // network->markOutput(*Delta2Bbox_layer->getOutput(0));
    // score_slice_layer_0->getOutput(0)->setName("cls_out");
    // network->markOutput(*score_slice_layer_0->getOutput(0));


    // // add batchedNMS plugin       
    std::vector<PluginField> batchedNMSPlugin_attr;
    batchedNMSPlugin_attr.emplace_back(PluginField("shareLocation", &bbox_shareLocation, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("backgroundLabelId", &bbox_backgroundLabelId, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("numClasses", &bbox_numClasses, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("topK", &bbox_topK, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("keepTopK", &bbox_keepTopK, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("scoreThreshold", &bbox_scoreThreshold, PluginFieldType::kFLOAT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("iouThreshold", &bbox_iouThreshold, PluginFieldType::kFLOAT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("isNormalized", &bbox_isNormalized, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("clipBoxes", &bbox_clipBoxes, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("scoreBits", &bbox_scoreBits, PluginFieldType::kINT32, 1));
    
    // PluginFieldCollection *pluginFC = new PluginFieldCollection();
    auto pluginFC = new PluginFieldCollection();
    pluginFC->nbFields = batchedNMSPlugin_attr.size();
    pluginFC->fields = batchedNMSPlugin_attr.data();

    // nvinfer1::REGISTER_TENSORRT_PLUGIN(BatchedNMSDynamicPluginCreator);

    auto creator = getPluginRegistry()->getPluginCreator("BatchedNMSDynamic_TRT", "1");
    auto pluginObj = creator->createPlugin("BatchedNMS",pluginFC); //自己给该layer取的名字


    ITensor* inputNMSTensors[] = {Delta2Bbox_layer->getOutput(0), rcnn_results[1]};

    IPluginV2Layer* NMS_layer = network->addPluginV2(inputNMSTensors, 2, *pluginObj);


    // network->unmarkOutput(*network->getOutput(0));
    // network->unmarkOutput(*network->getOutput(0));
    // network->unmarkOutput(*network->getOutput(0));
    // network->unmarkOutput(*network->getOutput(0));
    // network->unmarkOutput(*network->getOutput(0));

    NMS_layer->getOutput(0)->setName("num_detections");
    network->markOutput(*NMS_layer->getOutput(0));

    NMS_layer->getOutput(1)->setName("nmsed_boxes");
    network->markOutput(*NMS_layer->getOutput(1));

    NMS_layer->getOutput(2)->setName("nmsed_scores");
    network->markOutput(*NMS_layer->getOutput(2));

    NMS_layer->getOutput(3)->setName("nmsed_classes");
    network->markOutput(*NMS_layer->getOutput(3));    

}

bool FasterRCNN::constructNetwork(SampleUniquePtr<nvonnxparser::IParser>& parser,
    SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    IOptimizationProfile* profile = builder->createOptimizationProfile();

    // Set formats and data types of inputs

    auto input = network->getInput(0);

    //若要设置成支持动态尺寸，需要在pytorch导模型时进行设置。

    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4(1, 3, inputSize, inputSize));

    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4(1, 3, inputSize, inputSize));

    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4(1, 3, inputSize, inputSize));
    config->addOptimizationProfile(profile);

   if (mParams.fp16)

   {

       config->setFlag(BuilderFlag::kFP16);

   }

   if (mParams.int8)

   {

       config->setFlag(BuilderFlag::kINT8);
    //    config->setInt8Calibrator(calibrator.get());

    //    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);

   }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1ULL << 32);
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    // ITensor* rois = network->getOutput(0); //[1,*,1,4]
    // ITensor* roi_scores = network->getOutput(1); //[1,*,1]
    // ITensor* fp0 = network->getOutput(2);  //[1,256,*,*]
    // ITensor* fp1 = network->getOutput(3);
    // ITensor* fp2 = network->getOutput(4);
    // ITensor* fp3 = network->getOutput(5);


    auto roiAlign_results = roiAlignBuild(network);

    auto rcnn_results = rcnnBuild(network, roiAlign_results[1]);

    decodeRCNNOut(network,rcnn_results, roiAlign_results[0]);


    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool FasterRCNN::infer()
{

    auto context_1 = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); //Create some space to store intermediate activation values

    if (!context_1)

    {

        return false;

    }

    context_1->setOptimizationProfile(0);

    context_1->setBindingDimensions(0, Dims4(mParams.batchSize, mInputDims.d[1], mInputDims.d[2], mInputDims.d[3]));

    if (!context_1->allInputDimensionsSpecified())

    {

        return false;

    }



    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize,context_1.get());

    // Create RAII buffer manager object
    // samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    // auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    // if (!context)
    // {
    //     return false;
    // }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    // bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    // if (!status)
    // {
    //     return false;
    // }
 
   bool status = context_1->executeV2(buffers.getDeviceBindings().data());

   if (!status)

   {

       return false;

   }   

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Cleans up any state created in the sample class
//!
// bool FasterRCNN::teardown()
// {
//     //! Clean up the libprotobuf files as the parsing is complete
//     //! \note It is not safe to use any other part of the protocol buffers library after
//     //! ShutdownProtobufLibrary() has been called.
//     nvcaffeparser1::shutdownProtobufLibrary();
//     return true;
// }

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool FasterRCNN::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int batchSize = mParams.batchSize;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("input"));
   for(size_t i =0, volImg = inputC * inputH * inputW; i<mParams.batchSize; i++)
   {

        auto image_path = locateFile(imageList[i], mParams.dataDirs);
        cv::Mat image = cv::imread(image_path);
        uchar* data_origin = (uchar*)image.data;
        
        cv::Size sz = image.size();
        float ratio = std::min(( float )inputW / ( float )sz.width, ( float )inputH / ( float )sz.height);

        mImgInfo.emplace_back(sz.height);     // Number of rows
        mImgInfo.emplace_back(sz.width); // Number of columns
        mImgInfo.emplace_back(ratio);                 // Image scale
        auto matOut = LGT::spanner::PreProcess::resize_to_bgrmat_with_ratio(image, inputW, inputH); //HWC BGR
        LGT::spanner::PreProcess::image_normalize(matOut, img_mean, img_std);
        float* data = (float*)matOut.data;
        for (int c = 0; c < inputC; ++c)//
        {
           
           // The color image to input should be in BGR order
           for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)  //将HW看成一维
               hostDataBuffer[i * volImg + c * volChl + j] = data[j * inputC + 2 - c];//   相当一个二维数组转置, 2-c是BGR->RGB
        }

   }

/***
    // Available images
    const std::vector<std::string> imageList = {"000456.ppm", "000542.ppm", "001150.ppm", "001763.ppm", "004545.ppm"};
    mPPMs.resize(batchSize);
    assert(mPPMs.size() <= imageList.size());

    // Fill im_info buffer
    // float* hostImInfoBuffer = static_cast<float*>(buffers.getHostBuffer("im_info"));
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.dataDirs), mPPMs[i]);
        mImgInfo.emplace_back(mPPMs[i].h);     // Number of rows
        mImgInfo.emplace_back(mPPMs[i].w); // Number of columns
        mImgInfo.emplace_back(1);                 // Image scale
    }

    // Fill data buffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("input"));
    // Pixel mean used by the Faster R-CNN's author
    const float pixelMean[3]{102.9801f, 115.9465f, 122.7717f}; // Also in BGR order
    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
                // hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + 2 - c]) - pixelMean[c];// The color image to input should be in BGR order
                hostDataBuffer[i * volImg + c * volChl + j] = (float(mPPMs[i].buffer[j * inputC + c]) - pixelMean[c])/255.0;
        }
    }
***/
    return true;
}

//!
//! \brief Filters output detections and handles post-processing of bounding boxes, verify result
//!
//! \return whether the detection output matches expectations
//!
bool FasterRCNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int batchSize = mParams.batchSize;

    // const int* nmsed_nums_out = static_cast<const int*>(buffers.getHostBuffer("nmsed_nums_out"));
    // float* nmsed_rpns = static_cast<float*>(buffers.getHostBuffer("nmsed_rpns"));

    // float *softmax_out = static_cast<float*>(buffers.getHostBuffer("softmax_out")); 
    // float sum =0;
    // for(int i =0; i<81; i++)
    //     sum += softmax_out[i];
    // float sum1 =0;
    // for(int j = 81; j<162; j++)
    //     sum1 += softmax_out[j];   


    // float* cls_out = static_cast<float*>(buffers.getHostBuffer("cls_out"));        
    // std::vector<float> cls_vec(3600*81, 0);
    // memcpy(&cls_vec[0], cls_out, 3600*81*sizeof(float));
    // // float max_cls = *max_element(cls_vec.begin(), cls_vec.end());
    // // float min_cls = *min_element(cls_vec.begin(), cls_vec.end());
    // for(int i =0; i< 3; i++)
    // {
    //     for(int j = 0; j<81; j++)
    //     {
    //         std::cout<<cls_vec[i*3600+j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    // float* slice_cls_out_0 = static_cast<float*>(buffers.getHostBuffer("slice_cls_out_0")); 
    // std::vector<float> slice_cls_vec(3600*80, 0);
    // memcpy(&slice_cls_vec[0], slice_cls_out_0, 3600*80*sizeof(float));
    // // float max_cls = *max_element(cls_vec.begin(), cls_vec.end());
    // // float min_cls = *min_element(cls_vec.begin(), cls_vec.end());
    // for(int i =0; i< 3; i++)
    // {
    //     for(int j = 0; j<80; j++)
    //     {
    //         std::cout<<slice_cls_vec[i*3600+j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }


    // float* delta_out = static_cast<float*>(buffers.getHostBuffer("delta_out"));
    
    
    // float* bbox_out = static_cast<float*>(buffers.getHostBuffer("bbox_out"));
//     std::vector<std::vector<std::vector<float>>> bbox_vec(3600, std::vector<std::vector<float>>(80, std::vector<float>(4, 0)));
//     memcpy(&bbox_vec[0][0][0], bbox_out, 3600*80*4*sizeof(float));
//    // float max_box = *max_element(bbox_vec.begin(), bbox_vec.end());
//     //float min_box = *min_element(bbox_vec.begin(), bbox_vec.end());
//     for(int i =0; i< 3; i++){
//         for(int j = 0; j<80; j++){
//             for(int k =0; k< 4; k++)
//                 std::cout<<bbox_vec[i][j][k]<<" ";
//             std::cout<<std::endl;
//         }
//     }

/***
    float* rois = static_cast<float*>(buffers.getHostBuffer("rois"));
    for(int i =0; i< mParams.roiCount; i++){//取前1000个
        for(int k =0; k< 4; k++)
            std::cout<<rois[i*4 + k]<<" ";
        std::cout<<std::endl;

    }   
***/

    // float* roiAlign = static_cast<float*>(buffers.getHostBuffer("roiAlign"));
    
    // const float* imInfo = static_cast<const float*>(buffers.getHostBuffer("input"));
    const int* numDetections = static_cast<const int*>(buffers.getHostBuffer("num_detections"));
    float* nmsedBoxes = static_cast<float*>(buffers.getHostBuffer("nmsed_boxes"));
    float* nmsedScores = static_cast<float*>(buffers.getHostBuffer("nmsed_scores"));
    float* nmsedClasses = static_cast<float*>(buffers.getHostBuffer("nmsed_classes"));    
    
    // const float* boxes = static_cast<const float*>(buffers.getHostBuffer("boxes"));
    // const float* clsProbs = static_cast<const float*>(buffers.getHostBuffer("scores"));




    // float* rois = static_cast<float*>(buffers.getHostBuffer("rois"));


    // Unscale back to raw image space
    // for (int i = 0; i < batchSize; ++i)
    // {
    //     auto numBoxes = numDetections[i];
    //     for (int j = 0; j < numBoxes * 4 && mImgInfo.data()[i * 3 + 2] != 1;  ++j)
    //     {
    //         nmsedBoxes[j] /= mImgInfo[i * 3 + 2];
    //     }
    // }

    // std::vector<float> predBBoxes(batchSize * nmsMaxOut * outputBBoxSize, 0);
    // bboxTransformInvAndClip(rois, deltas, predBBoxes.data(), mImgInfo, batchSize, nmsMaxOut, outputClsSize);

    // const float nmsThreshold = 0.3f;
    // const float score_threshold = 0.8f;
    // const std::vector<std::string> classes{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    //     "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    //     "train", "tvmonitor"};





    for (int i = 0; i < batchSize; ++i)
    {
        int numBoxes = numDetections[i];
        cout<<"detect "<<numBoxes<<" objects!"<<std::endl;
        cv::Mat srcImg = cv::imread(locateFile(imageList[i], mParams.dataDirs));

        
        float* bbox = nmsedBoxes + bbox_keepTopK * i * 4;
        float* score = nmsedScores + bbox_keepTopK * i;
        float* classIndex = nmsedClasses + bbox_keepTopK * i;

        float scal_factor = mImgInfo.data()[i * 3 + 2];
        float origin_height = mImgInfo.data()[i * 3 + 0];
        float origin_width = mImgInfo.data()[i * 3 + 1];

        for (int j = 0; j < numBoxes; ++j) // Show results
        {   
            auto x1 = *bbox / scal_factor;
            auto y1 = *(bbox+1) / scal_factor;
            auto x2 = *(bbox+2) / scal_factor;
            auto y2 = *(bbox+3) / scal_factor;

            //clamp
            x1 = std::max(0.0f, x1);
            y1 = std::max(0.0f, y1);
            x2 = std::min(origin_width, x2);
            y2 = std::min(origin_height, y2);

            const std::string storeName = classes[*classIndex] + "-" + std::to_string(*score) + ".jpg";
            sample::gLogInfo << "Detected " << classes[*classIndex] << " in " << imageList[i] << " with confidence "
                                << *score * 100.0f << "% "
                                << " x1: "<<x1<<" y1: "<<y1<<" x2: "<<x2<<" y2: "<<y2<< std::endl;

            // const samplesCommon::BBox b{*bbox, *bbox+1, *bbox+2, *bbox+3};
            // writePPMFileWithBBox(storeName, mPPMs[i], b);

            cv::rectangle(srcImg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1, cv::LINE_8, 0);

            ostringstream oss;
            oss<<std::setprecision(2)<<*score;
            cv::putText(srcImg,classes[*classIndex]+":"+oss.str(),cv::Point(x1,y1),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,0,0),1,8);
            

            bbox +=  4;
            score ++;
            classIndex ++;
            
        }
        cv::imwrite(imageList[i], srcImg);

    }
    return true;
}

//!
//! \brief Performs inverse bounding box transform
//!
void FasterRCNN::bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes,
    const float* imInfo, const int N, const int nmsMaxOut, const int numCls)
{
    for (int i = 0; i < N * nmsMaxOut; ++i)
    {
        float width = rois[i * 4 + 2] - rois[i * 4] + 1;
        float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
        float ctr_x = rois[i * 4] + 0.5f * width;
        float ctr_y = rois[i * 4 + 1] + 0.5f * height;
        const float* imInfo_offset = imInfo + i / nmsMaxOut * 3;
        for (int j = 0; j < numCls; ++j)
        {
            float dx = deltas[i * numCls * 4 + j * 4];
            float dy = deltas[i * numCls * 4 + j * 4 + 1];
            float dw = deltas[i * numCls * 4 + j * 4 + 2];
            float dh = deltas[i * numCls * 4 + j * 4 + 3];
            float pred_ctr_x = dx * width + ctr_x;
            float pred_ctr_y = dy * height + ctr_y;
            float pred_w = exp(dw) * width;
            float pred_h = exp(dh) * height;
            predBBoxes[i * numCls * 4 + j * 4]
                = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 1]
                = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 2]
                = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 3]
                = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
        }
    }
}

//!
//! \brief Performs non maximum suppression on final bounding boxes
//!
std::vector<int> FasterRCNN::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
    const int classNum, const int numClasses, const float nmsThreshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : scoreIndex)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(
                    &bbox[(idx * numClasses + classNum) * 4], &bbox[(kept_idx * numClasses + classNum) * 4]);
                keep = overlap <= nmsThreshold;
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

}// namespace LGT