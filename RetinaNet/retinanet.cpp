#include "retinanet.h"
#include "../utils/pre_process.h"
#include "../utils/calibrator.hpp"

namespace LGT
{

bool RetinaNet::build()
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
    if(mParams.saveEngine)
    {

       std::ofstream p("RetinaNet101.plan", std::ios::binary);
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

    // network->destroy();
    return true;
}


bool RetinaNet::constructNetwork(SampleUniquePtr<nvonnxparser::IParser>& parser,
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
    mParams.inputTensorNames.push_back(input->getName());

    //若要设置成支持动态尺寸，需要在pytorch导模型时进行设置。

    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4(1,3,inputSize,inputSize));

    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4(8,3,inputSize,inputSize));

    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4(mParams.batchSize,3,inputSize,inputSize));
    config->addOptimizationProfile(profile);

    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.fp16)
    {
       config->setFlag(BuilderFlag::kFP16);
    }

    if (mParams.int8)
    {

       config->setFlag(BuilderFlag::kINT8);

        // MNISTBatchStream calibrationStream(mParams.calBatchSize, mParams.nbCalBatches, "train-images-idx3-ubyte",
        //     "train-labels-idx1-ubyte", mParams.dataDirs);
        // calibrator.reset(new Int8EntropyCalibrator2<MNISTBatchStream>(
        //     calibrationStream, 0, mParams.networkName.c_str(), mParams.inputTensorNames[0].c_str()));
        // config->setInt8Calibrator(calibrator.get());
    //    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);


        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(mParams.batchSize, inputSize, inputSize, "/home/ubuntu/data/coco/val2017/",
        "int8calib_retianet.table", input->getName());
        config->setInt8Calibrator(calibrator);

    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB);
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);




    
    std::vector<PluginField> batchedNMSPlugin_attr;
    batchedNMSPlugin_attr.emplace_back(PluginField("shareLocation", &shareLocation, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("backgroundLabelId", &backgroundLabelId, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("numClasses", &numClasses, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("topK", &topK, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("keepTopK", &keepTopK, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("scoreThreshold", &scoreThreshold, PluginFieldType::kFLOAT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("iouThreshold", &iouThreshold, PluginFieldType::kFLOAT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("isNormalized", &isNormalized, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("clipBoxes", &clipBoxes, PluginFieldType::kINT32, 1));
    batchedNMSPlugin_attr.emplace_back(PluginField("scoreBits", &scoreBits, PluginFieldType::kINT32, 1));
    
    std::unique_ptr< PluginFieldCollection > pluginFC(new PluginFieldCollection());
    // PluginFieldCollection *pluginFC = new PluginFieldCollection();
    pluginFC->nbFields = batchedNMSPlugin_attr.size();
    pluginFC->fields = batchedNMSPlugin_attr.data();

    // nvinfer1::REGISTER_TENSORRT_PLUGIN(BatchedNMSDynamicPluginCreator);

    auto creator = getPluginRegistry()->getPluginCreator("BatchedNMSDynamic_TRT", "1");
    IPluginV2 *pluginObj = creator->createPlugin("BatchedNMS",pluginFC.get()); //自己给该layer取的名字


    ITensor* bboxes = network->getOutput(0); //[batch,26761,1,4]
    ITensor* scores = network->getOutput(1);  //[batch,26761,80]

    ITensor* inputTensors[] = {bboxes, scores};

    IPluginV2Layer* NMS_layer = network->addPluginV2(inputTensors, 2, *pluginObj);

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


    // ITensor* bboxes = network->getOutput(0); //[batch,26761,4]
    // ITensor* scores = network->getOutput(1);  //[batch,26761,80]
    
    // const int num_dims = network->getOutput(1)->getDimensions().nbDims;
    // const uint32_t reduce_axes_cls = 1 << (num_dims - 1);
    // ITopKLayer* max_class_layer = network->addTopK(*network->getOutput(1), TopKOperation::kMAX, 1, reduce_axes_cls); 
    // ITensor* filtered_score = max_class_layer->getOutput(0); //[batch,27671,1]
 
    //  const uint32_t reduce_axes_index = (num_dims - 1);

    // ITopKLayer* topK_filtered_layer = network->addTopK(*filtered_score, TopKOperation::kMAX, mParams.nmsMaxOut, reduce_axes_index);//最后一个参数是bitmask

    // ITensor* filtered_index = topK_filtered_layer->getOutput(1);

    // auto dim7 = filtered_index->getDimensions();


    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool RetinaNet::infer()
{

    auto context_1 = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); //Create some space to store intermediate activation values

    if (!context_1)

    {

        return false;

    }

    context_1->setOptimizationProfile(0);

    context_1->setBindingDimensions(0, Dims4(mParams.batchSize, 3, 384, 384));

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
    // assert(mParams.inputTensorNames.size() == 1);
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
// bool RetinaNet::teardown()
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
bool RetinaNet::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int batchSize = mParams.batchSize;

    // float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("input"));
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
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
               hostDataBuffer[i * volImg + c * volChl + j] = data[j * inputC + 2 - c];//   相当一个二维数组转置 2-c是BGR->RGB
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
bool RetinaNet::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int batchSize = mParams.batchSize;
    // const int nmsMaxOut = mParams.nmsMaxOut;
    // const int outputClsSize = mParams.outputClsSize;
    // const int outputBBoxSize = mParams.outputClsSize * 4;

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
        cv::Mat srcImg = cv::imread(locateFile(imageList[i], mParams.dataDirs));

        
        float* bbox = nmsedBoxes + keepTopK * i * 4;
        float* score = nmsedScores + keepTopK * i;
        float* classIndex = nmsedClasses + keepTopK * i;

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
void RetinaNet::bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes,
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
std::vector<int> RetinaNet::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
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

}//namespace LGT