

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

// #include "NvCaffeParser.h"
#include "retinanet.h"

using namespace LGT;

RetinaNetParams initializeSampleParams(const samplesCommon::Args& args)
{
    RetinaNetParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        // params.dataDirs.push_back("/home/ubuntu/source_code/TensorRT-7.2.2.3/data/faster-rcnn/");
        params.dataDirs.push_back("/home/ubuntu/source_code/detection/mmdetection/onnx_file/");
        params.dataDirs.push_back("/home/ubuntu/data/coco/val2017/");
        // params.dataDirs.push_back("")
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "retinanet_new.onnx";
    params.inputTensorNames.push_back("input");
    // params.inputTensorNames.push_back("im_info");
    params.batchSize = 1;
    params.outputTensorNames.push_back("boxes");
    params.outputTensorNames.push_back("scores");
    // params.outputTensorNames.push_back("rois");
    params.dlaCore = args.useDLACore;

    params.outputClsSize = 80;
    params.nmsMaxOut
        = 300; // This value needs to be changed as per the nmsMaxOut value set in RPROI plugin parameters in prototxt

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_fasterRCNN [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/faster-rcnn/ and data/faster-rcnn/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    initLibNvInferPlugins(&sample::gLogger, "");

    

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    RetinaNet sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for RetinaNet" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    // if (!sample.teardown())
    // {
    //     return sample::gLogger.reportFail(sampleTest);
    // }

    return sample::gLogger.reportPass(sampleTest);
}
