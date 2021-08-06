
## Build TensorRT engine
nvinfer1::createInferBuilder

 builder->createNetworkV2(explicitBatch)

  builder->createBuilderConfig()

  nvonnxparser::createParser

  constructNetwork(parser, builder, network, config);

  builder->buildEngineWithConfig(*network, *config)

  mEngine->serialize()

## construct network using TensorRT APIs
Like other AI frameworks, TensorRT provide APIs for user to construct network for inference.
### save weights from pytorch
![image](https://user-images.githubusercontent.com/84964763/128491389-e5f692f0-26a2-4365-9519-379cd364d5f8.png)

![image](https://user-images.githubusercontent.com/84964763/128491396-7a847730-be19-41e1-861c-390be36591fc.png)


### load weights from 2.1

![image](https://user-images.githubusercontent.com/84964763/128491288-64e3ab4a-dca1-4f24-88b3-cc1574edc77c.png)


### 2.3	define the network using APIs

1.Refer the original network and find the right APIs in TenorRT. 

![image](https://user-images.githubusercontent.com/84964763/128491710-3eb94291-8e47-4d36-888e-b6c3ea2c8657.png)


2.Do not forget non weights layers such as activation, pooling, slice, etc.

3.Make sure input and output tensor of your added layers correct. 

4.If you add the customized plugin, followed this:

![image](https://user-images.githubusercontent.com/84964763/128491946-b42fe4f2-f2b8-4f31-8d4a-28ebd40446dd.png)

5.unmark and mark the graph outputs.

![image](https://user-images.githubusercontent.com/84964763/128491977-9dfed77f-3d69-4d11-9684-ff3d0ccd4dbf.png)

##  define customized plugin

### 3.1 implement plugin class
A custom layer is implemented by extending the class IPluginCreator and one of TensorRT’s base classes for plugins. IPluginCreator is a creator class for custom layers using which users can get plugin name, version, and plugin field parameters.

You should inherent a class from following and override the virtual methods.
IPluginV2Ext
IPluginV2DynamicExt


### 3.2 build shared library using cuda

put all the plugin execution in .cu file and compile it to .so file(libFasterRCNNKernels.so). refer it when you use the customized 

Implementation of enqueue

![image](https://user-images.githubusercontent.com/84964763/128492364-62330fc7-9c71-4a88-9df6-12ad5be2735d.png)

In FasterRCNNKernels.cu,   roiAlign call roiAlign_kernel

![image](https://user-images.githubusercontent.com/84964763/128492415-be9ffd99-24c7-474c-890a-f388e3682e59.png)


## 4.verify inference result

### 4.1 image_preprocess

1. Resize and pad
resize_to_bgrmat_with_ratio(cv::Mat& input_image, int toH, int toW, bool keep_ratio=true)

2. normalize
image_normalize(cv::Mat& input_image, const float* img_mean, const float* img_std)

  3.permute: HWC->CHW

### 4.2 post_process

Intra class nms:
class_nms(std::vector<BBox>& bboxes, float iou_threshold = 0.5, float score_threshold = 0.05)

## 5. Int8 quantization

TensorRT built-in support fp16 quantization, you only need to set the IBuilderConfiger:config->setFlag(BuilderFlag::kFP16);
	
w.r.t int8 quantization you need to do follows:
	
       config->setFlag(BuilderFlag::kINT8);
	
       samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	
If you want to do calibration, replace the above by this:
	
	    assert(builder->platformHasFastInt8());
	
        config->setFlag(BuilderFlag::kINT8);
	
        Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(mParams.batchSize, inputSize, inputSize, "/home/ubuntu/data/coco/val2017/",
        "int8calib_fasterrcnn.table", input->getName());
	
        config->setInt8Calibrator(calibrator); 

Rewrite two functions in class Int8EntropyCalibrator2: getBatchSize() getBatch()
	
in getBatch() you should implement the image file loading and image preprocess, the image preprocess shall identical to which in model inference. 







