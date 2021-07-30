

#include "DynamicPyramidROIAlignPlugin.h"
#include "../plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::DynamicPyramidROIAlign;
using nvinfer1::plugin::DynamicPyramidROIAlignPluginCreator;

namespace
{
const char* DYNAMICPYRAMIDROIALGIN_PLUGIN_VERSION{"1"};
const char* DYNAMICPYRAMIDROIALGIN_PLUGIN_NAME{"DynamicPyramidROIAlign_TRT"};
} // namespace
REGISTER_TENSORRT_PLUGIN(DynamicPyramidROIAlignPluginCreator);

PluginFieldCollection DynamicPyramidROIAlignPluginCreator::mFC{};
std::vector<PluginField> DynamicPyramidROIAlignPluginCreator::mPluginAttributes;

DynamicPyramidROIAlignPluginCreator::DynamicPyramidROIAlignPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("input_size", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DynamicPyramidROIAlignPluginCreator::getPluginName() const
{
    return DYNAMICPYRAMIDROIALGIN_PLUGIN_NAME;
};

const char* DynamicPyramidROIAlignPluginCreator::getPluginVersion() const
{
    return DYNAMICPYRAMIDROIALGIN_PLUGIN_VERSION;
};

const PluginFieldCollection* DynamicPyramidROIAlignPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* DynamicPyramidROIAlignPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "pooled_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mPooledSize = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "input_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mInputSize = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new DynamicPyramidROIAlign(mPooledSize, mInputSize);
};

IPluginV2DynamicExt* DynamicPyramidROIAlignPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
   

    auto* plugin = new DynamicPyramidROIAlign(data, length);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
};


//implementation of DynamicPyramidROIAlign

DynamicPyramidROIAlign::DynamicPyramidROIAlign(int pooled_size, int input_size)
    : mPooledSize({pooled_size, pooled_size}), mInputSize({input_size,input_size})
{

    assert(pooled_size > 0);
    // shape
    // mInputSize = MaskRCNNConfig::IMAGE_SHAPE.d[1];
    // mThresh = (224 * 224 * 2.0f / (mInputSize.x* mInputSize.y)) / (4.0 * 4.0f);
    // mThresh = (mInputSize.x/8.0)*(mInputSize.y/8.0); //相当于1/64，一条边的话就是1/8。 输入为384的话就是48*48
    mThresh = 112*112;//mmdetection中的设置，是个绝对量。
};

int DynamicPyramidROIAlign::getNbOutputs() const
{
    return 1;
};

int DynamicPyramidROIAlign::initialize()
{
    return 0;
};

void DynamicPyramidROIAlign::terminate(){

};

void DynamicPyramidROIAlign::destroy()
{
    delete this;
};

// size_t DynamicPyramidROIAlign::getWorkspaceSize(int) const
// {
//     return 0;
// }
size_t DynamicPyramidROIAlign::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept
{
    return 0;
}

// bool DynamicPyramidROIAlign::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// };
bool DynamicPyramidROIAlign:: supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(0 <= pos && pos < 6);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = in[0].type == in[pos].type;
    switch (pos)
    {
    case 0:
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT)
            && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;

    case 2:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
            
    case 3:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 4:
           return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
            
    case 5:
        return (out[0].type == DataType::kHALF || out[0].type == DataType::kFLOAT)
            && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}
;
const char* DynamicPyramidROIAlign::getPluginType() const
{
    return DYNAMICPYRAMIDROIALGIN_PLUGIN_NAME;
};

const char* DynamicPyramidROIAlign::getPluginVersion() const
{
    return DYNAMICPYRAMIDROIALGIN_PLUGIN_VERSION;
};

IPluginV2DynamicExt* DynamicPyramidROIAlign::clone() const noexcept
{
    auto plugin = new DynamicPyramidROIAlign(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
};

void DynamicPyramidROIAlign::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* DynamicPyramidROIAlign::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

void DynamicPyramidROIAlign::check_valid_inputs(const nvinfer1::DimsExprs* inputs, int nbInputDims)
{
    // roi: [N, anchors, 4],
    // feature_map list(4 maps): p2, p3, p4, p5
    assert(nbInputDims == 1 + mFeatureMapCount);

    nvinfer1::DimsExprs rois = inputs[0];
    assert(rois.nbDims == 3);
    // assert(rois.d[2] == exprBuilder.constant(4));

    for (int i = 1; i < nbInputDims; ++i)
    {
        nvinfer1::DimsExprs de = inputs[i];

        // CHW with the same #C
        assert(de.nbDims == 4 && de.d[1] == inputs[1].d[1]);
    }
}

// Dims DynamicPyramidROIAlign::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
// {

//     check_valid_inputs(inputs, nbInputDims);
//     assert(index == 0);

//     nvinfer1::Dims result;
//     result.nbDims = 4;

//     // mROICount
//     result.d[0] = inputs[0].d[0];
//     // mFeatureLength
//     result.d[1] = inputs[1].d[0];
//     // height
//     result.d[2] = mPooledSize.y;
//     // width
//     result.d[3] = mPooledSize.x;

//     return result;
// };
 DimsExprs DynamicPyramidROIAlign::getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    check_valid_inputs(inputs, nbInputs);
    // assert(index == 0);

    nvinfer1::DimsExprs result;
    result.nbDims = 5;

    //batch size 不是constant，所以不能使用xprBuilder.constant(inputs[1].d[1]->getConstantValue());,如果是constan则在build阶段就能看到其维度，否则显示-1
    result.d[0] = inputs[0].d[0];

    // mROICount
    result.d[1] = inputs[0].d[1];
    // result.d[1] = exprBuilder.constant(mROICount);
    // mFeatureLength, feature的通道数
    result.d[2] = exprBuilder.constant(inputs[1].d[1]->getConstantValue());
    // height
    result.d[3] = exprBuilder.constant(mPooledSize.y);
    // width
    result.d[4] = exprBuilder.constant(mPooledSize.x);

    return result;
}

// int DynamicPyramidROIAlign::enqueue(
//     int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
// {

//     void* pooled = outputs[0];

//     cudaError_t status = roiAlign(stream, batch_size, mFeatureLength, mROICount, mThresh,

//         inputs[0], &inputs[1], mFeatureSpatialSize,

//         pooled, mPooledSize);

//     assert(status == cudaSuccess);
//     return 0;
// };

int DynamicPyramidROIAlign::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    void* pooled = outputs[0];
    int batchSize = inputDesc[0].dims.d[0];

    cudaError_t status = roiAlign(stream, batchSize, mFeatureLength, mROICount, mThresh,

        inputs[0], &inputs[1], mFeatureSpatialSize,

        pooled, mPooledSize, mInputSize);
    // cudaError_t status = roiAlign(stream, batchSize, mFeatureLength, mROICount, mThresh,

    //     inputs[0], &inputs[1], mFeatureSpatialSize,

    //     pooled, mPooledSize);
        
    assert(status == cudaSuccess);
    return 0;
};


size_t DynamicPyramidROIAlign::getSerializationSize() const
{
    return sizeof(int) * 2 + sizeof(int) * 4 + sizeof(float) + sizeof(int) * 2 * 4;
};

void DynamicPyramidROIAlign::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPooledSize.y);
    write(d, mPooledSize.x);
    write(d, mFeatureLength);
    write(d, mROICount);
    write(d, mInputSize.y);
    write(d, mInputSize.x);
    write(d, mThresh);
    write(d, mFeatureSpatialSize[0].y);
    write(d, mFeatureSpatialSize[0].x);
    write(d, mFeatureSpatialSize[1].y);
    write(d, mFeatureSpatialSize[1].x);
    write(d, mFeatureSpatialSize[2].y);
    write(d, mFeatureSpatialSize[2].x);
    write(d, mFeatureSpatialSize[3].y);
    write(d, mFeatureSpatialSize[3].x);
    assert(d == a + getSerializationSize());
};

DynamicPyramidROIAlign::DynamicPyramidROIAlign(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mPooledSize = {read<int>(d), read<int>(d)};
    mFeatureLength = read<int>(d);
    mROICount = read<int>(d);
    mInputSize = {read<int>(d), read<int>(d)};
    mThresh = read<float>(d);
    mFeatureSpatialSize[0].y = read<int>(d);
    mFeatureSpatialSize[0].x = read<int>(d);
    mFeatureSpatialSize[1].y = read<int>(d);
    mFeatureSpatialSize[1].x = read<int>(d);
    mFeatureSpatialSize[2].y = read<int>(d);
    mFeatureSpatialSize[2].x = read<int>(d);
    mFeatureSpatialSize[3].y = read<int>(d);
    mFeatureSpatialSize[3].x = read<int>(d);

    assert(d == a + length);
}

// Return the DataType of the plugin output at the requested index
DataType DynamicPyramidROIAlign::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
// bool DynamicPyramidROIAlign::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
// {
//     return false;
// }

// // Return true if plugin can use input that is broadcast across batch without replication.
// bool DynamicPyramidROIAlign::canBroadcastInputAcrossBatch(int inputIndex) const
// {
//     return false;
// }

// Configure the layer with input and output data types.
// void DynamicPyramidROIAlign::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
//     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
//     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
// {
//     assert(supportsFormat(inputTypes[0], floatFormat));
//     check_valid_inputs(inputDims, nbInputs);

//     assert(nbOutputs == 1);
//     assert(nbInputs == 1 + mFeatureMapCount);

//     mROICount = inputDims[0].d[0];
//     mFeatureLength = inputDims[1].d[0];

//     for (size_t layer = 0; layer < mFeatureMapCount; ++layer)
//     {
//         mFeatureSpatialSize[layer] = {inputDims[layer + 1].d[1], inputDims[layer + 1].d[2]};
//     }
// }

void DynamicPyramidROIAlign:: configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept
{

    // assert((inputTypes[0], floatFormat));
    // check_valid_inputs(inputDims, nbInputs);

    assert(nbOutputs == 1);
    assert(nbInputs == 1 + mFeatureMapCount);

    mROICount = in[0].desc.dims.d[1];
    mFeatureLength = in[1].desc.dims.d[1];

    for (size_t layer = 0; layer < mFeatureMapCount; ++layer)
    {
        mFeatureSpatialSize[layer] = {in[layer + 1].desc.dims.d[2], in[layer + 1].desc.dims.d[3]};
    }
}    

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// void DynamicPyramidROIAlign::attachToContext(
//     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
// {
// }

// // Detach the plugin object from its execution context.
// void DynamicPyramidROIAlign::detachFromContext() {}

