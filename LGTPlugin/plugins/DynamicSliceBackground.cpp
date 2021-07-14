

#include "DynamicSliceBackground.h"
#include "../plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::DynamicSliceBackground;
using nvinfer1::plugin::DynamicSliceBackgroundPluginCreator;

namespace
{
const char* DYNAMICSLICEBACKGROUND_PLUGIN_VERSION{"1"};
const char* DYNAMICSLICEBACKGROUND_PLUGIN_NAME{"DynamicSliceBackground_TRT"};
} // namespace
REGISTER_TENSORRT_PLUGIN(DynamicSliceBackgroundPluginCreator);

PluginFieldCollection DynamicSliceBackgroundPluginCreator::mFC{};
std::vector<PluginField> DynamicSliceBackgroundPluginCreator::mPluginAttributes;

DynamicSliceBackgroundPluginCreator::DynamicSliceBackgroundPluginCreator()
{
    // mPluginAttributes.emplace_back(PluginField("input_h", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("input_w", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DynamicSliceBackgroundPluginCreator::getPluginName() const
{
    return DYNAMICSLICEBACKGROUND_PLUGIN_NAME;
};

const char* DynamicSliceBackgroundPluginCreator::getPluginVersion() const
{
    return DYNAMICSLICEBACKGROUND_PLUGIN_VERSION;
};

const PluginFieldCollection* DynamicSliceBackgroundPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* DynamicSliceBackgroundPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // const PluginField* fields = fc->fields;
    // for (int i = 0; i < fc->nbFields; ++i)
    // {
    //     const char* attrName = fields[i].name;
    //     if (!strcmp(attrName, "input_h"))
    //     {
    //         assert(fields[i].type == PluginFieldType::kINT32);
    //         mInputH = *(static_cast<const quad_t*>(fields[i].data));
    //     }
    //     if (!strcmp(attrName, "input_w"))
    //     {
    //         assert(fields[i].type == PluginFieldType::kINT32);
    //         mInputW = *(static_cast<const quad_t*>(fields[i].data));
    //     }
    // }
    return new DynamicSliceBackground();
};

IPluginV2DynamicExt* DynamicSliceBackgroundPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    return new DynamicSliceBackground(data, length);
};


//implementation of DynamicSliceBackground

DynamicSliceBackground::DynamicSliceBackground()
{


};

int DynamicSliceBackground::getNbOutputs() const
{
    return 1;
};

int DynamicSliceBackground::initialize()
{
    return 0;
};

void DynamicSliceBackground::terminate(){

};

void DynamicSliceBackground::destroy()
{
    delete this;
};

// size_t DynamicSliceBackground::getWorkspaceSize(int) const
// {
//     return 0;
// }
size_t DynamicSliceBackground::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept
{
    return 0;
}

// bool DynamicSliceBackground::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// };
bool DynamicSliceBackground:: supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(0 <= pos && pos < 2);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = in[0].type == in[pos].type;
    switch (pos)
    {
    case 0:
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT)
            && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (out[0].type == DataType::kHALF || out[0].type == DataType::kFLOAT)
            && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;

            
    }
    return false;
}
;
const char* DynamicSliceBackground::getPluginType() const
{
    return "DynamicSliceBackground_TRT";
};

const char* DynamicSliceBackground::getPluginVersion() const
{
    return "1";
};

IPluginV2DynamicExt* DynamicSliceBackground::clone() const noexcept
{
    auto plugin = new DynamicSliceBackground(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
};

void DynamicSliceBackground::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* DynamicSliceBackground::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

void DynamicSliceBackground::check_valid_inputs(const nvinfer1::DimsExprs* inputs, int nbInputDims)
{

    assert(nbInputDims == 1);

    nvinfer1::DimsExprs scores = inputs[0];
    assert(scores.nbDims == 3);
    // assert(deltas.d[1]==rois.d[1]);

}

 DimsExprs DynamicSliceBackground::getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    check_valid_inputs(inputs, nbInputs);
    // assert(index == 0);

    nvinfer1::DimsExprs result;
    result.nbDims = 3;

    //batch size 不是constant，所以不能使用exprBuilder.constant(inputs[1].d[1]->getConstantValue());,如果是constan则在build阶段就能看到其维度，否则显示-1
    result.d[0] = inputs[0].d[0];

    // mROICount
    result.d[1] = inputs[0].d[1];

    result.d[2] = exprBuilder.constant(inputs[0].d[2]->getConstantValue() -1 );

    return result;
}



int DynamicSliceBackground::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    
    int batchSize = inputDesc[0].dims.d[0];
    int roiCount = inputDesc[0].dims.d[1];
    int numCls = inputDesc[0].dims.d[2] -1;
    
    cudaError_t status = sliceBackground(stream, batchSize, roiCount, numCls, inputs[0], outputs[0]);

    assert(status == cudaSuccess);
    return 0;
};


size_t DynamicSliceBackground::getSerializationSize() const
{
    return 0;
};

void DynamicSliceBackground::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    // write(d, mInputH);
    // write(d, mInputW);
    // write(d, mTargetMean[2]);
    // write(d, mTargetMean[3]);
    // write(d, mTargetStd[0]);
    // write(d, mTargetStd[1]);
    // write(d, mTargetStd[2]);
    // write(d, mTargetStd[3]);    

    assert(d == a + getSerializationSize());
};

DynamicSliceBackground::DynamicSliceBackground(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;

    // mInputH = read<int>(d);
    // mInputW = read<int>(d);
    // mTargetMean[2] = read<float>d;
    // mTargetMean[3] = read<float>d;
    // mTargetStd[0] = read<float>d;
    // mTargetStd[1] = read<float>d;
    // mTargetStd[2] = read<float>d;
    // mTargetStd[3] = read<float>d;

    assert(d == a + length);
}

// Return the DataType of the plugin output at the requested index
DataType DynamicSliceBackground::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
// bool DynamicSliceBackground::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
// {
//     return false;
// }

// // Return true if plugin can use input that is broadcast across batch without replication.
// bool DynamicSliceBackground::canBroadcastInputAcrossBatch(int inputIndex) const
// {
//     return false;
// }

// Configure the layer with input and output data types.
// void DynamicSliceBackground::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
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

void DynamicSliceBackground:: configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept
{

    // assert((inputTypes[0], floatFormat));
    // check_valid_inputs(inputDims, nbInputs);

    assert(nbOutputs == 1);
    assert(nbInputs == 1);

    // mROICount = in[0].desc.dims.d[1];
    // mFeatureLength = in[1].desc.dims.d[1];


}    

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// void DynamicSliceBackground::attachToContext(
//     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
// {
// }

// // Detach the plugin object from its execution context.
// void DynamicSliceBackground::detachFromContext() {}

