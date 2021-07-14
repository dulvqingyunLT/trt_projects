

#include "DynamicDelta2BboxPlugin.h"
#include "../plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::DynamicDelta2Bbox;
using nvinfer1::plugin::DynamicDelta2BboxPluginCreator;

namespace
{
const char* DYNAMICPDELTA2BBOX_PLUGIN_VERSION{"1"};
const char* DYNAMICPDELTA2BBOX_PLUGIN_NAME{"DynamicDelta2Bbox_TRT"};
} // namespace
// REGISTER_TENSORRT_PLUGIN(DynamicDelta2BboxPluginCreator);

PluginFieldCollection DynamicDelta2BboxPluginCreator::mFC{};
std::vector<PluginField> DynamicDelta2BboxPluginCreator::mPluginAttributes;

DynamicDelta2BboxPluginCreator::DynamicDelta2BboxPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("input_h", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("input_w", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DynamicDelta2BboxPluginCreator::getPluginName() const
{
    return DYNAMICPDELTA2BBOX_PLUGIN_NAME;
};

const char* DynamicDelta2BboxPluginCreator::getPluginVersion() const
{
    return DYNAMICPDELTA2BBOX_PLUGIN_VERSION;
};

const PluginFieldCollection* DynamicDelta2BboxPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2DynamicExt* DynamicDelta2BboxPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "input_h"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mInputH = *(static_cast<const quad_t*>(fields[i].data));
        }
        if (!strcmp(attrName, "input_w"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mInputW = *(static_cast<const quad_t*>(fields[i].data));
        }
    }
    return new DynamicDelta2Bbox(mInputH, mInputW);
};

IPluginV2DynamicExt* DynamicDelta2BboxPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    return new DynamicDelta2Bbox(data, length);
};


//implementation of DynamicDelta2Bbox

DynamicDelta2Bbox::DynamicDelta2Bbox(int h, int w):mInputH(h),mInputW(w)
{


};

int DynamicDelta2Bbox::getNbOutputs() const
{
    return 1;
};

int DynamicDelta2Bbox::initialize()
{
    return 0;
};

void DynamicDelta2Bbox::terminate(){

};

void DynamicDelta2Bbox::destroy()
{
    delete this;
};

// size_t DynamicDelta2Bbox::getWorkspaceSize(int) const
// {
//     return 0;
// }
size_t DynamicDelta2Bbox::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept
{
    return 0;
}

// bool DynamicDelta2Bbox::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// };
bool DynamicDelta2Bbox:: supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(0 <= pos && pos < 3);
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
        return (out[0].type == DataType::kHALF || out[0].type == DataType::kFLOAT)
            && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
            
    }
    return false;
}
;
const char* DynamicDelta2Bbox::getPluginType() const
{
    return "DynamicDelta2Bbox_TRT";
};

const char* DynamicDelta2Bbox::getPluginVersion() const
{
    return "1";
};

IPluginV2DynamicExt* DynamicDelta2Bbox::clone() const noexcept
{
    auto plugin = new DynamicDelta2Bbox(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
};

void DynamicDelta2Bbox::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* DynamicDelta2Bbox::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

void DynamicDelta2Bbox::check_valid_inputs(const nvinfer1::DimsExprs* inputs, int nbInputDims)
{

    assert(nbInputDims == 2);

    nvinfer1::DimsExprs rois = inputs[0];
    nvinfer1::DimsExprs deltas = inputs[1];
    assert(rois.nbDims == 3);
    assert(deltas.nbDims == 4);
    // assert(deltas.d[1]==rois.d[1]);

}

 DimsExprs DynamicDelta2Bbox::getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    check_valid_inputs(inputs, nbInputs);
    // assert(index == 0);

    nvinfer1::DimsExprs result;
    result.nbDims = 4;

    //batch size 不是constant，所以不能使用xprBuilder.constant(inputs[1].d[1]->getConstantValue());,如果是constan则在build阶段就能看到其维度，否则显示-1
    result.d[0] = inputs[0].d[0];

    // mROICount
    result.d[1] = inputs[0].d[1];

    result.d[2] = inputs[1].d[2];
    // height
    result.d[3] = exprBuilder.constant(4);


    return result;
}



int DynamicDelta2Bbox::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    
    int batchSize = inputDesc[0].dims.d[0];
    int roiCount = inputDesc[0].dims.d[1];
    int numCls = inputDesc[1].dims.d[2];
    

    // cudaError_t status = ApplyDelta2Bboxes(stream, batchSize, roiCount,

    //      inputs[0], inputs[1], outputs[0]);
    
    cudaError_t status = ApplyDelta2Bboxes_extend(stream, batchSize, roiCount, numCls,

         mInputH, mInputW, inputs[0], inputs[1], outputs[0]
    );

    assert(status == cudaSuccess);
    return 0;
};


size_t DynamicDelta2Bbox::getSerializationSize() const
{
    return sizeof(int) * 2;
};

void DynamicDelta2Bbox::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mInputH);
    write(d, mInputW);
    // write(d, mTargetMean[2]);
    // write(d, mTargetMean[3]);
    // write(d, mTargetStd[0]);
    // write(d, mTargetStd[1]);
    // write(d, mTargetStd[2]);
    // write(d, mTargetStd[3]);    

    assert(d == a + getSerializationSize());
};

DynamicDelta2Bbox::DynamicDelta2Bbox(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;

    mInputH = read<int>(d);
    mInputW = read<int>(d);
    // mTargetMean[2] = read<float>d;
    // mTargetMean[3] = read<float>d;
    // mTargetStd[0] = read<float>d;
    // mTargetStd[1] = read<float>d;
    // mTargetStd[2] = read<float>d;
    // mTargetStd[3] = read<float>d;

    assert(d == a + length);
}

// Return the DataType of the plugin output at the requested index
DataType DynamicDelta2Bbox::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
// bool DynamicDelta2Bbox::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
// {
//     return false;
// }

// // Return true if plugin can use input that is broadcast across batch without replication.
// bool DynamicDelta2Bbox::canBroadcastInputAcrossBatch(int inputIndex) const
// {
//     return false;
// }

// Configure the layer with input and output data types.
// void DynamicDelta2Bbox::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
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

void DynamicDelta2Bbox:: configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept
{

    // assert((inputTypes[0], floatFormat));
    // check_valid_inputs(inputDims, nbInputs);

    assert(nbOutputs == 1);
    assert(nbInputs == 2);

    // mROICount = in[0].desc.dims.d[1];
    // mFeatureLength = in[1].desc.dims.d[1];


}    

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// void DynamicDelta2Bbox::attachToContext(
//     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
// {
// }

// // Detach the plugin object from its execution context.
// void DynamicDelta2Bbox::detachFromContext() {}

