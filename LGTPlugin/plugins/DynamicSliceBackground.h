

#ifndef TRT_SLICE_BACKGROUND_PLUGIN_H
#define TRT_SLICE_BACKGROUND_PLUGIN_H

#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "../kernels/FasterRCNNKernels.h"
// #include "mrcnn_config.h"

namespace nvinfer1
{
namespace plugin
{

class DynamicSliceBackground : public IPluginV2DynamicExt
{
public:
    DynamicSliceBackground();

    DynamicSliceBackground(const void* data, size_t length);

    ~DynamicSliceBackground() override = default;

    // IPluginV2 methods    
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;

    // IPluginV2Ext methods
    nvinfer1:: DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;  
    // Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    // bool supportsFormat(DataType type, PluginFormat format) const override;
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    // void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    //     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    //     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    // size_t getWorkspaceSize(int) const override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    // int enqueue(
    //     int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
   int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;


    // ohters 
    // bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    // bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    // void attachToContext(
    //     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;


    // void detachFromContext() override;

private:
    void check_valid_inputs(const nvinfer1::DimsExprs* inputs, int nbInputDims);

    // quad_t<float> mTargetMean;
    // quad_t<float> mTargetStd;
    // int mInputH;
    // int mInputW;
    std::string mNameSpace;
};

class DynamicSliceBackgroundPluginCreator : public BaseCreator
{
public:
    DynamicSliceBackgroundPluginCreator();

    ~DynamicSliceBackgroundPluginCreator(){};

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    // IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    // IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) override;
    
    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    // quad_t<float> mTargetMean;
    // quad_t<float> mTargetStd;
    // int mInputH;
    // int mInputW;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_SLICE_BACKGROUND_PLUGIN_H
