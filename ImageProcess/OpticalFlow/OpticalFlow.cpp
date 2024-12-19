/*****************************************************************//**
 * \file   OpticalFlow.cpp
 * \brief  计算光流
 * 
 * \author LUOJIAXUAN
 * \date   July 4th 2024
 *********************************************************************/
#include "OpticalFlow.h"

SparseSurfelFusion::OpticalFlow::OpticalFlow(size_t& inferMemorySize)
{


	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(TensorRTLogger);
	std::string ModelPath = GMFLOW_MODEL_PATH;
	bool didInitPlugins = initLibNvInferPlugins(nullptr, "");

	CudaInferEngine = DeserializeEngine(ModelPath, runtime);
	inferMemorySize = CudaInferEngine->getDeviceMemorySize();

	// 使用 CUDA 引擎对象的方法创建一个执行上下文对象，并将结果存储在 std::shared_ptr 类型的智能指针 ExecutionContext 中
	ExecutionContext = std::shared_ptr<nvinfer1::IExecutionContext>(CudaInferEngine->createExecutionContext());
	if (!ExecutionContext) {
		LOGGING(FATAL) << "创建执行上下文对象失败！" << std::endl;
	}
#ifdef DRAW_OPTICALFLOW
	// 绘制光流调试，完成调试即可删除
	draw = make_shared<DynamicallyDrawOpticalFlow>();
#endif // DRAW_OPTICALFLOW

}

void SparseSurfelFusion::OpticalFlow::AllocateInferBuffer()
{
	//std::cout << "IONum = " << CudaInferEngine->getNbIOTensors() << std::endl;;
	for (int i = 0; i < CudaInferEngine->getNbIOTensors(); i++) {
		TensorName2Index.insert(std::pair<std::string, int>(CudaInferEngine->getIOTensorName(i), i));
		nvinfer1::Dims dims = CudaInferEngine->getTensorShape(CudaInferEngine->getIOTensorName(i));
	}

	PreviousImageInputIndex = TensorName2Index["image_previous"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[PreviousImageInputIndex], PreviousImageSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[PreviousImageInputIndex], 0, PreviousImageSize * sizeof(float)));
	PreviousDepthInputIndex = TensorName2Index["depth_previous"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[PreviousDepthInputIndex], PreviousDepthSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[PreviousDepthInputIndex], 0, PreviousDepthSize * sizeof(float)));
	CurrentImageInputIndex = TensorName2Index["image_current"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[CurrentImageInputIndex], CurrentImageSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[CurrentImageInputIndex], 0, CurrentImageSize * sizeof(float)));
	CurrentDepthInputIndex = TensorName2Index["depth_current"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[CurrentDepthInputIndex], CurrentDepthSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[CurrentDepthInputIndex], 0, CurrentDepthSize * sizeof(float)));
	FlowOutputIndex = TensorName2Index["previous_to_current_flow"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[FlowOutputIndex], FlowSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[FlowOutputIndex], 0, FlowSize * sizeof(float)));

	SetAndBindTensor("image_previous", PreviousImageInputIndex, true);
	SetAndBindTensor("depth_previous", PreviousDepthInputIndex, true);
	SetAndBindTensor("image_current", CurrentImageInputIndex, true);
	SetAndBindTensor("depth_current", CurrentDepthInputIndex, true);
	SetAndBindTensor("previous_to_current_flow", FlowOutputIndex);

	InputPreviousImage.create(PreviousImageSize);
	InputPreviousDepth.create(PreviousDepthSize);
	InputCurrentImage.create(CurrentImageSize);
	InputCurrentDepth.create(CurrentDepthSize);

	Flow2D.create(rawImageSize * 2);
	PixelPairsMap.create(ImageRowsCliped, ImageColsCliped);
	//MarkEdgeValidPixelPairs.create(ImageRowsCliped, ImageColsCliped);

	correspondencePixelPair.AllocateBuffer(clipedImageSize);
	correspondencePixelPair.ResizeArrayOrException(clipedImageSize);
	validPixelPairs.AllocateBuffer(clipedImageSize);

	markValidPairs.AllocateBuffer(clipedImageSize);
	markValidPairs.ResizeArrayOrException(clipedImageSize);

#ifdef DRAW_OPTICALFLOW
	// 下述为可视化专用数组
	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&FlowPtrDevice), FlowSize * sizeof(float)));
	CHECKCUDA(cudaMemset(FlowPtrDevice, 0, FlowSize * sizeof(float)));

	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&FlowVector3D), clipedImageSize * sizeof(float4)));
	CHECKCUDA(cudaMemset(FlowVector3D, 0, clipedImageSize * sizeof(float4)));

	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&FlowVectorOpenGL), 2 * clipedImageSize * sizeof(float3)));
	CHECKCUDA(cudaMemset(FlowVectorOpenGL, 0, 2 * clipedImageSize * sizeof(float3)));

	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&markValidFlow), 2 * clipedImageSize * sizeof(bool)));
	CHECKCUDA(cudaMemset(markValidFlow, 0, 2 * clipedImageSize * sizeof(bool)));

	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&validFlowVector), 2 * clipedImageSize * sizeof(float3)));
	CHECKCUDA(cudaMemset(validFlowVector, 0, 2 * clipedImageSize * sizeof(float3)));

	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&colorVertexPtr), 2 * clipedImageSize * sizeof(ColorVertex)));
	CHECKCUDA(cudaMemset(colorVertexPtr, 0, 2 * clipedImageSize * sizeof(ColorVertex)));

	CHECKCUDA(cudaMalloc(reinterpret_cast<void**>(&validColorVertex), 2 * clipedImageSize * sizeof(ColorVertex)));
	CHECKCUDA(cudaMemset(validColorVertex, 0, 2 * clipedImageSize * sizeof(ColorVertex)));
#endif // DRAW_OPTICALFLOW


}

void SparseSurfelFusion::OpticalFlow::ReleaseInferBuffer()
{
	for (int i = 0; i < 5; i++) {
		CHECKCUDA(cudaFree(IOBuffers[i]));
	}
	InputPreviousImage.release();
	InputPreviousDepth.release();
	InputCurrentImage.release();
	InputCurrentDepth.release();

	Flow2D.release();
	PixelPairsMap.release();

	correspondencePixelPair.ReleaseBuffer();
	validPixelPairs.ReleaseBuffer();
	markValidPairs.ReleaseBuffer();

#ifdef DRAW_OPTICALFLOW
	cudaFree(FlowPtrDevice);
	cudaFree(FlowVector3D);
	cudaFree(FlowVectorOpenGL);
	cudaFree(markValidFlow);
	cudaFree(validFlowVector);
	cudaFree(colorVertexPtr);
	cudaFree(validColorVertex);
#endif // DRAW_OPTICALFLOW

}

void SparseSurfelFusion::OpticalFlow::setForegroundAndVertexMapTexture(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t preVertexMap, cudaTextureObject_t currVertexMap, cudaTextureObject_t preNormalMap, cudaTextureObject_t currNormalMap, cudaTextureObject_t currEdgeMask, const mat34 initialCameraSE3)
{
	PreviousForeground = preForeground;
	CurrentForeground = currForeground;
	PreviousVertexMap = preVertexMap;
	CurrentVertexMap = currVertexMap;
	PreviousNormalMap = preNormalMap;
	CurrentNormalMap = currNormalMap;
	CurrentEdgeMaskMap = currEdgeMask;
	InitialCameraSE3 = initialCameraSE3;
}

void SparseSurfelFusion::OpticalFlow::InferOpticalFlow(DeviceArray<uchar3>& previousImage, DeviceArray<unsigned short>& previousDepth, DeviceArray<uchar3>& currentImage, DeviceArray<unsigned short>& currentDepth, cudaStream_t stream)
{
	ConvertBuffer2InputTensor(previousImage, previousDepth, currentImage, currentDepth, stream);
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[PreviousImageInputIndex], InputPreviousImage.ptr(), PreviousImageSize * sizeof(float), cudaMemcpyHostToDevice), stream);
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[PreviousDepthInputIndex], InputPreviousDepth.ptr(), PreviousDepthSize * sizeof(float), cudaMemcpyHostToDevice), stream);
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[CurrentImageInputIndex], InputCurrentImage.ptr(), CurrentImageSize * sizeof(float), cudaMemcpyHostToDevice), stream);
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[CurrentDepthInputIndex], InputCurrentDepth.ptr(), CurrentDepthSize * sizeof(float), cudaMemcpyHostToDevice), stream);
	
	if (!ExecutionContext->enqueueV3(stream)) {
		std::cout << "推理失败！" << std::endl;
	}

	CHECKCUDA(cudaMemcpyAsync(Flow2D.ptr(), IOBuffers[FlowOutputIndex], FlowSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
#ifdef DRAW_OPTICALFLOW
	CalculatePixelPairAnd3DOpticalFlow(stream);
#else 
	CalculatePixelPairs(stream);
#endif // DRAW_OPTICALFLOW

	CHECKCUDA(cudaStreamSynchronize(stream));
}

std::shared_ptr<nvinfer1::ICudaEngine> SparseSurfelFusion::OpticalFlow::DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime)
{
	std::shared_ptr<nvinfer1::ICudaEngine> engine = std::make_shared<nvinfer1::ICudaEngine>();
	std::ifstream file(Path, std::ios::binary);

	if (file.good()) {
		// 获取文件大小
		file.seekg(0, file.end);
		size_t size = file.tellg();
		file.seekg(0, file.beg);

		// 分配内存
		std::vector<char> trtModelStream(size);
		assert(trtModelStream.data());

		// 读取文件内容
		file.read(trtModelStream.data(), size);
		file.close();

		// 反序列化引擎
		engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream.data(), size));
	}

	return engine;
}

void SparseSurfelFusion::OpticalFlow::createCliped3DOpticalFlowTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect)
{
	createCliped3DOpticalFlowTextureSurface(clipedRows, clipedCols, collect.texture, collect.surface, collect.cudaArray);
}

void SparseSurfelFusion::OpticalFlow::createCliped3DOpticalFlowTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
	cudaTextureDesc flowTextureDescription;
	createDefault2DTextureDescriptor(flowTextureDescription);
	cudaChannelFormatDesc flowChannelDescription = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);	// 三通道32位float

	CHECKCUDA(cudaMallocArray(&cudaArray, &flowChannelDescription, clipedCols, clipedRows));

	cudaResourceDesc resourceDescription;
	memset(&resourceDescription, 0, sizeof(cudaResourceDesc));
	resourceDescription.resType = cudaResourceTypeArray;
	resourceDescription.res.array.array = cudaArray;

	CHECKCUDA(cudaCreateTextureObject(&texture, &resourceDescription, &flowTextureDescription, 0));
	CHECKCUDA(cudaCreateSurfaceObject(&surface, &resourceDescription));
}

void SparseSurfelFusion::OpticalFlow::SetAndBindTensor(const std::string tensorName, const int tensorIndex, bool isInput)
{
	const char* name = tensorName.c_str();
	nvinfer1::Dims srcDims = CudaInferEngine->getTensorShape(name);
	if (isInput == true) ExecutionContext->setInputShape(name, srcDims);
	ExecutionContext->setTensorAddress(name, IOBuffers[tensorIndex]);// 设定数据地址
}