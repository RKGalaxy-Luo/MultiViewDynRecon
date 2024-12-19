/*****************************************************************//**
 * \file   ForegroundSegmenter.cpp
 * \brief  前景分割器
 * 
 * \author LUO
 * \date   March 22nd 2024
 *********************************************************************/
#include "ForegroundSegmenter.h"

SparseSurfelFusion::ForegroundSegmenter::ForegroundSegmenter(size_t& inferMemorySize)
{
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(TensorRTLogger);
	CudaInferEngine = DeserializeEngine(RVM_MODEL_PATH, runtime);
	inferMemorySize = CudaInferEngine->getDeviceMemorySize();

	// 使用 CUDA 引擎对象的方法创建一个执行上下文对象，并将结果存储在 std::shared_ptr 类型的智能指针 ExecutionContext 中
	ExecutionContext = std::shared_ptr<nvinfer1::IExecutionContext>(CudaInferEngine->createExecutionContext());
	if (!ExecutionContext) {
		LOGGING(FATAL) << "创建执行上下文对象失败！" << std::endl;
	}
}

bool SparseSurfelFusion::ForegroundSegmenter::SaveOnnxModel2EngineModel(const std::string& FilePath, const std::string& SavePath)
{
	auto start = std::chrono::high_resolution_clock::now();// 获取程序开始时间点
	// 构建推理器引擎
	auto builder = nvinfer1::createInferBuilder(TensorRTLogger);
	if (!builder) {
		std::cout << "构建推理器引擎失败！" << std::endl;
		return false;
	}

	// 创建构建器配置对象
	auto config = builder->createBuilderConfig();
	if (!config) {
		std::cout << "创建构建器配置对象失败！" << std::endl;
		return false;
	}

	// 显式批处理：指在网络定义中明确指定批处理大小，而不是使用动态批处理大小
	// 使用 dynamic shape 的前提是：The network definition must not have an implicit batch dimension.
	// 使用位运算符 << 将数值 1 左移nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 所代表的位数。这个位数是为了指示创建一个显式批处理的网络定义。
	auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = builder->createNetworkV2(explicitBatch);
	if (!network) {
		std::cout << "显式批处理的网络创建失败！" << std::endl;
		return false;
	}

	// 创建一个 ONNX 解析器对象：里面还包含着权重
	auto parser = nvonnxparser::createParser(*network, TensorRTLogger);
	if (!parser) {
		std::cout << "创建一个 ONNX 解析器对象失败！" << std::endl;
		return false;
	}
	parser->parseFromFile(FilePath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));	// 记录警告
	auto ErrorNum = parser->getNbErrors();
	if (ErrorNum != 0) {
		for (int32_t i = 0; i < ErrorNum; ++i) {
			std::cout << parser->getError(i)->desc() << std::endl;
		}
		std::cout << "加载 ONNX 文件失败！" << std::endl;
		return false;
	}

	config->setFlag(nvinfer1::BuilderFlag::kFP16);

	auto plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan) {
		std::cout << "序列化网络失败！" << std::endl;
		return false;
	}

	std::shared_ptr<nvinfer1::IRuntime> runtime =
		std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(TensorRTLogger));
	if (!runtime) {
		std::cout << "创建IRuntime实例失败！" << std::endl;
		return false;
	}

	std::shared_ptr<nvinfer1::ICudaEngine> engine =
		std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));

	SerializeEngine(SavePath, engine);

	auto end = std::chrono::high_resolution_clock::now();// 获取程序结束时间点
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// 计算程序运行时间（毫秒）
	std::cout << "文件存储成功！用时：" << duration / 1000.0f << " 秒" << std::endl;

	return true;
}

void SparseSurfelFusion::ForegroundSegmenter::AllocateInferBuffer()
{

	//std::cout << "IONum = " << CudaInferEngine->getNbIOTensors() << std::endl;;
	for (int i = 0; i < CudaInferEngine->getNbIOTensors(); i++) {
		TensorName2Index.insert(std::pair<std::string, int>(CudaInferEngine->getIOTensorName(i), i));
		nvinfer1::Dims dims = CudaInferEngine->getTensorShape(CudaInferEngine->getIOTensorName(i));
		//printf("index = %d 的绑定名称为：%s  Shape = (%lld, %lld, %lld, %lld)\n", i, CudaInferEngine->getIOTensorName(i), dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
	}

	srcInputIndex = TensorName2Index["src"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[srcInputIndex], srcInputSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[srcInputIndex], 0, srcInputSize * sizeof(float)));
	r1InputIndex = TensorName2Index["r1i"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r1InputIndex], R1IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r1InputIndex], 1, R1IOSize * sizeof(float)));
	r2InputIndex = TensorName2Index["r2i"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r2InputIndex], R2IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r2InputIndex], 1, R2IOSize * sizeof(float)));
	r3InputIndex = TensorName2Index["r3i"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r3InputIndex], R3IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r3InputIndex], 1, R3IOSize * sizeof(float)));
	r4InputIndex = TensorName2Index["r4i"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r4InputIndex], R4IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r4InputIndex], 1, R4IOSize * sizeof(float)));

	FgrOutputIndex = TensorName2Index["fgr"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[FgrOutputIndex], FgrOutputSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[FgrOutputIndex], 0, FgrOutputSize * sizeof(float)));
	PhaOutputIndex = TensorName2Index["pha"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[PhaOutputIndex], PhaOutputSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[PhaOutputIndex], 0, PhaOutputSize * sizeof(float)));
	r1OutputIndex = TensorName2Index["r1o"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r1OutputIndex], R1IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r1OutputIndex], 1, R1IOSize * sizeof(float)));
	r2OutputIndex = TensorName2Index["r2o"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r2OutputIndex], R2IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r2OutputIndex], 1, R2IOSize * sizeof(float)));
	r3OutputIndex = TensorName2Index["r3o"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r3OutputIndex], R3IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r3OutputIndex], 1, R3IOSize * sizeof(float)));
	r4OutputIndex = TensorName2Index["r4o"];
	CHECKCUDA(cudaMalloc((void**)&IOBuffers[r4OutputIndex], R4IOSize * sizeof(float)));
	CHECKCUDA(cudaMemset(IOBuffers[r4OutputIndex], 1, R4IOSize * sizeof(float)));

	SetAndBindTensor("src", srcInputIndex, true);
	SetAndBindTensor("r1i", r1InputIndex, true);
	SetAndBindTensor("r2i", r2InputIndex, true);
	SetAndBindTensor("r3i", r3InputIndex, true);
	SetAndBindTensor("r4i", r4InputIndex, true);

	SetAndBindTensor("fgr", FgrOutputIndex);
	SetAndBindTensor("pha", PhaOutputIndex);
	SetAndBindTensor("r1o", r1OutputIndex);
	SetAndBindTensor("r2o", r2OutputIndex);
	SetAndBindTensor("r3o", r3OutputIndex);
	SetAndBindTensor("r4o", r4OutputIndex);

	MaskDeviceArray.create(PhaOutputSize);
	InputSrc.create(srcInputSize);

	// 创建纹理，将Surface与Texture映射
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, RawMask);
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, ClipedMask);
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, PreviousMask);	
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, ClipedEdgeMask);
	createClipedFilteredMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, FilteredMask);
}

void SparseSurfelFusion::ForegroundSegmenter::ReleaseInferBuffer()
{
	for (int i = 0; i < 11; i++) {	// 11个IO
		CHECKCUDA(cudaFree(IOBuffers[i]));
	}
	releaseTextureCollect(RawMask);
	releaseTextureCollect(ClipedMask);
	releaseTextureCollect(PreviousMask);
	releaseTextureCollect(ClipedEdgeMask);
	releaseTextureCollect(FilteredMask);
}

void SparseSurfelFusion::ForegroundSegmenter::SetAndBindTensor(const std::string tensorName, const int tensorIndex, bool isInput)
{
	const char* name = tensorName.c_str();
	nvinfer1::Dims srcDims = CudaInferEngine->getTensorShape(name);
	if(isInput == true) ExecutionContext->setInputShape(name, srcDims);
	ExecutionContext->setTensorAddress(name, IOBuffers[tensorIndex]);// 设定数据地址
}

void SparseSurfelFusion::ForegroundSegmenter::SerializeEngine(const std::string& path, std::shared_ptr<nvinfer1::ICudaEngine> engine)
{
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	printf("主机内存数据类型: (%d) \n", modelStream->type());
	printf("规划文件大小 %.5f MB \n", modelStream->size() / 1024.0 / 1024.0);
	std::ofstream f(path, std::ios::binary);
	// 文本文件用"<<"    二进制文件用f.write
	f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	f.close();
}

std::shared_ptr<nvinfer1::ICudaEngine> SparseSurfelFusion::ForegroundSegmenter::DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime)
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


void SparseSurfelFusion::ForegroundSegmenter::InferMask(cudaStream_t stream, DeviceArray<uchar3>& colorImages)
{
	// 先收集上一帧的mask
	CollectPreviousForegroundMask(ClipedMask.texture, PreviousMask.surface, rawImageRowsCliped, rawImageColsCliped, stream);

	SwapChannelAndNormalizeValue(colorImages, InputSrc, stream);

	void* srcPtr = InputSrc.ptr();

	CHECKCUDA(cudaMemcpyAsync(IOBuffers[srcInputIndex], srcPtr, srcInputSize * sizeof(float), cudaMemcpyDeviceToDevice), stream);

	if (!ExecutionContext->enqueueV3(stream)) {
		std::cout << "推理失败！" << std::endl;
	}
	void* phaPtr = MaskDeviceArray.ptr();
	// 只关心mask，不关心fgr
	CHECKCUDA(cudaMemcpyAsync(phaPtr, IOBuffers[PhaOutputIndex], PhaOutputSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r1InputIndex], IOBuffers[r1OutputIndex], R1IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r2InputIndex], IOBuffers[r2OutputIndex], R2IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r3InputIndex], IOBuffers[r3OutputIndex], R3IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r4InputIndex], IOBuffers[r4OutputIndex], R4IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	
}

void SparseSurfelFusion::ForegroundSegmenter::SyncAndConvert2Texture(cudaStream_t stream)
{
	CHECKCUDA(cudaStreamSynchronize(stream));		// 阻塞该线程，为了获得Mask的纹理
	CollectAndClipMask(rawImageRowsCliped, rawImageColsCliped, RawMask, ClipedMask, stream);
	FilterForefroundMask(ClipedMask.texture, rawImageRowsCliped, rawImageColsCliped, Constants::kForegroundSigma, FilteredMask.surface);
}

cudaTextureObject_t SparseSurfelFusion::ForegroundSegmenter::getClipedMaskTexture()
{
	return ClipedMask.texture;
}

cudaTextureObject_t SparseSurfelFusion::ForegroundSegmenter::getFilteredMaskTexture()
{
	return FilteredMask.texture;
}

cudaTextureObject_t SparseSurfelFusion::ForegroundSegmenter::getPreviousMaskTexture()
{
	return PreviousMask.texture;
}

cudaTextureObject_t SparseSurfelFusion::ForegroundSegmenter::getForegroundEdgeMaskTexture()
{
	return ClipedEdgeMask.texture;
}

cudaTextureObject_t SparseSurfelFusion::ForegroundSegmenter::getRawClipedForegroundTexture()
{
	return RawMask.texture;
}
