/*****************************************************************//**
 * \file   ForegroundSegmenter.cpp
 * \brief  ǰ���ָ���
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

	// ʹ�� CUDA �������ķ�������һ��ִ�������Ķ��󣬲�������洢�� std::shared_ptr ���͵�����ָ�� ExecutionContext ��
	ExecutionContext = std::shared_ptr<nvinfer1::IExecutionContext>(CudaInferEngine->createExecutionContext());
	if (!ExecutionContext) {
		LOGGING(FATAL) << "����ִ�������Ķ���ʧ�ܣ�" << std::endl;
	}
}

bool SparseSurfelFusion::ForegroundSegmenter::SaveOnnxModel2EngineModel(const std::string& FilePath, const std::string& SavePath)
{
	auto start = std::chrono::high_resolution_clock::now();// ��ȡ����ʼʱ���
	// ��������������
	auto builder = nvinfer1::createInferBuilder(TensorRTLogger);
	if (!builder) {
		std::cout << "��������������ʧ�ܣ�" << std::endl;
		return false;
	}

	// �������������ö���
	auto config = builder->createBuilderConfig();
	if (!config) {
		std::cout << "�������������ö���ʧ�ܣ�" << std::endl;
		return false;
	}

	// ��ʽ������ָ�����綨������ȷָ���������С��������ʹ�ö�̬�������С
	// ʹ�� dynamic shape ��ǰ���ǣ�The network definition must not have an implicit batch dimension.
	// ʹ��λ����� << ����ֵ 1 ����nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH �������λ�������λ����Ϊ��ָʾ����һ����ʽ����������綨�塣
	auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = builder->createNetworkV2(explicitBatch);
	if (!network) {
		std::cout << "��ʽ����������紴��ʧ�ܣ�" << std::endl;
		return false;
	}

	// ����һ�� ONNX �������������滹������Ȩ��
	auto parser = nvonnxparser::createParser(*network, TensorRTLogger);
	if (!parser) {
		std::cout << "����һ�� ONNX ����������ʧ�ܣ�" << std::endl;
		return false;
	}
	parser->parseFromFile(FilePath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));	// ��¼����
	auto ErrorNum = parser->getNbErrors();
	if (ErrorNum != 0) {
		for (int32_t i = 0; i < ErrorNum; ++i) {
			std::cout << parser->getError(i)->desc() << std::endl;
		}
		std::cout << "���� ONNX �ļ�ʧ�ܣ�" << std::endl;
		return false;
	}

	config->setFlag(nvinfer1::BuilderFlag::kFP16);

	auto plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan) {
		std::cout << "���л�����ʧ�ܣ�" << std::endl;
		return false;
	}

	std::shared_ptr<nvinfer1::IRuntime> runtime =
		std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(TensorRTLogger));
	if (!runtime) {
		std::cout << "����IRuntimeʵ��ʧ�ܣ�" << std::endl;
		return false;
	}

	std::shared_ptr<nvinfer1::ICudaEngine> engine =
		std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));

	SerializeEngine(SavePath, engine);

	auto end = std::chrono::high_resolution_clock::now();// ��ȡ�������ʱ���
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// �����������ʱ�䣨���룩
	std::cout << "�ļ��洢�ɹ�����ʱ��" << duration / 1000.0f << " ��" << std::endl;

	return true;
}

void SparseSurfelFusion::ForegroundSegmenter::AllocateInferBuffer()
{

	//std::cout << "IONum = " << CudaInferEngine->getNbIOTensors() << std::endl;;
	for (int i = 0; i < CudaInferEngine->getNbIOTensors(); i++) {
		TensorName2Index.insert(std::pair<std::string, int>(CudaInferEngine->getIOTensorName(i), i));
		nvinfer1::Dims dims = CudaInferEngine->getTensorShape(CudaInferEngine->getIOTensorName(i));
		//printf("index = %d �İ�����Ϊ��%s  Shape = (%lld, %lld, %lld, %lld)\n", i, CudaInferEngine->getIOTensorName(i), dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
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

	// ����������Surface��Textureӳ��
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, RawMask);
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, ClipedMask);
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, PreviousMask);	
	createClipedMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, ClipedEdgeMask);
	createClipedFilteredMaskTextureSurface(rawImageRowsCliped, rawImageColsCliped, FilteredMask);
}

void SparseSurfelFusion::ForegroundSegmenter::ReleaseInferBuffer()
{
	for (int i = 0; i < 11; i++) {	// 11��IO
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
	ExecutionContext->setTensorAddress(name, IOBuffers[tensorIndex]);// �趨���ݵ�ַ
}

void SparseSurfelFusion::ForegroundSegmenter::SerializeEngine(const std::string& path, std::shared_ptr<nvinfer1::ICudaEngine> engine)
{
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	printf("�����ڴ���������: (%d) \n", modelStream->type());
	printf("�滮�ļ���С %.5f MB \n", modelStream->size() / 1024.0 / 1024.0);
	std::ofstream f(path, std::ios::binary);
	// �ı��ļ���"<<"    �������ļ���f.write
	f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	f.close();
}

std::shared_ptr<nvinfer1::ICudaEngine> SparseSurfelFusion::ForegroundSegmenter::DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime)
{
	std::shared_ptr<nvinfer1::ICudaEngine> engine = std::make_shared<nvinfer1::ICudaEngine>();
	std::ifstream file(Path, std::ios::binary);

	if (file.good()) {
		// ��ȡ�ļ���С
		file.seekg(0, file.end);
		size_t size = file.tellg();
		file.seekg(0, file.beg);

		// �����ڴ�
		std::vector<char> trtModelStream(size);
		assert(trtModelStream.data());

		// ��ȡ�ļ�����
		file.read(trtModelStream.data(), size);
		file.close();

		// �����л�����
		engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream.data(), size));
	}

	return engine;
}


void SparseSurfelFusion::ForegroundSegmenter::InferMask(cudaStream_t stream, DeviceArray<uchar3>& colorImages)
{
	// ���ռ���һ֡��mask
	CollectPreviousForegroundMask(ClipedMask.texture, PreviousMask.surface, rawImageRowsCliped, rawImageColsCliped, stream);

	SwapChannelAndNormalizeValue(colorImages, InputSrc, stream);

	void* srcPtr = InputSrc.ptr();

	CHECKCUDA(cudaMemcpyAsync(IOBuffers[srcInputIndex], srcPtr, srcInputSize * sizeof(float), cudaMemcpyDeviceToDevice), stream);

	if (!ExecutionContext->enqueueV3(stream)) {
		std::cout << "����ʧ�ܣ�" << std::endl;
	}
	void* phaPtr = MaskDeviceArray.ptr();
	// ֻ����mask��������fgr
	CHECKCUDA(cudaMemcpyAsync(phaPtr, IOBuffers[PhaOutputIndex], PhaOutputSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r1InputIndex], IOBuffers[r1OutputIndex], R1IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r2InputIndex], IOBuffers[r2OutputIndex], R2IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r3InputIndex], IOBuffers[r3OutputIndex], R3IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaMemcpyAsync(IOBuffers[r4InputIndex], IOBuffers[r4OutputIndex], R4IOSize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	
}

void SparseSurfelFusion::ForegroundSegmenter::SyncAndConvert2Texture(cudaStream_t stream)
{
	CHECKCUDA(cudaStreamSynchronize(stream));		// �������̣߳�Ϊ�˻��Mask������
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
