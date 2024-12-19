/*****************************************************************//**
 * \file   ForegroundSegmenter.h
 * \brief  ǰ���ָ���
 * 
 * \author LUO
 * \date   March 22nd 2024
 *********************************************************************/
#pragma once
#include <base/Constants.h>
#include <base/Logging.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>
namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief ����Rͨ����Bͨ�������ҹ�һ��UChar�ĺ˺���.
		 * 
		 * \param UCharInputArray ����ԭʼ��0-255��RGB����ͼ��
		 * \param NormalizedInputArray ������һ����������Rͨ����Bͨ����
		 * \param rawRows ԭʼͼ��ĸ�
		 * \param rawCols ԭʼͼ��Ŀ�
		 */
		__global__ void SwapChannelAndNormalizeValueKernel(PtrSize<uchar3> UCharInputArray, PtrSize<float> NormalizedInputArray, const unsigned int rawSize);

		/**
		 * \brief �ռ�������Mask�ĺ˺���.
		 * 
		 * \param MaskArray ����δ�����õ�Mask
		 * \param threshold ��õ�0-1��Χ��pha��ֵ���������threshold��Ϊ��ǰ������Mask��Ӧ���������Ϊ1
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 * \param MaskSurface ��ǰ��Mask���ݴ���cudaSurfaceObject_t(��ӳ�䵽cudaTextureObject_t)
		 */
		__global__ void CollectAndClipMaskKernel(PtrSize<float> MaskArray, float threshold, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t rawForeground);

		/**
		 * \brief ��ԭʼǰ������ʴ��������Ϊ�����Σ���ȥ��mask��Ե�쳣��.
		 */
		__global__ void ErodeRawForeground(cudaTextureObject_t rawForeground, const int erodeRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t MaskSurface);

		/**
		 * \brief �Դ����mask�������Ͳ���������Ϊ�����Σ�.
		 */
		__global__ void DilateRawForeground(cudaTextureObject_t rawMask, const int dilateRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t dilateMaskSurface);

		/**
		 * \brief ��ԭʼǰ������ʴ��������ΪԲ����ȥ��mask��Ե�쳣��.
		 */
		__global__ void ErodeRawForeground(cudaTextureObject_t rawForeground, const int erodeRadius, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t MaskSurface);

		/**
		 * \brief �Դ����mask�������Ͳ���������ΪԲ��.
		 */
		__global__ void DilateRawForeground(cudaTextureObject_t rawMask, const int dilateRadius, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t dilateMaskSurface);

		/**
		 * \brief ��ԭʼ��clipedMask(��������̬ѧ����)����Mask.
		 */
		__global__ void ForegroundEdgeMask(cudaTextureObject_t clipedMask, const int edgeThickness, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t edgeMask);

		/**
		 * \brief �ռ�ǰһ֡Mask�ĺ˺���.
		 * 
		 * \param mask ��һ֡��mask
		 * \param previousMask �ռ���һ֡��mask
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 */
		__global__ void CollectPreviousForegroundMaskKernel(cudaTextureObject_t mask, cudaSurfaceObject_t previousMask, const unsigned int clipedRows, const unsigned int clipedCols);
	
		/**
		 * \brief ǰ��Mask�˲�.
		 */
		__global__ void filterForegroundMaskKernel(cudaTextureObject_t foregroundMask, unsigned maskRows, unsigned maskCols, const float sigma, cudaSurfaceObject_t filteredMask);
	}

	class ForegroundSegmenter
	{
	public:
		using Ptr = std::shared_ptr<ForegroundSegmenter>;
		/**
		 * \brief ���캯���������Ƿ���Ҫ��onnx�ļ�ת��engine�ļ�.
		 * 
		 */
		ForegroundSegmenter(size_t& inferMemorySize);
		~ForegroundSegmenter() = default;

		//enum InferDataType {
		//	FP16,
		//	FP32
		//};

		enum GPURuntimeMemory {
			GB = 1 << 30,	// 1GB
			MB = 1 << 20,	// 1MB
			KB = 1 << 10,	// 1KB
			B = 1			// B
		};

		/**
		 * \brief ���������ڴ�.
		 *
		 */
		void AllocateInferBuffer();

		/**
		 * \brief �ͷ��ڴ�.
		 *
		 */
		void ReleaseInferBuffer();

		/**
		 * \brief ���ò���Tensor.
		 * 
		 * \param tensorName ���õ���������
		 * \param tensorIndex ����������Index
		 * \param isInput �Ƿ�����������
		 */
		void SetAndBindTensor(const std::string tensorName, const int tensorIndex, bool isInput = false);

		/**
		 * \brief ��Onnxģ��ת��tensorRT Engineģ��.
		 *
		 * \param FilePath Onnx�ļ�·��
		 * \param SavePath �����Engine·��
		 * \return �Ƿ񱣴�ɹ�
		 */
		bool SaveOnnxModel2EngineModel(const std::string& FilePath, const std::string& SavePath);


		/**
		 * \brief �Ƶ��ó�Mask.
		 *
		 * \param cuda�� CUDA��ID
		 * \param colorImages ����Device�д洢��ͼ����Ϣ
		 */
		void InferMask(cudaStream_t stream, DeviceArray<uchar3>& colorImages);

		/**
		 * \brief ��ͬ������ת��Ϊ�����ڴ�.
		 * 
		 * \param stream cuda��ID
		 */
		void SyncAndConvert2Texture(cudaStream_t stream);

		/**
		 * \brief ���(���ú�)ǰ��Texture.
		 * 
		 * \return ���ǰ��������
		 */
		cudaTextureObject_t getClipedMaskTexture();

		/**
		 * \brief ���(�˲���)��ǰ��Mask.
		 * 
		 * \return �˲�������
		 */
		cudaTextureObject_t getFilteredMaskTexture();

		/**
		 * \brief �����һ֡(���ú�)ǰ��Texture.
		 *
		 * \return ���ǰ��������
		 */
		cudaTextureObject_t getPreviousMaskTexture();

		/**
		 * \brief ���ǰ���ı�ԵMask.
		 * 
		 * \return 
		 */
		cudaTextureObject_t getForegroundEdgeMaskTexture();

		/**
		 * \brief ���ԭʼ��Foreground.
		 * 
		 * \return ԭʼ��Foreground
		 */
		cudaTextureObject_t getRawClipedForegroundTexture();
	private:

		Logger TensorRTLogger;			// Tensor������־

		std::shared_ptr<nvinfer1::IExecutionContext> ExecutionContext;	// ִ�������Ķ��󣺺���ģ���������
		std::shared_ptr<nvinfer1::ICudaEngine> CudaInferEngine;			// cuda��������

		std::map<std::string, int> TensorName2Index;	// TensorRT 10.3 �� nvinfer1::ICudaEngine::getBindingIndex����û���ˣ��Լ�����ӳ���ϵ

		CudaTextureSurface PreviousMask;		// ǰһ֡��MASK
		CudaTextureSurface RawMask;				// ԭʼ��Mask
		CudaTextureSurface ClipedMask;			// ���ò������̬ѧ������MASK
		CudaTextureSurface FilteredMask;		// �˲����ǰ��
		CudaTextureSurface ClipedEdgeMask;		// ���ü�Raw��ԵMask

		DeviceArray<float> InputSrc;			// ����ͼ��RGBת��BGR������һ����ת��Tensor
		DeviceArray<float> MaskDeviceArray;		// �����õ�Maskͼ��

		const int erodeRadius = 2;				// ��̬ѧ��ʴ�뾶
		const int dilateRadius = 7;				// ��̬ѧ���Ͱ뾶
		const int edgeThickness = 2;			// Mask�ı�Ե�뾶

		int srcInputIndex;		// ԭͼ��������
		int r1InputIndex;		// r1i��������
		int r2InputIndex;		// r2i��������
		int r3InputIndex;		// r3i��������
		int r4InputIndex;		// r4i��������

		int FgrOutputIndex;		// fgr���������
		int PhaOutputIndex;		// pha���������
		int r1OutputIndex;		// r1o��������
		int r2OutputIndex;		// r2o��������
		int r3OutputIndex;		// r3o��������
		int r4OutputIndex;		// r4o��������

		void* IOBuffers[11];	// �����������������buffer

		const unsigned int ImageSize = FRAME_HEIGHT * FRAME_WIDTH;
		const unsigned int rawImageRows = FRAME_HEIGHT;
		const unsigned int rawImageCols = FRAME_WIDTH;
		const unsigned int rawImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;

		const int srcInputSize = 1 * 3 * FRAME_HEIGHT * FRAME_WIDTH;

#ifdef HIGH_FRAME_PER_SECOND
		// ratio = 0.8
		const int R1IOSize = 1 * 16 * 160 * 256;
		const int R2IOSize = 1 * 32 * 80 * 128;
		const int R3IOSize = 1 * 64 * 40 * 64;
		const int R4IOSize = 1 * 128 * 20 * 32;
		const int FgrOutputSize = 1 * 3 * 400 * 640;
		const int PhaOutputSize = 1 * 1 * 400 * 640;
#else
		// ratio = 0.4
		const int R1IOSize = 1 * 16 * 144 * 256;
		const int R2IOSize = 1 * 32 * 72 * 128;
		const int R3IOSize = 1 * 64 * 36 * 64;
		const int R4IOSize = 1 * 128 * 18 * 32;

		const int FgrOutputSize = 1 * 3 * 720 * 1280;
		const int PhaOutputSize = 1 * 1 * 720 * 1280;
#endif // HIGH_FRAME_PER_SECOND


		/**
		 * \brief �������ú��Mask����ͱ��棬������ͱ����໥ӳ��.
		 * 
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 * \param collect �������
		 */
		void createClipedMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect);

		/**
		 * \brief �������ú��Mask����ͱ��棬������ͱ����໥ӳ��.
		 * 
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 * \param texture ����(GPU�ڴ��ַ)
		 * \param surface ����(GPU�ڴ��ַ)
		 * \param cudaArray ����(�洢��GPU)
		 */
		void createClipedMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

		/**
		 * \brief �������ú��Mask����ͱ��棬������ͱ����໥ӳ��.
		 *
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 * \param collect �������
		 */
		void createClipedFilteredMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect);


		/**
		 * \brief �������ú��FilteredMask����ͱ��棬������ͱ����໥ӳ��.
		 *
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 * \param texture ����(GPU�ڴ��ַ)
		 * \param surface ����(GPU�ڴ��ַ)
		 * \param cudaArray ����(�洢��GPU)
		 */
		void createClipedFilteredMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

		/**
		 * \brief ����ICudaEngine������ONNX���engine�ļ�.
		 *
		 * \param path ��ģ�ʹ���õ�ַ
		 * \param engine onnxģ���漰��cuda����
		 */
		void SerializeEngine(const std::string& path, std::shared_ptr<nvinfer1::ICudaEngine> engine);

		/**
		 * \brief �����л�.engine�ļ��������� cuda ��������.
		 *
		 * \param Path .engine�ļ�
		 * \param runtime �����ִ��������������
		 * \return cuda ��������
		 */
		std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime);

		/**
		 * \brief ��RGBת��BGR������һ��������infer��Tensor.
		 * 
		 * \param colorImages ԭʼ��ͼ��
		 * \param normalizedSrc ����ͨ������һ����src
		 * \param stream CUDA��ID
		 */
		void SwapChannelAndNormalizeValue(DeviceArray<uchar3>& colorImages, DeviceArray<float>& normalizedSrc, cudaStream_t stream);

		/**
		 * \brief ���ռ���һ֡������.
		 * 
		 * \param mask ��һ֡��Texture
		 * \param previousMask �洢��һ֡��Texture
		 * \param clipedRows ͼƬ�ĸ�
		 * \param clipedCols ͼƬ�Ŀ�
		 * \param stream cuda��id
		 */
		void CollectPreviousForegroundMask(cudaTextureObject_t mask, cudaSurfaceObject_t previousMask, const unsigned int clipedRows, const unsigned int clipedCols, cudaStream_t stream);

		/**
		 * \brief �ռ�������Mask.
		 * 
		 * \param rawMask ԭʼ��С��Mask
		 * \param clipedRows ���ú�ĸ�
		 * \param clipedCols ���ú�Ŀ�
		 * \param erodeRadius ��ʴ�����뾶
		 * \param maskSurface �����ú��mask��������������
		 * \param stream cuda��ID
		 */
		void CollectAndClipMask(const unsigned int clipedRows, const unsigned int clipedCols, CudaTextureSurface rawForeground, CudaTextureSurface maskSurface, cudaStream_t stream);

		/**
		 * \brief ��ǰ�������˲�.
		 * 
		 * \param foregroundMask ǰ��
		 * \param clipedRows ���ú����
		 * \param clipedCols ���ú����
		 * \param sigma �˲�����sigma
		 * \param filteredMask �˲����Mask
		 * \param stream cuda��
		 */
		void FilterForefroundMask(cudaTextureObject_t foregroundMask, const unsigned int clipedRows, const unsigned int clipedCols, float sigma, cudaSurfaceObject_t filteredMask, cudaStream_t stream = 0);
	};

}

