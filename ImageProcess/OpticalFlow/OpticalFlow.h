/*****************************************************************//**
 * \file   OpticalFlow.h
 * \brief  �������ģ��
 * 
 * \author LUOJIAXUAN
 * \date   July 4th 2024
 *********************************************************************/
#pragma once
#include <base/Logging.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <math/VectorUtils.h>
#include <string>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <NvInferPlugin.h>
#include "DynamicallyDrawOpticalFlow.h"
#include <core/AlgorithmTypes.h>
//#define DRAW_OPTICALFLOW

namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief �󴰿������ͼ�ļ���.
		 */
		__forceinline__ __device__ float windowRange(const unsigned short& x, const unsigned short& y, const short& windowRadius, const unsigned int& cols, const unsigned int& rows, cudaTextureObject_t vertexMap);

		/**
		 * \brief �������ԭʼͼ�����ת��ģ�������Tensor.
		 * 
		 * \param previousImage ��һ֡RGBͼ��
		 * \param previousDepth ��һ֡���ͼ��
		 * \param currentImage ��ǰ֡RGBͼ��
		 * \param currentDepth ��ǰ֡���ͼ��
		 * \param rawImageSize ԭʼͼ��Ĵ�С
		 * \param inputPreviousImage ����ģ�͵���һ֡RGBͼ��
		 * \param inputPreviousDepth ����ģ�͵���һ֡���ͼ��
		 * \param inputCurrentImage ����ģ�͵ĵ�ǰ֡RGBͼ��
		 * \param inputCurrentDepth ����ģ�͵ĵ�ǰ֡���ͼ��
		 */
		__global__ void ConvertBuffer2InputTensorKernel(const uchar3* previousImage, const unsigned short* previousDepth, const uchar3* currentImage, const unsigned short* currentDepth, const unsigned int rawImageSize, float* inputPreviousImage, float* inputPreviousDepth, float* inputCurrentImage, float* inputCurrentDepth);

		/**
		 * \brief ��������ƥ����Լ���ά�������������۹�������ƥ���ԣ���ֻȡǰ��.
		 * 
		 * \param preForeground ǰһ֡��(���ú�)ǰ��Mask
		 * \param currForeground ��ǰ֡��(���ú�)ǰ��Mask
		 * \param PreviousVertexMap ǰһ֡��(���ú�)��VertexConfidenceMap
		 * \param CurrentVertexMap ��ǰ֡��(���ú�)��VertexConfidenceMap
		 * \param clipedImageRows ���ú�ͼ���
		 * \param clipedImageCols ���ú�ͼ���
		 * \param pixelPair ƥ�����ص��
		 * \param markValidPair �����Ч�Ĺ�����
		 * \param flow3d ��ά��������
		 */
		__global__ void CalculatePixelPairAnd3DOpticalFlowKernel(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t PreviousVertexMap, cudaTextureObject_t CurrentVertexMap, const float* flow2d, const unsigned int clipedImageRows, const unsigned int clipedImageCols, const unsigned int rawImageSize, ushort4* pixelPair, bool* markValidPair, float4* FlowVector3D, float3* FlowVectorOpenGL, ColorVertex* colorVertex, bool* markValidFlow);

		/**
		 * \brief ��������ƥ����Լ���ά�������������۹�������ƥ���ԣ���ֻȡǰ��.
		 *
		 * \param preForeground ǰһ֡��(���ú�)ǰ��Mask
		 * \param currForeground ��ǰ֡��(���ú�)ǰ��Mask
		 * \param PreviousVertexMap ǰһ֡��(���ú�)��VertexConfidenceMap
		 * \param CurrentVertexMap ��ǰ֡��(���ú�)��VertexConfidenceMap
		 * \param PreviousNormalMap ǰһ֡��(���ú�)��NormalRadiusMap
		 * \param CurrentNormalMap ��ǰ֡��(���ú�)��NormalRadiusMap
		 * \param initialCameraSE3 ��ǰ�����λ��
		 * \param flow2d ����õ��Ķ�ά����
		 * \param clipedImageRows ���ú�ͼ���
		 * \param clipedImageCols ���ú�ͼ���
		 * \param pixelPair ƥ�����ص��
		 * \param markValidPair �����Ч�Ĺ�����
		 */
		__global__ void CalculatePixelPairsKernel(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t PreviousVertexMap, cudaTextureObject_t CurrentVertexMap, cudaTextureObject_t preNormalMap, cudaTextureObject_t currNormalMap, const mat34 initialCameraSE3, const float* flow2d, const unsigned int clipedImageRows, const unsigned int clipedImageCols, const unsigned int rawImageSize, ushort4* pixelPair, bool* markValidPair, PtrStepSize<ushort4> validEdgePixelPairs);

		/**
		 * \brief ��ñ�Ե����Чƥ���.
		 */
		__global__ void GetValidEdgePixelPairsKernel(cudaTextureObject_t currEdgeMaskMap, const ushort4* pixelPair, const bool* markValidPixelPairs, const unsigned int clipedImageRows, const unsigned int clipedImageCols, PtrStepSize<ushort4> validEdgePixelPairs);

		__global__ void LabelSortedCurrPairEncodedCoor(const PtrSize<const unsigned int> sortedCoor, unsigned int* label);

		__global__ void CompactDenseCorrespondence(const PtrSize<ushort4> sortedPairs, const unsigned int* pairsLabel, const unsigned int* prefixSum, ushort4* compactedCorrPairs);

		/**
		 * \brief У������Map.
		 * 
		 * \param markValidFlowSe3Map ��ǽ����˹����任��vertex
		 * \param correctSe3 У׼����
		 * \param clipedImageRows ���ú�ͼ���
		 * \param clipedImageCols ���ú�ͼ���
		 * \param vertexFlowSe3 У����Ĺ���Map
		 */
		__global__ void CorrectOpticalFlowSe3MapKernel(const DeviceArrayView2D<unsigned char> markValidFlowSe3Map, const mat34 correctSe3, const unsigned int clipedImageRows, const unsigned int clipedImageCols, PtrStepSize<mat34> vertexFlowSe3);
	}
	class OpticalFlow
	{
	public:
		using Ptr = std::shared_ptr<OpticalFlow>;

		OpticalFlow(size_t& inferMemorySize);

		~OpticalFlow() = default;

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
		 * \brief ����ǰһ֡�͵�ǰ֡��ǰ���Լ�ǰһ֡�͵�ǰ֡�Ķ���ͼ.
		 * 
		 * \param preForeground ǰһ֡ǰ��Mask
		 * \param currForeground ��ǰ֡ǰ��Mask
		 * \param PreviousVertexMap ǰһ֡��(���ú�)��VertexConfidenceMap
		 * \param CurrentVertexMap ��ǰ֡��(���ú�)��VertexConfidenceMap
		 * \param preNormalMap ��һ֡����Map
		 * \param currNormalMap ��ǰ֡����Map
		 * \param currEdgeMask ��ǰ֡��Ե��mask
		 * \param initialCameraSE3 ���λ��
		 */
		void setForegroundAndVertexMapTexture(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t preVertexMap, cudaTextureObject_t currVertexMap, cudaTextureObject_t preNormalMap, cudaTextureObject_t currNormalMap, cudaTextureObject_t currEdgeMask, const mat34 initialCameraSE3);

		/**
		 * \brief �������ģ��.
		 * 
		 * \param previousImage ǰһ֡ͼ��
		 * \param previousDepth ǰһ֡���
		 * \param currentImage ��ǰ֡ͼ��
		 * \param currentDepth ��ǰ֡���
		 * \param stream cuda��
		 */
		void InferOpticalFlow(DeviceArray<uchar3>& previousImage, DeviceArray<unsigned short>& previousDepth, DeviceArray<uchar3>& currentImage, DeviceArray<unsigned short>& currentDepth, cudaStream_t stream);

	private:
		DeviceBufferArray<ushort4> correspondencePixelPair;			// �������ض�
		DeviceBufferArray<ushort4> validPixelPairs;					// ѡ����Ч�����ص��(���㣺1��ǰ����֡�����ص����ǰ���ϣ�2������ƥ������ص�����������Ϊ0�������3������ƥ������ص����������ģ < ָ����ֵ��4���޳�ָ��ͬһ����Ĺ���)

		DeviceBufferArray<bool> markValidPairs;						// �����Ч�ĵ��
		Logger TensorRTLogger;										// Tensor������־

		std::shared_ptr<nvinfer1::IExecutionContext> ExecutionContext;	// ִ�������Ķ��󣺺���ģ���������
		std::shared_ptr<nvinfer1::ICudaEngine> CudaInferEngine;			// cuda��������

		std::map<std::string, int> TensorName2Index;	// TensorRT 10.3 �� nvinfer1::ICudaEngine::getBindingIndex����û���ˣ��Լ�����ӳ���ϵ


		cudaTextureObject_t PreviousForeground;	// ǰһ֡��(���ú�)ǰ��Mask
		cudaTextureObject_t CurrentForeground;	// ��ǰ֡��(���ú�)ǰ��Mask
		cudaTextureObject_t PreviousVertexMap;	// ǰһ֡��(���ú�)VertexMap
		cudaTextureObject_t CurrentVertexMap;	// ��ǰ֡��(���ú�)VertexMap
		cudaTextureObject_t PreviousNormalMap;	// ǰһ֡��(���ú�)NormalMap
		cudaTextureObject_t CurrentNormalMap;	// ��ǰ֡��(���ú�)NormalMap
		cudaTextureObject_t CurrentEdgeMaskMap;	// ��ǰ֡(���ú�)��Ե���ֵ�MaskMap

		mat34 InitialCameraSE3;

		void* IOBuffers[5];
		DeviceArray<float> InputPreviousImage;
		DeviceArray<float> InputPreviousDepth;
		DeviceArray<float> InputCurrentImage;
		DeviceArray<float> InputCurrentDepth;

		DeviceArray<float> Flow2D;								// ����
		DeviceArray2D<ushort4> PixelPairsMap;					// ƥ���Map


		int PreviousImageInputIndex;
		int PreviousDepthInputIndex;
		int CurrentImageInputIndex;
		int CurrentDepthInputIndex;

		int FlowOutputIndex;

		const int PreviousImageSize = 1 * 3 * FRAME_HEIGHT * FRAME_WIDTH;
		const int PreviousDepthSize = 1 * 1 * FRAME_HEIGHT * FRAME_WIDTH;
		const int CurrentImageSize = 1 * 3 * FRAME_HEIGHT * FRAME_WIDTH;
		const int CurrentDepthSize = 1 * 1 * FRAME_HEIGHT * FRAME_WIDTH;

		const int FlowSize = 1 * 2 * FRAME_HEIGHT * FRAME_WIDTH;

		const unsigned int ImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int ImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageSize = FRAME_HEIGHT * FRAME_WIDTH;
		const unsigned int clipedImageSize = ImageRowsCliped * ImageColsCliped;

#ifdef DRAW_OPTICALFLOW
		float* FlowPtrDevice = NULL;			// �����ӻ�����������
		float4* FlowVector3D = NULL;			// �����ӻ���������ά����, w��ʾǿ��(͸����)
		float3* FlowVectorOpenGL = NULL;		// �����ӻ���������ά������OpenGL��ʾ
		bool* markValidFlow = NULL;				// �����ӻ��������Ч�Ĺ���
		float3* validFlowVector = NULL;			// �����ӻ����ռ���Ч�Ĺ���
		ColorVertex* colorVertexPtr = NULL;		// �����ӻ�������ɫ�Ķ���
		ColorVertex* validColorVertex = NULL;	// �����ӻ�����Ч�Ĵ���ɫ�Ķ���
		int validFlowNum = 0;					// �����ӻ�����Ч�Ĺ�������
		int validVertexNum = 0;					// �����ӻ�����Ч�Ķ�������
		DynamicallyDrawOpticalFlow::Ptr draw;	// �����ӻ������ƹ����ķ���

	public:
		/**
		 * \brief ʹ��OpenGL���ƹ���.
		 *
		 * \param stream
		 */
		void drawOpticalFlowOpenGL(cudaStream_t stream) {
			draw->imshow(validFlowVector, validColorVertex, validFlowNum, stream);
		}

		float3* GetValidOpticalFlow(int& validFlowCount) {
			validFlowCount = validFlowNum;
			return validFlowVector;
		}

		ColorVertex* GetValidColorVertex() {
			return validColorVertex;
		}

		/**
		 * \brief �������ص�ƥ���.
		 *
		 * \param stream cuda��ID
		 */
		void CalculatePixelPairAnd3DOpticalFlow(cudaStream_t stream);

#endif // DRAW_OPTICALFLOW


	private:

		/**
		 * \brief �����л�.engine�ļ��������� cuda ��������.
		 *
		 * \param Path .engine�ļ�
		 * \param runtime �����ִ��������������
		 * \return cuda ��������
		 */
		std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime);

		/**
		 * \brief �������ú��3D����(����Ϊ��ά����).
		 * 
		 * \param clipedRows ����ͼ��ĸ�
		 * \param clipedCols ����ͼ��Ŀ�
		 * \param collect �����ڴ�
		 */
		void createCliped3DOpticalFlowTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect);

		/**
		 * \brief �������ú��3D����(����Ϊ��ά����).
		 * 
		 * \param clipedRows ����ͼ��ĸ�
		 * \param clipedCols ����ͼ��Ŀ�
		 * \param texture ֻ������
		 * \param surface ��д����
		 * \param cudaArray �����ڴ�����
		 */
		void createCliped3DOpticalFlowTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

		/**
		 * \brief �������ԭʼͼ�����ת��ģ�������Tensor.
		 *
		 * \param previousImage ��һ֡RGBͼ��
		 * \param previousDepth ��һ֡���ͼ��
		 * \param currentImage ��ǰ֡RGBͼ��
		 * \param currentDepth ��ǰ֡���ͼ��
		 * \param stream cuda��ID
		 */
		void ConvertBuffer2InputTensor(DeviceArray<uchar3>& previousImage, DeviceArray<unsigned short>& previousDepth, DeviceArray<uchar3>& currentImage, DeviceArray<unsigned short>& currentDepth, cudaStream_t stream);
		
		/**
		 * \brief ���ò���Tensor.
		 *
		 * \param tensorName ���õ���������
		 * \param tensorIndex ����������Index
		 * \param isInput �Ƿ�����������
		 */
		void SetAndBindTensor(const std::string tensorName, const int tensorIndex, bool isInput = false);

		/**
		 * \brief �������ص�ƥ���.
		 *
		 * \param stream cuda��ID
		 */
		void CalculatePixelPairs(cudaStream_t stream);


	public:
		/**
		 * \brief �����Чƥ����.
		 *
		 * \return ƥ����
		 */
		DeviceArray<ushort4> getCorrespondencePixelPair() const { return validPixelPairs.Array(); }

		/**
		 * \brief ��ñ�Ե�ϵ�ƥ���.
		 * 
		 * \return ��Ե�ϵ�ƥ���
		 */
		DeviceArrayView2D<ushort4> getCorrespondencePairsMap() const { return DeviceArrayView2D<ushort4>(PixelPairsMap); }

	};

}

