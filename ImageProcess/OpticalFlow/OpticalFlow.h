/*****************************************************************//**
 * \file   OpticalFlow.h
 * \brief  推理光流模型
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
		 * \brief 求窗口内深度图的极差.
		 */
		__forceinline__ __device__ float windowRange(const unsigned short& x, const unsigned short& y, const short& windowRadius, const unsigned int& cols, const unsigned int& rows, cudaTextureObject_t vertexMap);

		/**
		 * \brief 将传入的原始图像及深度转成模型输入的Tensor.
		 * 
		 * \param previousImage 上一帧RGB图像
		 * \param previousDepth 上一帧深度图像
		 * \param currentImage 当前帧RGB图像
		 * \param currentDepth 当前帧深度图像
		 * \param rawImageSize 原始图像的大小
		 * \param inputPreviousImage 输入模型的上一帧RGB图像
		 * \param inputPreviousDepth 输入模型的上一帧深度图像
		 * \param inputCurrentImage 输入模型的当前帧RGB图像
		 * \param inputCurrentDepth 输入模型的当前帧深度图像
		 */
		__global__ void ConvertBuffer2InputTensorKernel(const uchar3* previousImage, const unsigned short* previousDepth, const uchar3* currentImage, const unsigned short* currentDepth, const unsigned int rawImageSize, float* inputPreviousImage, float* inputPreviousDepth, float* inputCurrentImage, float* inputCurrentDepth);

		/**
		 * \brief 计算像素匹配对以及三维光流向量，不论光流还是匹配点对，均只取前景.
		 * 
		 * \param preForeground 前一帧的(剪裁后)前景Mask
		 * \param currForeground 当前帧的(剪裁后)前景Mask
		 * \param PreviousVertexMap 前一帧的(剪裁后)的VertexConfidenceMap
		 * \param CurrentVertexMap 当前帧的(剪裁后)的VertexConfidenceMap
		 * \param clipedImageRows 剪裁后图像高
		 * \param clipedImageCols 剪裁后图像宽
		 * \param pixelPair 匹配像素点对
		 * \param markValidPair 标记有效的光流点
		 * \param flow3d 三维光流向量
		 */
		__global__ void CalculatePixelPairAnd3DOpticalFlowKernel(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t PreviousVertexMap, cudaTextureObject_t CurrentVertexMap, const float* flow2d, const unsigned int clipedImageRows, const unsigned int clipedImageCols, const unsigned int rawImageSize, ushort4* pixelPair, bool* markValidPair, float4* FlowVector3D, float3* FlowVectorOpenGL, ColorVertex* colorVertex, bool* markValidFlow);

		/**
		 * \brief 计算像素匹配对以及三维光流向量，不论光流还是匹配点对，均只取前景.
		 *
		 * \param preForeground 前一帧的(剪裁后)前景Mask
		 * \param currForeground 当前帧的(剪裁后)前景Mask
		 * \param PreviousVertexMap 前一帧的(剪裁后)的VertexConfidenceMap
		 * \param CurrentVertexMap 当前帧的(剪裁后)的VertexConfidenceMap
		 * \param PreviousNormalMap 前一帧的(剪裁后)的NormalRadiusMap
		 * \param CurrentNormalMap 当前帧的(剪裁后)的NormalRadiusMap
		 * \param initialCameraSE3 当前相机的位姿
		 * \param flow2d 推理得到的二维光流
		 * \param clipedImageRows 剪裁后图像高
		 * \param clipedImageCols 剪裁后图像宽
		 * \param pixelPair 匹配像素点对
		 * \param markValidPair 标记有效的光流点
		 */
		__global__ void CalculatePixelPairsKernel(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t PreviousVertexMap, cudaTextureObject_t CurrentVertexMap, cudaTextureObject_t preNormalMap, cudaTextureObject_t currNormalMap, const mat34 initialCameraSE3, const float* flow2d, const unsigned int clipedImageRows, const unsigned int clipedImageCols, const unsigned int rawImageSize, ushort4* pixelPair, bool* markValidPair, PtrStepSize<ushort4> validEdgePixelPairs);

		/**
		 * \brief 获得边缘的有效匹配点.
		 */
		__global__ void GetValidEdgePixelPairsKernel(cudaTextureObject_t currEdgeMaskMap, const ushort4* pixelPair, const bool* markValidPixelPairs, const unsigned int clipedImageRows, const unsigned int clipedImageCols, PtrStepSize<ushort4> validEdgePixelPairs);

		__global__ void LabelSortedCurrPairEncodedCoor(const PtrSize<const unsigned int> sortedCoor, unsigned int* label);

		__global__ void CompactDenseCorrespondence(const PtrSize<ushort4> sortedPairs, const unsigned int* pairsLabel, const unsigned int* prefixSum, ushort4* compactedCorrPairs);

		/**
		 * \brief 校正光流Map.
		 * 
		 * \param markValidFlowSe3Map 标记进行了光流变换的vertex
		 * \param correctSe3 校准矩阵
		 * \param clipedImageRows 剪裁后图像高
		 * \param clipedImageCols 剪裁后图像宽
		 * \param vertexFlowSe3 校正后的光流Map
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
		 * \brief 分配推理内存.
		 *
		 */
		void AllocateInferBuffer();

		/**
		 * \brief 释放内存.
		 *
		 */
		void ReleaseInferBuffer();

		/**
		 * \brief 设置前一帧和当前帧的前景以及前一帧和当前帧的顶点图.
		 * 
		 * \param preForeground 前一帧前景Mask
		 * \param currForeground 当前帧前景Mask
		 * \param PreviousVertexMap 前一帧的(剪裁后)的VertexConfidenceMap
		 * \param CurrentVertexMap 当前帧的(剪裁后)的VertexConfidenceMap
		 * \param preNormalMap 上一帧法线Map
		 * \param currNormalMap 当前帧法线Map
		 * \param currEdgeMask 当前帧边缘的mask
		 * \param initialCameraSE3 相机位姿
		 */
		void setForegroundAndVertexMapTexture(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t preVertexMap, cudaTextureObject_t currVertexMap, cudaTextureObject_t preNormalMap, cudaTextureObject_t currNormalMap, cudaTextureObject_t currEdgeMask, const mat34 initialCameraSE3);

		/**
		 * \brief 推理光流模型.
		 * 
		 * \param previousImage 前一帧图像
		 * \param previousDepth 前一帧深度
		 * \param currentImage 当前帧图像
		 * \param currentDepth 当前帧深度
		 * \param stream cuda流
		 */
		void InferOpticalFlow(DeviceArray<uchar3>& previousImage, DeviceArray<unsigned short>& previousDepth, DeviceArray<uchar3>& currentImage, DeviceArray<unsigned short>& currentDepth, cudaStream_t stream);

	private:
		DeviceBufferArray<ushort4> correspondencePixelPair;			// 关联像素对
		DeviceBufferArray<ushort4> validPixelPairs;					// 选择有效的像素点对(满足：1、前后两帧的像素点均在前景上；2、两个匹配的像素点均不存在深度为0的情况；3、两个匹配的像素点光流向量的模 < 指定阈值；4、剔除指向同一个点的光流)

		DeviceBufferArray<bool> markValidPairs;						// 标记有效的点对
		Logger TensorRTLogger;										// Tensor错误日志

		std::shared_ptr<nvinfer1::IExecutionContext> ExecutionContext;	// 执行上下文对象：后续模型推理操作
		std::shared_ptr<nvinfer1::ICudaEngine> CudaInferEngine;			// cuda推理引擎

		std::map<std::string, int> TensorName2Index;	// TensorRT 10.3 中 nvinfer1::ICudaEngine::getBindingIndex函数没有了，自己构建映射关系


		cudaTextureObject_t PreviousForeground;	// 前一帧的(剪裁后)前景Mask
		cudaTextureObject_t CurrentForeground;	// 当前帧的(剪裁后)前景Mask
		cudaTextureObject_t PreviousVertexMap;	// 前一帧的(剪裁后)VertexMap
		cudaTextureObject_t CurrentVertexMap;	// 当前帧的(剪裁后)VertexMap
		cudaTextureObject_t PreviousNormalMap;	// 前一帧的(剪裁后)NormalMap
		cudaTextureObject_t CurrentNormalMap;	// 当前帧的(剪裁后)NormalMap
		cudaTextureObject_t CurrentEdgeMaskMap;	// 当前帧(剪裁后)边缘部分的MaskMap

		mat34 InitialCameraSE3;

		void* IOBuffers[5];
		DeviceArray<float> InputPreviousImage;
		DeviceArray<float> InputPreviousDepth;
		DeviceArray<float> InputCurrentImage;
		DeviceArray<float> InputCurrentDepth;

		DeviceArray<float> Flow2D;								// 光流
		DeviceArray2D<ushort4> PixelPairsMap;					// 匹配点Map


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
		float* FlowPtrDevice = NULL;			// 【可视化】光流像素
		float4* FlowVector3D = NULL;			// 【可视化】光流三维向量, w表示强度(透明度)
		float3* FlowVectorOpenGL = NULL;		// 【可视化】光流三维向量，OpenGL显示
		bool* markValidFlow = NULL;				// 【可视化】标记有效的光流
		float3* validFlowVector = NULL;			// 【可视化】收集有效的光流
		ColorVertex* colorVertexPtr = NULL;		// 【可视化】带颜色的顶点
		ColorVertex* validColorVertex = NULL;	// 【可视化】有效的带颜色的顶点
		int validFlowNum = 0;					// 【可视化】有效的光流数量
		int validVertexNum = 0;					// 【可视化】有效的顶点数量
		DynamicallyDrawOpticalFlow::Ptr draw;	// 【可视化】绘制光流的方法

	public:
		/**
		 * \brief 使用OpenGL绘制光流.
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
		 * \brief 计算像素点匹配对.
		 *
		 * \param stream cuda流ID
		 */
		void CalculatePixelPairAnd3DOpticalFlow(cudaStream_t stream);

#endif // DRAW_OPTICALFLOW


	private:

		/**
		 * \brief 反序列化.engine文件，并构建 cuda 推理引擎.
		 *
		 * \param Path .engine文件
		 * \param runtime 管理和执行推理任务引擎
		 * \return cuda 推理引擎
		 */
		std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime);

		/**
		 * \brief 创建剪裁后的3D光流(光流为三维向量).
		 * 
		 * \param clipedRows 剪裁图像的高
		 * \param clipedCols 剪裁图像的宽
		 * \param collect 纹理内存
		 */
		void createCliped3DOpticalFlowTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect);

		/**
		 * \brief 创建剪裁后的3D光流(光流为三维向量).
		 * 
		 * \param clipedRows 剪裁图像的高
		 * \param clipedCols 剪裁图像的宽
		 * \param texture 只读纹理
		 * \param surface 读写纹理
		 * \param cudaArray 纹理内存数组
		 */
		void createCliped3DOpticalFlowTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

		/**
		 * \brief 将传入的原始图像及深度转成模型输入的Tensor.
		 *
		 * \param previousImage 上一帧RGB图像
		 * \param previousDepth 上一帧深度图像
		 * \param currentImage 当前帧RGB图像
		 * \param currentDepth 当前帧深度图像
		 * \param stream cuda流ID
		 */
		void ConvertBuffer2InputTensor(DeviceArray<uchar3>& previousImage, DeviceArray<unsigned short>& previousDepth, DeviceArray<uchar3>& currentImage, DeviceArray<unsigned short>& currentDepth, cudaStream_t stream);
		
		/**
		 * \brief 设置并绑定Tensor.
		 *
		 * \param tensorName 设置的张量名称
		 * \param tensorIndex 设置张量的Index
		 * \param isInput 是否是输入张量
		 */
		void SetAndBindTensor(const std::string tensorName, const int tensorIndex, bool isInput = false);

		/**
		 * \brief 计算像素点匹配对.
		 *
		 * \param stream cuda流ID
		 */
		void CalculatePixelPairs(cudaStream_t stream);


	public:
		/**
		 * \brief 获得有效匹配点对.
		 *
		 * \return 匹配点对
		 */
		DeviceArray<ushort4> getCorrespondencePixelPair() const { return validPixelPairs.Array(); }

		/**
		 * \brief 获得边缘上的匹配点.
		 * 
		 * \return 边缘上的匹配点
		 */
		DeviceArrayView2D<ushort4> getCorrespondencePairsMap() const { return DeviceArrayView2D<ushort4>(PixelPairsMap); }

	};

}

