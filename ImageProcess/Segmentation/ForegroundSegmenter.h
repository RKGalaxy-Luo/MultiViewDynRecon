/*****************************************************************//**
 * \file   ForegroundSegmenter.h
 * \brief  前景分割器
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
		 * \brief 交换R通道和B通道，并且归一化UChar的核函数.
		 * 
		 * \param UCharInputArray 传入原始的0-255的RGB类型图像
		 * \param NormalizedInputArray 传出归一化并交换了R通道和B通道的
		 * \param rawRows 原始图像的高
		 * \param rawCols 原始图像的宽
		 */
		__global__ void SwapChannelAndNormalizeValueKernel(PtrSize<uchar3> UCharInputArray, PtrSize<float> NormalizedInputArray, const unsigned int rawSize);

		/**
		 * \brief 收集并剪裁Mask的核函数.
		 * 
		 * \param MaskArray 传入未经剪裁的Mask
		 * \param threshold 获得的0-1范围的pha数值，当其大于threshold认为是前景，将Mask对应的纹理点置为1
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 * \param MaskSurface 将前景Mask数据存入cudaSurfaceObject_t(可映射到cudaTextureObject_t)
		 */
		__global__ void CollectAndClipMaskKernel(PtrSize<float> MaskArray, float threshold, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t rawForeground);

		/**
		 * \brief 对原始前景做腐蚀处理（窗口为正方形），去除mask边缘异常点.
		 */
		__global__ void ErodeRawForeground(cudaTextureObject_t rawForeground, const int erodeRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t MaskSurface);

		/**
		 * \brief 对传入的mask进行膨胀操作（窗口为正方形）.
		 */
		__global__ void DilateRawForeground(cudaTextureObject_t rawMask, const int dilateRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t dilateMaskSurface);

		/**
		 * \brief 对原始前景做腐蚀处理（窗口为圆），去除mask边缘异常点.
		 */
		__global__ void ErodeRawForeground(cudaTextureObject_t rawForeground, const int erodeRadius, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t MaskSurface);

		/**
		 * \brief 对传入的mask进行膨胀操作（窗口为圆）.
		 */
		__global__ void DilateRawForeground(cudaTextureObject_t rawMask, const int dilateRadius, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t dilateMaskSurface);

		/**
		 * \brief 对原始的clipedMask(不进行形态学处理)进行Mask.
		 */
		__global__ void ForegroundEdgeMask(cudaTextureObject_t clipedMask, const int edgeThickness, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t edgeMask);

		/**
		 * \brief 收集前一帧Mask的核函数.
		 * 
		 * \param mask 上一帧的mask
		 * \param previousMask 收集上一帧的mask
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 */
		__global__ void CollectPreviousForegroundMaskKernel(cudaTextureObject_t mask, cudaSurfaceObject_t previousMask, const unsigned int clipedRows, const unsigned int clipedCols);
	
		/**
		 * \brief 前景Mask滤波.
		 */
		__global__ void filterForegroundMaskKernel(cudaTextureObject_t foregroundMask, unsigned maskRows, unsigned maskCols, const float sigma, cudaSurfaceObject_t filteredMask);
	}

	class ForegroundSegmenter
	{
	public:
		using Ptr = std::shared_ptr<ForegroundSegmenter>;
		/**
		 * \brief 构造函数，传入是否需要将onnx文件转成engine文件.
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
		 * \brief 设置并绑定Tensor.
		 * 
		 * \param tensorName 设置的张量名称
		 * \param tensorIndex 设置张量的Index
		 * \param isInput 是否是输入张量
		 */
		void SetAndBindTensor(const std::string tensorName, const int tensorIndex, bool isInput = false);

		/**
		 * \brief 将Onnx模型转成tensorRT Engine模型.
		 *
		 * \param FilePath Onnx文件路径
		 * \param SavePath 保存的Engine路径
		 * \return 是否保存成功
		 */
		bool SaveOnnxModel2EngineModel(const std::string& FilePath, const std::string& SavePath);


		/**
		 * \brief 推导得出Mask.
		 *
		 * \param cuda流 CUDA流ID
		 * \param colorImages 传入Device中存储的图像信息
		 */
		void InferMask(cudaStream_t stream, DeviceArray<uchar3>& colorImages);

		/**
		 * \brief 流同步并且转换为纹理内存.
		 * 
		 * \param stream cuda流ID
		 */
		void SyncAndConvert2Texture(cudaStream_t stream);

		/**
		 * \brief 获得(剪裁后)前景Texture.
		 * 
		 * \return 获得前景的纹理
		 */
		cudaTextureObject_t getClipedMaskTexture();

		/**
		 * \brief 获得(滤波后)的前景Mask.
		 * 
		 * \return 滤波后纹理
		 */
		cudaTextureObject_t getFilteredMaskTexture();

		/**
		 * \brief 获得上一帧(剪裁后)前景Texture.
		 *
		 * \return 获得前景的纹理
		 */
		cudaTextureObject_t getPreviousMaskTexture();

		/**
		 * \brief 获得前景的边缘Mask.
		 * 
		 * \return 
		 */
		cudaTextureObject_t getForegroundEdgeMaskTexture();

		/**
		 * \brief 获得原始的Foreground.
		 * 
		 * \return 原始的Foreground
		 */
		cudaTextureObject_t getRawClipedForegroundTexture();
	private:

		Logger TensorRTLogger;			// Tensor错误日志

		std::shared_ptr<nvinfer1::IExecutionContext> ExecutionContext;	// 执行上下文对象：后续模型推理操作
		std::shared_ptr<nvinfer1::ICudaEngine> CudaInferEngine;			// cuda推理引擎

		std::map<std::string, int> TensorName2Index;	// TensorRT 10.3 中 nvinfer1::ICudaEngine::getBindingIndex函数没有了，自己构建映射关系

		CudaTextureSurface PreviousMask;		// 前一帧的MASK
		CudaTextureSurface RawMask;				// 原始的Mask
		CudaTextureSurface ClipedMask;			// 剪裁并完成形态学操作的MASK
		CudaTextureSurface FilteredMask;		// 滤波后的前景
		CudaTextureSurface ClipedEdgeMask;		// 仅裁剪Raw边缘Mask

		DeviceArray<float> InputSrc;			// 输入图像RGB转成BGR，并归一化，转成Tensor
		DeviceArray<float> MaskDeviceArray;		// 输出获得的Mask图像

		const int erodeRadius = 2;				// 形态学腐蚀半径
		const int dilateRadius = 7;				// 形态学膨胀半径
		const int edgeThickness = 2;			// Mask的边缘半径

		int srcInputIndex;		// 原图输入索引
		int r1InputIndex;		// r1i输入索引
		int r2InputIndex;		// r2i输入索引
		int r3InputIndex;		// r3i输入索引
		int r4InputIndex;		// r4i输入索引

		int FgrOutputIndex;		// fgr输出的索引
		int PhaOutputIndex;		// pha输出的索引
		int r1OutputIndex;		// r1o输入索引
		int r2OutputIndex;		// r2o输入索引
		int r3OutputIndex;		// r3o输入索引
		int r4OutputIndex;		// r4o输入索引

		void* IOBuffers[11];	// 获得推理的输入输出的buffer

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
		 * \brief 创建剪裁后的Mask纹理和表面，将纹理和表面相互映射.
		 * 
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 * \param collect 纹理表面
		 */
		void createClipedMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect);

		/**
		 * \brief 创建剪裁后的Mask纹理和表面，将纹理和表面相互映射.
		 * 
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 * \param texture 纹理(GPU内存地址)
		 * \param surface 表面(GPU内存地址)
		 * \param cudaArray 数据(存储在GPU)
		 */
		void createClipedMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

		/**
		 * \brief 创建剪裁后的Mask纹理和表面，将纹理和表面相互映射.
		 *
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 * \param collect 纹理表面
		 */
		void createClipedFilteredMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect);


		/**
		 * \brief 创建剪裁后的FilteredMask纹理和表面，将纹理和表面相互映射.
		 *
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 * \param texture 纹理(GPU内存地址)
		 * \param surface 表面(GPU内存地址)
		 * \param cudaArray 数据(存储在GPU)
		 */
		void createClipedFilteredMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

		/**
		 * \brief 传入ICudaEngine，并将ONNX存成engine文件.
		 *
		 * \param path 将模型存入该地址
		 * \param engine onnx模型涉及的cuda引擎
		 */
		void SerializeEngine(const std::string& path, std::shared_ptr<nvinfer1::ICudaEngine> engine);

		/**
		 * \brief 反序列化.engine文件，并构建 cuda 推理引擎.
		 *
		 * \param Path .engine文件
		 * \param runtime 管理和执行推理任务引擎
		 * \return cuda 推理引擎
		 */
		std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const std::string& Path, nvinfer1::IRuntime* runtime);

		/**
		 * \brief 将RGB转成BGR，并归一化成输入infer的Tensor.
		 * 
		 * \param colorImages 原始的图像
		 * \param normalizedSrc 交换通道并归一化的src
		 * \param stream CUDA流ID
		 */
		void SwapChannelAndNormalizeValue(DeviceArray<uchar3>& colorImages, DeviceArray<float>& normalizedSrc, cudaStream_t stream);

		/**
		 * \brief 将收集上一帧的纹理.
		 * 
		 * \param mask 上一帧的Texture
		 * \param previousMask 存储上一帧的Texture
		 * \param clipedRows 图片的高
		 * \param clipedCols 图片的宽
		 * \param stream cuda流id
		 */
		void CollectPreviousForegroundMask(cudaTextureObject_t mask, cudaSurfaceObject_t previousMask, const unsigned int clipedRows, const unsigned int clipedCols, cudaStream_t stream);

		/**
		 * \brief 收集并剪裁Mask.
		 * 
		 * \param rawMask 原始大小的Mask
		 * \param clipedRows 剪裁后的高
		 * \param clipedCols 剪裁后的宽
		 * \param erodeRadius 腐蚀操作半径
		 * \param maskSurface 将剪裁后的mask存成纹理表面类型
		 * \param stream cuda流ID
		 */
		void CollectAndClipMask(const unsigned int clipedRows, const unsigned int clipedCols, CudaTextureSurface rawForeground, CudaTextureSurface maskSurface, cudaStream_t stream);

		/**
		 * \brief 对前景进行滤波.
		 * 
		 * \param foregroundMask 前景
		 * \param clipedRows 剪裁后的行
		 * \param clipedCols 剪裁后的列
		 * \param sigma 滤波参数sigma
		 * \param filteredMask 滤波后的Mask
		 * \param stream cuda流
		 */
		void FilterForefroundMask(cudaTextureObject_t foregroundMask, const unsigned int clipedRows, const unsigned int clipedCols, float sigma, cudaSurfaceObject_t filteredMask, cudaStream_t stream = 0);
	};

}

