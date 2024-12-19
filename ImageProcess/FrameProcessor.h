/*****************************************************************//**
 * \file   FrameProcessor.h
 * \brief  这个类主要是处理一些图像与GPU交互以及图像预处理的相关操作
 * 
 * \author LUO
 * \date   January 24th 2024
 *********************************************************************/
#pragma once
#include <atomic> 
#include <thread>

#include <opencv2/opencv.hpp>
#include <base/CommonTypes.h>
#include <base/CommonUtils.h>
#include <base/ConfigParser.h>
#include <base/CameraObservation.h>
#include <base/SolverMethodConstants.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

#include <ImageProcess/ImageProcessByGPU.h>
#include <ImageProcess/FrameIO/GetFrameFromCamera.h>
#include <ImageProcess/FrameIO/GetImageFromLocal.h>
#include <ImageProcess/Segmentation/ForegroundSegmenter.h>
#include <ImageProcess/OpticalFlow/OpticalFlow.h>
#include <ImageProcess/GlobalPatchCollider/PatchColliderRGBCorrespondence.h>
#include <ImageProcess/Gradient/ImageGradient.h>
#include <core/AlgorithmTypes.h>
#include <core/RigidSolver/RigidSolver.h>
#include <core/Geometry/SurfelsProcessor.h>
#include <core/MergeSurface/MergeSurface.h>
#include <core/CrossViewEdgeCorrespondence/CrossViewEdgeCorrespondence.h>
#include <core/CrossViewEdgeCorrespondence/CrossViewMatchingInterpolation.h>
#include <render/Renderer.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <visualization/Visualizer.h>
#include <base/data_transfer.h>


#define A_IMAGEPROCESSOR_NEED_CUDA_STREAM 4		// 每一个图片最多需要流的数量

#define USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
//#define WITH_NODE_CORRECTION

namespace SparseSurfelFusion {
	class FrameProcessor
	{
	public:
		using Ptr = std::shared_ptr<FrameProcessor>;	// FrameProcessor类的智能指针(shared_ptr)
		/**
		 * \brief 显式声明构造函数，构造函数不能隐式调用.
		 * 
		 */
		explicit FrameProcessor(std::shared_ptr<ThreadPool> threadPool);

		/**
		 * \brief 析构函数，释放内存.
		 * 
		 */
		~FrameProcessor();
/*********************************************     相机调整     *********************************************/
	private:

		GetFrameFromCamera::Ptr frame;		// 获得相机帧
		ConfigParser::Ptr configParser;		// 相机及图像基本参数
		std::shared_ptr<ThreadPool> pool;

		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];
		std::atomic<int> numThreadsCompleted;  // 完成的线程数量

		//直接把原始深度面元映射到这三个里，另外三个直接写入融合后的深度面源映射。
		CudaTextureSurface icp_vertex_confidence[MAX_CAMERA_COUNT];
		CudaTextureSurface icp_normal_radius[MAX_CAMERA_COUNT];
		CudaTextureSurface icp_color_time[MAX_CAMERA_COUNT];

		unsigned int beginIndex = 0;

	public:
		/**
		 * \brief 获取是否所有相机都已准备完毕.
		 * 
		 * \return 所有相机都已准备完毕返回true
		 */
		bool CamerasReady();

		/**
		 * \brief 显示所有图像RGB和Depth.
		 * 
		 */
		void showAllImages();

		/**
		 * \brief 获取当前编号相机的彩色图.
		 * 
		 * \param CameraID 相机编号
		 * \return 返回当前编号的相机的彩色图
		 */
		cv::Mat getCurrentColorImage(int CameraID);

		/**
		 * \brief 获取当前编号相机的深度图.
		 * 
		 * \param CameraID 相机编号
		 * \return 返回当前编号相机的深度图
		 */
		cv::Mat getCurrentDepthImage(int CameraID);

		/**
		 * \brief 返回是否结束Cameras.
		 * 
		 * \return true是关闭了Camera
		 */
		bool isStop() {
			return !frame->ContinueCapturingCamera();
		}

		/**
		 * \brief 停止所有相机.
		 * 
		 */
		void StopAllCameras() {
			frame->StopIt();
		}

		void SaveData(size_t frameIdx) {
			frame->setFrameIndex(frameIdx);
		}

		/**
		 * \brief 获取接入相机的个数.
		 * 
		 * \return 
		 */
		int getCameraCount() {
			return deviceCount;
		}

		ConfigParser::Ptr getCameraConfigParser() {
			return configParser;
		}

		/**
		 * \brief 获得对应相机RGB的窗口名字.
		 * 
		 * \param CameraID 相机ID
		 * \return RGB窗口名字
		 */
		cv::String GetColorWindowsName(int CameraID) {
			return frame->GetColorWindowName(CameraID);
		}

		/**
		 * \brief 获得对应相机深度图的窗口名字.
		 * 
		 * \param CameraID 相机ID
		 * \return Depth窗口名字
		 */
		cv::String GetDepthWindowsName(int CameraID) {
			return frame->GetDepthWindowName(CameraID);
		}

		/**
		 * \brief 设置算法开始的index.
		 * 
		 * \param beginIdx
		 */
		void setBeginFrameIndex(const unsigned int beginIdx) {
			beginIndex = beginIdx;
		}

		/**
		 * \brief 处理当前图片的任务，每个相机单分出一个线程.
		 * 
		 * \param CameraID 相机ID
		 * \param observation 当前观察相机获得的参数收集器
		 */
		void ProcessCurrentCameraImageTask(const unsigned int CameraID, CameraObservation& observation);


		/**
		 * \brief 处理第一帧图像，并将多视角相机深度面元融合.
		 * 
		 * \param frameIndex 传入帧ID
		 * \param rigidSolver 求解相机的初始相对位置
		 * \return 返回融合后的面元
		 */
		DeviceArrayView<DepthSurfel> ProcessFirstFrame(size_t frameIndex, RigidSolver::Ptr rigidSolver);

		/**
		 * \brief 处理后续图像(非第一帧).
		 * 
		 * \param observation 传出通过相机获得并计算得到的用于对齐的参数
		 * \param frameIndex 传入帧ID
		 * \param rigidSolver 传入刚性求解器
		 * \return 返回融合后的面元
		 */
		void ProcessCurrentFrame(CameraObservation& observation, size_t frameIndex);



		/**
		 * \brief 获得剪裁后的IntrinsicArray.
		 * 
		 * \return 剪裁后的IntrinsicArray.
		 */
		Intrinsic* GetClipedIntrinsicArray() { return clipColorIntrinsic; }

/*********************************************     整体参数     *********************************************/
	public:
		const unsigned int rawImageRows = FRAME_HEIGHT;
		const unsigned int rawImageCols = FRAME_WIDTH;
		const unsigned int rawImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;

	private:
		const unsigned int clipNear = MIN_DEPTH_THRESHOLD;
		const unsigned int clipFar = MAX_DEPTH_THRESHOLD;

		int deviceCount = 0;
		size_t FrameIndex = 0;
		Intrinsic rawColorIntrinsic[MAX_CAMERA_COUNT];
		Intrinsic rawDepthIntrinsic[MAX_CAMERA_COUNT];
		Intrinsic clipColorIntrinsic[MAX_CAMERA_COUNT];

		cv::Mat depthImage[MAX_CAMERA_COUNT];						// 传入深度图像
		cv::Mat colorImage[MAX_CAMERA_COUNT];						// 传入RGB图像
		cv::Mat previousDepthImage[MAX_CAMERA_COUNT];				// 前一帧深度图像
		cv::Mat previousColorImage[MAX_CAMERA_COUNT];				// 前一帧RGB图像

		std::vector<cv::Mat>* ColorOfflineData;
		std::vector<cv::Mat>* DepthOfflineData;
		std::vector<cv::String> ColorOfflinePath[MAX_CAMERA_COUNT];
		std::vector<cv::String> DepthOfflinePath[MAX_CAMERA_COUNT];

		bool intoBuffer = true;
		std::string dataSuffixPath;

		/**
		 * \brief 收集通过图像观察视角的参数.
		 * 
		 * \param observation 参数收集器
		 * \param CameraID 相机ID
		 */
		void CollectObservationParameters(cudaStream_t stream, CameraObservation& observation, const unsigned int CameraID);
		void CollectMergeObservationParameters(CameraObservation& observation);
/*********************************************     流模式处理     *********************************************/

	private:
		// cuda处理的流个数
		cudaStream_t ProcessorStream[A_IMAGEPROCESSOR_NEED_CUDA_STREAM * MAX_CAMERA_COUNT];
		const unsigned int TotalProcessingStreams = A_IMAGEPROCESSOR_NEED_CUDA_STREAM * MAX_CAMERA_COUNT;
		/**
		 * \brief 初始化cuda处理的流.
		 */
		void initProcessorStream();
		/**
		 * \brief 释放处理的流.
		 * 
		 */
		void releaseProcessorStream();
		/**
		 * \brief 同步所有的流.
		 * 
		 */
		void syncAllProcessorStream();


/*********************************************     图像获取     *********************************************/
	private:

		/**
		 * \brief 预先分配获取图像的锁页内存.
		 */
		void allocateImageHostBuffer();

		/**
		 * \brief 释放锁页内存.
		 * 
		 */
		void releaseImageHostBuffer();
	public:
		
		void GetOfflineData(std::vector<cv::Mat>* ColorOffline, std::vector<cv::Mat>* DepthOffline, std::string type, bool intoBuffer = true);

		/**
		 * \brief 获取相机RGB和深度图片信息，前一帧RGB图片.
		 * 
		 * \param frameIndex 第几帧
		 * \param CameraID 相机ID
		 */
		void FetchFrame(size_t frameIndex, const unsigned int CameraID);

		/**
		 * \brief 获得原始不同视角Mat图片.
		 * 
		 * \return 原始不同视角Mat图片数组
		 */
		cv::Mat* GetRawImageMatArray() { return colorImage; }


/*********************************************     深度图     *********************************************/

	public:
		/**
		 * \brief 上传深度图片到GPU中.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 摄像机ID，每一个相机的Depth图片均要一个流进行处理
		 */
		void UploadDepthImageToGPU(cudaStream_t stream, int CameraID);

		/**
		 * \brief 在.cu文件中实现深度图裁剪函数，具体函数实现在ImageProcessorByGPU.cu中.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 摄像机ID，每一个相机的Depth图片均要一个流进行处理
		 */
		void ClipFilterDepthImage(cudaStream_t stream, int CameraID);

		/**
		 * \brief 获得原始深度纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 原始深度纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getRawDepthCUDATexture(int CameraID) const { return rawDepthCUDATexture[CameraID].texture; }

		/**
		 * \brief 获得剪裁后的深度纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 剪裁后的深度纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getClipedDepthCUDATexture(int CameraID) const { return clipedDepthCUDATexture[CameraID].texture; }

	private:
		void* depthBufferPagelock[MAX_CAMERA_COUNT];									// 深度图像锁页内存
		void* depthBufferPagelockPrevious[MAX_CAMERA_COUNT];							// 前一帧深度图像锁页内存
		DeviceArray<unsigned short> rawDepthImage[MAX_CAMERA_COUNT];					// 原始图像在GPU上的buffer
		DeviceArray<unsigned short> rawDepthImagePrevious[MAX_CAMERA_COUNT];			// 前一帧图像在GPU上的buffer
		CudaTextureSurface rawDepthCUDATexture[MAX_CAMERA_COUNT];						// 原始图片的纹理
		CudaTextureSurface clipedDepthCUDATexture[MAX_CAMERA_COUNT];					// 剪裁图片的纹理(多视角限制后)
		/**
		 * \brief 分配深度图纹理缓存.
		 * 
		 */
		void allocateDepthTexture();

		/**
		 * \brief 释放深度图纹理缓存.
		 * 
		 */
		void releaseDepthTexture();

/*********************************************     RGB图和密度(灰度)图     *********************************************/
	// 需要注意的是密度(灰度)图GrayScaleImage是用来在离线情况下寻找前景ForeGround
	public:
		/**
		 * \brief 上传RGB图像到GPU缓存中.
		 *
		 * \param stream cuda流ID
		 */
		void UploadRawColorImageToGPU(cudaStream_t stream, int CameraID);

		/**
		 * \brief 剪裁并归一化RGB图像，具体函数实现在ImageProcessorByGPU.cu中.
		 *
		 * \param stream cuda流ID
		 * \param CameraID 摄像机ID，每个相机的RGB图片需要一个流来处理
		 */
		void ClipNormalizeColorImage(cudaStream_t stream, int CameraID);

		/**
		 * \brief 获得GPU上的原始为裁剪和滤波的RGB图像.
		 * 
		 * \param CameraID 相机ID
		 * \return 原始存储在GPU上的图像数据(传出以DeviceArray<uchar3>& 类型)
		 */
		const DeviceArray<uchar3>& getRawColorImage(int CameraID) const { return rawColorImage[CameraID]; }

		/**
		 * \brief 获得剪裁后的RGB图像的纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 剪裁滤波归一化后RGB图像的纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getClipedNormalizeColorImageTexture(int CameraID) const { return clipedNormalizeColorImage[CameraID].texture; }

		/**
		 * \brief 获得剪裁滤波归一化后的RGB图像的纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 剪裁滤波归一化后的RGB图像的纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getClipedNormalizeColorImagePreviousTexture(int CameraID) const { return clipedNormalizeColorImagePrevious[CameraID].texture; }

		/**
		 * \brief 获得RGB剪裁后转成的灰度(密度)图像纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return RGB剪裁后转成的灰度(密度)图像纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getGrayScaleImageTexture(int CameraID) const { return GrayScaleImage[CameraID].texture; }

	private:
		void* colorBufferPagelock[MAX_CAMERA_COUNT];								// RGB图像锁页内存
		void* colorBufferPagelockPrevious[MAX_CAMERA_COUNT];						// 前一帧RGB图像锁页内存
		//RGB图像的buffer，用在GPU上，以flatten buffer的形式(一维)存储
		DeviceArray<uchar3> rawColorImage[MAX_CAMERA_COUNT];						// 原始图像在GPU上的buffer
		DeviceArray<uchar3> rawColorImagePrevious[MAX_CAMERA_COUNT];				// 前一帧图像在GPU上的buffer
		//裁剪和标准化的rgb图像(float4)												  
		CudaTextureSurface clipedNormalizeColorImage[MAX_CAMERA_COUNT];				// 归一化剪裁的RGB图像，存储在GPU的纹理及表面内存中
		CudaTextureSurface clipedNormalizeColorImagePrevious[MAX_CAMERA_COUNT];		// 前一帧归一化剪裁的RGB图像，存储在GPU的纹理及表面内存中
		//RGB和RGB_Previous的密度(灰度)图(float1)									    
		CudaTextureSurface GrayScaleImage[MAX_CAMERA_COUNT];						// 密度(灰度)图，存储在GPU的纹理及表面内存中
		CudaTextureSurface GrayScaleImageFiltered[MAX_CAMERA_COUNT];				// 滤波后的密度(灰度)图，存储在GPU的纹理及表面内存中

		/**
		 * \brief 分配RGB图像的缓存区域，创建RGB图像及GrayScaleImage图像在GPU中的纹理内存.
		 * 
		 */
		void allocateColorTexture();

		/**
		 * \brief 释放RGB图像的缓存区域.
		 * 
		 */
		void releaseColorTexture();

/*********************************************     面元属性     *********************************************/
	private:
		CudaTextureSurface previousVertexConfidenceTexture[MAX_CAMERA_COUNT];		// 上一帧float4纹理，(x, y, z)为顶点，w为置信度值
		CudaTextureSurface previousNormalRadiusTexture[MAX_CAMERA_COUNT];			// 上一帧float4纹理，(x, y, z)为法线，w为半径

		CudaTextureSurface vertexConfidenceTexture[MAX_CAMERA_COUNT];				// float4纹理，(x, y, z)为顶点，w为置信度值
		CudaTextureSurface normalRadiusTexture[MAX_CAMERA_COUNT];					// float4纹理，(x, y, z)为法线，w为半径

		/**
		 * \brief 分配面元属性纹理：顶点、法线、半径以及置信度.
		 * 
		 */
		void allocateSurfelAttributeTexture();

		/**
		 * \brief 释放面元属性纹理.
		 * 
		 */
		void releaseSurfelAttributeTexture();

	public:

		/**
		 * \brief 构造顶点和置信度的Map(一种抽象的float4纹理：x,y,z是顶点坐标，w是顶点置信度).
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void buildVertexConfidenceMap(cudaStream_t stream, int CameraID);

		/**
		 * \brief 构造法线和半径的Map(一种抽象的float4纹理：x,y,z是法线，w是半径).
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void buildNormalRadiusMap(cudaStream_t stream, int CameraID);

		/**
		 * \brief 返回前一帧的顶点纹理.
		 *
		 * \param CameraID 相机ID
		 * \return 前一帧的顶点纹理
		 */
		cudaTextureObject_t getPreviousVertexConfidenceTexture(int CameraID) { return previousVertexConfidenceTexture[CameraID].texture; }

		/**
		 * \brief 返回前一帧的法线纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 前一帧的法线纹理
		 */
		cudaTextureObject_t getPreviousNormalRadiusTexture(int CameraID) { return previousNormalRadiusTexture[CameraID].texture; }

	

		/**
		 * \brief 返回顶点及置信度纹理.
		 *
		 * \param CameraID 相机ID
		 * \return 顶点及置信度纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getBackgroundVertexConfidenceTexture(int CameraID) { return icp_vertex_confidence[CameraID].texture; }
		cudaTextureObject_t getVertexConfidenceTexture(int CameraID) { return vertexConfidenceTexture[CameraID].texture; }

		/**
		 * \brief 返回法线及半径纹理.
		 *
		 * \param CameraID 相机ID
		 * \return 线及半径纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getBackgroundNormalRadiusTexture(int CameraID) { return icp_normal_radius[CameraID].texture; }
		cudaTextureObject_t getNormalRadiusTexture(int CameraID) { return normalRadiusTexture[CameraID].texture; }

/*********************************************     Color及Time属性Map     *********************************************/
		/**
		 * \brief 从RGB图像计算ColorTime图，应该与面元数组格式相同float4中x是RGBA的编码，.
		 */
	private:
		CudaTextureSurface colorTimeTexture[MAX_CAMERA_COUNT];						// 面元及上一次看见这个面元的时间(第几帧看到)
		
		/**
		 * \brief 分配面元颜色及上一次看到这个面元的时间纹理缓存.
		 */
		void allocateColorTimeTexture();

		/**
		 * \brief 释放面元颜色及上一次看到这个面元的时间纹理缓存.
		 * 
		 */
		void releaseColorTimeTexture();

	public:
		/**
		 * \brief 构建ColorTime图(从RGB图像计算ColorTime图，应该与面元数组格式相同x,y,z是RGB，w是time).
		 * 
		 * \param frameIdx 当前帧
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void buildColorTimeMap(size_t frameIdx, cudaStream_t stream, int CameraID);

		/**
		 * \brief 返回Color-Time[RGB颜色及上一帧见到这个面元的时间]纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return Color-Time纹理(传出以cudaTextureObject_t类型)
		 */
		cudaTextureObject_t getBackgroundColorTimeTexture(int CameraID) { return icp_color_time[CameraID].texture; }
		cudaTextureObject_t getColorTimeTexture(int CameraID) { return colorTimeTexture[CameraID].texture; }

/*********************************************     构造并选择有效的深度面元     *********************************************/
	private:
		SurfelsProcessor::Ptr surfelsProcessor;
		//MergeSurface::Ptr mergeSurface;
		DeviceBufferArray<DepthSurfel> depthSurfel[MAX_CAMERA_COUNT];				// 有效的深度面元
		DeviceArrayView<DepthSurfel> depthSurfelView[MAX_CAMERA_COUNT];				// 只可读的depthSurfel
		FlagSelection validDepthPixelSelector[MAX_CAMERA_COUNT];					// 有效深度面元筛选器
		DeviceBufferArray<DepthSurfel> preAlignedSurfel;							// 已经完成刚性对齐的面元
		DeviceArrayView<DepthSurfel> preAlignedSurfelView;							// 只读稠密融合面元
		DeviceBufferArray<float4> subsampleSparseVertice[MAX_CAMERA_COUNT];			// 可读写的稀疏顶点



		/**
		 * \brief 分配有效面元筛选器的缓存，有效面元筛选器可以复用.
		 * 
		 */
		void allocateValidSurfelSelectionBuffer();

		/**
		 * \brief 释放有效面元筛选器的缓存.
		 * 
		 */
		void releaseValidSurfelSelectionBuffer();

	public:
		/**
		 * \brief 收集有效深度面元.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void collectValidDepthSurfel(cudaStream_t stream, int CameraID);



		/**
		 * \brief 根据第一帧计算的相机的相对位置，将深度面元转换到同一个Canonical空间.
		 * 
		 * \param solver 刚性求解器
		 * \param stream CUDA流ID
		 * \param CameraID 相机ID
		 */
		void alignedDenseValidDepthSurfel(RigidSolver::Ptr solver, cudaStream_t stream, int CameraID);

		/**
		 * \brief 获得当前相机有效面元的数组(只读数组DeviceArrayView<DepthSurfel>).
		 * 
		 * \param CameraID 相机ID
		 * \return 当前相机有效面元的数组
		 */
		DeviceArrayView<DepthSurfel> getValidDepthSurfelArray(int CameraID) { return depthSurfelView[CameraID]; }

		/**
		 * \brief 获得当前多相机位姿调整对齐后的面元的数组(只读数组DeviceArrayView<DepthSurfel>).
		 *
		 * \return 当前多相机位姿调整对齐后的面元的数组
		 */
		DeviceArrayView<DepthSurfel> getAlignedSurfelArray() { return preAlignedSurfelView; }

		/**
		 * \brief 获得经过下采样的顶点.
		 * 
		 * \param CameraID
		 * \return 
		 */
		DeviceArrayView<float4> getSubsampledSparseVertexArray(const unsigned int CameraID) { return subsampleSparseVertice[CameraID].ArrayView(); }
	
		
/*********************************************     前景分割     *********************************************/
	private:
		bool OnlineSegmenter = true;
		ForegroundSegmenter::Ptr foregroundSegmenter[MAX_CAMERA_COUNT]; //前景分割器，为每一张图片创建一个分割器，分割结果纹理存储在对象之中

		/**
		 * \brief 分配前景分割器的运行内存，加载模型.
		 * 
		 */
		void allocateForegroundSegmentationBuffer();

		/**
		 * \brief 释放前景分割器的运行内存.
		 * 
		 */
		void releaseForegroundSegmentationBuffer();
	public:
		/**
		 * \brief 对Color图像进行前景分割，每一张图像单独分配一个分割器，并给分割器单独分配一个stream【函数内部已经流同步(用时10ms，在读取图像完成之前，找到mask，不安全，应该增加事件响应？)】.
		 * 
		 * \param stream CUDA流ID
		 * \param CameraID 相机ID
		 */
		void SegmentForeground(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief 手动同步segment stream，也可以写cuda事件触发.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void SyncAndConvertForeground2Texture(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief 获得当前相机的(剪裁后)前景Mask.
		 * 
		 * \param CameraID 相机ID
		 * \return 前景mask纹理
		 */
		cudaTextureObject_t getForegroundMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getClipedMaskTexture(); }

		/**
		 * \brief 获得滤波后的前景Mask.
		 * 
		 * \param CameraID 相机ID
		 * \return 滤波后的前景mask纹理
		 */
		cudaTextureObject_t getFilteredForegroundMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getFilteredMaskTexture(); }

		/**
		 * \brief 获得前一帧的(剪裁后)前景Mask.
		 * 
		 * \param CameraID 相机ID
		 * \return 上一帧的前景纹理
		 */
		cudaTextureObject_t getPreviousForegroundMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getPreviousMaskTexture(); }

		/**
		 * \brief 获得前景边缘的Mask.
		 * 
		 * \param CameraID 相机ID
		 * \return 当前帧前景边缘Mask
		 */
		cudaTextureObject_t getCurrentForegroundEdgeMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getForegroundEdgeMaskTexture(); }

/*********************************************     稀疏特征点匹配     *********************************************/
	private:
		PatchColliderRGBCorrespondence::Ptr featureCorrespondenceFinder[MAX_CAMERA_COUNT];

		/**
		 * \brief 分配寻找特征点的方法的内存，每一个相机均对应一个方法.
		 * 
		 */
		void allocateFeatureCorrespondenceBuffer();



		/**
		 * \brief 释放寻找特征点方法的内存.
		 * 
		 */
		void releaseFeatureCorrespondenceBuffer();
	
	public:
		/**
		 * \brief 寻找特征点阻塞，新开一个Stream，在内部同步.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void FindCorrespondence(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief 从分割器中获得匹配点对.
		 * 
		 * \param CameraID 相机ID
		 * \return 匹配点对
		 */
		DeviceArray<ushort4> getCorrespondencePixelPair(const unsigned int CameraID) const { 
#ifndef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
			return featureCorrespondenceFinder[CameraID]->CorrespondedPixelPairs();
#else
			return opticalFlow[CameraID]->getCorrespondencePixelPair();
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

		}

/*********************************************     计算视频光流     *********************************************/
	private:
		OpticalFlow::Ptr opticalFlow[MAX_CAMERA_COUNT];
		DeviceArrayView2D<mat34> correctSe3Map[MAX_CAMERA_COUNT];
		DeviceArrayView2D<unsigned char> markValidSe3Map[MAX_CAMERA_COUNT];

		/**
		 * \brief 分配光流内存.
		 * 
		 */
		void allocateOpticalFlowBuffer();

		/**
		 * \brief 释放光流内存.
		 * 
		 */
		void releaseOpticalFlowBuffer();

		/**
		 * \brief 计算光流.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void computeOpticalFlow(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief 获得光流所匹配的像素点.
		 * 
		 * \param CameraID 相机ID
		 * \return 光流所匹配的像素点
		 */
		DeviceArray<ushort4> getOpticalFlowPixelPair(const unsigned int CameraID) const { return opticalFlow[CameraID]->getCorrespondencePixelPair(); }
	public:
		/**
		 * \brief 获得光流引导图.
		 *
		 * \return 光流引导图
		 */
		DeviceArrayView2D<mat34>* GetCorrectedSe3Maps() { return correctSe3Map; }

		/**
		 * \brief 获得稀疏匹配点有效值标记图.
		 *
		 * \return 稀疏匹配点有效值标记图
		 */
		DeviceArrayView2D<unsigned char>* GetValidSe3Maps() { return markValidSe3Map; }

/*********************************************     计算跨镜匹配点并插值     *********************************************/
	private:
		CrossViewEdgeCorrespondence::Ptr CrossViewMatching;
		CrossViewMatchingInterpolation::Ptr CrossViewInterpolation;
		/**
		 * \brief 分类跨镜匹配内存.
		 * 
		 */
		void allocateCrossViewMatchingBuffer();

		/**
		 * \brief 释放跨镜匹配内存.
		 * 
		 */
		void releaseCrossViewMatchingBuffer();

/*********************************************     计算灰度图和Mask的梯度     *********************************************/
	private:
		ImageGradient::Ptr imageGradient;										// 计算图片的梯度

		CudaTextureSurface ForegroundMaskGradientMap[MAX_CAMERA_COUNT];			// 记录前景掩膜的梯度图
		CudaTextureSurface GrayscaleGradientMap[MAX_CAMERA_COUNT];				// 记录灰度图的梯度图
		/**
		 * \brief 分配梯度图的缓存.
		 * 
		 */
		void allocateGradientMapBuffer();

		/**
		 * \brief 释放梯度图的缓存.
		 * 
		 */
		void releaseGradientMapBuffer();

		/**
		 * \brief 计算ForegroundMask和Grayscale的梯度.
		 * 
		 * \param stream cuda流ID
		 * \param CameraID 相机ID
		 */
		void ComputeGradientMap(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief 获得灰度图的梯度纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 灰度图的梯度纹理
		 */
		cudaTextureObject_t getGrayScaleGradientTexture(const unsigned int CameraID) const { return GrayscaleGradientMap[CameraID].texture; }

		/**
		 * \brief 获得前景Mask的梯度纹理.
		 * 
		 * \param CameraID 相机ID
		 * \return 前景Mask的梯度纹理
		 */
		cudaTextureObject_t getForegroundMaskGradientTexture(const unsigned int CameraID) const { return ForegroundMaskGradientMap[CameraID].texture; }

	public:
		void setInput(vector<string>* colorpath, vector<string>* depthpath);

	};

}


