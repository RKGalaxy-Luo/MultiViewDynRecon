/*****************************************************************//**
 * \file   RigidSolver.h
 * \brief  ICP刚性配准
 * 
 * \author LUO
 * \date   February 2nd 2024
 *********************************************************************/
#pragma once
#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/CommonUtils.h>
#include <base/Constants.h>
#include <base/CameraObservation.h>
#include <base/ThreadPool.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <render/Renderer.h>

#include <math/MatUtils.h>

//#include <core/KNNSearchFunction.h>
#include <core/Geometry/VoxelSubsampler.h>

#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {

	namespace device { // 核函数以及GPU上运行的数据类型
		/**
		 * \brief ICP【Point - to - Plane】方法及参数.
		 */
		struct RigidSolverDevice {
			// 需要传入静态变量，静态变量又得在类外声明，直接声明枚举变量
			enum {
				// 6 + 5 + 4 + 3 + 2 + 1 = 21
				lhsMatrixSize = 21,									// 左手矩阵：线性方程组中，左手边矩阵是由系数组成的矩阵，它描述了未知数与方程之间的关系
				rhsVectorSize = 6,									// 右手向量：线性方程组中，右手边向量是由常数项组成的向量，它描述了方程的等式右侧
				totalSharedSize = lhsMatrixSize + rhsVectorSize,	// 左手边矩阵（lhsMatrixSize）和右手边向量（rhsVectorSize）在共享内存（shared memory）中所占用的总空间大小

				// 设置线程块和线程束的大小
				blockSize = 256,									// GPU中线程块的大小：每个线程块中有256个线程
				warpsSize = 32,										// 每个线程束中有32个线程(在内存中占据五个bit)
				warpsNum = blockSize / warpsSize,					// GPU中线程束的大小：每个线程块中有8个线程束

				noReduceBlockSize = 512,							// 针对特征点较少，不使用归并算法，开辟足够大的block，以便在一个block中将匹配点算完

			};


			/*
			 * 在GPU架构中，同一个线程束中的线程同时执行相同的指令，以SIMD（Single Instruction, Multiple Data）的方式执行。
			 * 这意味着线程束中的每个线程都执行相同的指令，但可以操作不同的数据
			 */

			 /**
			  * \brief 这个是ICP刚性对齐的参考Model，计算mat34是计算相对于reference的mat34.
			  */
			struct {
				cudaTextureObject_t vertexMap;
				cudaTextureObject_t normalMap;
				cudaTextureObject_t foregroundMask;
			} referenceMaps;

			/**
			 * \brief 这个是ICP刚性对齐的变换Model，计算mat34是计算这个conversion相对于reference的mat34.
			 */
			struct {
				cudaTextureObject_t vertexMap;
				cudaTextureObject_t normalMap;
				cudaTextureObject_t foregroundMask;
			} conversionMaps;

			unsigned int imageRows;
			unsigned int imageCols;
			unsigned index;//当前是第几次迭代
			unsigned max;//设定的最大迭代次数


			mat34 currentWorldToCamera;		// 当前世界坐标系转到相机坐标系矩阵(在求两个相机相对位置时，相当于ReferenceCameraToConversionCamera变换矩阵)
			Intrinsic intrinsic;			// 相机的内参(预处理中是参考相机内参)
			DeviceArrayView<ushort4> pixelpair;

			/*
			 * 原始 ICP 算法的代价函数中使用的 point-to-point 距离，point-to-plane 则是考虑源顶点到目标顶点所在面的距离，
			 * 比起直接计算点到点距离，考虑了点云的局部结构，精度更高，不容易陷入局部最优；但要注意 point-to-plane 的优化
			 * 是一个非线性问题，速度比较慢，一般使用其线性化近似。
			 * 这里需要注意的是，根据https://zhuanlan.zhihu.com/p/385414929中推导得到Ax=b的形式仍然无法很好求解，因为A可能不可逆
			 * 因此我们求ATA*x=AT*b  <=>  JTJ*x = JT*b
			 */

			 /**
			  * \brief 只能运行在GPU上，在核函数中调用的函数，ICP【Point-to-Plane ICP】迭代获得相机相对位置.
			  *
			  * \param reduceBuffer 归约buffer
			  * \return
			  */
			__device__ __forceinline__ void rigidSolveIteration(PtrStep<float> reduceBuffer) const;

			__device__ __forceinline__ void rigidSolveIterationUseOpticalFlows(PtrStep<float> reduceBuffer) const;
		};
			
		

		/**
		 * \brief 主要计算的是ICP【Point - to - Point】.
		 */
		struct Point2PointICP {
			// 需要传入静态变量，静态变量又得在类外声明，直接声明枚举变量
			enum {
				// 6 + 5 + 4 + 3 + 2 + 1 = 21
				lhsMatrixSize = 21,									// 左手矩阵：线性方程组中，左手边矩阵是由系数组成的矩阵，它描述了未知数与方程之间的关系
				rhsVectorSize = 6,									// 右手向量：线性方程组中，右手边向量是由常数项组成的向量，它描述了方程的等式右侧
				totalSharedSize = lhsMatrixSize + rhsVectorSize,	// 左手边矩阵（lhsMatrixSize）和右手边向量（rhsVectorSize）在共享内存（shared memory）中所占用的总空间大小

				// 设置线程块和线程束的大小
				blockSize = 256,									// GPU中线程块的大小：每个线程块中有256个线程
				warpsSize = 32,										// 每个线程束中有32个线程(在内存中占据五个bit)
				warpsNum = blockSize / warpsSize					// GPU中线程束的大小：每个线程块中有8个线程束
			};

			unsigned int imageRows;
			unsigned int imageCols;



			mat34 currentPositionMat34;		// 当在求两个相机相对位置时，相当于ReferenceCamera To ConversionCamera变换矩阵
			Intrinsic referenceIntrinsic;	// 参考相机的内参
			Intrinsic conversionIntrinsic;	// 变换相机的内参

			DeviceArrayView<float4> referenceSparseVertices;		// 进行算法的参考相机的稀疏顶点
			DeviceArrayView<float4> conversionSparseVertices;		// 进行算法的变换相机的稀疏顶点

			DeviceArrayView<ushort4> knnIndex;						// reference中距离conversion点最近的点，knnIndex数量和conversion点数一样

			DeviceArrayView<float4> matchedPoints;					// 特征点匹配后的点对

			/**
			 * \brief 只能运行在GPU上，在核函数中调用的函数，ICP【Point-to-Point ICP】迭代获得相机相对位置.
			 *
			 * \param referenceVertices 参考相机的顶点
			 * \param conversionVertices 变换相机的顶点
			 * \param vertexKNN 最近点
			 * \param pointsNum 点的数量(变换相机的点)
			 * \param reduceBuffer 进行归约的容器
			 */
			__device__ __forceinline__ void solverIterationCamerasPosition(DeviceArrayView<float4> referenceVertices, DeviceArrayView<float4> conversionVertices, DeviceArrayView<ushort4> vertexKNN, const unsigned int pointsNum, PtrStep<float> reduceBuffer) const;

			/**
			 * \brief 只能运行在GPU上，在核函数中调用的函数，ICP【Point-to-Point ICP】迭代获得相机相对位置.
			 *
			 * \param pointsPairs 特征匹配点对
			 * \param pointsNum 点的数量(变换相机的点)
			 * \param MatrixBuffer 记录增广矩阵[A|b]的数值
			 */
			__device__ __forceinline__ void solverIterationFeatureMatchCamerasPosition(DeviceArrayView<float4> pointsPairs, const unsigned int pairsNum, PtrStep<float> MatrixBuffer) const;

		};

		/**
		 * \brief ICP迭代获得相机位姿的核函数.
		 *
		 * \param solver Point2PointICP类型求解器，可以访问相关参数，调用归约计算ATA矩阵和ATb向量
		 * \param reduceBuffer 归约得到的矩阵，二维转一维存储，ptr(i)[j] i是行,j是列。i行表示ATA和ATb构成的上三角矩阵和向量矩阵对应元素(27个元素按顺序标号了)，j表示不同的block得到的不同的27个元素
		 */
		__global__ void rigidSolveIterationCamerasPositionKernel(const Point2PointICP solver, PtrStep<float> reduceBuffer);

		/**
		 * \brief ICP迭代获得相机位姿的核函数【通过特征点提取】.
		 *
		 * \param solver Point2PointICP类型求解器，可以访问相关参数，调用归约计算ATA矩阵和ATb向量
		 * \param pairsNum 匹配点对数
		 * \param reduceBuffer 归约得到的矩阵，二维转一维存储，ptr(i)[j] i是行,j是列。i行表示ATA和ATb构成的上三角矩阵和向量矩阵对应元素(27个元素按顺序标号了)，j表示不同的block得到的不同的27个元素
		 */
		__global__ void rigidSolveIterationFeatureMatchCamerasPositionKernel(const Point2PointICP solver, const unsigned int pairsNum, PtrStep<float> MatrixBuffer);

		/**
		 * \brief 将每行对应的多个block，按列归约，得到唯一一个的27个元素的ATA和ATb构成的上三角矩阵和向量矩阵.
		 *
		 * \param globalBuffer 全局的缓存，辅助归约将27行所有的block加起来的值存起来
		 * \param target 存储最小二乘法ATA和ATb构成的上三角矩阵和向量矩阵对应元素(遍历所有点)
		 */
		__global__ void columnReduceKernel(const PtrStepSize<const float> globalBuffer, float* target);

		/**
		 * \brief 将匹配点对应的矩阵元素相加得到最后的矩阵.
		 *
		 * \param globalBuffer 传入全局缓存
		 * \param target 存储最小二乘法ATA和ATb构成的上三角矩阵和向量矩阵对应元素(遍历所有点)
		 */
		__global__ void columnReduceFeatureMatchKernel(const PtrStepSize<const float> globalBuffer, const unsigned int blockNum, float* target);

		/**
		 * \brief ICP迭代求相对于上一帧的变换.
		 *
		 * \param solver Point2Plane类型求解器，可以访问相关参数，调用归约计算ATA矩阵和ATb向量
		 * \param 归约得到的矩阵，二维转一维存储，ptr(i)[j] i是行,j是列。i行表示ATA和ATb构成的上三角矩阵和向量矩阵对应元素(27个元素按顺序标号了)，j表示不同的block得到的不同的27个元素
		 */
		__global__ void RigidSolveInterationKernel(const RigidSolverDevice solver, PtrStep<float> reduceBuffer);
		__global__ void RigidSolveInterationKernelUseOpticalFlows(const RigidSolverDevice solver, PtrStep<float> reduceBuffer);


		/**
		 * \brief 将深度面元转化成稠密顶点.
		 *
		 * \param 传入深度面元
		 * \param 传入稠密顶点
		 */
		__global__ void getVertexFromDepthSurfelKernel(DeviceArrayView<DepthSurfel> denseDepthSurfel, unsigned int denseSurfelNum, float4* denseVertex);

		/**
		 * \brief 将多个摄像头的数据融合到preAlignedSurfel.
		 *
		 * \param preAlignedSurfel 融合多摄像头的Surfel到此处
		 * \param depthSurfel 需要转换到preAlignedSurfel的面元
		 * \param relativePose 初始摄像头位姿
		 * \param pointsNum 当前相机的点的数量
		 * \param offset preAlignedSurfel的偏移量，此时应该从哪个位置开始存储
		 */
		__global__ void addDensePointsToCanonicalFieldKernel(DeviceArrayHandle<DepthSurfel> preAlignedSurfel, DeviceArrayView<DepthSurfel> depthSurfel, mat34 relativePose, const unsigned int pointsNum, const unsigned int offset);

		__global__ void checkPreviousAndCurrentTextureKernel(const unsigned int rows, const unsigned int cols, cudaTextureObject_t previousVertex, cudaTextureObject_t previousNormal, cudaTextureObject_t currentVertex, cudaTextureObject_t currentNormal);

	}





	class RigidSolver
	{
	private:

		cudaStream_t SolverStreams[MAX_CAMERA_COUNT];		// 求解器最大的流为相机最大个数，每个流归约一个相机

		struct {
			cudaTextureObject_t vertexMap;
			cudaTextureObject_t normalMap;
			cudaTextureObject_t foreground;
		} referenceMap[MAX_CAMERA_COUNT];	// 参考的

		struct {
			cudaTextureObject_t vertexMap;
			cudaTextureObject_t normalMap;
			cudaTextureObject_t foreground;
		} conversionMap[MAX_CAMERA_COUNT];	// 变换的

	private:
		int deviceCount = 0;										// 相机个数
		unsigned int imageRows, imageCols;							// 图像的长宽

		/**
		 * \brief 预处理刚性对齐.
		 */
		struct PreprocessingRigidSolver
		{
			Intrinsic clipedIntrinsic[MAX_CAMERA_COUNT];						// 剪裁后的内参
			mat34 CamerasRelativePose[MAX_CAMERA_COUNT];						// 记录每个相机的相对于0号相机的位姿，后续看能否接IMU，判断位姿是否变化，变化则重新计算
			DeviceBufferArray<ushort4> VertexKNN[MAX_CAMERA_COUNT];				// 记录相机顶点的KNN，conversion在reference中寻找
			VoxelSubsampler vertexSubsampler[MAX_CAMERA_COUNT];					// 每个相机的采样器
			DeviceBufferArray<float4> denseVertices[MAX_CAMERA_COUNT];			// 稠密顶点
			DeviceArrayView<float4> sparseVertices[MAX_CAMERA_COUNT];			// 下采样得到的稀疏顶点(每个稀疏点均对应一个体素中)
			DeviceArrayView<DepthSurfel> denseSurfel[MAX_CAMERA_COUNT];			// 稠密的面元
			DeviceArrayView<float4> matchedPoints[MAX_CAMERA_COUNT];			// 匹配的点对，previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum)
		}PreRigidSolver;


	public:
		// RigidSolver共享指针
		using Ptr = std::shared_ptr<RigidSolver>;

		/**
		 * \brief RigidSolver构造函数，将ICP刚性配准参数传入.
		 *
		 * \param devCount 接入相机个数
		 * \param clipIntrinsic 剪裁后相机的内参
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 */
		explicit RigidSolver(int devCount, Intrinsic* clipIntrinsic, unsigned int rows, unsigned int cols);

		/**
		 * \brief RigidSolver析构函数.
		 *
		 */
		~RigidSolver();

		NO_COPY_ASSIGN_MOVE(RigidSolver);


	public:
		/**
		 * \brief 对稠密点进行下采样.
		 *
		 * \param CameraID 哪一个相机的稠密点
		 * \param denseDepthSurfel 稠密面元
		 * \param sparseVertex 稀疏顶点
		 * \param stream CUDA流ID
		 */
		void performVertexSubsamplingSync(const unsigned int CameraID, DeviceArrayView<DepthSurfel>& denseDepthSurfel, DeviceBufferArray<float4>& sparseVertex, cudaStream_t stream = 0);


	private:

		/**
		 * \brief 分配用来缓存.
		 *
		 */
		void allocateSubsampleBuffer();

		/**
		 * \brief 释放下采样的缓存.
		 *
		 */
		void releaseSubsampleBuffer();

		/**
		 * \brief 分配寻找KNN所需的缓存.
		 *
		 */
		void allocateKNNBuffer();

		/**
		 * \brief 释放寻找KNN所需的缓存.
		 *
		 */
		void releaseKNNBuffer();

		/**
		 * \brief 分配用来进行ICP刚性求解，涉及到归约的内存.
		 *
		 */
		void allocateReduceBuffer();

		/**
		 * \brief 释放涉及到归约的内存.
		 *
		 */
		void releaseReduceBuffer();

		/**
		 * \brief 初始化流.
		 *
		 */
		void initSolverStreams();

		/**
		 * \brief 释放流.
		 *
		 */
		void releaseSolverStreams();

		/**
		 * \brief 将DepthSurfel提取float4，以供下采样.
		 *
		 * \param denseDepthSurfel 稠密的深度面元
		 * \param denseVertex 稠密的顶点
		 * \param stream CUDA流ID
		 */
		void getVertexFromDepthSurfel(DeviceArrayView<DepthSurfel>& denseDepthSurfel, DeviceBufferArray<float4>& denseVertex, cudaStream_t stream = 0);

		/**
		 * \brief 将CameraID号相机中的深度面元，根据刚性对齐的结果，转化到同一个域中.
		 *
		 * \param preAlignedSurfel 将不同Canonical域中的面元对齐到preAlignedSurfel中
		 * \param CameraID 需要融合的CameraID面元
		 * \param stream CUDA流ID
		 */
		void addDensePointsToCanonicalField(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, unsigned int CameraID, cudaStream_t stream = 0);

		/**
		 * \brief 设置相机初始化位姿.
		 *
		 * \param CameraID 相机ID
		 */
		void setInitialCameraPose(const unsigned int CameraID);

	public:
		/**
		 * \brief 求解相机的初始相对位置，设置Solver的输出.
		 *
		 * \param initialPose 超参数，根据摄像头实际摆放位置，预先设置，防止ICP陷入局部最优，没有初始位置则全为mat34单位矩阵
		 */
		void setPreRigidSolverSE3(mat34* initialPose = nullptr);

		/**
		 * \brief 设置相机的初始化位姿.
		 *
		 * \param CameraID 设置相机的ID
		 */
		void setCamerasInitialSE3(const unsigned int CameraID);

		/**
		 * \brief 获得所有相机相对于0号相机的位置.
		 *
		 * \return 相机相对于0号相机的位置
		 */
		mat34* getCamerasInitialSE3();

		/**
		 * \brief 【预处理】设置稠密面元，设置特征匹配点对.
		 *
		 * \param CameraID 相机ID
		 * \param denseSurfel 稠密面元
		 * \param pointsPairs 匹配点对
		 */
		void setPreRigidSolverInput(const unsigned int CameraID, DeviceArrayView<DepthSurfel>& denseSurfel, DeviceArrayView<float4>& pointsPairs);
		/**
		 * \brief 【预处理】对图像进行ICP【Point - to - Point】求解.
		 *
		 * \param 【输出】预处理多摄像头ICP刚性对齐完成的面元
		 * \param maxIteration 最大迭代次数
		 * \param stream cuda流ID(纹理内存无法原子操作，只能依次处理纹理内存)
		 * \return 位姿矩阵mat34
		 */
		 //void PreSolvePoint2PointICP(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, int maxIteration = 3, cudaStream_t stream = 0);

		 /**
		  * \brief 根据特征点对求解SE3.
		  *
		  * \param preAlignedSurfel 【输出】预处理多摄像头ICP刚性对齐完成的面元
		  * \param maxIteration 最大迭代次数
		  * \param stream cuda流ID(纹理内存无法原子操作，只能依次处理纹理内存)
		  */
		  //void PreSolveMatchedPairsPoint2PointICP(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, int maxIteration = 3, cudaStream_t stream = 0);
		  /**
		   * \brief 将稠密的有效面元，根据首帧得到的相机位姿，融入到同一个空间.
		   *
		   * \param preAlignedSurfel 将不同Canonical域中的面元对齐到preAlignedSurfel中
		   * \param currentValidDepthSurfel 当前相机ID的稠密面元
		   * \param CameraID 相机ID
		   * \param offset 存入preAlignedSurfel的位置偏移
		   * \param stream CUDA流ID
		   */
		void mergeDenseSurfelToCanonicalField(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, DeviceArrayView<DepthSurfel>& currentValidDepthSurfel, const unsigned int CameraID, const unsigned int offset, cudaStream_t stream = 0);

		/**
		 * \brief 获得预处理刚性对齐的位姿矩阵，多个相机相对于第一个相机的位姿.
		 *
		 * \param 需要查看的预处理刚性对齐的位姿矩阵
		 */
		mat34 getPreSolverMat34(unsigned int CameraID);

		/**
		 * \brief 同步所有流为下一步做准备.
		 *
		 * \param 所有用于矩阵归约的流
		 * \param 流的数量
		 */
		void SynchronizeAllStreams(cudaStream_t* streams, const unsigned int streamsNum);

		/**
		 * \brief 刚性对齐的输入，直接与上一帧对齐.
		 *
		 * \param observation 输入摄像机观察并处理的参数
		 */
		void SetRigidInput(CameraObservation& observation, const mat34 world2camera0, const mat34 world2camera1);

		/**
		 * \brief 将刚性对齐的求解每个相机的增广矩阵做成并行，然后将每个相机的增广矩阵(求解过程中没有阻塞Host过程，顺次添加到不同的Stream即可，不需要线程池).
		 *
		 * \param observation 传入摄像机收集的参数
		 * \param isfirstframe 是否是刷新帧
		 * \param MaxInteration 算法最大迭代次数
		 */
		void SolveRigidAlignment(CameraObservation& observation, bool& isfirstframe, unsigned int MaxInteration = 6);

		/**
		 * \brief 将刚性对齐的求解每个相机的增广矩阵做成并行，然后将每个相机的增广矩阵(求解过程中没有阻塞Host过程，顺次添加到不同的Stream即可，不需要线程池).
		 * 
		 * \param solverMaps 内部存储着Live域中的点
		 * \param observation 存储着当前帧观察到的点
		 * \param LiveField2ObservedField 将Live域的点对齐到当前观察帧
		 * \param MaxInteration 最大迭代次数
		 */
		void SolveRigidAlignment(Renderer::SolverMaps* solverMaps, CameraObservation& observation, const mat34* LiveField2ObservedField, unsigned int MaxInteration = 3);

		/************************************************** GPU迭代获得ATA矩阵和ATb向量 **************************************************/
	public:
		mat34 GetWorld2Camera() const {
			return Canonical2Live;
		}

		mat34 GetWorld2Camera(unsigned int cameraID) const {
			return World2Camera[cameraID];
		}

		mat34* GetFrame2FrameWorld2Camera() {
			return Frame2FrameWorld2Camera;
		}

	private:
		/********************************************** 运行帧非刚性对齐前刚性对齐 **********************************************/
		DeviceArray2D<float> reduceAugmentedMatrix[MAX_CAMERA_COUNT];			// 归约增广矩阵[A|b]元素值
		SynchronizeArray<float> finalAugmentedMatrixVector[MAX_CAMERA_COUNT];	// 最终的增广矩阵的值
		mat34 World2Camera[MAX_CAMERA_COUNT];									// 这个是记录从can到当前帧的累计位姿变换
		mat34 Canonical2Live;													// 将每个相机的刚性SE3全部转换到0号坐标系，算所有相机的平均刚性SE3作为最终Model的SE3
		mat34 AccumulatePrevious2Current[MAX_CAMERA_COUNT];						// 这个是当前帧与上一帧的对齐。
		mat34 Frame2FrameWorld2Camera[MAX_CAMERA_COUNT];						// 两帧之间的刚性变换
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];								// 相机初始化位姿

	/********************************************** 预处理对齐算法 **********************************************/
		// 缓冲区用于在ICP求解的迭代过程中进行数据的归约，在ICP算法中，归约操作通常用于计算误差或距离的累加、求和或平均值等。通过将每个线程块（block）的计算结果归约到一个单独的值，可以减少全局通信和同步的开销
		DeviceArray2D<float> reduceBuffer;				// 这里是用来作为全局内存：存储每个block计算得到的，使用最小二乘法中的ATA矩阵和ATb向量
		SynchronizeArray<float> reduceMatrixVector;		// 将reduceBuffer数据相加，利用归约将每个block相加，得到最终的最小二乘法中的ATA矩阵和ATb向量
	/********************************************** 特征点，Point - to - Point对齐算法 **********************************************/
		DeviceArray2D<float> MatrixBuffer;				// 用来存储每个block中的增广矩阵[A|b]元素值
		SynchronizeArray<float> finalMatrixVector;		// 最终的增广矩阵的值

		/**
		 * \brief 【非0帧】与前一帧的ICP刚性对齐，算法内部非阻塞.
		 *
		 * \param stream cuda流ID   index是当前帧的
		 */
		void rigidSolveDeviceIteration(const unsigned int CameraID, cudaStream_t stream = 0);

		/**
		 * \brief 【非0帧】与前一帧的ICP刚性对齐，算法内部非阻塞.
		 *
		 * \param stream cuda流ID   index是当前帧的
		 */
		void rigidSolveDeviceIteration(const unsigned int CameraID, const mat34* world2Camera, cudaStream_t stream = 0);

		/**
		 * \brief 这里采用的刚性对齐方式是：Live域点对齐当前帧.
		 * 
		 * \param CameraID 相机ID
		 * \param stream cuda流ID
		 */
		void rigidSolveDeviceIterationLive2Observation(const unsigned int CameraID, cudaStream_t stream = 0);

		//这个光流对齐用
		void rigidSolveDeviceIterationUseOpticalFlows(const unsigned int CameraID, unsigned index, unsigned max, cudaStream_t stream = 0);
		/**
		 * \brief 【预处理】GPU迭代刚性求解器，注意返回的是conversion相对的reference的位置，reference是参考的相机，conversion是移动变换的相机.
		 *
		 * \param referenceIndex 参考对象的纹理索引
		 * \param conversionIndex 变换对象的纹理索引
		 * \param stream cuda流ID(纹理内存无法原子操作，只能依次处理纹理内存)
		 */
		void rigidSolveDeviceIteration(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);



		/**
		 * \brief 【预处理】传入相机的匹配点对，进行刚性求解.
		 *
		 * \param referenceIndex 参考对象的纹理索引
		 * \param conversionIndex 变换对象的纹理索引
		 * \param stream cuda流ID(纹理内存无法原子操作，只能依次处理纹理内存)
		 */
		void rigidSolveDeviceIterationFeatureMatch(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);


		/************************************************** CPU迭代获得位姿矩阵 **************************************************/
		Eigen::Matrix<float, 6, 6> ATA_[MAX_CAMERA_COUNT];	// 最小二乘法的ATA矩阵


		Eigen::Matrix<float, 6, 1> ATb_[MAX_CAMERA_COUNT];	// 最小二乘法的ATb向量

		/**
		 * \brief 检查ICP计算得到rot和trans是否是NaN，如果是则将其置为0.
		 *
		 * \param rot 传入的旋转参数
		 * \param trans 传入的平移参数
		 */
		void CheckNaNValue(float3& rot, float3& trans);

		/**
		 * \brief 收集每个相机的归约矩阵，并相加做最后归约.
		 *
		 * \param stream 使用默认流做计算
		 */
		void rigidSolveHostIterationSync(cudaStream_t stream = 0);

		/**
		 * \brief 收集每个相机的归约矩阵，并相加做最后归约.
		 *
		 * \param stream 使用默认流做计算
		 */
		void rigidSolveHostIterationSync(mat34* world2Camera, cudaStream_t stream = 0);

		/**
		 * \brief 收集每个相机的归约矩阵，并相加做最后归约.
		 *
		 * \param stream 使用默认流做计算
		 */
		void rigidSolveLive2ObservedHostIterationSync(cudaStream_t stream = 0);

		/**
		 * \brief 三个视角一起做求解.
		 * 
		 * \param stream 使用默认流做计算
		 */
		void rigidSolveMultiViewHostIterationSync(cudaStream_t stream = 0);

		/**
		 * \brief 在CPU中使用Eigen库迭代获得mat34位姿矩阵.
		 *
		 * \param referenceIndex 参考对象的纹理索引
		 * \param conversionIndex 变换对象的纹理索引
		 * \param stream cuda流ID
		 */
		 //void rigidSolveHostIterationSync(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);

		 /**
		  * \brief 在CPU中使用Eigen库迭代获得mat34位姿矩阵.
		  *
		  * \param referenceIndex 参考对象的纹理索引
		  * \param conversionIndex 变换对象的纹理索引
		  * \param stream cuda流ID
		  */
		  //void rigidSolveHostIterationFeatureMatchSync(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);

		void checkPreviousAndCurrentTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t previousVertex, cudaTextureObject_t previousNormal, cudaTextureObject_t currentVertex, cudaTextureObject_t currentNormal, cudaStream_t stream);

		/**
		 * \brief 将每个相机将针对该相机视角下求的刚性SE3，转成0号相机视角下的刚性SE3表达，然后将所有相机的SE3求平均.
		 * 
		 * \return 平均刚性SE3
		 */
		mat34 AverageCanonicalFieldRigidSE3(const mat34* world2Camera);
	};
}










