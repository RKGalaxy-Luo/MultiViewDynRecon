/*****************************************************************//**
 * \file   RigidSolver.h
 * \brief  ICP������׼
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

	namespace device { // �˺����Լ�GPU�����е���������
		/**
		 * \brief ICP��Point - to - Plane������������.
		 */
		struct RigidSolverDevice {
			// ��Ҫ���뾲̬��������̬�����ֵ�������������ֱ������ö�ٱ���
			enum {
				// 6 + 5 + 4 + 3 + 2 + 1 = 21
				lhsMatrixSize = 21,									// ���־������Է������У����ֱ߾�������ϵ����ɵľ�����������δ֪���뷽��֮��Ĺ�ϵ
				rhsVectorSize = 6,									// �������������Է������У����ֱ��������ɳ�������ɵ��������������˷��̵ĵ�ʽ�Ҳ�
				totalSharedSize = lhsMatrixSize + rhsVectorSize,	// ���ֱ߾���lhsMatrixSize�������ֱ�������rhsVectorSize���ڹ����ڴ棨shared memory������ռ�õ��ܿռ��С

				// �����߳̿���߳����Ĵ�С
				blockSize = 256,									// GPU���߳̿�Ĵ�С��ÿ���߳̿�����256���߳�
				warpsSize = 32,										// ÿ���߳�������32���߳�(���ڴ���ռ�����bit)
				warpsNum = blockSize / warpsSize,					// GPU���߳����Ĵ�С��ÿ���߳̿�����8���߳���

				noReduceBlockSize = 512,							// �����������٣���ʹ�ù鲢�㷨�������㹻���block���Ա���һ��block�н�ƥ�������

			};


			/*
			 * ��GPU�ܹ��У�ͬһ���߳����е��߳�ͬʱִ����ͬ��ָ���SIMD��Single Instruction, Multiple Data���ķ�ʽִ�С�
			 * ����ζ���߳����е�ÿ���̶߳�ִ����ͬ��ָ������Բ�����ͬ������
			 */

			 /**
			  * \brief �����ICP���Զ���Ĳο�Model������mat34�Ǽ��������reference��mat34.
			  */
			struct {
				cudaTextureObject_t vertexMap;
				cudaTextureObject_t normalMap;
				cudaTextureObject_t foregroundMask;
			} referenceMaps;

			/**
			 * \brief �����ICP���Զ���ı任Model������mat34�Ǽ������conversion�����reference��mat34.
			 */
			struct {
				cudaTextureObject_t vertexMap;
				cudaTextureObject_t normalMap;
				cudaTextureObject_t foregroundMask;
			} conversionMaps;

			unsigned int imageRows;
			unsigned int imageCols;
			unsigned index;//��ǰ�ǵڼ��ε���
			unsigned max;//�趨������������


			mat34 currentWorldToCamera;		// ��ǰ��������ϵת���������ϵ����(��������������λ��ʱ���൱��ReferenceCameraToConversionCamera�任����)
			Intrinsic intrinsic;			// ������ڲ�(Ԥ�������ǲο�����ڲ�)
			DeviceArrayView<ushort4> pixelpair;

			/*
			 * ԭʼ ICP �㷨�Ĵ��ۺ�����ʹ�õ� point-to-point ���룬point-to-plane ���ǿ���Դ���㵽Ŀ�궥��������ľ��룬
			 * ����ֱ�Ӽ���㵽����룬�����˵��Ƶľֲ��ṹ�����ȸ��ߣ�����������ֲ����ţ���Ҫע�� point-to-plane ���Ż�
			 * ��һ�����������⣬�ٶȱȽ�����һ��ʹ�������Ի����ơ�
			 * ������Ҫע����ǣ�����https://zhuanlan.zhihu.com/p/385414929���Ƶ��õ�Ax=b����ʽ��Ȼ�޷��ܺ���⣬��ΪA���ܲ�����
			 * ���������ATA*x=AT*b  <=>  JTJ*x = JT*b
			 */

			 /**
			  * \brief ֻ��������GPU�ϣ��ں˺����е��õĺ�����ICP��Point-to-Plane ICP���������������λ��.
			  *
			  * \param reduceBuffer ��Լbuffer
			  * \return
			  */
			__device__ __forceinline__ void rigidSolveIteration(PtrStep<float> reduceBuffer) const;

			__device__ __forceinline__ void rigidSolveIterationUseOpticalFlows(PtrStep<float> reduceBuffer) const;
		};
			
		

		/**
		 * \brief ��Ҫ�������ICP��Point - to - Point��.
		 */
		struct Point2PointICP {
			// ��Ҫ���뾲̬��������̬�����ֵ�������������ֱ������ö�ٱ���
			enum {
				// 6 + 5 + 4 + 3 + 2 + 1 = 21
				lhsMatrixSize = 21,									// ���־������Է������У����ֱ߾�������ϵ����ɵľ�����������δ֪���뷽��֮��Ĺ�ϵ
				rhsVectorSize = 6,									// �������������Է������У����ֱ��������ɳ�������ɵ��������������˷��̵ĵ�ʽ�Ҳ�
				totalSharedSize = lhsMatrixSize + rhsVectorSize,	// ���ֱ߾���lhsMatrixSize�������ֱ�������rhsVectorSize���ڹ����ڴ棨shared memory������ռ�õ��ܿռ��С

				// �����߳̿���߳����Ĵ�С
				blockSize = 256,									// GPU���߳̿�Ĵ�С��ÿ���߳̿�����256���߳�
				warpsSize = 32,										// ÿ���߳�������32���߳�(���ڴ���ռ�����bit)
				warpsNum = blockSize / warpsSize					// GPU���߳����Ĵ�С��ÿ���߳̿�����8���߳���
			};

			unsigned int imageRows;
			unsigned int imageCols;



			mat34 currentPositionMat34;		// ����������������λ��ʱ���൱��ReferenceCamera To ConversionCamera�任����
			Intrinsic referenceIntrinsic;	// �ο�������ڲ�
			Intrinsic conversionIntrinsic;	// �任������ڲ�

			DeviceArrayView<float4> referenceSparseVertices;		// �����㷨�Ĳο������ϡ�趥��
			DeviceArrayView<float4> conversionSparseVertices;		// �����㷨�ı任�����ϡ�趥��

			DeviceArrayView<ushort4> knnIndex;						// reference�о���conversion������ĵ㣬knnIndex������conversion����һ��

			DeviceArrayView<float4> matchedPoints;					// ������ƥ���ĵ��

			/**
			 * \brief ֻ��������GPU�ϣ��ں˺����е��õĺ�����ICP��Point-to-Point ICP���������������λ��.
			 *
			 * \param referenceVertices �ο�����Ķ���
			 * \param conversionVertices �任����Ķ���
			 * \param vertexKNN �����
			 * \param pointsNum �������(�任����ĵ�)
			 * \param reduceBuffer ���й�Լ������
			 */
			__device__ __forceinline__ void solverIterationCamerasPosition(DeviceArrayView<float4> referenceVertices, DeviceArrayView<float4> conversionVertices, DeviceArrayView<ushort4> vertexKNN, const unsigned int pointsNum, PtrStep<float> reduceBuffer) const;

			/**
			 * \brief ֻ��������GPU�ϣ��ں˺����е��õĺ�����ICP��Point-to-Point ICP���������������λ��.
			 *
			 * \param pointsPairs ����ƥ����
			 * \param pointsNum �������(�任����ĵ�)
			 * \param MatrixBuffer ��¼�������[A|b]����ֵ
			 */
			__device__ __forceinline__ void solverIterationFeatureMatchCamerasPosition(DeviceArrayView<float4> pointsPairs, const unsigned int pairsNum, PtrStep<float> MatrixBuffer) const;

		};

		/**
		 * \brief ICP����������λ�˵ĺ˺���.
		 *
		 * \param solver Point2PointICP��������������Է�����ز��������ù�Լ����ATA�����ATb����
		 * \param reduceBuffer ��Լ�õ��ľ��󣬶�άתһά�洢��ptr(i)[j] i����,j���С�i�б�ʾATA��ATb���ɵ������Ǿ�������������ӦԪ��(27��Ԫ�ذ�˳������)��j��ʾ��ͬ��block�õ��Ĳ�ͬ��27��Ԫ��
		 */
		__global__ void rigidSolveIterationCamerasPositionKernel(const Point2PointICP solver, PtrStep<float> reduceBuffer);

		/**
		 * \brief ICP����������λ�˵ĺ˺�����ͨ����������ȡ��.
		 *
		 * \param solver Point2PointICP��������������Է�����ز��������ù�Լ����ATA�����ATb����
		 * \param pairsNum ƥ������
		 * \param reduceBuffer ��Լ�õ��ľ��󣬶�άתһά�洢��ptr(i)[j] i����,j���С�i�б�ʾATA��ATb���ɵ������Ǿ�������������ӦԪ��(27��Ԫ�ذ�˳������)��j��ʾ��ͬ��block�õ��Ĳ�ͬ��27��Ԫ��
		 */
		__global__ void rigidSolveIterationFeatureMatchCamerasPositionKernel(const Point2PointICP solver, const unsigned int pairsNum, PtrStep<float> MatrixBuffer);

		/**
		 * \brief ��ÿ�ж�Ӧ�Ķ��block�����й�Լ���õ�Ψһһ����27��Ԫ�ص�ATA��ATb���ɵ������Ǿ������������.
		 *
		 * \param globalBuffer ȫ�ֵĻ��棬������Լ��27�����е�block��������ֵ������
		 * \param target �洢��С���˷�ATA��ATb���ɵ������Ǿ�������������ӦԪ��(�������е�)
		 */
		__global__ void columnReduceKernel(const PtrStepSize<const float> globalBuffer, float* target);

		/**
		 * \brief ��ƥ����Ӧ�ľ���Ԫ����ӵõ����ľ���.
		 *
		 * \param globalBuffer ����ȫ�ֻ���
		 * \param target �洢��С���˷�ATA��ATb���ɵ������Ǿ�������������ӦԪ��(�������е�)
		 */
		__global__ void columnReduceFeatureMatchKernel(const PtrStepSize<const float> globalBuffer, const unsigned int blockNum, float* target);

		/**
		 * \brief ICP�������������һ֡�ı任.
		 *
		 * \param solver Point2Plane��������������Է�����ز��������ù�Լ����ATA�����ATb����
		 * \param ��Լ�õ��ľ��󣬶�άתһά�洢��ptr(i)[j] i����,j���С�i�б�ʾATA��ATb���ɵ������Ǿ�������������ӦԪ��(27��Ԫ�ذ�˳������)��j��ʾ��ͬ��block�õ��Ĳ�ͬ��27��Ԫ��
		 */
		__global__ void RigidSolveInterationKernel(const RigidSolverDevice solver, PtrStep<float> reduceBuffer);
		__global__ void RigidSolveInterationKernelUseOpticalFlows(const RigidSolverDevice solver, PtrStep<float> reduceBuffer);


		/**
		 * \brief �������Ԫת���ɳ��ܶ���.
		 *
		 * \param ���������Ԫ
		 * \param ������ܶ���
		 */
		__global__ void getVertexFromDepthSurfelKernel(DeviceArrayView<DepthSurfel> denseDepthSurfel, unsigned int denseSurfelNum, float4* denseVertex);

		/**
		 * \brief ���������ͷ�������ںϵ�preAlignedSurfel.
		 *
		 * \param preAlignedSurfel �ں϶�����ͷ��Surfel���˴�
		 * \param depthSurfel ��Ҫת����preAlignedSurfel����Ԫ
		 * \param relativePose ��ʼ����ͷλ��
		 * \param pointsNum ��ǰ����ĵ������
		 * \param offset preAlignedSurfel��ƫ��������ʱӦ�ô��ĸ�λ�ÿ�ʼ�洢
		 */
		__global__ void addDensePointsToCanonicalFieldKernel(DeviceArrayHandle<DepthSurfel> preAlignedSurfel, DeviceArrayView<DepthSurfel> depthSurfel, mat34 relativePose, const unsigned int pointsNum, const unsigned int offset);

		__global__ void checkPreviousAndCurrentTextureKernel(const unsigned int rows, const unsigned int cols, cudaTextureObject_t previousVertex, cudaTextureObject_t previousNormal, cudaTextureObject_t currentVertex, cudaTextureObject_t currentNormal);

	}





	class RigidSolver
	{
	private:

		cudaStream_t SolverStreams[MAX_CAMERA_COUNT];		// �����������Ϊ�����������ÿ������Լһ�����

		struct {
			cudaTextureObject_t vertexMap;
			cudaTextureObject_t normalMap;
			cudaTextureObject_t foreground;
		} referenceMap[MAX_CAMERA_COUNT];	// �ο���

		struct {
			cudaTextureObject_t vertexMap;
			cudaTextureObject_t normalMap;
			cudaTextureObject_t foreground;
		} conversionMap[MAX_CAMERA_COUNT];	// �任��

	private:
		int deviceCount = 0;										// �������
		unsigned int imageRows, imageCols;							// ͼ��ĳ���

		/**
		 * \brief Ԥ������Զ���.
		 */
		struct PreprocessingRigidSolver
		{
			Intrinsic clipedIntrinsic[MAX_CAMERA_COUNT];						// ���ú���ڲ�
			mat34 CamerasRelativePose[MAX_CAMERA_COUNT];						// ��¼ÿ������������0�������λ�ˣ��������ܷ��IMU���ж�λ���Ƿ�仯���仯�����¼���
			DeviceBufferArray<ushort4> VertexKNN[MAX_CAMERA_COUNT];				// ��¼��������KNN��conversion��reference��Ѱ��
			VoxelSubsampler vertexSubsampler[MAX_CAMERA_COUNT];					// ÿ������Ĳ�����
			DeviceBufferArray<float4> denseVertices[MAX_CAMERA_COUNT];			// ���ܶ���
			DeviceArrayView<float4> sparseVertices[MAX_CAMERA_COUNT];			// �²����õ���ϡ�趥��(ÿ��ϡ������Ӧһ��������)
			DeviceArrayView<DepthSurfel> denseSurfel[MAX_CAMERA_COUNT];			// ���ܵ���Ԫ
			DeviceArrayView<float4> matchedPoints[MAX_CAMERA_COUNT];			// ƥ��ĵ�ԣ�previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum)
		}PreRigidSolver;


	public:
		// RigidSolver����ָ��
		using Ptr = std::shared_ptr<RigidSolver>;

		/**
		 * \brief RigidSolver���캯������ICP������׼��������.
		 *
		 * \param devCount �����������
		 * \param clipIntrinsic ���ú�������ڲ�
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 */
		explicit RigidSolver(int devCount, Intrinsic* clipIntrinsic, unsigned int rows, unsigned int cols);

		/**
		 * \brief RigidSolver��������.
		 *
		 */
		~RigidSolver();

		NO_COPY_ASSIGN_MOVE(RigidSolver);


	public:
		/**
		 * \brief �Գ��ܵ�����²���.
		 *
		 * \param CameraID ��һ������ĳ��ܵ�
		 * \param denseDepthSurfel ������Ԫ
		 * \param sparseVertex ϡ�趥��
		 * \param stream CUDA��ID
		 */
		void performVertexSubsamplingSync(const unsigned int CameraID, DeviceArrayView<DepthSurfel>& denseDepthSurfel, DeviceBufferArray<float4>& sparseVertex, cudaStream_t stream = 0);


	private:

		/**
		 * \brief ������������.
		 *
		 */
		void allocateSubsampleBuffer();

		/**
		 * \brief �ͷ��²����Ļ���.
		 *
		 */
		void releaseSubsampleBuffer();

		/**
		 * \brief ����Ѱ��KNN����Ļ���.
		 *
		 */
		void allocateKNNBuffer();

		/**
		 * \brief �ͷ�Ѱ��KNN����Ļ���.
		 *
		 */
		void releaseKNNBuffer();

		/**
		 * \brief ������������ICP������⣬�漰����Լ���ڴ�.
		 *
		 */
		void allocateReduceBuffer();

		/**
		 * \brief �ͷ��漰����Լ���ڴ�.
		 *
		 */
		void releaseReduceBuffer();

		/**
		 * \brief ��ʼ����.
		 *
		 */
		void initSolverStreams();

		/**
		 * \brief �ͷ���.
		 *
		 */
		void releaseSolverStreams();

		/**
		 * \brief ��DepthSurfel��ȡfloat4���Թ��²���.
		 *
		 * \param denseDepthSurfel ���ܵ������Ԫ
		 * \param denseVertex ���ܵĶ���
		 * \param stream CUDA��ID
		 */
		void getVertexFromDepthSurfel(DeviceArrayView<DepthSurfel>& denseDepthSurfel, DeviceBufferArray<float4>& denseVertex, cudaStream_t stream = 0);

		/**
		 * \brief ��CameraID������е������Ԫ�����ݸ��Զ���Ľ����ת����ͬһ������.
		 *
		 * \param preAlignedSurfel ����ͬCanonical���е���Ԫ���뵽preAlignedSurfel��
		 * \param CameraID ��Ҫ�ںϵ�CameraID��Ԫ
		 * \param stream CUDA��ID
		 */
		void addDensePointsToCanonicalField(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, unsigned int CameraID, cudaStream_t stream = 0);

		/**
		 * \brief ���������ʼ��λ��.
		 *
		 * \param CameraID ���ID
		 */
		void setInitialCameraPose(const unsigned int CameraID);

	public:
		/**
		 * \brief �������ĳ�ʼ���λ�ã�����Solver�����.
		 *
		 * \param initialPose ����������������ͷʵ�ʰڷ�λ�ã�Ԥ�����ã���ֹICP����ֲ����ţ�û�г�ʼλ����ȫΪmat34��λ����
		 */
		void setPreRigidSolverSE3(mat34* initialPose = nullptr);

		/**
		 * \brief ��������ĳ�ʼ��λ��.
		 *
		 * \param CameraID ���������ID
		 */
		void setCamerasInitialSE3(const unsigned int CameraID);

		/**
		 * \brief ���������������0�������λ��.
		 *
		 * \return ��������0�������λ��
		 */
		mat34* getCamerasInitialSE3();

		/**
		 * \brief ��Ԥ�������ó�����Ԫ����������ƥ����.
		 *
		 * \param CameraID ���ID
		 * \param denseSurfel ������Ԫ
		 * \param pointsPairs ƥ����
		 */
		void setPreRigidSolverInput(const unsigned int CameraID, DeviceArrayView<DepthSurfel>& denseSurfel, DeviceArrayView<float4>& pointsPairs);
		/**
		 * \brief ��Ԥ������ͼ�����ICP��Point - to - Point�����.
		 *
		 * \param �������Ԥ���������ͷICP���Զ�����ɵ���Ԫ
		 * \param maxIteration ����������
		 * \param stream cuda��ID(�����ڴ��޷�ԭ�Ӳ�����ֻ�����δ��������ڴ�)
		 * \return λ�˾���mat34
		 */
		 //void PreSolvePoint2PointICP(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, int maxIteration = 3, cudaStream_t stream = 0);

		 /**
		  * \brief ��������������SE3.
		  *
		  * \param preAlignedSurfel �������Ԥ���������ͷICP���Զ�����ɵ���Ԫ
		  * \param maxIteration ����������
		  * \param stream cuda��ID(�����ڴ��޷�ԭ�Ӳ�����ֻ�����δ��������ڴ�)
		  */
		  //void PreSolveMatchedPairsPoint2PointICP(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, int maxIteration = 3, cudaStream_t stream = 0);
		  /**
		   * \brief �����ܵ���Ч��Ԫ��������֡�õ������λ�ˣ����뵽ͬһ���ռ�.
		   *
		   * \param preAlignedSurfel ����ͬCanonical���е���Ԫ���뵽preAlignedSurfel��
		   * \param currentValidDepthSurfel ��ǰ���ID�ĳ�����Ԫ
		   * \param CameraID ���ID
		   * \param offset ����preAlignedSurfel��λ��ƫ��
		   * \param stream CUDA��ID
		   */
		void mergeDenseSurfelToCanonicalField(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, DeviceArrayView<DepthSurfel>& currentValidDepthSurfel, const unsigned int CameraID, const unsigned int offset, cudaStream_t stream = 0);

		/**
		 * \brief ���Ԥ������Զ����λ�˾��󣬶���������ڵ�һ�������λ��.
		 *
		 * \param ��Ҫ�鿴��Ԥ������Զ����λ�˾���
		 */
		mat34 getPreSolverMat34(unsigned int CameraID);

		/**
		 * \brief ͬ��������Ϊ��һ����׼��.
		 *
		 * \param �������ھ����Լ����
		 * \param ��������
		 */
		void SynchronizeAllStreams(cudaStream_t* streams, const unsigned int streamsNum);

		/**
		 * \brief ���Զ�������룬ֱ������һ֡����.
		 *
		 * \param observation ����������۲첢����Ĳ���
		 */
		void SetRigidInput(CameraObservation& observation, const mat34 world2camera0, const mat34 world2camera1);

		/**
		 * \brief �����Զ�������ÿ�����������������ɲ��У�Ȼ��ÿ��������������(��������û������Host���̣�˳����ӵ���ͬ��Stream���ɣ�����Ҫ�̳߳�).
		 *
		 * \param observation ����������ռ��Ĳ���
		 * \param isfirstframe �Ƿ���ˢ��֡
		 * \param MaxInteration �㷨����������
		 */
		void SolveRigidAlignment(CameraObservation& observation, bool& isfirstframe, unsigned int MaxInteration = 6);

		/**
		 * \brief �����Զ�������ÿ�����������������ɲ��У�Ȼ��ÿ��������������(��������û������Host���̣�˳����ӵ���ͬ��Stream���ɣ�����Ҫ�̳߳�).
		 * 
		 * \param solverMaps �ڲ��洢��Live���еĵ�
		 * \param observation �洢�ŵ�ǰ֡�۲쵽�ĵ�
		 * \param LiveField2ObservedField ��Live��ĵ���뵽��ǰ�۲�֡
		 * \param MaxInteration ����������
		 */
		void SolveRigidAlignment(Renderer::SolverMaps* solverMaps, CameraObservation& observation, const mat34* LiveField2ObservedField, unsigned int MaxInteration = 3);

		/************************************************** GPU�������ATA�����ATb���� **************************************************/
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
		/********************************************** ����֡�Ǹ��Զ���ǰ���Զ��� **********************************************/
		DeviceArray2D<float> reduceAugmentedMatrix[MAX_CAMERA_COUNT];			// ��Լ�������[A|b]Ԫ��ֵ
		SynchronizeArray<float> finalAugmentedMatrixVector[MAX_CAMERA_COUNT];	// ���յ���������ֵ
		mat34 World2Camera[MAX_CAMERA_COUNT];									// ����Ǽ�¼��can����ǰ֡���ۼ�λ�˱任
		mat34 Canonical2Live;													// ��ÿ������ĸ���SE3ȫ��ת����0������ϵ�������������ƽ������SE3��Ϊ����Model��SE3
		mat34 AccumulatePrevious2Current[MAX_CAMERA_COUNT];						// ����ǵ�ǰ֡����һ֡�Ķ��롣
		mat34 Frame2FrameWorld2Camera[MAX_CAMERA_COUNT];						// ��֮֡��ĸ��Ա任
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];								// �����ʼ��λ��

	/********************************************** Ԥ��������㷨 **********************************************/
		// ������������ICP���ĵ��������н������ݵĹ�Լ����ICP�㷨�У���Լ����ͨ�����ڼ������������ۼӡ���ͻ�ƽ��ֵ�ȡ�ͨ����ÿ���߳̿飨block���ļ�������Լ��һ��������ֵ�����Լ���ȫ��ͨ�ź�ͬ���Ŀ���
		DeviceArray2D<float> reduceBuffer;				// ������������Ϊȫ���ڴ棺�洢ÿ��block����õ��ģ�ʹ����С���˷��е�ATA�����ATb����
		SynchronizeArray<float> reduceMatrixVector;		// ��reduceBuffer������ӣ����ù�Լ��ÿ��block��ӣ��õ����յ���С���˷��е�ATA�����ATb����
	/********************************************** �����㣬Point - to - Point�����㷨 **********************************************/
		DeviceArray2D<float> MatrixBuffer;				// �����洢ÿ��block�е��������[A|b]Ԫ��ֵ
		SynchronizeArray<float> finalMatrixVector;		// ���յ���������ֵ

		/**
		 * \brief ����0֡����ǰһ֡��ICP���Զ��룬�㷨�ڲ�������.
		 *
		 * \param stream cuda��ID   index�ǵ�ǰ֡��
		 */
		void rigidSolveDeviceIteration(const unsigned int CameraID, cudaStream_t stream = 0);

		/**
		 * \brief ����0֡����ǰһ֡��ICP���Զ��룬�㷨�ڲ�������.
		 *
		 * \param stream cuda��ID   index�ǵ�ǰ֡��
		 */
		void rigidSolveDeviceIteration(const unsigned int CameraID, const mat34* world2Camera, cudaStream_t stream = 0);

		/**
		 * \brief ������õĸ��Զ��뷽ʽ�ǣ�Live�����뵱ǰ֡.
		 * 
		 * \param CameraID ���ID
		 * \param stream cuda��ID
		 */
		void rigidSolveDeviceIterationLive2Observation(const unsigned int CameraID, cudaStream_t stream = 0);

		//�������������
		void rigidSolveDeviceIterationUseOpticalFlows(const unsigned int CameraID, unsigned index, unsigned max, cudaStream_t stream = 0);
		/**
		 * \brief ��Ԥ����GPU���������������ע�ⷵ�ص���conversion��Ե�reference��λ�ã�reference�ǲο��������conversion���ƶ��任�����.
		 *
		 * \param referenceIndex �ο��������������
		 * \param conversionIndex �任�������������
		 * \param stream cuda��ID(�����ڴ��޷�ԭ�Ӳ�����ֻ�����δ��������ڴ�)
		 */
		void rigidSolveDeviceIteration(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);



		/**
		 * \brief ��Ԥ�������������ƥ���ԣ����и������.
		 *
		 * \param referenceIndex �ο��������������
		 * \param conversionIndex �任�������������
		 * \param stream cuda��ID(�����ڴ��޷�ԭ�Ӳ�����ֻ�����δ��������ڴ�)
		 */
		void rigidSolveDeviceIterationFeatureMatch(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);


		/************************************************** CPU�������λ�˾��� **************************************************/
		Eigen::Matrix<float, 6, 6> ATA_[MAX_CAMERA_COUNT];	// ��С���˷���ATA����


		Eigen::Matrix<float, 6, 1> ATb_[MAX_CAMERA_COUNT];	// ��С���˷���ATb����

		/**
		 * \brief ���ICP����õ�rot��trans�Ƿ���NaN�������������Ϊ0.
		 *
		 * \param rot �������ת����
		 * \param trans �����ƽ�Ʋ���
		 */
		void CheckNaNValue(float3& rot, float3& trans);

		/**
		 * \brief �ռ�ÿ������Ĺ�Լ���󣬲����������Լ.
		 *
		 * \param stream ʹ��Ĭ����������
		 */
		void rigidSolveHostIterationSync(cudaStream_t stream = 0);

		/**
		 * \brief �ռ�ÿ������Ĺ�Լ���󣬲����������Լ.
		 *
		 * \param stream ʹ��Ĭ����������
		 */
		void rigidSolveHostIterationSync(mat34* world2Camera, cudaStream_t stream = 0);

		/**
		 * \brief �ռ�ÿ������Ĺ�Լ���󣬲����������Լ.
		 *
		 * \param stream ʹ��Ĭ����������
		 */
		void rigidSolveLive2ObservedHostIterationSync(cudaStream_t stream = 0);

		/**
		 * \brief �����ӽ�һ�������.
		 * 
		 * \param stream ʹ��Ĭ����������
		 */
		void rigidSolveMultiViewHostIterationSync(cudaStream_t stream = 0);

		/**
		 * \brief ��CPU��ʹ��Eigen��������mat34λ�˾���.
		 *
		 * \param referenceIndex �ο��������������
		 * \param conversionIndex �任�������������
		 * \param stream cuda��ID
		 */
		 //void rigidSolveHostIterationSync(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);

		 /**
		  * \brief ��CPU��ʹ��Eigen��������mat34λ�˾���.
		  *
		  * \param referenceIndex �ο��������������
		  * \param conversionIndex �任�������������
		  * \param stream cuda��ID
		  */
		  //void rigidSolveHostIterationFeatureMatchSync(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream = 0);

		void checkPreviousAndCurrentTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t previousVertex, cudaTextureObject_t previousNormal, cudaTextureObject_t currentVertex, cudaTextureObject_t currentNormal, cudaStream_t stream);

		/**
		 * \brief ��ÿ���������Ը�����ӽ�����ĸ���SE3��ת��0������ӽ��µĸ���SE3��Ȼ�����������SE3��ƽ��.
		 * 
		 * \return ƽ������SE3
		 */
		mat34 AverageCanonicalFieldRigidSE3(const mat34* world2Camera);
	};
}










