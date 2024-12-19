/*****************************************************************//**
 * \file   FrameProcessor.h
 * \brief  �������Ҫ�Ǵ���һЩͼ����GPU�����Լ�ͼ��Ԥ�������ز���
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


#define A_IMAGEPROCESSOR_NEED_CUDA_STREAM 4		// ÿһ��ͼƬ�����Ҫ��������

#define USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
//#define WITH_NODE_CORRECTION

namespace SparseSurfelFusion {
	class FrameProcessor
	{
	public:
		using Ptr = std::shared_ptr<FrameProcessor>;	// FrameProcessor�������ָ��(shared_ptr)
		/**
		 * \brief ��ʽ�������캯�������캯��������ʽ����.
		 * 
		 */
		explicit FrameProcessor(std::shared_ptr<ThreadPool> threadPool);

		/**
		 * \brief �����������ͷ��ڴ�.
		 * 
		 */
		~FrameProcessor();
/*********************************************     �������     *********************************************/
	private:

		GetFrameFromCamera::Ptr frame;		// ������֡
		ConfigParser::Ptr configParser;		// �����ͼ���������
		std::shared_ptr<ThreadPool> pool;

		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];
		std::atomic<int> numThreadsCompleted;  // ��ɵ��߳�����

		//ֱ�Ӱ�ԭʼ�����Ԫӳ�䵽���������������ֱ��д���ںϺ�������Դӳ�䡣
		CudaTextureSurface icp_vertex_confidence[MAX_CAMERA_COUNT];
		CudaTextureSurface icp_normal_radius[MAX_CAMERA_COUNT];
		CudaTextureSurface icp_color_time[MAX_CAMERA_COUNT];

		unsigned int beginIndex = 0;

	public:
		/**
		 * \brief ��ȡ�Ƿ������������׼�����.
		 * 
		 * \return �����������׼����Ϸ���true
		 */
		bool CamerasReady();

		/**
		 * \brief ��ʾ����ͼ��RGB��Depth.
		 * 
		 */
		void showAllImages();

		/**
		 * \brief ��ȡ��ǰ�������Ĳ�ɫͼ.
		 * 
		 * \param CameraID ������
		 * \return ���ص�ǰ��ŵ�����Ĳ�ɫͼ
		 */
		cv::Mat getCurrentColorImage(int CameraID);

		/**
		 * \brief ��ȡ��ǰ�����������ͼ.
		 * 
		 * \param CameraID ������
		 * \return ���ص�ǰ�����������ͼ
		 */
		cv::Mat getCurrentDepthImage(int CameraID);

		/**
		 * \brief �����Ƿ����Cameras.
		 * 
		 * \return true�ǹر���Camera
		 */
		bool isStop() {
			return !frame->ContinueCapturingCamera();
		}

		/**
		 * \brief ֹͣ�������.
		 * 
		 */
		void StopAllCameras() {
			frame->StopIt();
		}

		void SaveData(size_t frameIdx) {
			frame->setFrameIndex(frameIdx);
		}

		/**
		 * \brief ��ȡ��������ĸ���.
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
		 * \brief ��ö�Ӧ���RGB�Ĵ�������.
		 * 
		 * \param CameraID ���ID
		 * \return RGB��������
		 */
		cv::String GetColorWindowsName(int CameraID) {
			return frame->GetColorWindowName(CameraID);
		}

		/**
		 * \brief ��ö�Ӧ������ͼ�Ĵ�������.
		 * 
		 * \param CameraID ���ID
		 * \return Depth��������
		 */
		cv::String GetDepthWindowsName(int CameraID) {
			return frame->GetDepthWindowName(CameraID);
		}

		/**
		 * \brief �����㷨��ʼ��index.
		 * 
		 * \param beginIdx
		 */
		void setBeginFrameIndex(const unsigned int beginIdx) {
			beginIndex = beginIdx;
		}

		/**
		 * \brief ����ǰͼƬ������ÿ��������ֳ�һ���߳�.
		 * 
		 * \param CameraID ���ID
		 * \param observation ��ǰ�۲������õĲ����ռ���
		 */
		void ProcessCurrentCameraImageTask(const unsigned int CameraID, CameraObservation& observation);


		/**
		 * \brief �����һ֡ͼ�񣬲������ӽ���������Ԫ�ں�.
		 * 
		 * \param frameIndex ����֡ID
		 * \param rigidSolver �������ĳ�ʼ���λ��
		 * \return �����ںϺ����Ԫ
		 */
		DeviceArrayView<DepthSurfel> ProcessFirstFrame(size_t frameIndex, RigidSolver::Ptr rigidSolver);

		/**
		 * \brief �������ͼ��(�ǵ�һ֡).
		 * 
		 * \param observation ����ͨ�������ò�����õ������ڶ���Ĳ���
		 * \param frameIndex ����֡ID
		 * \param rigidSolver ������������
		 * \return �����ںϺ����Ԫ
		 */
		void ProcessCurrentFrame(CameraObservation& observation, size_t frameIndex);



		/**
		 * \brief ��ü��ú��IntrinsicArray.
		 * 
		 * \return ���ú��IntrinsicArray.
		 */
		Intrinsic* GetClipedIntrinsicArray() { return clipColorIntrinsic; }

/*********************************************     �������     *********************************************/
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

		cv::Mat depthImage[MAX_CAMERA_COUNT];						// �������ͼ��
		cv::Mat colorImage[MAX_CAMERA_COUNT];						// ����RGBͼ��
		cv::Mat previousDepthImage[MAX_CAMERA_COUNT];				// ǰһ֡���ͼ��
		cv::Mat previousColorImage[MAX_CAMERA_COUNT];				// ǰһ֡RGBͼ��

		std::vector<cv::Mat>* ColorOfflineData;
		std::vector<cv::Mat>* DepthOfflineData;
		std::vector<cv::String> ColorOfflinePath[MAX_CAMERA_COUNT];
		std::vector<cv::String> DepthOfflinePath[MAX_CAMERA_COUNT];

		bool intoBuffer = true;
		std::string dataSuffixPath;

		/**
		 * \brief �ռ�ͨ��ͼ��۲��ӽǵĲ���.
		 * 
		 * \param observation �����ռ���
		 * \param CameraID ���ID
		 */
		void CollectObservationParameters(cudaStream_t stream, CameraObservation& observation, const unsigned int CameraID);
		void CollectMergeObservationParameters(CameraObservation& observation);
/*********************************************     ��ģʽ����     *********************************************/

	private:
		// cuda�����������
		cudaStream_t ProcessorStream[A_IMAGEPROCESSOR_NEED_CUDA_STREAM * MAX_CAMERA_COUNT];
		const unsigned int TotalProcessingStreams = A_IMAGEPROCESSOR_NEED_CUDA_STREAM * MAX_CAMERA_COUNT;
		/**
		 * \brief ��ʼ��cuda�������.
		 */
		void initProcessorStream();
		/**
		 * \brief �ͷŴ������.
		 * 
		 */
		void releaseProcessorStream();
		/**
		 * \brief ͬ�����е���.
		 * 
		 */
		void syncAllProcessorStream();


/*********************************************     ͼ���ȡ     *********************************************/
	private:

		/**
		 * \brief Ԥ�ȷ����ȡͼ�����ҳ�ڴ�.
		 */
		void allocateImageHostBuffer();

		/**
		 * \brief �ͷ���ҳ�ڴ�.
		 * 
		 */
		void releaseImageHostBuffer();
	public:
		
		void GetOfflineData(std::vector<cv::Mat>* ColorOffline, std::vector<cv::Mat>* DepthOffline, std::string type, bool intoBuffer = true);

		/**
		 * \brief ��ȡ���RGB�����ͼƬ��Ϣ��ǰһ֡RGBͼƬ.
		 * 
		 * \param frameIndex �ڼ�֡
		 * \param CameraID ���ID
		 */
		void FetchFrame(size_t frameIndex, const unsigned int CameraID);

		/**
		 * \brief ���ԭʼ��ͬ�ӽ�MatͼƬ.
		 * 
		 * \return ԭʼ��ͬ�ӽ�MatͼƬ����
		 */
		cv::Mat* GetRawImageMatArray() { return colorImage; }


/*********************************************     ���ͼ     *********************************************/

	public:
		/**
		 * \brief �ϴ����ͼƬ��GPU��.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID �����ID��ÿһ�������DepthͼƬ��Ҫһ�������д���
		 */
		void UploadDepthImageToGPU(cudaStream_t stream, int CameraID);

		/**
		 * \brief ��.cu�ļ���ʵ�����ͼ�ü����������庯��ʵ����ImageProcessorByGPU.cu��.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID �����ID��ÿһ�������DepthͼƬ��Ҫһ�������д���
		 */
		void ClipFilterDepthImage(cudaStream_t stream, int CameraID);

		/**
		 * \brief ���ԭʼ�������.
		 * 
		 * \param CameraID ���ID
		 * \return ԭʼ�������(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getRawDepthCUDATexture(int CameraID) const { return rawDepthCUDATexture[CameraID].texture; }

		/**
		 * \brief ��ü��ú���������.
		 * 
		 * \param CameraID ���ID
		 * \return ���ú���������(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getClipedDepthCUDATexture(int CameraID) const { return clipedDepthCUDATexture[CameraID].texture; }

	private:
		void* depthBufferPagelock[MAX_CAMERA_COUNT];									// ���ͼ����ҳ�ڴ�
		void* depthBufferPagelockPrevious[MAX_CAMERA_COUNT];							// ǰһ֡���ͼ����ҳ�ڴ�
		DeviceArray<unsigned short> rawDepthImage[MAX_CAMERA_COUNT];					// ԭʼͼ����GPU�ϵ�buffer
		DeviceArray<unsigned short> rawDepthImagePrevious[MAX_CAMERA_COUNT];			// ǰһ֡ͼ����GPU�ϵ�buffer
		CudaTextureSurface rawDepthCUDATexture[MAX_CAMERA_COUNT];						// ԭʼͼƬ������
		CudaTextureSurface clipedDepthCUDATexture[MAX_CAMERA_COUNT];					// ����ͼƬ������(���ӽ����ƺ�)
		/**
		 * \brief �������ͼ������.
		 * 
		 */
		void allocateDepthTexture();

		/**
		 * \brief �ͷ����ͼ������.
		 * 
		 */
		void releaseDepthTexture();

/*********************************************     RGBͼ���ܶ�(�Ҷ�)ͼ     *********************************************/
	// ��Ҫע������ܶ�(�Ҷ�)ͼGrayScaleImage�����������������Ѱ��ǰ��ForeGround
	public:
		/**
		 * \brief �ϴ�RGBͼ��GPU������.
		 *
		 * \param stream cuda��ID
		 */
		void UploadRawColorImageToGPU(cudaStream_t stream, int CameraID);

		/**
		 * \brief ���ò���һ��RGBͼ�񣬾��庯��ʵ����ImageProcessorByGPU.cu��.
		 *
		 * \param stream cuda��ID
		 * \param CameraID �����ID��ÿ�������RGBͼƬ��Ҫһ����������
		 */
		void ClipNormalizeColorImage(cudaStream_t stream, int CameraID);

		/**
		 * \brief ���GPU�ϵ�ԭʼΪ�ü����˲���RGBͼ��.
		 * 
		 * \param CameraID ���ID
		 * \return ԭʼ�洢��GPU�ϵ�ͼ������(������DeviceArray<uchar3>& ����)
		 */
		const DeviceArray<uchar3>& getRawColorImage(int CameraID) const { return rawColorImage[CameraID]; }

		/**
		 * \brief ��ü��ú��RGBͼ�������.
		 * 
		 * \param CameraID ���ID
		 * \return �����˲���һ����RGBͼ�������(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getClipedNormalizeColorImageTexture(int CameraID) const { return clipedNormalizeColorImage[CameraID].texture; }

		/**
		 * \brief ��ü����˲���һ�����RGBͼ�������.
		 * 
		 * \param CameraID ���ID
		 * \return �����˲���һ�����RGBͼ�������(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getClipedNormalizeColorImagePreviousTexture(int CameraID) const { return clipedNormalizeColorImagePrevious[CameraID].texture; }

		/**
		 * \brief ���RGB���ú�ת�ɵĻҶ�(�ܶ�)ͼ������.
		 * 
		 * \param CameraID ���ID
		 * \return RGB���ú�ת�ɵĻҶ�(�ܶ�)ͼ������(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getGrayScaleImageTexture(int CameraID) const { return GrayScaleImage[CameraID].texture; }

	private:
		void* colorBufferPagelock[MAX_CAMERA_COUNT];								// RGBͼ����ҳ�ڴ�
		void* colorBufferPagelockPrevious[MAX_CAMERA_COUNT];						// ǰһ֡RGBͼ����ҳ�ڴ�
		//RGBͼ���buffer������GPU�ϣ���flatten buffer����ʽ(һά)�洢
		DeviceArray<uchar3> rawColorImage[MAX_CAMERA_COUNT];						// ԭʼͼ����GPU�ϵ�buffer
		DeviceArray<uchar3> rawColorImagePrevious[MAX_CAMERA_COUNT];				// ǰһ֡ͼ����GPU�ϵ�buffer
		//�ü��ͱ�׼����rgbͼ��(float4)												  
		CudaTextureSurface clipedNormalizeColorImage[MAX_CAMERA_COUNT];				// ��һ�����õ�RGBͼ�񣬴洢��GPU�����������ڴ���
		CudaTextureSurface clipedNormalizeColorImagePrevious[MAX_CAMERA_COUNT];		// ǰһ֡��һ�����õ�RGBͼ�񣬴洢��GPU�����������ڴ���
		//RGB��RGB_Previous���ܶ�(�Ҷ�)ͼ(float1)									    
		CudaTextureSurface GrayScaleImage[MAX_CAMERA_COUNT];						// �ܶ�(�Ҷ�)ͼ���洢��GPU�����������ڴ���
		CudaTextureSurface GrayScaleImageFiltered[MAX_CAMERA_COUNT];				// �˲�����ܶ�(�Ҷ�)ͼ���洢��GPU�����������ڴ���

		/**
		 * \brief ����RGBͼ��Ļ������򣬴���RGBͼ��GrayScaleImageͼ����GPU�е������ڴ�.
		 * 
		 */
		void allocateColorTexture();

		/**
		 * \brief �ͷ�RGBͼ��Ļ�������.
		 * 
		 */
		void releaseColorTexture();

/*********************************************     ��Ԫ����     *********************************************/
	private:
		CudaTextureSurface previousVertexConfidenceTexture[MAX_CAMERA_COUNT];		// ��һ֡float4����(x, y, z)Ϊ���㣬wΪ���Ŷ�ֵ
		CudaTextureSurface previousNormalRadiusTexture[MAX_CAMERA_COUNT];			// ��һ֡float4����(x, y, z)Ϊ���ߣ�wΪ�뾶

		CudaTextureSurface vertexConfidenceTexture[MAX_CAMERA_COUNT];				// float4����(x, y, z)Ϊ���㣬wΪ���Ŷ�ֵ
		CudaTextureSurface normalRadiusTexture[MAX_CAMERA_COUNT];					// float4����(x, y, z)Ϊ���ߣ�wΪ�뾶

		/**
		 * \brief ������Ԫ�����������㡢���ߡ��뾶�Լ����Ŷ�.
		 * 
		 */
		void allocateSurfelAttributeTexture();

		/**
		 * \brief �ͷ���Ԫ��������.
		 * 
		 */
		void releaseSurfelAttributeTexture();

	public:

		/**
		 * \brief ���춥������Ŷȵ�Map(һ�ֳ����float4����x,y,z�Ƕ������꣬w�Ƕ������Ŷ�).
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void buildVertexConfidenceMap(cudaStream_t stream, int CameraID);

		/**
		 * \brief ���취�ߺͰ뾶��Map(һ�ֳ����float4����x,y,z�Ƿ��ߣ�w�ǰ뾶).
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void buildNormalRadiusMap(cudaStream_t stream, int CameraID);

		/**
		 * \brief ����ǰһ֡�Ķ�������.
		 *
		 * \param CameraID ���ID
		 * \return ǰһ֡�Ķ�������
		 */
		cudaTextureObject_t getPreviousVertexConfidenceTexture(int CameraID) { return previousVertexConfidenceTexture[CameraID].texture; }

		/**
		 * \brief ����ǰһ֡�ķ�������.
		 * 
		 * \param CameraID ���ID
		 * \return ǰһ֡�ķ�������
		 */
		cudaTextureObject_t getPreviousNormalRadiusTexture(int CameraID) { return previousNormalRadiusTexture[CameraID].texture; }

	

		/**
		 * \brief ���ض��㼰���Ŷ�����.
		 *
		 * \param CameraID ���ID
		 * \return ���㼰���Ŷ�����(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getBackgroundVertexConfidenceTexture(int CameraID) { return icp_vertex_confidence[CameraID].texture; }
		cudaTextureObject_t getVertexConfidenceTexture(int CameraID) { return vertexConfidenceTexture[CameraID].texture; }

		/**
		 * \brief ���ط��߼��뾶����.
		 *
		 * \param CameraID ���ID
		 * \return �߼��뾶����(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getBackgroundNormalRadiusTexture(int CameraID) { return icp_normal_radius[CameraID].texture; }
		cudaTextureObject_t getNormalRadiusTexture(int CameraID) { return normalRadiusTexture[CameraID].texture; }

/*********************************************     Color��Time����Map     *********************************************/
		/**
		 * \brief ��RGBͼ�����ColorTimeͼ��Ӧ������Ԫ�����ʽ��ͬfloat4��x��RGBA�ı��룬.
		 */
	private:
		CudaTextureSurface colorTimeTexture[MAX_CAMERA_COUNT];						// ��Ԫ����һ�ο��������Ԫ��ʱ��(�ڼ�֡����)
		
		/**
		 * \brief ������Ԫ��ɫ����һ�ο��������Ԫ��ʱ��������.
		 */
		void allocateColorTimeTexture();

		/**
		 * \brief �ͷ���Ԫ��ɫ����һ�ο��������Ԫ��ʱ��������.
		 * 
		 */
		void releaseColorTimeTexture();

	public:
		/**
		 * \brief ����ColorTimeͼ(��RGBͼ�����ColorTimeͼ��Ӧ������Ԫ�����ʽ��ͬx,y,z��RGB��w��time).
		 * 
		 * \param frameIdx ��ǰ֡
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void buildColorTimeMap(size_t frameIdx, cudaStream_t stream, int CameraID);

		/**
		 * \brief ����Color-Time[RGB��ɫ����һ֡���������Ԫ��ʱ��]����.
		 * 
		 * \param CameraID ���ID
		 * \return Color-Time����(������cudaTextureObject_t����)
		 */
		cudaTextureObject_t getBackgroundColorTimeTexture(int CameraID) { return icp_color_time[CameraID].texture; }
		cudaTextureObject_t getColorTimeTexture(int CameraID) { return colorTimeTexture[CameraID].texture; }

/*********************************************     ���첢ѡ����Ч�������Ԫ     *********************************************/
	private:
		SurfelsProcessor::Ptr surfelsProcessor;
		//MergeSurface::Ptr mergeSurface;
		DeviceBufferArray<DepthSurfel> depthSurfel[MAX_CAMERA_COUNT];				// ��Ч�������Ԫ
		DeviceArrayView<DepthSurfel> depthSurfelView[MAX_CAMERA_COUNT];				// ֻ�ɶ���depthSurfel
		FlagSelection validDepthPixelSelector[MAX_CAMERA_COUNT];					// ��Ч�����Ԫɸѡ��
		DeviceBufferArray<DepthSurfel> preAlignedSurfel;							// �Ѿ���ɸ��Զ������Ԫ
		DeviceArrayView<DepthSurfel> preAlignedSurfelView;							// ֻ�������ں���Ԫ
		DeviceBufferArray<float4> subsampleSparseVertice[MAX_CAMERA_COUNT];			// �ɶ�д��ϡ�趥��



		/**
		 * \brief ������Ч��Ԫɸѡ���Ļ��棬��Ч��Ԫɸѡ�����Ը���.
		 * 
		 */
		void allocateValidSurfelSelectionBuffer();

		/**
		 * \brief �ͷ���Ч��Ԫɸѡ���Ļ���.
		 * 
		 */
		void releaseValidSurfelSelectionBuffer();

	public:
		/**
		 * \brief �ռ���Ч�����Ԫ.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void collectValidDepthSurfel(cudaStream_t stream, int CameraID);



		/**
		 * \brief ���ݵ�һ֡�������������λ�ã��������Ԫת����ͬһ��Canonical�ռ�.
		 * 
		 * \param solver ���������
		 * \param stream CUDA��ID
		 * \param CameraID ���ID
		 */
		void alignedDenseValidDepthSurfel(RigidSolver::Ptr solver, cudaStream_t stream, int CameraID);

		/**
		 * \brief ��õ�ǰ�����Ч��Ԫ������(ֻ������DeviceArrayView<DepthSurfel>).
		 * 
		 * \param CameraID ���ID
		 * \return ��ǰ�����Ч��Ԫ������
		 */
		DeviceArrayView<DepthSurfel> getValidDepthSurfelArray(int CameraID) { return depthSurfelView[CameraID]; }

		/**
		 * \brief ��õ�ǰ�����λ�˵�����������Ԫ������(ֻ������DeviceArrayView<DepthSurfel>).
		 *
		 * \return ��ǰ�����λ�˵�����������Ԫ������
		 */
		DeviceArrayView<DepthSurfel> getAlignedSurfelArray() { return preAlignedSurfelView; }

		/**
		 * \brief ��þ����²����Ķ���.
		 * 
		 * \param CameraID
		 * \return 
		 */
		DeviceArrayView<float4> getSubsampledSparseVertexArray(const unsigned int CameraID) { return subsampleSparseVertice[CameraID].ArrayView(); }
	
		
/*********************************************     ǰ���ָ�     *********************************************/
	private:
		bool OnlineSegmenter = true;
		ForegroundSegmenter::Ptr foregroundSegmenter[MAX_CAMERA_COUNT]; //ǰ���ָ�����Ϊÿһ��ͼƬ����һ���ָ������ָ�������洢�ڶ���֮��

		/**
		 * \brief ����ǰ���ָ����������ڴ棬����ģ��.
		 * 
		 */
		void allocateForegroundSegmentationBuffer();

		/**
		 * \brief �ͷ�ǰ���ָ����������ڴ�.
		 * 
		 */
		void releaseForegroundSegmentationBuffer();
	public:
		/**
		 * \brief ��Colorͼ�����ǰ���ָÿһ��ͼ�񵥶�����һ���ָ����������ָ�����������һ��stream�������ڲ��Ѿ���ͬ��(��ʱ10ms���ڶ�ȡͼ�����֮ǰ���ҵ�mask������ȫ��Ӧ�������¼���Ӧ��)��.
		 * 
		 * \param stream CUDA��ID
		 * \param CameraID ���ID
		 */
		void SegmentForeground(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief �ֶ�ͬ��segment stream��Ҳ����дcuda�¼�����.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void SyncAndConvertForeground2Texture(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief ��õ�ǰ�����(���ú�)ǰ��Mask.
		 * 
		 * \param CameraID ���ID
		 * \return ǰ��mask����
		 */
		cudaTextureObject_t getForegroundMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getClipedMaskTexture(); }

		/**
		 * \brief ����˲����ǰ��Mask.
		 * 
		 * \param CameraID ���ID
		 * \return �˲����ǰ��mask����
		 */
		cudaTextureObject_t getFilteredForegroundMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getFilteredMaskTexture(); }

		/**
		 * \brief ���ǰһ֡��(���ú�)ǰ��Mask.
		 * 
		 * \param CameraID ���ID
		 * \return ��һ֡��ǰ������
		 */
		cudaTextureObject_t getPreviousForegroundMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getPreviousMaskTexture(); }

		/**
		 * \brief ���ǰ����Ե��Mask.
		 * 
		 * \param CameraID ���ID
		 * \return ��ǰ֡ǰ����ԵMask
		 */
		cudaTextureObject_t getCurrentForegroundEdgeMaskTexture(const unsigned int CameraID) const { return foregroundSegmenter[CameraID]->getForegroundEdgeMaskTexture(); }

/*********************************************     ϡ��������ƥ��     *********************************************/
	private:
		PatchColliderRGBCorrespondence::Ptr featureCorrespondenceFinder[MAX_CAMERA_COUNT];

		/**
		 * \brief ����Ѱ��������ķ������ڴ棬ÿһ���������Ӧһ������.
		 * 
		 */
		void allocateFeatureCorrespondenceBuffer();



		/**
		 * \brief �ͷ�Ѱ�������㷽�����ڴ�.
		 * 
		 */
		void releaseFeatureCorrespondenceBuffer();
	
	public:
		/**
		 * \brief Ѱ���������������¿�һ��Stream�����ڲ�ͬ��.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void FindCorrespondence(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief �ӷָ����л��ƥ����.
		 * 
		 * \param CameraID ���ID
		 * \return ƥ����
		 */
		DeviceArray<ushort4> getCorrespondencePixelPair(const unsigned int CameraID) const { 
#ifndef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
			return featureCorrespondenceFinder[CameraID]->CorrespondedPixelPairs();
#else
			return opticalFlow[CameraID]->getCorrespondencePixelPair();
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

		}

/*********************************************     ������Ƶ����     *********************************************/
	private:
		OpticalFlow::Ptr opticalFlow[MAX_CAMERA_COUNT];
		DeviceArrayView2D<mat34> correctSe3Map[MAX_CAMERA_COUNT];
		DeviceArrayView2D<unsigned char> markValidSe3Map[MAX_CAMERA_COUNT];

		/**
		 * \brief ��������ڴ�.
		 * 
		 */
		void allocateOpticalFlowBuffer();

		/**
		 * \brief �ͷŹ����ڴ�.
		 * 
		 */
		void releaseOpticalFlowBuffer();

		/**
		 * \brief �������.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void computeOpticalFlow(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief ��ù�����ƥ������ص�.
		 * 
		 * \param CameraID ���ID
		 * \return ������ƥ������ص�
		 */
		DeviceArray<ushort4> getOpticalFlowPixelPair(const unsigned int CameraID) const { return opticalFlow[CameraID]->getCorrespondencePixelPair(); }
	public:
		/**
		 * \brief ��ù�������ͼ.
		 *
		 * \return ��������ͼ
		 */
		DeviceArrayView2D<mat34>* GetCorrectedSe3Maps() { return correctSe3Map; }

		/**
		 * \brief ���ϡ��ƥ�����Чֵ���ͼ.
		 *
		 * \return ϡ��ƥ�����Чֵ���ͼ
		 */
		DeviceArrayView2D<unsigned char>* GetValidSe3Maps() { return markValidSe3Map; }

/*********************************************     ����羵ƥ��㲢��ֵ     *********************************************/
	private:
		CrossViewEdgeCorrespondence::Ptr CrossViewMatching;
		CrossViewMatchingInterpolation::Ptr CrossViewInterpolation;
		/**
		 * \brief ����羵ƥ���ڴ�.
		 * 
		 */
		void allocateCrossViewMatchingBuffer();

		/**
		 * \brief �ͷſ羵ƥ���ڴ�.
		 * 
		 */
		void releaseCrossViewMatchingBuffer();

/*********************************************     ����Ҷ�ͼ��Mask���ݶ�     *********************************************/
	private:
		ImageGradient::Ptr imageGradient;										// ����ͼƬ���ݶ�

		CudaTextureSurface ForegroundMaskGradientMap[MAX_CAMERA_COUNT];			// ��¼ǰ����Ĥ���ݶ�ͼ
		CudaTextureSurface GrayscaleGradientMap[MAX_CAMERA_COUNT];				// ��¼�Ҷ�ͼ���ݶ�ͼ
		/**
		 * \brief �����ݶ�ͼ�Ļ���.
		 * 
		 */
		void allocateGradientMapBuffer();

		/**
		 * \brief �ͷ��ݶ�ͼ�Ļ���.
		 * 
		 */
		void releaseGradientMapBuffer();

		/**
		 * \brief ����ForegroundMask��Grayscale���ݶ�.
		 * 
		 * \param stream cuda��ID
		 * \param CameraID ���ID
		 */
		void ComputeGradientMap(cudaStream_t stream, const unsigned int CameraID);

		/**
		 * \brief ��ûҶ�ͼ���ݶ�����.
		 * 
		 * \param CameraID ���ID
		 * \return �Ҷ�ͼ���ݶ�����
		 */
		cudaTextureObject_t getGrayScaleGradientTexture(const unsigned int CameraID) const { return GrayscaleGradientMap[CameraID].texture; }

		/**
		 * \brief ���ǰ��Mask���ݶ�����.
		 * 
		 * \param CameraID ���ID
		 * \return ǰ��Mask���ݶ�����
		 */
		cudaTextureObject_t getForegroundMaskGradientTexture(const unsigned int CameraID) const { return ForegroundMaskGradientMap[CameraID].texture; }

	public:
		void setInput(vector<string>* colorpath, vector<string>* depthpath);

	};

}


