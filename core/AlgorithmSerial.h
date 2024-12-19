/*****************************************************************//**
 * \file   AlgorithmSerial.h
 * \brief  ��Ҫ�������̺߳����㷨
 * 
 * \author LUO
 * \date   January 24th 2024
 *********************************************************************/
#pragma once
#include <base/ConfigParser.h>
#include <base/Camera.h>
#include <ImageProcess/FrameProcessor.h>
#include <render/Renderer.h>
#include <render/DynamicallyDrawPoints.h>
#include <render/DynamicallyRenderSurfels.h>
#include <core/Geometry/SurfelGeometry.h>
#include <core/NonRigidSolver/WarpField.h>
#include <core/NonRigidSolver/NonRigidSolver.h>
#include <core/NonRigidSolver/CanonicalNodesSkinner.h>
#include <core/Geometry/SurfelNodeDeformer.h>
#include <core/Geometry/KNNBruteForceLiveNodes.h>
#include <core/Geometry/LiveGeometryUpdater.h>
#include <core/Geometry/WarpFieldExtender.h>
#include <core/Geometry/GeometryReinitProcessor.h>
#include <mesh/PoissonReconstruction.h>
#include <core/Geometry/FusionDepthGeometry.h>
#include <visualization/Visualizer.h>

namespace SparseSurfelFusion{	

	/**
	 * \brief ��Ҫ�������̺߳����㷨.
	 */
	class AlgorithmSerial
	{
	public:

		struct visualizerIO {
			//geometry
			DeviceArrayView<float4> canonicalVerticesConfidence;
			DeviceArrayView<float4> canonicalNormalRadius;
			DeviceArrayView<float4> liveVertexConfidence;			
			DeviceArrayView<float4> liveNormalRadius;			
			DeviceArrayView<float4> colorTime;	
			//Depthsurfel
			//DeviceArrayView<DepthSurfel> mergedDepthSurfels;
		};

		visualizerIO getVisualizerIO();

		/**
		 * \brief ��ʽ���ã�����ConfigParser��������.
		 * 
		 * \param config ����Լ�ͼ����û�������
		 */
		explicit AlgorithmSerial(std::shared_ptr<ThreadPool> threadPool, bool intoBuffer);
		~AlgorithmSerial();

		/**
		 * \brief ���õ�ǰ��֡ID.
		 * 
		 */
		void setFrameIndex(size_t frameidx);

		void setFrameIndex(size_t frameidx, const unsigned int beginIdx);

		/**
		 * \brief �Ƿ�����������Ѿ����Ի�ȡͼ��.
		 * 
		 * \return 
		 */
		bool AllCamerasReady() { return frameProcessor->CamerasReady(); }

		/**
		 * \brief �����һ֡������.
		 * 
		 */
		void ProcessFirstFrame();

		/**
		 * \brief ����֡��(�ǵ�һ֡).
		 * 
		 * \param SaveResult �Ƿ񱣴汾��
		 * \param RealTimeDisplay �Ƿ�ʵʱ��ʾ
		 * \param drawRecent �Ƿ�������
		 */
		void ProcessFrameStream(bool SaveResult , bool RealTimeDisplay, bool drawRecent = true);

		/**
		 * \brief ��ʾCamera�ĵ�RGB�����ͼ.
		 * 
		 */
		void showAllCameraImages();

		/**
		 * \brief �㷨�Ƿ�ֹͣ.
		 * 
		 * \return true �㷨ֹͣ�� false�㷨ûֹͣ�����ɼ�����
		 */
		bool AlgorithmStop() {
			return frameProcessor->isStop();
		}

		/**
		 * \brief ֹͣ���������ֹͣ�㷨.
		 * 
		 */
		void StopIt() {
			frameProcessor->StopAllCameras();

		}
		/**
		 * \brief ��ý������������.
		 * 
		 * \return �������������
		 */
		int getCameraCount() {
			return frameProcessor->getCameraCount();
		}


		void LoadImages(bool intoBuffer = true);
		


	private:

		bool intoBuffer = true;										// ���ݼ���ȡ��ʽ��true��һ����ȫ�����룬false��ÿ֡��һ��ͼƬ
		std::vector<cv::String> ColorOfflinePath[MAX_CAMERA_COUNT];	// ȫ�������ڴ��RGBͼ��ַ
		std::vector<cv::String> DepthOfflinePath[MAX_CAMERA_COUNT];	// ȫ�������ڴ�����ͼ��ַ
		std::vector<cv::Mat> ColorOffline[MAX_CAMERA_COUNT];		// ȫ�������ڴ��RGBͼ
		std::vector<cv::Mat> DepthOffline[MAX_CAMERA_COUNT];		// ȫ�������ڴ�����ͼ
		Intrinsic ClipedIntrinsic[MAX_CAMERA_COUNT];				// ���õ�����ڲ�

		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];				// �����λ��(�����0���������)
		Matrix4f initialWorld2Camera[MAX_CAMERA_COUNT];			// �����SE3��canical�������һ֡���ģ�͵ģ�������һ֡�Ļ�û����
		mat34 World2Camera[MAX_CAMERA_COUNT];					//
		mat34 Camera2World[MAX_CAMERA_COUNT];
		Renderer::SolverMaps solverMaps[MAX_CAMERA_COUNT];		// 
		FrameProcessor::Ptr frameProcessor;						// ͼ����
		ConfigParser::Ptr configParser;							// �����ͼ���������
		RigidSolver::Ptr rigidSolver;							// ICP���Զ��������
		Renderer::Ptr renderer;									// ��Ⱦ��
		SurfelGeometry::Ptr surfelGeometry[MAX_CAMERA_COUNT][2];// ��Ԫ���Σ��󶨵���Ⱦ�����ˣ����ﷴӳ���ǳ���sufel
		WarpField::Ptr warpField;								// Ť����������ؼ��ڵ㼰����
		NonRigidSolver::Ptr nonRigidSolver;						// �Ǹ��������
		CanonicalNodesSkinner::Ptr canonicalNodesSkinner;		// Canonical���еĽڵ���Ƥ����	
		Camera camera;											// 0����������е��SE�Ĵ洢��������
		int deviceCount = 0;									// �������
		size_t frameIndex = -1;									// ��ǰ֡��
		unsigned int updatedGeometryIndex = 0;					// ˫���弸�θ���


		const unsigned int imageClipedWidth = CLIP_WIDTH;		// ���к�ͼ��Ŀ��
		const unsigned int imageClipedHeight = CLIP_HEIGHT;		// ���к�ͼ��ĸ߶�
		const unsigned int FusionMapScale = d_fusion_map_scale;	// �ϲ��õĳ߶�


		unsigned int maxDenseSurfels = 0;
		unsigned int maxNodesNum = 0;
		unsigned int refreshFrameNum = 0;
		unsigned int fusedFrameNum = 0;

		unsigned int continuouslyFusedFramesCount = 0;
		unsigned int maxcontinuouslyFusedFramesNum = 0;
		size_t reinitFrameIndex;

		//The knn index for live and reference nodes  OK
		KNNBruteForceLiveNodes::Ptr m_live_nodes_knn_skinner;

		//The surfel geometry and their updater   
		LiveGeometryUpdater::Ptr m_live_geometry_updater;

		WarpFieldExtender::Ptr m_warpfield_extender;

		//����ˢ��ref  reinit
		GeometryReinitProcessor::Ptr m_geometry_reinit_processor;

		//������������
		PoissonReconstruction::Ptr m_poissonreconstruction;

		DynamicallyDrawPoints::Ptr dynamicallyDraw;

		DynamicallyRenderSurfels::Ptr renderPhong;
		DynamicallyRenderSurfels::Ptr renderNormal;
		DynamicallyRenderSurfels::Ptr renderAlbedo;
		//The decision function for integration and reinit
		bool shouldDoIntegration() const;
		bool shouldDoReinit(const size_t refreshFrameIndex) const;
		bool shouldDrawRecentObservation() const;

		//���ڼ�¼�ǲ���ˢ��can��ĵ�һ֡����һ֡��Ҫ�����ۻ��������
		bool isfirstframe;
		bool isRefreshFrame;

		void showRanderImage(const unsigned int num_vertex, int vao_idx, const Eigen::Matrix4f& world2camera, const Eigen::Matrix4f& init_world2camera, bool with_recent = true);
	
		void showAlignmentErrorMap(std::string renderType, float scale = 10.0f);
	};

}

