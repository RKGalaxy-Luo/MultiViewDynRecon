/*****************************************************************//**
 * \file   AlgorithmSerial.h
 * \brief  主要处理主线程核心算法
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
	 * \brief 主要处理主线程核心算法.
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
		 * \brief 显式调用，传入ConfigParser基本参数.
		 * 
		 * \param config 相机以及图像剪裁基本参数
		 */
		explicit AlgorithmSerial(std::shared_ptr<ThreadPool> threadPool, bool intoBuffer);
		~AlgorithmSerial();

		/**
		 * \brief 设置当前的帧ID.
		 * 
		 */
		void setFrameIndex(size_t frameidx);

		void setFrameIndex(size_t frameidx, const unsigned int beginIdx);

		/**
		 * \brief 是否所有相机都已经可以获取图像.
		 * 
		 * \return 
		 */
		bool AllCamerasReady() { return frameProcessor->CamerasReady(); }

		/**
		 * \brief 处理第一帧的数据.
		 * 
		 */
		void ProcessFirstFrame();

		/**
		 * \brief 处理帧流(非第一帧).
		 * 
		 * \param SaveResult 是否保存本地
		 * \param RealTimeDisplay 是否实时显示
		 * \param drawRecent 是否绘制最近
		 */
		void ProcessFrameStream(bool SaveResult , bool RealTimeDisplay, bool drawRecent = true);

		/**
		 * \brief 显示Camera拍的RGB和深度图.
		 * 
		 */
		void showAllCameraImages();

		/**
		 * \brief 算法是否停止.
		 * 
		 * \return true 算法停止， false算法没停止继续采集数据
		 */
		bool AlgorithmStop() {
			return frameProcessor->isStop();
		}

		/**
		 * \brief 停止所有相机，停止算法.
		 * 
		 */
		void StopIt() {
			frameProcessor->StopAllCameras();

		}
		/**
		 * \brief 获得接入相机的数量.
		 * 
		 * \return 接入相机的数量
		 */
		int getCameraCount() {
			return frameProcessor->getCameraCount();
		}


		void LoadImages(bool intoBuffer = true);
		


	private:

		bool intoBuffer = true;										// 数据集读取方式，true：一次性全部读入，false：每帧读一个图片
		std::vector<cv::String> ColorOfflinePath[MAX_CAMERA_COUNT];	// 全部读入内存的RGB图地址
		std::vector<cv::String> DepthOfflinePath[MAX_CAMERA_COUNT];	// 全部读入内存的深度图地址
		std::vector<cv::Mat> ColorOffline[MAX_CAMERA_COUNT];		// 全部读入内存的RGB图
		std::vector<cv::Mat> DepthOffline[MAX_CAMERA_COUNT];		// 全部读入内存的深度图
		Intrinsic ClipedIntrinsic[MAX_CAMERA_COUNT];				// 剪裁的相机内参

		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];				// 相机的位姿(相对于0号相机而言)
		Matrix4f initialWorld2Camera[MAX_CAMERA_COUNT];			// 这里的SE3是canical域对齐上一帧相机模型的，对齐这一帧的还没计算
		mat34 World2Camera[MAX_CAMERA_COUNT];					//
		mat34 Camera2World[MAX_CAMERA_COUNT];
		Renderer::SolverMaps solverMaps[MAX_CAMERA_COUNT];		// 
		FrameProcessor::Ptr frameProcessor;						// 图像处理
		ConfigParser::Ptr configParser;							// 相机及图像基本参数
		RigidSolver::Ptr rigidSolver;							// ICP刚性对齐求解器
		Renderer::Ptr renderer;									// 渲染器
		SurfelGeometry::Ptr surfelGeometry[MAX_CAMERA_COUNT][2];// 面元几何，绑定到渲染器上了，这里反映的是稠密sufel
		WarpField::Ptr warpField;								// 扭曲场，处理关键节点及参数
		NonRigidSolver::Ptr nonRigidSolver;						// 非刚性求解器
		CanonicalNodesSkinner::Ptr canonicalNodesSkinner;		// Canonical域中的节点蒙皮方法	
		Camera camera;											// 0号相机：所有点的SE的存储迭代在这
		int deviceCount = 0;									// 相机数量
		size_t frameIndex = -1;									// 当前帧数
		unsigned int updatedGeometryIndex = 0;					// 双缓冲几何更新


		const unsigned int imageClipedWidth = CLIP_WIDTH;		// 剪切后图像的宽度
		const unsigned int imageClipedHeight = CLIP_HEIGHT;		// 剪切后图像的高度
		const unsigned int FusionMapScale = d_fusion_map_scale;	// 上采用的尺度


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

		//用于刷新ref  reinit
		GeometryReinitProcessor::Ptr m_geometry_reinit_processor;

		//用于生成网格
		PoissonReconstruction::Ptr m_poissonreconstruction;

		DynamicallyDrawPoints::Ptr dynamicallyDraw;

		DynamicallyRenderSurfels::Ptr renderPhong;
		DynamicallyRenderSurfels::Ptr renderNormal;
		DynamicallyRenderSurfels::Ptr renderAlbedo;
		//The decision function for integration and reinit
		bool shouldDoIntegration() const;
		bool shouldDoReinit(const size_t refreshFrameIndex) const;
		bool shouldDrawRecentObservation() const;

		//用于记录是不是刷新can后的第一帧，第一帧需要重新累积对齐矩阵
		bool isfirstframe;
		bool isRefreshFrame;

		void showRanderImage(const unsigned int num_vertex, int vao_idx, const Eigen::Matrix4f& world2camera, const Eigen::Matrix4f& init_world2camera, bool with_recent = true);
	
		void showAlignmentErrorMap(std::string renderType, float scale = 10.0f);
	};

}

