/*****************************************************************//**
 * \file   NonRigidSolver.h
 * \brief  非刚性变换求解器
 * 
 * \author LUO
 * \date   March 28th 2024
 *********************************************************************/
#pragma once
#include <base/Camera.h>
#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/CommonUtils.h>
#include <base/GlobalConfigs.h>
#include <base/CameraObservation.h>

#include <render/Renderer.h>
#include "WarpField.h"
#include "CanonicalNodesSkinner.h"
#include "SolverIterationData.h"
#include "ImageTermKNNFetcher.h"
#include "DenseDepthHandler.h"
#include "DensityForegroundMapHandler.h"
#include "SparseCorrespondenceHandler.h"
#include "Node2TermsIndex.h"
#include "NodePair2TermsIndex.h"
#include "NodeGraphSmoothHandler.h"
#include "PreconditionerRhsBuilder.h"
#include "ResidualEvaluator.h"
#include "ApplyJtJMatrixFreeHandler.h"
#include "JtJMaterializer.h"
#include "BlockPCG.h"
#include "CrossViewCorrespondenceHandler.h"
#include <vector>
#include <visualization/Visualizer.h>

namespace SparseSurfelFusion {
	namespace device {

		enum {
			FlowSearchWindowHalfSize = 4,
			CorrSearchWindowHalfSize = 2
		};

		struct GuidedNodesInput {
			cudaTextureObject_t preVertexMap[MAX_CAMERA_COUNT];					// 当前帧，当前视角观察的数据
			cudaTextureObject_t preNormalMap[MAX_CAMERA_COUNT];					// 当前帧，当前视角观察的数据
			mat34 initialCameraSE3[MAX_CAMERA_COUNT];							// 位姿变换
			mat34 initialCameraSE3Inverse[MAX_CAMERA_COUNT];					// 位姿反变换
			mat34 world2Camera[MAX_CAMERA_COUNT];								// world2Camera
			Intrinsic intrinsic[MAX_CAMERA_COUNT];								// 内参矩阵
			DeviceArrayView2D<mat34> guidedSe3Map[MAX_CAMERA_COUNT];			// 用于引导的SE3矩阵
			DeviceArrayView2D<unsigned char> markValidSe3Map[MAX_CAMERA_COUNT];	// 主要是针对稀疏特征构造的Se3Map，标记有效值，稠密光流无需该Map
		};

		/**
		 * \brief 引导校正打包输入.
		 */
		struct CorrectInput {
			// SurfelGeometry
			float4* denseLiveSurfelsVertex[MAX_CAMERA_COUNT];	// 稠密面元Canonical域Vertex
			float4* denseLiveSurfelsNormal[MAX_CAMERA_COUNT];	// 稠密面元Canonical域Normal
			ushort4* surfelKnn[MAX_CAMERA_COUNT];				// 【Knn关系不变】面元的KNN
			float4* surfelKnnWeight[MAX_CAMERA_COUNT];			// 【Knn关系不变】面元KNN的权重

			// WarpField
			DualQuaternion* nodesSE3;							// 节点SE3，Canonical转Live
			float4* liveNodesCoordinate;						// Live域节点坐标
			ushort4* nodesKNN;									// 【Knn关系不变】与节点邻近的点的index
			float4* nodesKNNWeight;								// 【Knn关系不变】与节点邻近的点的Weight(权重)

			// Corretion
			DualQuaternion* dqCorrectNodes;						// 用作校正的节点SE3
			mat34* correctedCanonicalSurfelsSE3;				// 作用到节点上的SE3
		};

	}

	class NonRigidSolver {
/*********************************************     整体参数     *********************************************/
	private:
		const unsigned int imageWidth = FRAME_WIDTH - 2 * CLIP_BOUNDARY;		// 图片的宽
		const unsigned int imageHeight = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;		// 图片的高
		const unsigned int devicesCount = MAX_CAMERA_COUNT;						// 相机数量
		CameraObservation observation;								// 相机获取的参数输入
		SurfelGeometry::NonRigidSolverInput denseSurfelsInput;		// 稠密面元输入
		WarpField::NonRigidSolverInput sparseSurfelsInput;			// 稀疏节点输入

		SolverIterationData iterationData;
		Renderer::SolverMaps solverMap[MAX_CAMERA_COUNT];
		mat34 world2Camera[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		Intrinsic ClipColorIntrinsic[MAX_CAMERA_COUNT];				// 剪裁图像的内参

		DeviceBufferArray<DualQuaternion> dqCorrectNodes;
		DeviceBufferArray<mat34> correctedCanonicalSurfelsSE3;

	public:
		using Ptr = std::shared_ptr<NonRigidSolver>;
		/**
		 * \brief 构造函数.
		 * 
		 */
		NonRigidSolver(Intrinsic* intr = nullptr);
		/**
		 * \brief 默认析构函数.
		 * 
		 */
		~NonRigidSolver() = default;

		// 不可隐式复制/赋值/移动
		NO_COPY_ASSIGN_MOVE(NonRigidSolver);

		/**
		 * \brief 分配内存.
		 * 
		 */
		void AllocateBuffer();

		/**
		 * \brief 释放内存.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief 设置非刚性求解器的输入.
		 * 
		 * \param cameraObservation 相机获得的数据
		 * \param denseInput 稠密点
		 * \param sparseInput 稀疏节点
		 * \param canonical2Live 从Canonical转到Live的刚性位姿变换
		 */
		void SetSolverInput(
			CameraObservation cameraObservation, 
			SurfelGeometry::NonRigidSolverInput denseInput,
			WarpField::NonRigidSolverInput sparseInput, 
			mat34* canonical2Live,
			Renderer::SolverMaps* solverMaps, 
			Intrinsic* ClipColorIntrinsic
		);

		device::GuidedNodesInput guidedNodesInput;

		/**
		 * \brief 构造计算引导节点SE3的输入.
		 * 
		 * \param opticalMap 光流Se3Map
		 * \param observation 观察的数据
		 * \param InitialCameraSE3Array 相机位姿SE3矩阵
		 * \param intrinsicArray 相机内参Array
		 * \param devicesCount 相机数量
		 * \param input 构造的核函数输入
		 */
		void SetComputeOpticalGuideNodesSe3Input(const DeviceArrayView2D<mat34>* opticalMap, CameraObservation observation, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, const unsigned int devicesCount, device::GuidedNodesInput& input);

		/**
		 * \brief 构造计算引导节点SE3的输入.
		 * 
		 * \param corrMap 稀疏相关点构成的Map
		 * \param markValidSe3Map 标记有效Se3的Map
		 * \param observation 观察的数据
		 * \param InitialCameraSE3Array 相机位姿SE3矩阵
		 * \param world2camera Live转camera
		 * \param intrinsicArray 相机内参Array
		 * \param devicesCount 相机数量
		 * \param input 构造的核函数输入
		 */
		void SetComputeCorrespondenceGuideNodeSe3Input(const DeviceArrayView2D<mat34>* corrMap, const DeviceArrayView2D<unsigned char>* markValidSe3Map, CameraObservation observation, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, const unsigned int devicesCount, device::GuidedNodesInput& input);

		/**
		 * \brief 构造对live域node、vertex以及node和nodeSe3的引导变换.
		 * 
		 * \param warpField 扭曲场：存储着非刚性变换的node以及nodeSe3
		 * \param geometry 几何场：存储着稠密面元的相关信息，涉及SolverMap和FusionMap
		 * \param devicesCount 相机数量
		 * \param updatedGeometryIndex 当前双缓冲index
		 * \param correctInput 构造的输入
		 */
		void SetOpticalGuideInput(WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int devicesCount, const unsigned int updatedGeometryIndex, device::CorrectInput& correctInput);

		/**
		 * \brief 光流引导节点，校正：1、live域Vertex和Normal  2、node和nodeSE3.
		 * 
		 * \param opticalMap 光流Se3Map
		 * \param observation observation观察到的vertex
		 * \param nodeUnitedError 以该节点为邻居节点的稠密点的误差(fabs_diff_xyz * weight)全部累计到这个节点，并将其归一化
		 * \param InitialCameraSE3Array 相机位姿SE3矩阵
		 * \param intrinsicArray 相机内参Array
		 * \param warpField 扭曲场：存储着非刚性变换的node以及nodeSe3
		 * \param geometry 几何场：存储着稠密面元的相关信息，涉及SolverMap和FusionMap
		 * \param updatedGeometryIndex 当前双缓冲index
		 * \param  
		 */
		void CorrectLargeErrorNode(const unsigned int frameIdx, const DeviceArrayView2D<mat34>* opticalMap, CameraObservation observation, DeviceArrayView<float> nodeUnitedError, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int updatedGeometryIndex, cudaStream_t stream = 0);

		/**
		 * \brief 稀疏特征引导大误差节点校正.
		 * 
		 * \param frameIdx
		 * \param correspondenceSe3Map 稀疏特征计算得到的Se3Map
		 * \param markValidSe3Map 标记是否是有效的Se3Map值
		 * \param observation  observation观察到的vertex
		 * \param nodeUnitedError 以该节点为邻居节点的稠密点的误差(fabs_diff_xyz * weight)全部累计到这个节点，并将其归一化
		 * \param InitialCameraSE3Array 相机位姿SE3矩阵
		 * \param world2camera Live域转Camera域
		 * \param intrinsicArray 相机内参Array
		 * \param warpField 扭曲场：存储着非刚性变换的node以及nodeSe3
		 * \param geometry 几何场：存储着稠密面元的相关信息，涉及SolverMap和FusionMap
		 * \param updatedGeometryIndex 当前双缓冲index
		 * \param stream stream流
		 */
		void CorrectLargeErrorNode(const unsigned int frameIdx, const DeviceArrayView2D<mat34>* correspondenceSe3Map, const DeviceArrayView2D<unsigned char>* markValidSe3Map, CameraObservation observation, DeviceArrayView<float> nodeUnitedError, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int updatedGeometryIndex, cudaStream_t stream = 0);
		//最终输出结果
		DeviceArrayView<DualQuaternion> SolvedNodeSE3() const { return iterationData.CurrentWarpFieldInput(); }

		// 光流校正reference的结果
		DeviceArrayView<mat34> CorrectedCanonicalSurfelsSE3() const { return correctedCanonicalSurfelsSE3.ArrayView(); }
		//求解器
		void SolveSerial();


/*********************************************     流模式处理     *********************************************/
	private:
		cudaStream_t solverStreams[MAX_NONRIGID_CUDA_STREAM];	// 求解器流处理器

		/**
		 * \brief 初始化求解的流.
		 * 
		 */
		void initSolverStreams();

		/**
		 * \brief 释放求解器的流.
		 * 
		 */
		void releaseSolverStreams();
		
		/**
		 * \brief 同步所有求解器的流.
		 * 
		 */
		void synchronizeAllSolverStreams();

		/**
		 * \brief 绘制跨镜和上下两帧的Target和Canonical.
		 * 
		 */
		void showTargetAndCanonical(DeviceArrayView<float4> corrPairsTarget, DeviceArrayView<float4> corrPairsCan, DeviceArrayView<float4> corssPairsTarget, DeviceArrayView<float4> corssPairsCan);
	public:
		/**
		 * \brief 执行非刚性对齐求解.
		 * 
		 */
		void SolveNonRigidAlignment();

	private:
		void buildSolverIndexStreamed();
		void solverIterationGlobalIterationStreamed();
		void solverIterationLocalIterationStreamed();

		
		/* The pcg solver
		 */
	private:
		BlockPCG<6>::Ptr m_pcg_solver;
		void allocatePCGSolverBuffer();
		void releasePCGSolverBuffer();
	public:
		void UpdatePCGSolverStream(cudaStream_t stream);
		void SolvePCGMatrixFree();
		void SolvePCGMaterialized(int pcg_iterations = 10);



		/* Materialize the JtJ matrix
		 */
	private:
		JtJMaterializer::Ptr m_jtj_materializer;
		void allocateMaterializedJtJBuffer();
		void releaseMaterializedJtJBuffer();
	public:
		void SetJtJMaterializerInput();
		void MaterializeJtJNondiagonalBlocks(cudaStream_t stream = 0);
		void MaterializeJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream = 0);


		/* The method to apply JtJ to a vector
		 */
	private:
		ApplyJtJHandlerMatrixFree::Ptr m_apply_jtj_handler;
		void allocateJtJApplyBuffer();
		void releaseJtJApplyBuffer();






		/* The residual evaluator, the input is the same as above
		 */
	private:
		ResidualEvaluator::Ptr m_residual_evaluator;
		void allocateResidualEvaluatorBuffer();
		void releaseResidualEvaluatorBuffer();
	public:
		float ComputeTotalResidualSynced(cudaStream_t stream = 0);






		/* Construct the preconditioner and rhs of the method
		 */
	private:
		PreconditionerRhsBuilder::Ptr m_preconditioner_rhs_builder;
		void allocatePreconditionerRhsBuffer();
		void releasePreconditionerRhsBuffer();
	public:
		void SetPreconditionerBuilderAndJtJApplierInput();

		//The interface for diagonal pre-conditioner
		void BuildPreconditioner(cudaStream_t stream = 0);
		void BuildPreconditionerGlobalIteration(cudaStream_t stream = 0);

		//The interface for jt residual
		void ComputeJtResidual(cudaStream_t stream = 0);
		void ComputeJtResidualGlobalIteration(cudaStream_t stream = 0);
		DeviceArrayView<float> JtResidualValue() const { return m_preconditioner_rhs_builder->JtDotResidualValue(); }
		DeviceArrayView<float> JtJDiagonalBlockValue() const { return m_preconditioner_rhs_builder->JtJDiagonalBlocks(); }







       /* Hand in the value to node graph term handler
		 */
	private:
		NodeGraphSmoothHandler::Ptr m_graph_smooth_handler;
		void allocateSmoothTermHandlerBuffer();
		void releaseSmoothTermHandlerBuffer();
		void computeSmoothTermNode2Jacobian(cudaStream_t stream);






		/* Build the node to term index
		 * Depends: correspond depth, valid pixel, node graph, sparse feature
		 */
	private:
		Node2TermsIndex::Ptr m_node2term_index;			
		NodePair2TermsIndex::Ptr m_nodepair2term_index;
		void allocateNode2TermIndexBuffer();
		void releaseNode2TermIndexBuffer();
	public:
		void SetNode2TermIndexInput();
		void BuildNode2TermIndex(cudaStream_t stream = 0);
		void BuildNodePair2TermIndexBlocked(cudaStream_t stream = 0);

		/* Compute the jacobians for all terms
		 */
	public:
		void ComputeTermJacobiansFreeIndex(
			cudaStream_t dense_depth = 0,
			cudaStream_t density_map = 0,
			cudaStream_t foreground_mask = 0,
			cudaStream_t sparse_feature = 0,
			cudaStream_t crossViewCorrStream = 0
		);
		void ComputeTermJacobianFixedIndex(
			cudaStream_t denseDepthStream = 0,
			cudaStream_t densityMapStream = 0,
			cudaStream_t foregroundMaskStream = 0,
			cudaStream_t sparseFeatureStream = 0,
			cudaStream_t crossViewCorrStream = 0
		);







		/* Hand in the vertex maps and pixel pairs
		 * to sparse feature handler
		 */
	private:
		SparseCorrespondenceHandler::Ptr m_sparse_correspondence_handler;
		void allocateSparseFeatureBuffer();
		void releaseSparseFeatureBuffer();
	public:
		void SetSparseFeatureHandlerFullInput();
		void SelectValidSparseFeatureMatchedPairs(cudaStream_t stream = 0);

		unsigned int frameIdx = 0;


		/**
		 *  跨镜匹配点.
		 */
	private:
		CrossViewCorrespondenceHandler::Ptr cross_view_correspondence_handler;
		void allocateCrossViewCorrPairsBuffer();
		void releaseCrossViewCorrPairsBuffer();
	public:
		void SetCrossViewMatchingHandlerInput();
		



		/* Hand in the color and foreground
		 * mask to valid pixel compactor
		 * Depends: QueryPixelKNN
		 */
	private:
		DensityForegroundMapHandler::Ptr m_density_foreground_handler;
		void allocateDensityForegroundMapBuffer();
		void releaseDensityForegroundMapBuffer();
		void setDensityForegroundHandlerFullInput();
	public:
		void FindValidColorForegroundMaskPixel(cudaStream_t color_stream = 0, cudaStream_t mask_stream = 0);//4.19这个函数需要改
		void FindPotentialForegroundMaskPixelSynced(cudaStream_t stream = 0);







		/* Hand in the geometry maps to
		 * depth correspondence finder
		 * Depends: QueryPixelKNN, FetchPotentialDenseImageTermPixels
		 */
	private:
		DenseDepthHandler::Ptr m_dense_depth_handler;
		void allocateDenseDepthBuffer();
		void releaseDenseDepthBuffer();
		void setDenseDepthHandlerFullInput();
	/*	Intrinsic ClipColorIntrinsic[MAX_CAMERA_COUNT];   */                 //densedepthhandler要用,这个只是0号相机的内参，可能需要修改成两个的

	public:
		void setClipColorIntrinsic(Intrinsic* intr);  //给nonrigidsolver赋值用的，已经用了

		void FindCorrespondDepthPixelPairsFreeIndex(cudaStream_t stream = 0);

		///**
		// * \brief 计算刚性对齐在节点上的误差.
		// * 
		// * \param stream cuda流
		// */
		//void ComputeRigidAlignmentErrorOnNodes(CameraObservation observation, mat34* world2, cudaStream_t stream = 0);


		/**
		 * \brief 计算非刚性求解稠密点对齐误差，映射到误差图片上.
		 * 
		 * \param stream cuda流
		 */
		void ComputeAlignmentErrorMapDirect(cudaStream_t stream = 0);
		/**
		 * \brief 计算非刚性求解节点对齐误差，映射到误差图片上.
		 * 
		 * \param stream cuda流
		 */
		void ComputeAlignmentErrorOnNodes(const bool printNodeError = false, cudaStream_t stream = 0);
		/**
		 * \brief 计算节点的对齐误差.
		 * 
		 * \param stream cuda流
		 */
		void ComputeAlignmentErrorMapFromNode(cudaStream_t stream = 0);
		cudaTextureObject_t GetAlignmentErrorMap(const unsigned int CameraID) const { return m_dense_depth_handler->GetAlignmentErrorMap(CameraID); }
		NodeAlignmentError GetNodeAlignmentError() const { return m_dense_depth_handler->GetNodeAlignmentError(); }
		DeviceArrayView<float> GetNodeUnitAlignmentError() { return m_dense_depth_handler->GetNodeUnitAccumulatedError(); }






		/* Fetch the potential valid image term pixels, knn and weight
		 */
	private:
		ImageTermKNNFetcher::Ptr imageKnnFetcher;
		//void allocateImageKNNFetcherBuffer();直接加到了allocate中了
		//void releaseImageKNNFetcherBuffer();
	public:
		void FetchPotentialDenseImageTermPixelsFixedIndexSynced(cudaStream_t stream = 0);//4.19这函数还没用，里边需要改下



		/* Query the KNN for pixels given index map
		 * The knn map is in the same resolution as image
		 */
	private:
		DeviceArray2D<KNNAndWeight> knnMap[MAX_CAMERA_COUNT];

	public:
		void QueryPixelKNN(cudaStream_t stream = 0);





	//	//These are private interface
	//private:
	//	//The matrix-free index free solver interface
	//	void solveMatrixIndexFreeSerial(cudaStream_t stream = 0);
	//	void fullSolverIterationMatrixFreeSerial(cudaStream_t stream = 0);

	//	//The matrix-free solver which build index in the first iteration, and reuse the index
	//	void solveMatrixFreeFixedIndexSerial(cudaStream_t stream = 0);
	//	void fullSolverIterationMatrixFreeFixedIndexSerial(cudaStream_t stream = 0);
	//	void matrixFreeFixedIndexSolverIterationSerial(cudaStream_t stream = 0);

	//	//The materialized index-free solver interface
	//	void solveMaterializedIndexFreeSerial(cudaStream_t stream = 0);
	//	void fullSolverIterationMaterializedIndexFreeSerial(cudaStream_t stream = 0);

	//	//The materialized fixed index solver interface
	//	void solveMaterializedFixedIndexSerial(cudaStream_t stream = 0);
	//	void fullSolverIterationMaterializedFixedIndexSerial(cudaStream_t stream = 0);
	//	void materializedFixedIndexSolverIterationSerial(cudaStream_t stream = 0);

	//	//The one distinguish between local and global iteration
	//	void solveMaterializedFixedIndexGlobalLocalSerial(cudaStream_t stream = 0);
	//	void fullGlobalSolverIterationMaterializedFixedIndexSerial(cudaStream_t stream = 0);
	//	void materializedFixedIndexSolverGlobalIterationSerial(cudaStream_t stream = 0);


	//	//The materialized, fixed-index and lazy evaluation solver interface
	//	void solveMaterializedLazyEvaluateSerial(cudaStream_t stream = 0);
	//	void fullSolverIterationMaterializedLazyEvaluateSerial(cudaStream_t stream = 0);
	//	void materializedLazyEvaluateSolverIterationSerial(cudaStream_t stream = 0);


	};
}


