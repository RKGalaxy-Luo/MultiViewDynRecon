/*****************************************************************//**
 * \file   NonRigidSolver.h
 * \brief  �Ǹ��Ա任�����
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
			cudaTextureObject_t preVertexMap[MAX_CAMERA_COUNT];					// ��ǰ֡����ǰ�ӽǹ۲������
			cudaTextureObject_t preNormalMap[MAX_CAMERA_COUNT];					// ��ǰ֡����ǰ�ӽǹ۲������
			mat34 initialCameraSE3[MAX_CAMERA_COUNT];							// λ�˱任
			mat34 initialCameraSE3Inverse[MAX_CAMERA_COUNT];					// λ�˷��任
			mat34 world2Camera[MAX_CAMERA_COUNT];								// world2Camera
			Intrinsic intrinsic[MAX_CAMERA_COUNT];								// �ڲξ���
			DeviceArrayView2D<mat34> guidedSe3Map[MAX_CAMERA_COUNT];			// ����������SE3����
			DeviceArrayView2D<unsigned char> markValidSe3Map[MAX_CAMERA_COUNT];	// ��Ҫ�����ϡ�����������Se3Map�������Чֵ�����ܹ��������Map
		};

		/**
		 * \brief ����У���������.
		 */
		struct CorrectInput {
			// SurfelGeometry
			float4* denseLiveSurfelsVertex[MAX_CAMERA_COUNT];	// ������ԪCanonical��Vertex
			float4* denseLiveSurfelsNormal[MAX_CAMERA_COUNT];	// ������ԪCanonical��Normal
			ushort4* surfelKnn[MAX_CAMERA_COUNT];				// ��Knn��ϵ���䡿��Ԫ��KNN
			float4* surfelKnnWeight[MAX_CAMERA_COUNT];			// ��Knn��ϵ���䡿��ԪKNN��Ȩ��

			// WarpField
			DualQuaternion* nodesSE3;							// �ڵ�SE3��CanonicalתLive
			float4* liveNodesCoordinate;						// Live��ڵ�����
			ushort4* nodesKNN;									// ��Knn��ϵ���䡿��ڵ��ڽ��ĵ��index
			float4* nodesKNNWeight;								// ��Knn��ϵ���䡿��ڵ��ڽ��ĵ��Weight(Ȩ��)

			// Corretion
			DualQuaternion* dqCorrectNodes;						// ����У���Ľڵ�SE3
			mat34* correctedCanonicalSurfelsSE3;				// ���õ��ڵ��ϵ�SE3
		};

	}

	class NonRigidSolver {
/*********************************************     �������     *********************************************/
	private:
		const unsigned int imageWidth = FRAME_WIDTH - 2 * CLIP_BOUNDARY;		// ͼƬ�Ŀ�
		const unsigned int imageHeight = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;		// ͼƬ�ĸ�
		const unsigned int devicesCount = MAX_CAMERA_COUNT;						// �������
		CameraObservation observation;								// �����ȡ�Ĳ�������
		SurfelGeometry::NonRigidSolverInput denseSurfelsInput;		// ������Ԫ����
		WarpField::NonRigidSolverInput sparseSurfelsInput;			// ϡ��ڵ�����

		SolverIterationData iterationData;
		Renderer::SolverMaps solverMap[MAX_CAMERA_COUNT];
		mat34 world2Camera[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		Intrinsic ClipColorIntrinsic[MAX_CAMERA_COUNT];				// ����ͼ����ڲ�

		DeviceBufferArray<DualQuaternion> dqCorrectNodes;
		DeviceBufferArray<mat34> correctedCanonicalSurfelsSE3;

	public:
		using Ptr = std::shared_ptr<NonRigidSolver>;
		/**
		 * \brief ���캯��.
		 * 
		 */
		NonRigidSolver(Intrinsic* intr = nullptr);
		/**
		 * \brief Ĭ����������.
		 * 
		 */
		~NonRigidSolver() = default;

		// ������ʽ����/��ֵ/�ƶ�
		NO_COPY_ASSIGN_MOVE(NonRigidSolver);

		/**
		 * \brief �����ڴ�.
		 * 
		 */
		void AllocateBuffer();

		/**
		 * \brief �ͷ��ڴ�.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief ���÷Ǹ��������������.
		 * 
		 * \param cameraObservation �����õ�����
		 * \param denseInput ���ܵ�
		 * \param sparseInput ϡ��ڵ�
		 * \param canonical2Live ��Canonicalת��Live�ĸ���λ�˱任
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
		 * \brief ������������ڵ�SE3������.
		 * 
		 * \param opticalMap ����Se3Map
		 * \param observation �۲������
		 * \param InitialCameraSE3Array ���λ��SE3����
		 * \param intrinsicArray ����ڲ�Array
		 * \param devicesCount �������
		 * \param input ����ĺ˺�������
		 */
		void SetComputeOpticalGuideNodesSe3Input(const DeviceArrayView2D<mat34>* opticalMap, CameraObservation observation, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, const unsigned int devicesCount, device::GuidedNodesInput& input);

		/**
		 * \brief ������������ڵ�SE3������.
		 * 
		 * \param corrMap ϡ����ص㹹�ɵ�Map
		 * \param markValidSe3Map �����ЧSe3��Map
		 * \param observation �۲������
		 * \param InitialCameraSE3Array ���λ��SE3����
		 * \param world2camera Liveתcamera
		 * \param intrinsicArray ����ڲ�Array
		 * \param devicesCount �������
		 * \param input ����ĺ˺�������
		 */
		void SetComputeCorrespondenceGuideNodeSe3Input(const DeviceArrayView2D<mat34>* corrMap, const DeviceArrayView2D<unsigned char>* markValidSe3Map, CameraObservation observation, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, const unsigned int devicesCount, device::GuidedNodesInput& input);

		/**
		 * \brief �����live��node��vertex�Լ�node��nodeSe3�������任.
		 * 
		 * \param warpField Ť�������洢�ŷǸ��Ա任��node�Լ�nodeSe3
		 * \param geometry ���γ����洢�ų�����Ԫ�������Ϣ���漰SolverMap��FusionMap
		 * \param devicesCount �������
		 * \param updatedGeometryIndex ��ǰ˫����index
		 * \param correctInput ���������
		 */
		void SetOpticalGuideInput(WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int devicesCount, const unsigned int updatedGeometryIndex, device::CorrectInput& correctInput);

		/**
		 * \brief ���������ڵ㣬У����1��live��Vertex��Normal  2��node��nodeSE3.
		 * 
		 * \param opticalMap ����Se3Map
		 * \param observation observation�۲쵽��vertex
		 * \param nodeUnitedError �Ըýڵ�Ϊ�ھӽڵ�ĳ��ܵ�����(fabs_diff_xyz * weight)ȫ���ۼƵ�����ڵ㣬�������һ��
		 * \param InitialCameraSE3Array ���λ��SE3����
		 * \param intrinsicArray ����ڲ�Array
		 * \param warpField Ť�������洢�ŷǸ��Ա任��node�Լ�nodeSe3
		 * \param geometry ���γ����洢�ų�����Ԫ�������Ϣ���漰SolverMap��FusionMap
		 * \param updatedGeometryIndex ��ǰ˫����index
		 * \param  
		 */
		void CorrectLargeErrorNode(const unsigned int frameIdx, const DeviceArrayView2D<mat34>* opticalMap, CameraObservation observation, DeviceArrayView<float> nodeUnitedError, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int updatedGeometryIndex, cudaStream_t stream = 0);

		/**
		 * \brief ϡ���������������ڵ�У��.
		 * 
		 * \param frameIdx
		 * \param correspondenceSe3Map ϡ����������õ���Se3Map
		 * \param markValidSe3Map ����Ƿ�����Ч��Se3Mapֵ
		 * \param observation  observation�۲쵽��vertex
		 * \param nodeUnitedError �Ըýڵ�Ϊ�ھӽڵ�ĳ��ܵ�����(fabs_diff_xyz * weight)ȫ���ۼƵ�����ڵ㣬�������һ��
		 * \param InitialCameraSE3Array ���λ��SE3����
		 * \param world2camera Live��תCamera��
		 * \param intrinsicArray ����ڲ�Array
		 * \param warpField Ť�������洢�ŷǸ��Ա任��node�Լ�nodeSe3
		 * \param geometry ���γ����洢�ų�����Ԫ�������Ϣ���漰SolverMap��FusionMap
		 * \param updatedGeometryIndex ��ǰ˫����index
		 * \param stream stream��
		 */
		void CorrectLargeErrorNode(const unsigned int frameIdx, const DeviceArrayView2D<mat34>* correspondenceSe3Map, const DeviceArrayView2D<unsigned char>* markValidSe3Map, CameraObservation observation, DeviceArrayView<float> nodeUnitedError, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int updatedGeometryIndex, cudaStream_t stream = 0);
		//����������
		DeviceArrayView<DualQuaternion> SolvedNodeSE3() const { return iterationData.CurrentWarpFieldInput(); }

		// ����У��reference�Ľ��
		DeviceArrayView<mat34> CorrectedCanonicalSurfelsSE3() const { return correctedCanonicalSurfelsSE3.ArrayView(); }
		//�����
		void SolveSerial();


/*********************************************     ��ģʽ����     *********************************************/
	private:
		cudaStream_t solverStreams[MAX_NONRIGID_CUDA_STREAM];	// �������������

		/**
		 * \brief ��ʼ��������.
		 * 
		 */
		void initSolverStreams();

		/**
		 * \brief �ͷ����������.
		 * 
		 */
		void releaseSolverStreams();
		
		/**
		 * \brief ͬ���������������.
		 * 
		 */
		void synchronizeAllSolverStreams();

		/**
		 * \brief ���ƿ羵��������֡��Target��Canonical.
		 * 
		 */
		void showTargetAndCanonical(DeviceArrayView<float4> corrPairsTarget, DeviceArrayView<float4> corrPairsCan, DeviceArrayView<float4> corssPairsTarget, DeviceArrayView<float4> corssPairsCan);
	public:
		/**
		 * \brief ִ�зǸ��Զ������.
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
		 *  �羵ƥ���.
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
		void FindValidColorForegroundMaskPixel(cudaStream_t color_stream = 0, cudaStream_t mask_stream = 0);//4.19���������Ҫ��
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
	/*	Intrinsic ClipColorIntrinsic[MAX_CAMERA_COUNT];   */                 //densedepthhandlerҪ��,���ֻ��0��������ڲΣ�������Ҫ�޸ĳ�������

	public:
		void setClipColorIntrinsic(Intrinsic* intr);  //��nonrigidsolver��ֵ�õģ��Ѿ�����

		void FindCorrespondDepthPixelPairsFreeIndex(cudaStream_t stream = 0);

		///**
		// * \brief ������Զ����ڽڵ��ϵ����.
		// * 
		// * \param stream cuda��
		// */
		//void ComputeRigidAlignmentErrorOnNodes(CameraObservation observation, mat34* world2, cudaStream_t stream = 0);


		/**
		 * \brief ����Ǹ��������ܵ������ӳ�䵽���ͼƬ��.
		 * 
		 * \param stream cuda��
		 */
		void ComputeAlignmentErrorMapDirect(cudaStream_t stream = 0);
		/**
		 * \brief ����Ǹ������ڵ������ӳ�䵽���ͼƬ��.
		 * 
		 * \param stream cuda��
		 */
		void ComputeAlignmentErrorOnNodes(const bool printNodeError = false, cudaStream_t stream = 0);
		/**
		 * \brief ����ڵ�Ķ������.
		 * 
		 * \param stream cuda��
		 */
		void ComputeAlignmentErrorMapFromNode(cudaStream_t stream = 0);
		cudaTextureObject_t GetAlignmentErrorMap(const unsigned int CameraID) const { return m_dense_depth_handler->GetAlignmentErrorMap(CameraID); }
		NodeAlignmentError GetNodeAlignmentError() const { return m_dense_depth_handler->GetNodeAlignmentError(); }
		DeviceArrayView<float> GetNodeUnitAlignmentError() { return m_dense_depth_handler->GetNodeUnitAccumulatedError(); }






		/* Fetch the potential valid image term pixels, knn and weight
		 */
	private:
		ImageTermKNNFetcher::Ptr imageKnnFetcher;
		//void allocateImageKNNFetcherBuffer();ֱ�Ӽӵ���allocate����
		//void releaseImageKNNFetcherBuffer();
	public:
		void FetchPotentialDenseImageTermPixelsFixedIndexSynced(cudaStream_t stream = 0);//4.19�⺯����û�ã������Ҫ����



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


