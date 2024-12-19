/*****************************************************************//**
 * \file   NonRigidSolver.cpp
 * \brief  非刚性求解器方法实现
 * 
 * \author LUO
 * \date   March 28th 2024
 *********************************************************************/
#include "NonRigidSolver.h"

SparseSurfelFusion::NonRigidSolver::NonRigidSolver(Intrinsic* intr)
{
	imageKnnFetcher = std::make_shared<ImageTermKNNFetcher>();
	m_node2term_index = std::make_shared<Node2TermsIndex>();
	m_nodepair2term_index = std::make_shared<NodePair2TermsIndex>();
	memset(&observation, 0, sizeof(observation));
	memset(&denseSurfelsInput, 0, sizeof(denseSurfelsInput));
	memset(&sparseSurfelsInput, 0, sizeof(sparseSurfelsInput));
	for (int i = 0; i < devicesCount; i++) {
		ClipColorIntrinsic[i] = intr[i];
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}
	AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::AllocateBuffer()
{
	for (int i = 0; i < devicesCount; i++) {
		knnMap[i].create(imageHeight, imageWidth);
	}
	dqCorrectNodes.AllocateBuffer(Constants::maxNodesNum);
	correctedCanonicalSurfelsSE3.AllocateBuffer(Constants::maxSurfelsNum);
	allocateDenseDepthBuffer();
	allocateDensityForegroundMapBuffer();
	allocateSparseFeatureBuffer();
	allocateCrossViewCorrPairsBuffer();
	allocateSmoothTermHandlerBuffer();
	allocateNode2TermIndexBuffer();
	allocatePreconditionerRhsBuffer();
	allocateResidualEvaluatorBuffer();
	allocateJtJApplyBuffer();

#if defined(USE_MATERIALIZED_JTJ)
	allocateMaterializedJtJBuffer();
#endif
	allocatePCGSolverBuffer();

	//Init the stream for cuda
	initSolverStreams();
}

void SparseSurfelFusion::NonRigidSolver::ReleaseBuffer()
{
	for (int i = 0; i < devicesCount; i++) {
		knnMap[i].release();
	}
	dqCorrectNodes.ReleaseBuffer();
	correctedCanonicalSurfelsSE3.ReleaseBuffer();
	//Destroy the stream
	releaseSolverStreams();
	//源代码也没有imageKnnFetcher的释放代码
	releaseDenseDepthBuffer();
	releaseDensityForegroundMapBuffer();
	releaseSparseFeatureBuffer();
	releaseCrossViewCorrPairsBuffer();
	releaseSmoothTermHandlerBuffer();
	releaseNode2TermIndexBuffer();
	releasePreconditionerRhsBuffer();
	releaseResidualEvaluatorBuffer();
	releaseJtJApplyBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	releaseMaterializedJtJBuffer();
#endif
	releasePCGSolverBuffer();

}

void SparseSurfelFusion::NonRigidSolver::SetSolverInput(
	CameraObservation cameraObservation, 
	SurfelGeometry::NonRigidSolverInput denseInput, 
	WarpField::NonRigidSolverInput sparseInput, 
	mat34* canonical2Live, 
	Renderer::SolverMaps* solverMaps,
	Intrinsic* ClipIntrinsic
)
{
	observation = cameraObservation;
	denseSurfelsInput = denseInput;
	sparseSurfelsInput = sparseInput;
	for (int i = 0; i < devicesCount; i++) {
		world2Camera[i] = canonical2Live[i];
		ClipColorIntrinsic[i] = ClipIntrinsic[i];
		solverMap[i] = solverMaps[i];
	}
	iterationData.SetWarpFieldInitialValue(sparseInput.nodesSE3);
	m_sparse_correspondence_handler->frameIdx = frameIdx;
	cross_view_correspondence_handler->frameIdx = frameIdx;
}






/* The method to materialized the JtJ matrix
 */
void SparseSurfelFusion::NonRigidSolver::allocateMaterializedJtJBuffer() {
	m_jtj_materializer = std::make_shared<JtJMaterializer>();
	m_jtj_materializer->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releaseMaterializedJtJBuffer() {
	m_jtj_materializer->ReleaseBuffer();
}

void SparseSurfelFusion::NonRigidSolver::SetJtJMaterializerInput() {
	// Map from node to term
	const Node2TermsIndex::Node2TermMap node2term = m_node2term_index->GetNode2TermMap();

	// Map from nodepair to term
	const NodePair2TermsIndex::NodePair2TermMap nodepair2term = m_nodepair2term_index->GetNodePair2TermMap();

	// The dense depth term
	const DenseDepthTerm2Jacobian dense_depth_term2jacobian = m_dense_depth_handler->Term2JacobianMap();

	// The node graph term
	const NodeGraphSmoothTerm2Jacobian smooth_term2jacobian = m_graph_smooth_handler->Term2JacobianMap();

	// The image map term
	DensityMapTerm2Jacobian density_term2jacobian;
	ForegroundMaskTerm2Jacobian foreground_term2jacobian;
	m_density_foreground_handler->Term2JacobianMaps(density_term2jacobian, foreground_term2jacobian);

	// The sparse feature term
	const Point2PointICPTerm2Jacobian feature_term2jacobian = m_sparse_correspondence_handler->Term2JacobianMap();

	const Point2PointICPTerm2Jacobian cross_corr_term2jacobian = cross_view_correspondence_handler->Term2JacobianMap();

	// The penalty constants
	const auto penalty_constants = iterationData.CurrentPenaltyConstants();

	//Hand in to materializer
	m_jtj_materializer->SetInputs(
		nodepair2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		cross_corr_term2jacobian,
		node2term,
		penalty_constants
	);
}

void SparseSurfelFusion::NonRigidSolver::MaterializeJtJNondiagonalBlocks(cudaStream_t stream) {
	m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocks(stream);
}

void SparseSurfelFusion::NonRigidSolver::MaterializeJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream) {
	m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocksGlobalIteration(stream);
}







/* The method to apply JtJ to a given vector
 */
void SparseSurfelFusion::NonRigidSolver::allocateJtJApplyBuffer()
{
	m_apply_jtj_handler = std::make_shared<ApplyJtJHandlerMatrixFree>();
	m_apply_jtj_handler->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releaseJtJApplyBuffer()
{
	m_apply_jtj_handler->ReleaseBuffer();
}







/* Compute the preconditioner and linear equation rhs for later use
 */
void SparseSurfelFusion::NonRigidSolver::allocatePreconditionerRhsBuffer() {
	m_preconditioner_rhs_builder = std::make_shared<PreconditionerRhsBuilder>();
	m_preconditioner_rhs_builder->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releasePreconditionerRhsBuffer() {
	m_preconditioner_rhs_builder->ReleaseBuffer();
}

void SparseSurfelFusion::NonRigidSolver::SetPreconditionerBuilderAndJtJApplierInput() {
	//Map from node to term
	const Node2TermsIndex::Node2TermMap node2term = m_node2term_index->GetNode2TermMap();

	// 稠密深度面元项
	const DenseDepthTerm2Jacobian dense_depth_term2jacobian = m_dense_depth_handler->Term2JacobianMap();

	// 节点项(平滑项)
	const NodeGraphSmoothTerm2Jacobian smooth_term2jacobian = m_graph_smooth_handler->Term2JacobianMap();

	// 梯度前景项
	DensityMapTerm2Jacobian density_term2jacobian;
	ForegroundMaskTerm2Jacobian foreground_term2jacobian;
	m_density_foreground_handler->Term2JacobianMaps(density_term2jacobian, foreground_term2jacobian);

	// 匹配点项
	const Point2PointICPTerm2Jacobian feature_term2jacobian = m_sparse_correspondence_handler->Term2JacobianMap();

	// 跨镜匹配项
	const Point2PointICPTerm2Jacobian cross_view_term2jacobian = cross_view_correspondence_handler->Term2JacobianMap();

	//The penalty constants
	const PenaltyConstants penalty_constants = iterationData.CurrentPenaltyConstants();

	//Hand in the input to preconditioner builder
	m_preconditioner_rhs_builder->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		cross_view_term2jacobian,
		penalty_constants
	);

	//Hand in to residual evaluator
	m_residual_evaluator->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		cross_view_term2jacobian,
		penalty_constants
	);

	//Hand in the input to jtj applier
	m_apply_jtj_handler->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		penalty_constants
	);
}

void SparseSurfelFusion::NonRigidSolver::BuildPreconditioner(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeDiagonalPreconditioner(stream);
}

void SparseSurfelFusion::NonRigidSolver::BuildPreconditionerGlobalIteration(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeDiagonalPreconditionerGlobalIteration(stream);
}

//The method to compute jt residual
void SparseSurfelFusion::NonRigidSolver::ComputeJtResidual(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeJtResidual(stream);
}

void SparseSurfelFusion::NonRigidSolver::ComputeJtResidualGlobalIteration(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeJtResidualGlobalIteration(stream);
}


/* The method for smooth term handler
 */
void SparseSurfelFusion::NonRigidSolver::allocateSmoothTermHandlerBuffer()
{
	m_graph_smooth_handler = std::make_shared<NodeGraphSmoothHandler>();
}

void SparseSurfelFusion::NonRigidSolver::releaseSmoothTermHandlerBuffer() {

}
void SparseSurfelFusion::NonRigidSolver::computeSmoothTermNode2Jacobian(cudaStream_t stream) {
	// Prepare the input
	m_graph_smooth_handler->SetInputs(
		iterationData.CurrentWarpFieldInput(),
		sparseSurfelsInput.nodesGraph,
		sparseSurfelsInput.canonicalNodesCoordinate
	);

	//Do it
	m_graph_smooth_handler->BuildTerm2Jacobian(stream);
	// 里边就一个前向扭曲,实际就是按照节点图，把每个节点的邻点用邻点自己的和中心点的se3扭曲了下,结果在这里边
	// Ti_xj_array[idx] = Ti_xj;邻点用中心点的se3扭曲结果
	// Tj_xj_array[idx] = Tj_xj;邻点用邻点自己的se3的扭曲结果
	// validity_indicator_array[idx] = validity_indicator;
	
}

void SparseSurfelFusion::NonRigidSolver::allocateNode2TermIndexBuffer()
{
	m_node2term_index->setColRow(imageWidth, imageHeight);
	m_node2term_index->AllocateBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	m_nodepair2term_index->setColRow(imageWidth, imageHeight);
	m_nodepair2term_index->AllocateBuffer();
#endif
}
void SparseSurfelFusion::NonRigidSolver::releaseNode2TermIndexBuffer()
{
	m_node2term_index->ReleaseBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	m_nodepair2term_index->ReleaseBuffer();
#endif
}
void SparseSurfelFusion::NonRigidSolver::SetNode2TermIndexInput()
{
	const DeviceArrayView<ushort4> dense_depth_knn = imageKnnFetcher->DenseImageTermKNNArray();
	const DeviceArrayView<ushort2> node_graph = sparseSurfelsInput.nodesGraph;
	const DeviceArrayView<ushort4> foreground_mask_knn = m_density_foreground_handler->ForegroundMaskTermKNN();
	const DeviceArrayView<ushort4> sparse_feature_knn = m_sparse_correspondence_handler->SparseFeatureKNN();
	const DeviceArrayView<ushort4> cross_corr_knn = cross_view_correspondence_handler->CrossViewCorrKNN();
	//全部都是knn，只有节点
	//放到了m_node2term_index中的m_term2node里了
	m_node2term_index->SetInputs(
		dense_depth_knn,
		node_graph,
		sparseSurfelsInput.nodesSE3.Size(),
		foreground_mask_knn,
		sparse_feature_knn,
		cross_corr_knn
	);

#if defined(USE_MATERIALIZED_JTJ)
	const auto num_nodes = sparseSurfelsInput.nodesSE3.Size();
	m_nodepair2term_index->SetInputs(
		num_nodes,
		dense_depth_knn,
		node_graph,
		foreground_mask_knn,
		sparse_feature_knn,
		cross_corr_knn
	);
#endif
}

void SparseSurfelFusion::NonRigidSolver::BuildNode2TermIndex(cudaStream_t stream)
{
	m_node2term_index->BuildIndex(stream);
}

void SparseSurfelFusion::NonRigidSolver::BuildNodePair2TermIndexBlocked(cudaStream_t stream)
{

	m_nodepair2term_index->BuildHalfIndex(stream);						// 这里将节点对构造成key，并且压缩相同的节点对
	m_nodepair2term_index->QueryValidNodePairSize(stream);				// 这将被阻塞
	m_nodepair2term_index->BuildSymmetricAndRowBlocksIndex(stream);		// 后面的计算取决于大小

	//做一次完整性检查
	//m_nodepair2term_index->CheckHalfIndex();
	//m_nodepair2term_index->CompactedIndexSanityCheck();
	//m_nodepair2term_index->IndexStatistics();

	//做一个平滑项的统计:不能这样做
	//非图像的项对解的稳定性有重要影响
	//m_nodepair2term_index->CheckSmoothTermIndexCompleteness();
}


/* The method to filter the sparse feature term
 */
void SparseSurfelFusion::NonRigidSolver::allocateSparseFeatureBuffer() {
	m_sparse_correspondence_handler = std::make_shared<SparseCorrespondenceHandler>();
	m_sparse_correspondence_handler->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releaseSparseFeatureBuffer() {
	m_sparse_correspondence_handler->ReleaseBuffer();
}

void SparseSurfelFusion::NonRigidSolver::SetSparseFeatureHandlerFullInput() {
	//The current node se3 from iteraion data
	const DeviceArrayView<DualQuaternion> node_se3 = iterationData.CurrentWarpFieldInput();

	m_sparse_correspondence_handler->SetInputs(
		node_se3,
		knnMap,
		observation.vertexConfidenceMap,
		observation.edgeMaskMap,
		observation.correspondencePixelPairs,
		solverMap,
		world2Camera, 
		InitialCameraSE3
	);

}

void SparseSurfelFusion::NonRigidSolver::SelectValidSparseFeatureMatchedPairs(cudaStream_t stream) {
	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->BuildCorrespondVertexKNN(stream);
}


void SparseSurfelFusion::NonRigidSolver::allocateCrossViewCorrPairsBuffer()
{
	cross_view_correspondence_handler = std::make_shared<CrossViewCorrespondenceHandler>();
	cross_view_correspondence_handler->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releaseCrossViewCorrPairsBuffer()
{
	cross_view_correspondence_handler->ReleaseBuffer();
}

void SparseSurfelFusion::NonRigidSolver::SetCrossViewMatchingHandlerInput()
{
	const DeviceArrayView<DualQuaternion> nodeSe3 = iterationData.CurrentWarpFieldInput();	// 当前帧的NodeSe3
	cross_view_correspondence_handler->SetInputs(nodeSe3, knnMap, observation.vertexConfidenceMap, observation.crossCorrPairs, observation.corrMap, solverMap, world2Camera, InitialCameraSE3);
}



/* The buffer and method for density and foreground mask pixel finder
 */
void SparseSurfelFusion::NonRigidSolver::allocateDensityForegroundMapBuffer() {
	m_density_foreground_handler = std::make_shared<DensityForegroundMapHandler>();
	m_density_foreground_handler->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releaseDensityForegroundMapBuffer() {
	m_density_foreground_handler->ReleaseBuffer();
}

void SparseSurfelFusion::NonRigidSolver::setDensityForegroundHandlerFullInput() {
	//The current node se3 from iteraion data
	const auto node_se3 = iterationData.CurrentWarpFieldInput();//warpfieldinit

	//Hand in the information to handler
#if defined(USE_RENDERED_RGBA_MAP_SOLVER)
	m_density_foreground_handler->SetInputs(
		node_se3,
		knnMap,
		observation.foregroundMask,
		observation.filteredForegroundMask,
		observation.foregroundMaskGradientMap,
		observation.grayScaleMap,
		observation.grayScaleGradientMap,
		observation.normalizedRGBAMap,
		world2Camera,
		ClipColorIntrinsic,
		solverMap,
		imageKnnFetcher->GetImageTermPixelAndKNN()
	);
	
#else
	for (size_t i = 0; i < devicesCount; i++)
	{
		m_density_foreground_handler->SetInputs(
			node_se3,
			i,
			knnMap[i],
			observation.foregroundMask[i],
			//observation.filterForegroundMask,//没写
			observation.foregroundMaskGradientMap[i],
			observation.grayScaleMap[i],
			observation.grayScaleGradientMap[i],
			/*m_rendered_maps.reference_vertex_map,
			m_rendered_maps.reference_normal_map,
			m_rendered_maps.index_map,*/
			solverMap[i],
			observation.normalizedRGBAPrevious[i],
			world2Camera[i],
			ClipColorIntrinsic[i],
			imageKnnFetcher->GetImageTermPixelAndKNN()
		);
	}
#endif
}

void SparseSurfelFusion::NonRigidSolver::FindValidColorForegroundMaskPixel(cudaStream_t color_stream, cudaStream_t mask_stream)
{
	//Provide to input
	setDensityForegroundHandlerFullInput();

	//Do it
	m_density_foreground_handler->FindValidColorForegroundMaskPixels(color_stream, mask_stream);
}

void SparseSurfelFusion::NonRigidSolver::FindPotentialForegroundMaskPixelSynced(cudaStream_t stream) {
	//Provide to input
	setDensityForegroundHandlerFullInput();

	//Do it
	m_density_foreground_handler->FindPotentialForegroundMaskPixelSynced(stream);
}





void SparseSurfelFusion::NonRigidSolver::FetchPotentialDenseImageTermPixelsFixedIndexSynced(cudaStream_t stream) {
	//Hand in the input
	imageKnnFetcher->SetInputs(knnMap, solverMap);

	//Do processing
	//imageKnnFetcher->MarkPotentialMatchedPixels(stream);//要改
	//imageKnnFetcher->CompactPotentialValidPixels(stream);//要改
	//imageKnnFetcher->SyncQueryCompactedPotentialPixelSize(stream);

	//The sanity check: seems correct
	//Call this after dense depth handler
	//const auto& dense_depth_knn = m_dense_depth_handler->DenseDepthTermsKNNArray();
	//m_image_knn_fetcher->CheckDenseImageTermKNN(dense_depth_knn);
}









/* The buffer and method for correspondence finder
 */
void SparseSurfelFusion::NonRigidSolver::setClipColorIntrinsic(Intrinsic * intr) {
	for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
      ClipColorIntrinsic[i] = intr[i];
	}
	
}

void SparseSurfelFusion::NonRigidSolver::allocateDenseDepthBuffer() {
	m_dense_depth_handler = std::make_shared<DenseDepthHandler>();
	m_dense_depth_handler->AllocateBuffer();
	m_dense_depth_handler->SetInitalData(ClipColorIntrinsic);
}

void SparseSurfelFusion::NonRigidSolver::releaseDenseDepthBuffer() {
	m_dense_depth_handler->ReleaseBuffer();
}



void SparseSurfelFusion::NonRigidSolver::setDenseDepthHandlerFullInput() {
	const DeviceArrayView<DualQuaternion> node_se3 = iterationData.CurrentWarpFieldInput();
	//Construct the input
	m_dense_depth_handler->SetInputs(
		node_se3,
		knnMap,
		observation.vertexConfidenceMap,
		observation.normalRadiusMap,
		observation.filteredForegroundMask,
		solverMap,
		world2Camera,
		imageKnnFetcher->GetImageTermPixelAndKNN()
	);
}

void SparseSurfelFusion::NonRigidSolver::FindCorrespondDepthPixelPairsFreeIndex(cudaStream_t stream) {
	setDenseDepthHandlerFullInput();
	m_dense_depth_handler->FindCorrespondenceSynced(stream);
}

void SparseSurfelFusion::NonRigidSolver::ComputeAlignmentErrorMapDirect(cudaStream_t stream) {
	const DeviceArrayView<DualQuaternion> nodeSE3 = iterationData.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeAlignmentErrorMapDirect(nodeSE3, stream);
}


void SparseSurfelFusion::NonRigidSolver::ComputeAlignmentErrorOnNodes(const bool printNodeError, cudaStream_t stream) {
	const DeviceArrayView<DualQuaternion> node_se3 = iterationData.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeNodewiseError(node_se3, printNodeError, stream);
}


void SparseSurfelFusion::NonRigidSolver::ComputeAlignmentErrorMapFromNode(cudaStream_t stream) {
	const DeviceArrayView<DualQuaternion> nodeSE3 = iterationData.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeAlignmentErrorMapFromNode(nodeSE3, stream);
}

/* 准备好雅可比矩阵供以后使用
 */
void SparseSurfelFusion::NonRigidSolver::ComputeTermJacobiansFreeIndex(
	cudaStream_t dense_depth, cudaStream_t density_map,
	cudaStream_t foreground_mask, cudaStream_t sparse_feature, cudaStream_t crossViewCorrStream
) {
	//m_dense_depth_handler->ComputeJacobianTermsFreeIndex(dense_depth);
	//computeSmoothTermNode2Jacobian(sparse_feature);
	//m_density_foreground_handler->ComputeTwistGradient(density_map, foreground_mask);
	//m_sparse_correspondence_handler->BuildTerm2Jacobian(sparse_feature);
}


// Assume the SE3 for each term expepted smooth term is updated
void SparseSurfelFusion::NonRigidSolver::ComputeTermJacobianFixedIndex(cudaStream_t denseDepthStream, cudaStream_t densityMapStream, cudaStream_t foregroundMaskStream, cudaStream_t sparseFeatureStream, cudaStream_t crossViewCorrStream) {

	std::vector<unsigned int> differenceOffsetImageKnnFetcher, differenceOffsetForegroundHandler;
	imageKnnFetcher->getDifferenceViewOffset(differenceOffsetImageKnnFetcher);
	m_dense_depth_handler->ComputeJacobianTermsFixedIndex(differenceOffsetImageKnnFetcher, denseDepthStream);
	computeSmoothTermNode2Jacobian(sparseFeatureStream);
	m_density_foreground_handler->getDiffoffsetDensityForegroundMapHandler(differenceOffsetForegroundHandler);
	m_density_foreground_handler->ComputeTwistGradient(differenceOffsetImageKnnFetcher, differenceOffsetForegroundHandler, densityMapStream, foregroundMaskStream);
	m_sparse_correspondence_handler->BuildTerm2Jacobian(sparseFeatureStream);
	cross_view_correspondence_handler->BuildTerm2Jacobian(crossViewCorrStream);
}


/* The method to compute residual
 */
void SparseSurfelFusion::NonRigidSolver::allocateResidualEvaluatorBuffer() {
	m_residual_evaluator = std::make_shared<ResidualEvaluator>();
	m_residual_evaluator->AllocateBuffer();
}

void SparseSurfelFusion::NonRigidSolver::releaseResidualEvaluatorBuffer() {
	m_residual_evaluator->ReleaseBuffer();
}

float SparseSurfelFusion::NonRigidSolver::ComputeTotalResidualSynced(cudaStream_t stream) {
	return m_residual_evaluator->ComputeTotalResidualSynced(stream);
}

