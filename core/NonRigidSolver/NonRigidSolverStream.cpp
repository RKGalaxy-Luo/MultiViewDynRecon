#include "NonRigidSolver.h"



void SparseSurfelFusion::NonRigidSolver::initSolverStreams()
{
	for (int i = 0; i < MAX_NONRIGID_CUDA_STREAM; i++) {
		CHECKCUDA(cudaStreamCreate(&solverStreams[i]));
	}
}

void SparseSurfelFusion::NonRigidSolver::releaseSolverStreams()
{
	for (int i = 0; i < MAX_NONRIGID_CUDA_STREAM; i++) {
		CHECKCUDA(cudaStreamDestroy(solverStreams[i]));
		solverStreams[i] = 0;
	}

}

void SparseSurfelFusion::NonRigidSolver::synchronizeAllSolverStreams()
{
	//std::cout << "进入了最外层的流同步"<<std::endl;
	for (int i = 0; i < MAX_NONRIGID_CUDA_STREAM; i++) {
		CHECKCUDA(cudaStreamSynchronize(solverStreams[i]));
	}
}

void SparseSurfelFusion::NonRigidSolver::showTargetAndCanonical(DeviceArrayView<float4> corrPairsTarget, DeviceArrayView<float4> corrPairsCan, DeviceArrayView<float4> corssPairsTarget, DeviceArrayView<float4> corssPairsCan)
{
	CHECKCUDA(cudaDeviceSynchronize());
	DeviceArray<float4> togetherTarget;
	togetherTarget.create(corrPairsTarget.Size() + corssPairsTarget.Size());
	CHECKCUDA(cudaMemcpy(togetherTarget.ptr(), corrPairsTarget.RawPtr(), corrPairsTarget.Size() * sizeof(float4), cudaMemcpyDeviceToDevice));
	CHECKCUDA(cudaMemcpy(togetherTarget.ptr() + corrPairsTarget.Size(), corssPairsTarget.RawPtr(), corssPairsTarget.Size() * sizeof(float4), cudaMemcpyDeviceToDevice));
	DeviceArrayView<float4> togetherTargetView(togetherTarget.ptr(), togetherTarget.size());

	DeviceArray<float4> togetherCan;
	togetherCan.create(corrPairsCan.Size() + corssPairsCan.Size());
	CHECKCUDA(cudaMemcpy(togetherCan.ptr(), corrPairsCan.RawPtr(), corrPairsCan.Size() * sizeof(float4), cudaMemcpyDeviceToDevice));
	CHECKCUDA(cudaMemcpy(togetherCan.ptr() + corrPairsCan.Size(), corssPairsCan.RawPtr(), corssPairsCan.Size() * sizeof(float4), cudaMemcpyDeviceToDevice));
	DeviceArrayView<float4> togetherCanView(togetherCan.ptr(), togetherCan.size());

	if (frameIdx % 80 == 0) {
		Visualizer::DrawMatchedReferenceAndObseveredPointsPair(togetherCanView, togetherTargetView);
	}
	togetherTarget.release();
	togetherCan.release();
}

void SparseSurfelFusion::NonRigidSolver::SolveNonRigidAlignment()
{
	synchronizeAllSolverStreams();
}

void SparseSurfelFusion::NonRigidSolver::SolveSerial() {
	//计算前同步
	synchronizeAllSolverStreams();
	buildSolverIndexStreamed();

	for (int i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		if (iterationData.IsGlobalIteration()) {
			solverIterationGlobalIterationStreamed();
		}
		else {
			solverIterationLocalIterationStreamed();
		}
	}
	//再次同步调试
	synchronizeAllSolverStreams();
}


void SparseSurfelFusion::NonRigidSolver::buildSolverIndexStreamed()
{
	//std::cout << "进入了buildSolverIndexStreamed" << std::endl;
	QueryPixelKNN(solverStreams[0]); //这里需要同步
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
	//std::cout << "完成：生成KNNMap" << std::endl;

	imageKnnFetcher->SetInputs(knnMap, solverMap);
	imageKnnFetcher->MarkPotentialMatchedPixels(solverStreams[0]);
	imageKnnFetcher->CompactPotentialValidPixels(solverStreams[0]);	
	//std::cout << "完成：imageKnnFetcher 获取潜在有效像素" << std::endl;

	setDensityForegroundHandlerFullInput();//把深度观测到的数据（第一帧的）和ref中的数据传递给handler
	m_density_foreground_handler->MarkValidColorForegroundMaskPixels(solverStreams[1]);//这地方最终得到的是上一帧动作模型对齐到当前帧后，如果对应的像素点在当前帧的前景里，那么前景有效。然后只要indexmap里有效，就rgb有效
	m_density_foreground_handler->CompactValidMaskPixel(solverStreams[1]);//这一步得到了上一帧动作模型对齐到当前帧中前景有效的indexmap里面元的压缩数组及其knn													  
    //std::cout << "完成：m_density_foreground_handler 获取有效掩膜像素" << std::endl;

	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->ChooseValidPixelPairs(solverStreams[2]);//这个找的只是在处理深度图时找到的前后两帧对应得上的像素点，所以是稀疏的
	m_sparse_correspondence_handler->CompactQueryPixelPairs(solverStreams[2]);
	//std::cout << "完成：m_sparse_correspondence_handler 获取像素对" << std::endl;

	SetCrossViewMatchingHandlerInput();
	cross_view_correspondence_handler->ChooseValidCrossCorrPairs(solverStreams[3]);
	cross_view_correspondence_handler->CompactCrossViewCorrPairs(solverStreams[3]);

	imageKnnFetcher->SyncQueryCompactedPotentialPixelSize(solverStreams[0]);				// 在此处同步这个流
	m_density_foreground_handler->QueryCompactedMaskPixelArraySize(solverStreams[1]);		// 在此处同步这个流
	m_sparse_correspondence_handler->QueryCompactedArraySize(solverStreams[2]);				// 在此处同步这个流
	cross_view_correspondence_handler->QueryCompactedCrossViewArraySize(solverStreams[3]);	// 在此处同步这个流

	//showTargetAndCanonical(m_sparse_correspondence_handler->GetTargetVertex(), m_sparse_correspondence_handler->GetCanVertex(), cross_view_correspondence_handler->GetTargetVertex(), cross_view_correspondence_handler->GetCanVertex());
	
	
	//m_dense_depth_handler
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();
	//std::cout << "完成：三项计算结果的同步与输入" << std::endl;

	SetNode2TermIndexInput();
	BuildNode2TermIndex(solverStreams[0]); //这不会阻塞
	BuildNodePair2TermIndexBlocked(solverStreams[1]); //这会阻塞
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
}

void SparseSurfelFusion::NonRigidSolver::solverIterationGlobalIterationStreamed() {
	// 将新的SE3交给处理器
	m_dense_depth_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	cross_view_correspondence_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());

	// 雅可比矩阵的计算
	ComputeTermJacobianFixedIndex(solverStreams[0], solverStreams[1], solverStreams[2], solverStreams[3], solverStreams[4]); //A sync should happened here
	synchronizeAllSolverStreams();
	// 对角线块JTJ和JTe的计算
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();
	BuildPreconditionerGlobalIteration(solverStreams[0]);
	ComputeJtResidualGlobalIteration(solverStreams[1]);
	MaterializeJtJNondiagonalBlocksGlobalIteration(solverStreams[2]);
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
	CHECKCUDA(cudaStreamSynchronize(solverStreams[1]));
	CHECKCUDA(cudaStreamSynchronize(solverStreams[2]));
	//矩阵的集合:这里有一个同步
	const DeviceArrayView<float> diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, solverStreams[0]);

	//解决并更新
	SolvePCGMaterialized();
	iterationData.ApplyWarpFieldUpdate(solverStreams[0]);
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
}

void SparseSurfelFusion::NonRigidSolver::solverIterationLocalIterationStreamed() {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	cross_view_correspondence_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());

	//The computation of jacobian
	ComputeTermJacobianFixedIndex(solverStreams[0], solverStreams[1], solverStreams[2], solverStreams[3], solverStreams[4]); // A sync should happend here
	synchronizeAllSolverStreams();

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();
	BuildPreconditioner(solverStreams[0]);
	ComputeJtResidual(solverStreams[1]);
	MaterializeJtJNondiagonalBlocks(solverStreams[2]);
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
	CHECKCUDA(cudaStreamSynchronize(solverStreams[1]));
	CHECKCUDA(cudaStreamSynchronize(solverStreams[2]));


	//The assemble of matrix: a sync here
	const DeviceArrayView<float> diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, solverStreams[0]);

	//Debug methods
	//LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(m_solver_stream[0]);

	//Solve it and update
	SolvePCGMaterialized();
	iterationData.ApplyWarpFieldUpdate(solverStreams[0]);
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
}