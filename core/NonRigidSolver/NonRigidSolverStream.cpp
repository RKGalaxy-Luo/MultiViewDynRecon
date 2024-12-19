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
	//std::cout << "��������������ͬ��"<<std::endl;
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
	//����ǰͬ��
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
	//�ٴ�ͬ������
	synchronizeAllSolverStreams();
}


void SparseSurfelFusion::NonRigidSolver::buildSolverIndexStreamed()
{
	//std::cout << "������buildSolverIndexStreamed" << std::endl;
	QueryPixelKNN(solverStreams[0]); //������Ҫͬ��
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
	//std::cout << "��ɣ�����KNNMap" << std::endl;

	imageKnnFetcher->SetInputs(knnMap, solverMap);
	imageKnnFetcher->MarkPotentialMatchedPixels(solverStreams[0]);
	imageKnnFetcher->CompactPotentialValidPixels(solverStreams[0]);	
	//std::cout << "��ɣ�imageKnnFetcher ��ȡǱ����Ч����" << std::endl;

	setDensityForegroundHandlerFullInput();//����ȹ۲⵽�����ݣ���һ֡�ģ���ref�е����ݴ��ݸ�handler
	m_density_foreground_handler->MarkValidColorForegroundMaskPixels(solverStreams[1]);//��ط����յõ�������һ֡����ģ�Ͷ��뵽��ǰ֡�������Ӧ�����ص��ڵ�ǰ֡��ǰ�����ôǰ����Ч��Ȼ��ֻҪindexmap����Ч����rgb��Ч
	m_density_foreground_handler->CompactValidMaskPixel(solverStreams[1]);//��һ���õ�����һ֡����ģ�Ͷ��뵽��ǰ֡��ǰ����Ч��indexmap����Ԫ��ѹ�����鼰��knn													  
    //std::cout << "��ɣ�m_density_foreground_handler ��ȡ��Ч��Ĥ����" << std::endl;

	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->ChooseValidPixelPairs(solverStreams[2]);//����ҵ�ֻ���ڴ������ͼʱ�ҵ���ǰ����֡��Ӧ���ϵ����ص㣬������ϡ���
	m_sparse_correspondence_handler->CompactQueryPixelPairs(solverStreams[2]);
	//std::cout << "��ɣ�m_sparse_correspondence_handler ��ȡ���ض�" << std::endl;

	SetCrossViewMatchingHandlerInput();
	cross_view_correspondence_handler->ChooseValidCrossCorrPairs(solverStreams[3]);
	cross_view_correspondence_handler->CompactCrossViewCorrPairs(solverStreams[3]);

	imageKnnFetcher->SyncQueryCompactedPotentialPixelSize(solverStreams[0]);				// �ڴ˴�ͬ�������
	m_density_foreground_handler->QueryCompactedMaskPixelArraySize(solverStreams[1]);		// �ڴ˴�ͬ�������
	m_sparse_correspondence_handler->QueryCompactedArraySize(solverStreams[2]);				// �ڴ˴�ͬ�������
	cross_view_correspondence_handler->QueryCompactedCrossViewArraySize(solverStreams[3]);	// �ڴ˴�ͬ�������

	//showTargetAndCanonical(m_sparse_correspondence_handler->GetTargetVertex(), m_sparse_correspondence_handler->GetCanVertex(), cross_view_correspondence_handler->GetTargetVertex(), cross_view_correspondence_handler->GetCanVertex());
	
	
	//m_dense_depth_handler
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();
	//std::cout << "��ɣ������������ͬ��������" << std::endl;

	SetNode2TermIndexInput();
	BuildNode2TermIndex(solverStreams[0]); //�ⲻ������
	BuildNodePair2TermIndexBlocked(solverStreams[1]); //�������
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
}

void SparseSurfelFusion::NonRigidSolver::solverIterationGlobalIterationStreamed() {
	// ���µ�SE3����������
	m_dense_depth_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());
	cross_view_correspondence_handler->UpdateNodeSE3(iterationData.CurrentWarpFieldInput());

	// �ſɱȾ���ļ���
	ComputeTermJacobianFixedIndex(solverStreams[0], solverStreams[1], solverStreams[2], solverStreams[3], solverStreams[4]); //A sync should happened here
	synchronizeAllSolverStreams();
	// �Խ��߿�JTJ��JTe�ļ���
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();
	BuildPreconditionerGlobalIteration(solverStreams[0]);
	ComputeJtResidualGlobalIteration(solverStreams[1]);
	MaterializeJtJNondiagonalBlocksGlobalIteration(solverStreams[2]);
	CHECKCUDA(cudaStreamSynchronize(solverStreams[0]));
	CHECKCUDA(cudaStreamSynchronize(solverStreams[1]));
	CHECKCUDA(cudaStreamSynchronize(solverStreams[2]));
	//����ļ���:������һ��ͬ��
	const DeviceArrayView<float> diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, solverStreams[0]);

	//���������
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