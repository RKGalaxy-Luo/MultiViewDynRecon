/*****************************************************************//**
 * \file   AlgorithmSerial.cpp
 * \brief  ��Ҫ���ĵ����߳��㷨��ʵ��
 * 
 * \author LUO
 * \date   January 24th 2024
 *********************************************************************/
#include "AlgorithmSerial.h"

SparseSurfelFusion::AlgorithmSerial::AlgorithmSerial(std::shared_ptr<ThreadPool> threadPool, bool intoBuffer)
{
	this->intoBuffer = intoBuffer;

	frameProcessor = make_shared<FrameProcessor>(threadPool);	// �������̣߳��������

	deviceCount = frameProcessor->getCameraCount();				// ����������
	for (int i = 0; i < deviceCount; i++) {
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}

	configParser = frameProcessor->getCameraConfigParser();		// ��������ͼ�����

	camera.SetDevicesCount(deviceCount);						// ����camera�������������

	// ��ø��Զ������
	rigidSolver = make_shared<RigidSolver>(deviceCount, configParser->getClipedIntrinsicArray(), configParser->getImageClipedRows(), configParser->getImageClipedCols());

	// �������������Ⱦ��
	renderer = make_shared<Renderer>(frameProcessor->rawImageRowsCliped, frameProcessor->rawImageColsCliped, configParser);	

	//dynamicallyDraw = make_shared<DynamicallyDrawPoints>();

	//renderAlbedo = make_shared<DynamicallyRenderSurfels>(DynamicallyRenderSurfels::RenderSurfelsType::Albedo, ClipedIntrinsic[0]);
	renderPhong = make_shared<DynamicallyRenderSurfels>(DynamicallyRenderSurfels::RenderSurfelsType::Phong, ClipedIntrinsic[0]);
	//renderNormal = make_shared<DynamicallyRenderSurfels>(DynamicallyRenderSurfels::RenderSurfelsType::Normal, ClipedIntrinsic[0]);

	// ��SurfelGeometryӳ�䵽CUDA���Ա���Ⱦ��ʾ��ͬʱ����SurfelGeometry�����Ե��ڴ�
	for (int i = 0; i < deviceCount; i++) {
		ClipedIntrinsic[i] = configParser->getClipColorIntrinsic(i);
		surfelGeometry[i][0] = std::make_shared<SurfelGeometry>();
		surfelGeometry[i][1] = std::make_shared<SurfelGeometry>();
		World2Camera[i] = mat34::identity();
		Camera2World[i] = mat34::identity();
	}

	renderer->MapSurfelGeometryToCUDA(0, surfelGeometry);
	renderer->MapSurfelGeometryToCUDA(1, surfelGeometry);
	 
	renderer->UnmapSurfelGeometryFromCUDA(0);
	renderer->UnmapSurfelGeometryFromCUDA(1);

	m_live_nodes_knn_skinner = KNNBruteForceLiveNodes::Instance();
	m_live_geometry_updater = std::make_shared<LiveGeometryUpdater>(surfelGeometry, configParser->getClipedIntrinsicArray(), deviceCount);

	updatedGeometryIndex = -1;	// ��ʼ����Ⱦindex
	reinitFrameIndex = frameIndex;
	// ����Ť����
	warpField = std::make_shared<WarpField>();
	m_warpfield_extender = std::make_shared<WarpFieldExtender>();
	// ������Ƥ��
	canonicalNodesSkinner = std::make_shared<CanonicalNodesSkinner>(deviceCount);

	//�������� 
	//m_poissonreconstruction = std::make_shared<PoissonReconstruction>();
	
	nonRigidSolver = std::make_shared<NonRigidSolver>(configParser->getClipedIntrinsicArray());

	m_geometry_reinit_processor = std::make_shared<GeometryReinitProcessor>(surfelGeometry, configParser->getClipedIntrinsicArray());
	isfirstframe = false;

//����surfelgeometry�е�ַ

#ifdef CUDA_DEBUG_SYNC_CHECK
	CHECKCUDA(cudaDeviceSynchronize());
#endif 
}

SparseSurfelFusion::AlgorithmSerial::~AlgorithmSerial()
{
	nonRigidSolver->ReleaseBuffer();

}

void SparseSurfelFusion::AlgorithmSerial::setFrameIndex(size_t frameidx)
{
	frameIndex = frameidx;
}

void SparseSurfelFusion::AlgorithmSerial::setFrameIndex(size_t frameidx, const unsigned int beginIdx)
{
	frameIndex = frameidx;
	frameProcessor->setBeginFrameIndex(beginIdx);
}

void SparseSurfelFusion::AlgorithmSerial::ProcessFirstFrame()
{
	//�����surfelgeometry�е�ַ
	DeviceArrayView<DepthSurfel> mergedSurfels = frameProcessor->ProcessFirstFrame(frameIndex, rigidSolver);	// ���ںϺ���Ԫ�������ӽ����ͼ�ϴ���GPU��������ں���Ԫ
	//Visualizer::DrawPointCloud(mergedSurfels);
	//// ֱ���������ʼ��ͼ��
	//m_poissonreconstruction->SolvePoissionReconstructionMesh(mergedSurfels);
	//m_poissonreconstruction->DrawRebuildMesh();
	//Visualizer::DrawPointCloud(mergedSurfels);
	updatedGeometryIndex = 0;

	renderer->MapSurfelGeometryToCUDA(updatedGeometryIndex);

	for (int i = 0; i < deviceCount; i++) {
		surfelGeometry[i][updatedGeometryIndex]->initGeometryFromCamera(mergedSurfels);	// ��������һ�������surfelGeometry[i]��Ķ������еĵ�
	}

	// ��ñ�׼��Ķ��㣬ע����������ĵ�ʵ������һ���ģ���˲���0��1��2�����������
	DeviceArrayView<float4> CanonicalVertices = surfelGeometry[0][updatedGeometryIndex]->getCanonicalVertexConfidence();
	DeviceArrayView<float4> ColorViewTime = surfelGeometry[0][updatedGeometryIndex]->getColorTime();
	warpField->InitializeCanonicalNodesAndSE3FromMergedVertices(CanonicalVertices, ColorViewTime);	// ��Canonical��ĳ��ܵ���в��������ϡ��ڵ�
	warpField->BuildNodesGraph();	// �������������ϡ��ڵ㣬�����ڵ�ͼ
	//�ڵ�ͼ���鳤����node���鳤�ȵ�8����ÿһ���ڵ���8���ڽڵ㣬�ڵ�ͼ���������ţ��ڵ㣬����һ���ڵ㣩��

	DeviceArrayView<float4> canonicalNodes = warpField->getCanonicalNodesCoordinate();
	canonicalNodesSkinner->BuildInitialSkinningIndex(canonicalNodes);
	// ��ó�����Ԫ�й�Skinner��ӳ�����ݰ�
	SurfelGeometry::SkinnerInput denseGeometrySurfelSkinnerInfo = surfelGeometry[0][updatedGeometryIndex]->BindSurfelGeometrySkinnerInfo();// ����ֱ�ӿ���д��Ȩ��
	// ���Canonical��ڵ��й�Skinner��ӳ�����ݰ�
	WarpField::SkinnerInput sparseCanonicalNodesSkinnerInfo = warpField->BindWarpFieldSkinnerInfo();	// ����ֱ�ӿ���д��Ȩ��
	canonicalNodesSkinner->PerformSkinning(denseGeometrySurfelSkinnerInfo, sparseCanonicalNodesSkinnerInfo);//ÿ������ͽڵ㶼��4������ڼ���Ȩ�ء�

	renderer->UnmapSurfelGeometryFromCUDA(updatedGeometryIndex);

}

void SparseSurfelFusion::AlgorithmSerial::ProcessFrameStream(bool SaveResult, bool RealTimeDisplay, bool drawRecent) {

	const size_t numVertex = surfelGeometry[0][updatedGeometryIndex]->ValidSurfelsNum();
	const float currentTime = frameIndex - 1;
	renderer->SetFrameIndex(frameIndex);

	for (int i = 0; i < deviceCount; i++) {
		initialWorld2Camera[i] = camera.GetWorld2CameraEigen(i);
	}

	if (drawRecent) {
		renderer->DrawSolverMapsWithRecentObservation(numVertex, updatedGeometryIndex, currentTime, initialWorld2Camera);
	}
	else {
		renderer->DrawSolverMapsConfidentObservation(numVertex, updatedGeometryIndex, currentTime, initialWorld2Camera);
	}

	renderer->MapSolverMapsToCuda(solverMaps);
	renderer->MapSurfelGeometryToCUDA(updatedGeometryIndex);	

	CameraObservation observation;
	frameProcessor->ProcessCurrentFrame(observation, frameIndex);	// �ϴ�ͼ��cuda�У���ֵfirstObservation����ͼ�����ͼ�ͷ���ͼ

#ifdef REBUILD_WITHOUT_BACKGROUND
	rigidSolver->SolveRigidAlignment(observation, isRefreshFrame);
#else
	rigidSolver->SolveRigidAlignment(solverMaps, observation, World2Camera);
#endif // REBUILD_WITHOUT_BACKGROUND

	for (int i = 0; i < deviceCount; i++) {
		World2Camera[i] = rigidSolver->GetWorld2Camera();
		Camera2World[i] = World2Camera[i].inverse();
		camera.SetWorld2Camera(World2Camera[i], i);
	}
	//Visualizer::DrawPointCloud(observation.PreviousVertexConfidenceMap[1]);
	//Visualizer::DrawPointCloud(surfelGeometry[0][updatedGeometryIndex]->getCanonicalVertexConfidence());
	//Visualizer::DrawPointCloud(surfelGeometry[0][updatedGeometryIndex]->getLiveVertexConfidence());

	//Visualizer::DrawFusedSurfelCloud(
	//	surfelGeometry[0][updatedGeometryIndex]->getLiveVertexConfidence(),
	//	observation.vertexConfidenceMap[0], toEigen(Camera2World[0] * InitialCameraSE3[0]),
	//	observation.vertexConfidenceMap[1], toEigen(Camera2World[1] * InitialCameraSE3[1]),
	//	observation.vertexConfidenceMap[2], toEigen(Camera2World[2] * InitialCameraSE3[2]));

	// ׼���Ǹ�����������, ��ó�����Ԫ�й�Skinner��ӳ�����ݰ�
	SurfelGeometry::NonRigidSolverInput denseSurfelsNonRigidInfo = surfelGeometry[0][updatedGeometryIndex]->BindNonRigidSolverInfo();
	// ���Canonical��ڵ��й�Skinner��ӳ�����ݰ�
	WarpField::NonRigidSolverInput sparseCanonicalNodesNonRigidInfo = warpField->BindNonRigidSolverInfo();
	
	//debugIndexMap(solverMaps[0].index_map, solverMaps[1].index_map, solverMaps[2].index_map, "SolverMapIndexMap");
	//Visualizer::DrawMatchedCloudPair(solverMaps[1].warp_vertex_map, observation.PreviousVertexConfidenceMap[1], toEigen(Camera2World[1]));
	renderer->FilterSolverMapsIndexMap(solverMaps, observation, Camera2World, 0.3f);
	//debugIndexMap(solverMaps[0].index_map, solverMaps[1].index_map, solverMaps[2].index_map, "FilteredSolverMapIndexMap");
	
	
	// ǰ�죬���
	//if (335 < frameIndex && frameIndex <= 340) Visualizer::DrawMatchedCloudPair(solverMaps[1].warp_vertex_map, observation.vertexConfidenceMap[1], toEigen(Camera2World[1]));
	//Visualizer::DrawMatchedCloudPair(observation.vertexConfidenceMap[1], surfelGeometry[1][updatedGeometryIndex]->getCanonicalVertexConfidence(), toEigen(Camera2World[1] * Constants::getInitialCameraSE3(1)));
	//Visualizer::DrawFusedProcessInCanonicalField(surfelGeometry[1][updatedGeometryIndex]->getCanonicalVertexConfidence(), surfelGeometry[1][updatedGeometryIndex]->getLiveVertexConfidence());

	nonRigidSolver->frameIdx = frameIndex;

	nonRigidSolver->SetSolverInput(
		observation, 
		denseSurfelsNonRigidInfo, 
		sparseCanonicalNodesNonRigidInfo, 
		World2Camera,
		solverMaps,
		ClipedIntrinsic	
	);
	//Visualizer::DrawValidIndexMap(solverMaps[0].index_map, -1);

	// ��ɫ�ǹ۲쵽�ĵ㣬��ɫ��solverMap��Reference�㣬����ǰ����֡�Ĳ��
	//Visualizer::DrawFusedProcessInCameraView(solverMaps[1].warp_vertex_map, World2Camera[1], observation.vertexConfidenceMap[1]);
	//Visualizer::DrawFusedProcessInCameraView(solverMaps[1].reference_vertex_map, World2Camera[1], observation.vertexConfidenceMap[1]);
/*********************************************  ����ƥ���Ӧ�÷ǳ����Ŷ�  *********************************************/
	nonRigidSolver->SolveSerial();
	const DeviceArrayView<DualQuaternion> solvedSe3 = nonRigidSolver->SolvedNodeSE3();

/*********************************************  ����֤����ʵ�� �� 1 ֡ observation��Canonical���Ǻܽӽ���  *********************************************/
	//Visualizer::DrawMatchedCloudPair(observation.vertexConfidenceMap[1], surfelGeometry[1][updatedGeometryIndex]->getCanonicalVertexConfidence(), toEigen(Camera2World[1] * Constants::GetInitialCameraSE3(1)));
	warpField->UpdateHostDeviceNodeSE3NoSync(solvedSe3);
	bool showRender = false;
	if (frameIndex % 80 == 0) showRender = false;
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(showRender, *warpField, surfelGeometry, deviceCount, updatedGeometryIndex, solvedSe3);
/*********************************************  ����Live��CurrObservation����ǳ�������ȷ  *********************************************/
	//Visualizer::DrawMatchedCloudPair(observation.vertexConfidenceMap[0], surfelGeometry[0][updatedGeometryIndex]->getLiveVertexConfidence(), toEigen(Camera2World[0] * Constants::getInitialCameraSE3(0)));
	nonRigidSolver->ComputeAlignmentErrorOnNodes(false);
	//Visualizer::DrawUnitNodeError(warpField->getLiveNodesCoordinate(), nonRigidSolver->GetNodeUnitAlignmentError());

	// ��ýڵ��������ݽڵ������Ը���ڵ���н������ڵ������Ѿ��Ǵ����ӽ���Դ�����ˣ��ο� observation �� opticalflow 
#ifdef WITH_NODE_CORRECTION
	frameProcessor->GetCorrectedSe3Map();	// У������SE3Map
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	//nonRigidSolver->CorrectLargeErrorNode(frameIndex, frameProcessor->GetCorrectedSe3Maps(), observation, nonRigidSolver->GetNodeUnitAlignmentError(), InitialCameraSE3, World2Camera, ClipedIntrinsic, *warpField, surfelGeometry, updatedGeometryIndex);	// ����Map
#else
	nonRigidSolver->CorrectLargeErrorNode(frameIndex, frameProcessor->GetCorrectedSe3Maps(), frameProcessor->GetValidSe3Maps(), observation, nonRigidSolver->GetNodeUnitAlignmentError(), InitialCameraSE3, World2Camera, ClipedIntrinsic, *warpField, surfelGeometry, updatedGeometryIndex);	// ϡ��ƥ���map
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

	nonRigidSolver->ComputeAlignmentErrorOnNodes(true);	// �ٴμ��ڵ����
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

	//Visualizer::DrawUnitNodeError(warpField->getLiveNodesCoordinate(), nonRigidSolver->GetNodeUnitAlignmentError());

	const DeviceArrayView<float4> liveNodes = warpField->getLiveNodesCoordinate();	// Live��Ľڵ�
	//Visualizer::DrawPointCloud(liveNodes);
	//Visualizer::DrawFusedSurfelCloud(
	//	surfelGeometry[0][updatedGeometryIndex]->getLiveVertexConfidence(),
	//	observation.vertexConfidenceMap[0], toEigen(Camera2World[0] * InitialCameraSE3[0]),
	//	observation.vertexConfidenceMap[1], toEigen(Camera2World[1] * InitialCameraSE3[1]),
	//	observation.vertexConfidenceMap[2], toEigen(Camera2World[2] * InitialCameraSE3[2]));
	m_live_nodes_knn_skinner->BuildIndex(liveNodes);

    renderer->UnmapSolverMapsFromCuda();
    renderer->UnmapSurfelGeometryFromCUDA(updatedGeometryIndex);

	//surfelGeometryIndexmap��surfelGeometry[updatedGeometryIndex]�Ĳ�������ͬ����
	renderer->DrawFusionMap(numVertex, updatedGeometryIndex, camera.GetWorld2CameraMat34Array());
	//���ں�ͼӳ�䵽cuda
	Renderer::FusionMaps fusionMaps[MAX_CAMERA_COUNT];
	//fusionmap��buffer����ȫ��������4����
	renderer->MapFusionMapsToCuda(fusionMaps);
	//������Map��ӳ�䵽surfelwarp����Ϊ���Ƕ��Ǳ����
	renderer->MapSurfelGeometryToCUDA(0);
	renderer->MapSurfelGeometryToCUDA(1);

	//debugIndexMap(fusionMaps[0].index_map, fusionMaps[1].index_map, fusionMaps[2].index_map, "FusionMapIndexMap");
	//Visualizer::DrawFusedProcessInCameraView(fusionMaps[2].warp_vertex_map, Camera2World[2].inverse(), observation.vertexConfidenceMap[2]);
	renderer->FilterFusionMapsIndexMap(fusionMaps, observation, World2Camera);
	//debugIndexMap(fusionMaps[0].index_map, fusionMaps[1].index_map, fusionMaps[2].index_map, "FilteredFusionMapIndexMap");


	//std::cout << "��ɣ�DrawFusionMaps��ӳ��" << std::endl;

#ifdef USE_DYNAMICAL_REFRESH
	// ����Ӧѡ���Ƿ����ˢ��֡
	isRefreshFrame = shouldDoReinit(frameIndex - reinitFrameIndex) || renderer->ShouldDoRefresh();
#else
	isRefreshFrame = shouldDoReinit(frameIndex - reinitFrameIndex);
#endif // USE_DYNAMICAL_REFRESH



	//fusion��reinit��Ҫд��ļ������������û��д��������ǰ�ļ�������
	unsigned int fused_geometry_idx = updatedGeometryIndex;

	//ȡ���ڸ������黹�Ǽ���
	if (isRefreshFrame) {
		//printf("*****************   ˢ��   *****************\n");
		refreshFrameNum++;
		continuouslyFusedFramesCount = 0;
		//First setup the idx
		reinitFrameIndex = frameIndex;
		fused_geometry_idx = (updatedGeometryIndex + 1) % 2;
		//Hand in the input to reinit processor
		m_geometry_reinit_processor->SetInputs(
			fusionMaps,
			observation,
			updatedGeometryIndex,
			float(frameIndex),
			World2Camera
		);

		unsigned int num_remaining_surfel, num_appended_surfel;
		unsigned int numberView[MAX_CAMERA_COUNT];

		m_geometry_reinit_processor->ProcessReinitObservedOnlySerial(num_remaining_surfel, num_appended_surfel, numberView, frameIndex);
#ifdef DEBUG_RUNNING_INFO
		printf("ˢ��֡����һ����Canonical������Ԫ���� = %u   ��ǰˢ��֡ÿ���ӽ������Ԫ(%u + %u + %u) = %u \n", num_remaining_surfel, numberView[0], numberView[1], numberView[2], num_appended_surfel);
#endif // DEBUG_RUNNING_INFO

		//Reinit the warp field
		DeviceArrayView<float4> referenceVertex = surfelGeometry[0][fused_geometry_idx]->getCanonicalVertexConfidence();
		DeviceArrayView<float4> colorViewTime = surfelGeometry[0][fused_geometry_idx]->getColorTime();
		warpField->InitializeCanonicalNodesAndSE3FromMergedVertices(referenceVertex, colorViewTime);	// ��Canonical��ĳ��ܵ���в��������ϡ��ڵ�

		warpField->BuildNodesGraph();	// �������������ϡ��ڵ㣬�����ڵ�ͼ
		//�ڵ�ͼ���鳤����node���鳤�ȵ�8����ÿһ���ڵ���8���ڽڵ㣬�ڵ�ͼ���������ţ��ڵ㣬����һ���ڵ㣩��

		DeviceArrayView<float4> canonicalNodes = warpField->getCanonicalNodesCoordinate();
		canonicalNodesSkinner->BuildInitialSkinningIndex(canonicalNodes);

		// ��ó�����Ԫ�й�Skinner��ӳ�����ݰ�
		SurfelGeometry::SkinnerInput denseGeometrySurfelSkinnerInfo[MAX_CAMERA_COUNT];
		for (int i = 0; i < deviceCount; i++) {
			denseGeometrySurfelSkinnerInfo[i] = surfelGeometry[i][fused_geometry_idx]->BindSurfelGeometrySkinnerInfo();//�������arrayhandle������
		}
		// ���Canonical��ڵ��й�Skinner��ӳ�����ݰ�
		WarpField::SkinnerInput sparseCanonicalNodesSkinnerInfo = warpField->BindWarpFieldSkinnerInfo();//�������arrayhandle������
		//�����ˢ��surfelGeometryIndexMap��knn
		canonicalNodesSkinner->PerformSkinning(denseGeometrySurfelSkinnerInfo, sparseCanonicalNodesSkinnerInfo);//ÿ������ͽڵ㶼��4������ڼ���Ȩ�ء�
	}
	else {
		fusedFrameNum++;
		continuouslyFusedFramesCount++;
		if (maxcontinuouslyFusedFramesNum < continuouslyFusedFramesCount) maxcontinuouslyFusedFramesNum = continuouslyFusedFramesCount;
		fused_geometry_idx = (updatedGeometryIndex + 1) % 2;// ��ǰ֡�ں���ɵĽ���洢��֡����˫����
		const WarpField::LiveGeometryUpdaterInput warpfield_input = warpField->GeometryUpdaterAccess();//���ݳ�ref�ڵ��ѡ�Լ�live�ڵ��ѡ�Լ�SE3

		m_live_geometry_updater->SetInputs(
			fusionMaps,
			observation,
			warpfield_input,
			m_live_nodes_knn_skinner,
			updatedGeometryIndex,
			float(frameIndex),
			World2Camera
		);
	
		unsigned int num_remaining_surfel, num_appended_surfel;

		m_live_geometry_updater->ProcessFusionStreamed(num_remaining_surfel, num_appended_surfel);
#ifdef DEBUG_RUNNING_INFO
		printf("�ں�֡��������Canonical��Ԫ���� = %u   ��ӵ������Ԫ���� = %u   ��Ԫ�ܹ� = %u\n", num_remaining_surfel, num_appended_surfel, num_remaining_surfel + num_appended_surfel);
#endif // DEBUG_RUNNING_INFO

		SurfelNodeDeformer::InverseWarpSurfels(*warpField, surfelGeometry, deviceCount, fused_geometry_idx, solvedSe3);

#ifdef CUDA_DEBUG_SYNC_CHECK
		CHECKCUDA(cudaDeviceSynchronize());
#endif 

		//Extend the warp field reference nodes and SE3
		const unsigned int previousNodesSize = warpField->CheckAndGetNodeSize();
#ifdef DEBUG_RUNNING_INFO
		printf("��չǰ�ڵ���� = %u \n", previousNodesSize);
#endif // DEBUG_RUNNING_INFO
		const float4* appended_vertex_ptr = surfelGeometry[0][fused_geometry_idx]->getCanonicalVertexConfidence().RawPtr() + num_remaining_surfel;
		const float4* appended_cameraId_ptr = surfelGeometry[0][fused_geometry_idx]->getColorTime().RawPtr() + num_remaining_surfel;
		DeviceArrayView<float4> appended_vertex_view(appended_vertex_ptr, num_appended_surfel);
		DeviceArrayView<float4> appended_cameraId_view(appended_cameraId_ptr, num_appended_surfel);
		const ushort4* appended_knn_ptr = surfelGeometry[0][fused_geometry_idx]->SurfelKNNArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<ushort4> appended_surfel_knn(appended_knn_ptr, num_appended_surfel);

		m_warpfield_extender->ExtendReferenceNodesAndSE3Sync(appended_vertex_view, appended_cameraId_view, appended_surfel_knn, warpField);

		warpField->BuildNodesGraph();

		if (warpField->CheckAndGetNodeSize() > previousNodesSize) {
			// ��ó�����Ԫ�й�Skinner��ӳ�����ݰ�
			SurfelGeometry::SkinnerInput denseGeometrySurfelSkinnerInfo[MAX_CAMERA_COUNT];
			for (int i = 0; i < deviceCount; i++) {
				denseGeometrySurfelSkinnerInfo[i] = surfelGeometry[i][fused_geometry_idx]->BindSurfelGeometrySkinnerInfo();//�������arrayhandle������
			}

			canonicalNodesSkinner->UpdateBruteForceSkinningIndexWithNewNodes(warpField->getCanonicalNodesCoordinate(), previousNodesSize);
			
			WarpField::SkinnerInput skinnerWarpField = warpField->BindWarpFieldSkinnerInfo();
			canonicalNodesSkinner->PerformSkinningUpdate(denseGeometrySurfelSkinnerInfo, skinnerWarpField, previousNodesSize);
		}
	}
	unsigned int totalNodesNum = warpField->CheckAndGetNodeSize();
	unsigned int totalSurfels = surfelGeometry[0][fused_geometry_idx]->collectLiveandCanDepthSurfel();

	if (maxNodesNum < totalNodesNum) maxNodesNum = totalNodesNum;
	if (maxDenseSurfels < totalSurfels) maxDenseSurfels = totalSurfels;

#ifdef DEBUG_RUNNING_INFO
	float refreshRatio = (refreshFrameNum * 1.0f / (refreshFrameNum + fusedFrameNum)) * 100.0f;
	printf("Frame Index = %lld              �ڵ����� = %d\n", frameIndex, totalNodesNum);
	printf("���ϡ��ڵ����� = %d        �����ܶ������� = %d\n", maxNodesNum, maxDenseSurfels);
	printf("��ǰˢ��֡ռ�� = %.3f%%       �ں�֡ռ�� = %.3f%%      ��������ں�֡�� = %u\n", refreshRatio, 100.0f - refreshRatio, maxcontinuouslyFusedFramesNum);
#endif // DEBUG_RUNNING_INFO

	//����
	//m_poissonreconstruction->SolvePoissionReconstructionMesh(surfelGeometry[0][fused_geometry_idx]->getLiveDepthSurfels());
	//m_poissonreconstruction->DrawRebuildMesh();

	//bool checkPeriod = false;
	//if (frameIndex % 9999 == 2) { checkPeriod = true; }
	//else { checkPeriod = false; }
	//dynamicallyDraw->DrawLiveFusedPoints(surfelGeometry[0][fused_geometry_idx]->getLiveDepthSurfels(), checkPeriod);

	//Unmap attributes
	renderer->UnmapFusionMapsFromCuda();//buffer��unmap
	renderer->UnmapSurfelGeometryFromCUDA(0);//�±�������VBO��unmap
	renderer->UnmapSurfelGeometryFromCUDA(1);

	if (RealTimeDisplay) {
		const bool withRecent = drawRecent || isRefreshFrame;
		showRanderImage(totalSurfels, fused_geometry_idx, camera.GetWorld2CameraEigen(0), camera.GetInitWorld2CameraEigen(), withRecent);
		//showAlignmentErrorMap("node", 10.0f); 
		showAlignmentErrorMap("direct", 10.0f);
		//Visualizer::DrawUnitNodeError(warpField->getLiveNodesCoordinate(), nonRigidSolver->GetNodeUnitAlignmentError());
	}
	//if (frameIndex % 50 == 0) {
		std::cout << "�������������������������������� " << frameIndex << " ֡������������������������������" << std::endl;
		//renderAlbedo->DrawRenderedSurfels(surfelGeometry[0][fused_geometry_idx]->getLiveDepthSurfels(), frameIndex);
		renderPhong->DrawRenderedSurfels(surfelGeometry[0][fused_geometry_idx]->getLiveDepthSurfels(), frameIndex);
		//renderNormal->DrawRenderedSurfels(surfelGeometry[0][fused_geometry_idx]->getLiveDepthSurfels(), frameIndex);
	//}

	//Update the index
	updatedGeometryIndex = fused_geometry_idx;
}



void SparseSurfelFusion::AlgorithmSerial::showAllCameraImages()
{
	frameProcessor->showAllImages();
}


bool SparseSurfelFusion::AlgorithmSerial::shouldDoIntegration() const
{
	return true;
}

bool SparseSurfelFusion::AlgorithmSerial::shouldDoReinit(const size_t refreshDuration) const
{
	return configParser->ShouldDoReinitConfig(refreshDuration);
}

bool SparseSurfelFusion::AlgorithmSerial::shouldDrawRecentObservation() const
{
	return frameIndex - reinitFrameIndex <= Constants::kStableSurfelConfidenceThreshold + 1;
}

void SparseSurfelFusion::AlgorithmSerial::showRanderImage(const unsigned int num_vertex, int vao_idx, const Eigen::Matrix4f& world2camera, const Eigen::Matrix4f& init_world2camera, bool with_recent)
{
	renderer->ShowLiveNormalMap(num_vertex, vao_idx, frameIndex, world2camera, with_recent);
	renderer->ShowLiveAlbedoMap(num_vertex, vao_idx, frameIndex, world2camera, with_recent);
	renderer->ShowLivePhongMap(num_vertex, vao_idx, frameIndex, world2camera, with_recent);
	

	renderer->ShowReferenceNormalMap(num_vertex, vao_idx, frameIndex, init_world2camera, with_recent);
	renderer->ShowReferenceAlbedoMap(num_vertex, vao_idx, frameIndex, init_world2camera, with_recent);
	renderer->ShowReferencePhongMap(num_vertex, vao_idx, frameIndex, init_world2camera, with_recent);
}
void SparseSurfelFusion::AlgorithmSerial::showAlignmentErrorMap(std::string renderType, float scale)
{
	if (renderType == "node" || renderType == "Node") {
		nonRigidSolver->ComputeAlignmentErrorMapFromNode();
		for (int i = 0; i < deviceCount; i++) {
			cudaTextureObject_t alignmentErrorMap = nonRigidSolver->GetAlignmentErrorMap(i);
			Visualizer::DrawAlignmentErrorMap(alignmentErrorMap, frameProcessor->getForegroundMaskTexture(i), "AlignmentErrorMapNode_" + to_string(i), scale);
		}
	}
	else {
		nonRigidSolver->ComputeAlignmentErrorMapDirect();
		for (int i = 0; i < deviceCount; i++) {
			cudaTextureObject_t alignmentErrorMap = nonRigidSolver->GetAlignmentErrorMap(i);
			Visualizer::DrawAlignmentErrorMap(alignmentErrorMap, frameProcessor->getForegroundMaskTexture(i), "AlignmentErrorMapDirect_" + to_string(i), scale);
		}
	}
}

SparseSurfelFusion::AlgorithmSerial::visualizerIO SparseSurfelFusion::AlgorithmSerial::getVisualizerIO()
{
	visualizerIO io;
	io.canonicalVerticesConfidence = surfelGeometry[0][updatedGeometryIndex]->getCanonicalVertexConfidence();
	io.canonicalNormalRadius = surfelGeometry[0][updatedGeometryIndex]->getCanonicalNormalRadius();
	io.liveVertexConfidence = surfelGeometry[0][updatedGeometryIndex]->getLiveVertexConfidence();
	io.liveNormalRadius = surfelGeometry[0][updatedGeometryIndex]->getLiveNormalRadius();
	io.colorTime = surfelGeometry[0][updatedGeometryIndex]->getColorTime();
	return io;
}


void SparseSurfelFusion::AlgorithmSerial::LoadImages(bool intoBuffer)
{

	std::string dis, res, speed, action;
	std::string prefixPath = DATA_PATH_PREFIX;
	Constants::ReadDatasetParameter(prefixPath, res, dis, speed, action);
	std::string suffixPath = dis + "/" + action + "/" + res + "/" + speed;
	if (intoBuffer) {
		unsigned int maxFramesNum = 0;

		for (int i = 0; i < deviceCount; i++) {
			std::string path;
			if (action == "calibration") {
				path = prefixPath + dis + "/" + action + "/Camera_" + std::to_string(i);
			}
			else {
				path = prefixPath + dis + "/" + action + "/" + res + "/" + speed + "/Camera_" + std::to_string(i);
			}

			cv::String saveColorPath = cv::String(std::string(path + "/color"));
			cv::String saveDepthPath = cv::String(std::string(path + "/depth"));

			cv::glob(saveDepthPath, DepthOfflinePath[i]);
			cv::glob(saveColorPath, ColorOfflinePath[i]);
			std::sort(DepthOfflinePath[i].begin(), DepthOfflinePath[i].end());
			std::sort(ColorOfflinePath[i].begin(), ColorOfflinePath[i].end());

			// Replace backslashes with forward slashes
			for (auto& path : DepthOfflinePath[i]) {
				std::replace(path.begin(), path.end(), '\\', '/');
			}

			for (auto& path : ColorOfflinePath[i]) {
				std::replace(path.begin(), path.end(), '\\', '/');
			}
		}
		frameProcessor->setInput(ColorOfflinePath, DepthOfflinePath);

		if (speed == "slow") maxFramesNum = 1000;
		else if (speed == "fast") maxFramesNum = 500;
		else LOGGING(FATAL) << "MaxFramesNum ���ó���";

	
		for (int k = 0; k < deviceCount; k++) {
			for (int i = 0; i < maxFramesNum; i++) {
				std::cout << ColorOfflinePath << std::endl;
				while (true) {	// ��ȡͼ��ֱ������ǿ�ֵ����ʱ����ȡʧ��
					cv::Mat color;
					color = cv::imread(ColorOfflinePath[k][i], cv::IMREAD_ANYCOLOR);
					cv::waitKey(5);
					if (!color.empty()) {
						ColorOffline[k].push_back(color);
						break;
					}
				}
				while (true) {	// ��ȡͼ��ֱ������ǿ�ֵ����ʱ����ȡʧ��
					cv::Mat depth;
					depth = cv::imread(DepthOfflinePath[k][i], cv::IMREAD_ANYDEPTH);
					cv::waitKey(5);
					if (!depth.empty()) {
						DepthOffline[k].push_back(depth);
						break;
					}
				}
			}
		}
#ifdef CUDA_DEBUG_SYNC_CHECK
		CHECKCUDA(cudaDeviceSynchronize());
#endif 
		std::cout << "������ݼ���..." << std::endl;
	}	
	else {
		std::cout << "��֡��������..." << std::endl;
	}
	frameProcessor->GetOfflineData(ColorOffline, DepthOffline, suffixPath, intoBuffer);
}


