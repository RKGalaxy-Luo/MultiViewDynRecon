//#include <base/ConfigParser.h>
#include <base/Constants.h>
#include "GeometryReinitProcessor.h"


SparseSurfelFusion::GeometryReinitProcessor::GeometryReinitProcessor(SurfelGeometry::Ptr surfel_geometry[MAX_CAMERA_COUNT][2], Intrinsic * intrinsicArray){
	
	for (int i = 0; i < devicesCount; i++) {
		m_surfel_geometry[i][0] = surfel_geometry[i][0];
		m_surfel_geometry[i][1] = surfel_geometry[i][1];
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}

	m_updated_idx = 0;

	//Init of other attributes
	m_surfel_fusion_handler = std::make_shared<SurfelFusionHandler>();
	m_remaining_surfel_marker = std::make_shared<ReinitRemainingSurfelMarker>(intrinsicArray);
	
	//The buffer for prefix sum
	//const auto& config = ConfigParser::Instance();
	//const auto image_size = config.clip_image_rows() * config.clip_image_cols();

	appendedIndicatorPrefixsum.AllocateBuffer(devicesCount * clipedImageSize * 2);
	remainingIndicatorPrefixsum.AllocateBuffer(Constants::maxSurfelsNum);
	
	//The buffer for compactor
	m_surfel_compactor = std::make_shared<DoubleBufferCompactor>();
}

SparseSurfelFusion::GeometryReinitProcessor::~GeometryReinitProcessor()
{
}

void SparseSurfelFusion::GeometryReinitProcessor::SetInputs(
	Renderer::FusionMaps* fusionMap,
	const CameraObservation& observation,
	int updated_idx, 
	float current_time,
	const mat34* world2camera
) {
	for (int i = 0; i < devicesCount; i++) {
		m_fusion_maps[i] = fusionMap[i];
		m_world2camera[i] = world2camera[i];
	}
	
	m_observation = observation;
	m_updated_idx = updated_idx % 2;
	m_current_time = current_time;

}

void SparseSurfelFusion::GeometryReinitProcessor::ProcessReinitObservedOnlySerial(
	unsigned& num_remaining_surfel,
	unsigned& num_appended_surfel,
	unsigned* number,
	const unsigned int frameIdx,
	cudaStream_t stream
) { 
	const size_t num_surfels = m_surfel_geometry[0][m_updated_idx]->ValidSurfelsNum();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, stream);

	//Visualizer::DrawFusedProcessInCameraView(m_fusion_maps[1].warp_vertex_map, Constants::getInitialCameraSE3(1), m_fusion_maps[0].warp_vertex_map);
	//Visualizer::DrawFusedProcessInCameraView(m_fusion_maps[2].warp_vertex_map, m_world2camera[2], m_observation.vertexConfidenceMap[2]);

	// 将Live域RenderMap面元与观测帧稠密面元融合，如果RenderMap面元与观测面元能融合则保留，
	// 观测面元中没有被融合的一律添加到候选新增面元中
	FuseCameraObservationNoSync(stream);

	DeviceArrayHandle<unsigned int> remainingIndicator = m_surfel_fusion_handler->GetRemainingSurfelIndicator();
	//这是第一次融合的点 红色是融合后的，白色是未融合的，红色前，白色后
	//Visualizer::DrawFusedSurfelCloud(
	//	m_surfel_geometry[0][m_updated_idx]->getLiveVertexConfidence(),
	//	remainingIndicator.ArrayView(),
	//	m_observation.vertexConfidenceMap[0], toEigen(Constants::GetInitialCameraSE3(0)),
	//	m_observation.vertexConfidenceMap[1], toEigen(Constants::GetInitialCameraSE3(1)),
	//	m_observation.vertexConfidenceMap[2], toEigen(Constants::GetInitialCameraSE3(2))
	//);
	//Visualizer::DrawFusedSurfelCloud(
	//	m_surfel_geometry[0][m_updated_idx]->getLiveVertexConfidence(),
	//	remainingIndicator.ArrayView(),
	//	m_observation.vertexConfidenceMap[1], toEigen(m_world2camera[1].inverse() * Constants::getInitialCameraSE3(1))
	//);
	////当前帧融合后的can和live，红色前，白色后
	//Visualizer::DrawPointCloudcanandlive(m_surfel_geometry[0][m_updated_idx]->getCanonicalVertexConfidence(), m_surfel_geometry[0][m_updated_idx]->getLiveVertexConfidence());
	//CHECKCUDA(cudaDeviceSynchronize());
	// Live域中没有成功融合的面元，将这些面元与观测深度图进行对比，如果相似且在Foreground上，那么在Indicator上面标记保留
	MarkRemainingSurfelObservedOnly(stream);
	//CHECKCUDA(cudaDeviceSynchronize());


	//remainingIndicator = m_surfel_fusion_handler->GetRemainingSurfelIndicator();
	////这是第二次， 红色是融合的+保留的可观测的，白色是应该舍弃的点
	//Visualizer::DrawFusedSurfelCloud(m_surfel_geometry[0][m_updated_idx]->getLiveVertexConfidence(), remainingIndicator.ArrayView());
	// 对候选新增的面元指示器做前缀和
	processAppendedIndicatorPrefixsum(stream);
	//CHECKCUDA(cudaDeviceSynchronize());


	// 对保留的面元指示器做前缀和
	processRemainingIndicatorPrefixsum(stream);
	//CHECKCUDA(cudaDeviceSynchronize());


	//Ready for compaction
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, number, stream);

	//Visualizer::DrawPointCloudcanandlive(m_surfel_geometry[0][(m_updated_idx + 1) % 2]->getCanonicalVertexConfidence(), m_surfel_geometry[0][(m_updated_idx + 1) % 2]->getLiveVertexConfidence());

}


void SparseSurfelFusion::GeometryReinitProcessor::ProcessReinitNodeErrorSerial(
	unsigned & num_remaining_surfel, 
	unsigned & num_appended_surfel, 
	const NodeAlignmentError & node_error, 
	float threshold, 
	cudaStream_t stream
) {
	////没用这个带节点误差的
	//const auto num_surfels = m_surfel_geometry[m_updated_idx]->ValidSurfelsNum();
	//m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, stream);
	//FuseCameraObservationNoSync(stream);
	//MarkRemainingSurfelNodeError(node_error, threshold, stream);
	//
	////Do prefix sum
	//processAppendedIndicatorPrefixsum(stream);
	//processRemainingIndicatorPrefixsum(stream);
	//
	////Ready for compaction
	//CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, stream);
}

//Fuse camera observation into surfel geometry
void SparseSurfelFusion::GeometryReinitProcessor::FuseCameraObservationNoSync(cudaStream_t stream) {
	m_surfel_fusion_handler->SetInputs(m_fusion_maps, m_observation, m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess(), m_current_time, m_world2camera, false);
	
	//Do not use the internal candidate builder
	m_surfel_fusion_handler->ProcessFusionReinit(stream);
}


//Update the remaining indicator of the surfels
void SparseSurfelFusion::GeometryReinitProcessor::MarkRemainingSurfelObservedOnly(cudaStream_t stream) {
	// 暴露原始地址
	DeviceArrayHandle<unsigned int> remainingIndicator = m_surfel_fusion_handler->GetRemainingSurfelIndicator();

	//现在是只有融合的面元
	//hand in to marker
	m_remaining_surfel_marker->SetInputs(
		m_fusion_maps,
		m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess(),
		m_observation,
		m_current_time,
		m_world2camera,
		remainingIndicator
	);

	m_remaining_surfel_marker->MarkRemainingSurfelObservedOnly(stream);

}

void SparseSurfelFusion::GeometryReinitProcessor::MarkRemainingSurfelNodeError(
	const NodeAlignmentError & node_error, 
	float threshold, 
	cudaStream_t stream
) {
	// 暴露原始地址
	DeviceArrayHandle<unsigned int> remainingIndicator = m_surfel_fusion_handler->GetRemainingSurfelIndicator();
	
	m_remaining_surfel_marker->SetInputs(
		m_fusion_maps,
		m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess(),
		m_observation,
		m_current_time,
		m_world2camera,
		remainingIndicator
	);
	
	m_remaining_surfel_marker->MarkRemainingSurfelNodeError(node_error, threshold, stream);
}


//Compute the prefixsum for remaining indicator and appended indicator
void SparseSurfelFusion::GeometryReinitProcessor::processRemainingIndicatorPrefixsum(cudaStream_t stream) {
	const auto remaining_indicator = m_remaining_surfel_marker->GetRemainingSurfelIndicator();
	remainingIndicatorPrefixsum.InclusiveSum(remaining_indicator.ArrayView(), stream);
}

void SparseSurfelFusion::GeometryReinitProcessor::processAppendedIndicatorPrefixsum(cudaStream_t stream) {
	const DeviceArrayView<unsigned int> appendedIndicator = m_surfel_fusion_handler->GetAppendedObservationCandidateIndicator();
	appendedIndicatorPrefixsum.InclusiveSum(appendedIndicator, stream);
}

SparseSurfelFusion::RemainingLiveSurfel SparseSurfelFusion::GeometryReinitProcessor::getCompactionRemainingSurfel() const {
	RemainingLiveSurfel remaining_surfel;
	
	//The indicator after marker
	remaining_surfel.remainingIndicator = m_remaining_surfel_marker->GetRemainingSurfelIndicator().ArrayView();
	
	//Check and attach prefix sum
	const DeviceArray<unsigned int>& prefixsum_array = remainingIndicatorPrefixsum.valid_prefixsum_array;
	FUNCTION_CHECK(remaining_surfel.remainingIndicator.Size() == prefixsum_array.size());
	remaining_surfel.remainingIndicatorPrefixsum = prefixsum_array.ptr();
	
	//Check and attach geometry
	const SurfelGeometry::SurfelFusionInput& fusion_access = m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess();
	FUNCTION_CHECK(remaining_surfel.remainingIndicator.Size() == fusion_access.liveVertexConfidence.Size());
	FUNCTION_CHECK(remaining_surfel.remainingIndicator.Size() == fusion_access.liveNormalRadius.Size());
	FUNCTION_CHECK(remaining_surfel.remainingIndicator.Size() == fusion_access.colorTime.Size());
	remaining_surfel.liveVertexConfidence = fusion_access.liveVertexConfidence.RawPtr();
	remaining_surfel.liveNormalRadius = fusion_access.liveNormalRadius.RawPtr();
	remaining_surfel.colorTime = fusion_access.colorTime.RawPtr();
	
	//Everything is ready
	return remaining_surfel;
}

SparseSurfelFusion::ReinitAppendedObservationSurfel SparseSurfelFusion::GeometryReinitProcessor::getCompactionAppendedSurfel() const {
	ReinitAppendedObservationSurfel appended_observation;
	
	//The indicator directly from the fuser
	appended_observation.validityIndicator = m_surfel_fusion_handler->GetAppendedObservationCandidateIndicator();
	
	//Check and attach prefixsum
	const DeviceArray<unsigned int>& prefixsum_array = appendedIndicatorPrefixsum.valid_prefixsum_array;
	FUNCTION_CHECK(appended_observation.validityIndicator.Size() == prefixsum_array.size());
	appended_observation.validityIndicatorPrefixsum = prefixsum_array.ptr();
	
	//The texture object from observation
	for (int i = 0; i < devicesCount; i++) {
		appended_observation.observedVertexMap[i] = m_observation.vertexConfidenceMap[i];
		appended_observation.observedNormalMap[i] = m_observation.normalRadiusMap[i];
		appended_observation.observedColorMap[i] = m_observation.colorTimeMap[i];
		appended_observation.initialCameraSE3[i] = InitialCameraSE3[i];

		appended_observation.interVertexMap[i] = m_observation.interpolatedVertexMap[i];
		appended_observation.interNormalMap[i] = m_observation.interpolatedNormalMap[i];
		appended_observation.interColorMap[i] = m_observation.interpolatedColorMap[i];
	}
	return appended_observation;
}

void SparseSurfelFusion::GeometryReinitProcessor::CompactSurfelToAnotherBufferSync(
	unsigned int& num_remaining_surfels,
	unsigned int& num_appended_surfels,
	unsigned int* number, 
	cudaStream_t stream
) {
	//The input
	const RemainingLiveSurfel& remaining_surfels = getCompactionRemainingSurfel();
	const ReinitAppendedObservationSurfel& appended_surfels = getCompactionAppendedSurfel();
	
	//The output
	const int compacted_to_idx = (m_updated_idx + 1) % 2;

	mat34 World2Camera[MAX_CAMERA_COUNT];
	for (int i = 0; i < devicesCount; i++) {
		World2Camera[i] = m_world2camera[i];
	}
	
	//Do compaction在这里给另一个m_surfel_geometry初始化了
	m_surfel_compactor->SetReinitInputs(remaining_surfels, appended_surfels, compacted_to_idx, m_surfel_geometry);
	m_surfel_compactor->PerformComapctionGeometryOnlySync(num_remaining_surfels, num_appended_surfels, number, World2Camera, stream);
}
