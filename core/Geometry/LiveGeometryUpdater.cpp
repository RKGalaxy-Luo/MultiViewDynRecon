#include "fusion_types.h"
#include "LiveGeometryUpdater.h"
#include "SurfelFusionHandler.h"
#include "SurfelNodeDeformer.h"
#include "KNNSearch.h"
#include "LiveGeometryUpdater.h"


SparseSurfelFusion::LiveGeometryUpdater::LiveGeometryUpdater(SurfelGeometry::Ptr surfel_geometry[MAX_CAMERA_COUNT][2], Intrinsic *clipedintrinsic, const unsigned int devCount) : devicesCount(devCount)
{
	for (int i = 0; i < devicesCount; i++) {
		m_surfel_geometry[i][0] = surfel_geometry[i][0];
		m_surfel_geometry[i][1] = surfel_geometry[i][1];
	}


	m_updated_idx = 0;

	//The buffer allocation mehtods
	m_surfel_fusion_handler = std::make_shared<SurfelFusionHandler>();
	m_fusion_remaining_surfel_marker = std::make_shared<FusionRemainingSurfelMarker>(clipedintrinsic);
	m_appended_surfel_processor = std::make_shared<AppendSurfelProcessor>();
	m_surfel_compactor = std::make_shared<DoubleBufferCompactor>();

	initFusionStream();
}

SparseSurfelFusion::LiveGeometryUpdater::~LiveGeometryUpdater() {
	releaseFusionStream();
}


void SparseSurfelFusion::LiveGeometryUpdater::SetInputs(
	const Renderer::FusionMaps* maps,
	const CameraObservation& observation,
	const WarpField::LiveGeometryUpdaterInput& warpfield_input,
	const KNNSearch::Ptr& live_node_skinner,
	int updated_idx,
	float current_time,
	const mat34* world2camera
) {
	for (int i = 0; i < devicesCount; i++) {
		m_fusion_maps[i] = maps[i];
		m_world2camera[i] = world2camera[i];
	}
	m_observation = observation;
	m_warpfield_input = warpfield_input;
	m_live_node_skinner = live_node_skinner;
	m_updated_idx = updated_idx % 2;
	m_current_time = current_time;

}

void SparseSurfelFusion::LiveGeometryUpdater::TestFusion() {
	const auto num_surfels = m_surfel_geometry[0][m_updated_idx]->ValidSurfelsNum();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels);
	FuseCameraObservationSync();
	MarkRemainingSurfels();
	ProcessAppendedSurfels();

	//Do compaction
	unsigned num_remaining_surfel, num_appended_surfel;
	//CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel);

	//Do some checking on the compacted geometry
	//TestCompactionKNNFirstIter(num_remaining_surfel, num_appended_surfel);
}

void SparseSurfelFusion::LiveGeometryUpdater::ProcessFusionSerial(
	unsigned& num_remaining_surfel,
	unsigned& num_appended_surfel,
	cudaStream_t stream
) {
	const auto num_surfels = m_surfel_geometry[0][m_updated_idx]->ValidSurfelsNum();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, stream);
	FuseCameraObservationSync(stream);
	MarkRemainingSurfels(stream);
	ProcessAppendedSurfels(stream);
	//CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, stream);
}

/* The map to perform surfel fusion
 */
void SparseSurfelFusion::LiveGeometryUpdater::FuseCameraObservationSync(cudaStream_t stream) {
	m_surfel_fusion_handler->SetInputs(m_fusion_maps, m_observation, m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess(), m_current_time, m_world2camera, true);
	m_surfel_fusion_handler->ProcessFusion(stream);
	m_surfel_fusion_handler->BuildCandidateAppendedPixelsSync(); //This requires sync
}


/* The buffer and method to clear the existing surfel based on knn
 */
void SparseSurfelFusion::LiveGeometryUpdater::MarkRemainingSurfels(cudaStream_t stream) {
	auto remaining_surfel_indicator = m_surfel_fusion_handler->GetFusionIndicator().remaining_surfel_indicator;
	m_fusion_remaining_surfel_marker->SetInputs(m_fusion_maps, m_world2camera);
	m_fusion_remaining_surfel_marker->SetInputsNoNeedfor(
		m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess(),
		m_current_time,
		remaining_surfel_indicator//ArrayHandle的引用！！
	);


	m_fusion_remaining_surfel_marker->UpdateRemainingSurfelIndicator(stream);

	//Do prefixsum in another stream
	m_fusion_remaining_surfel_marker->RemainingSurfelIndicatorPrefixSum(stream);
}


SparseSurfelFusion::RemainingLiveSurfelKNN SparseSurfelFusion::LiveGeometryUpdater::GetRemainingLiveSurfelKNN() const {
	//From which geometry
	const auto geometry_from = m_surfel_geometry[0][m_updated_idx]->SurfelFusionAccess();

	RemainingLiveSurfelKNN remaining_surfel_knn;
	//The indicator part
	const auto& indicator = m_fusion_remaining_surfel_marker->GetRemainingSurfelIndicator();
	const auto& indicator_prefixsum = m_fusion_remaining_surfel_marker->GetRemainingSurfelIndicatorPrefixsum();
	FUNCTION_CHECK(indicator.Size() == indicator_prefixsum.Size());
	remaining_surfel_knn.liveGeometry.remainingIndicator = indicator;
	remaining_surfel_knn.liveGeometry.remainingIndicatorPrefixsum = indicator_prefixsum.RawPtr();

	//The geometry part
	remaining_surfel_knn.liveGeometry.liveVertexConfidence = geometry_from.liveVertexConfidence.RawPtr();
	remaining_surfel_knn.liveGeometry.liveNormalRadius = geometry_from.liveNormalRadius.RawPtr();
	remaining_surfel_knn.liveGeometry.colorTime = geometry_from.colorTime.RawPtr();

	//The knn part
	remaining_surfel_knn.remainingKnn.surfelKnn = geometry_from.surfelKnn.RawPtr();
	remaining_surfel_knn.remainingKnn.surfelKnnWeight = geometry_from.surfelKnnWeight.RawPtr();
	return remaining_surfel_knn;
}

/* Check inconsistent skinning and collision at the appended surfels
 */
void SparseSurfelFusion::LiveGeometryUpdater::ProcessAppendedSurfels(cudaStream_t stream) {
	const DeviceArrayView<ushort4> appended_pixel = m_surfel_fusion_handler->GetFusionIndicator().appended_pixels;
#ifdef DEBUG_RUNNING_INFO
	printf("待添加的appended_pixel = %lld \n", appended_pixel.Size());
#endif // DEBUG_RUNNING_INFO
	m_appended_surfel_processor->SetInputs(m_observation, m_world2camera, m_warpfield_input, m_live_node_skinner, appended_pixel);
	
	//Do processing
	//m_appended_surfel_processor->BuildVertexForFiniteDifference(stream);
	m_appended_surfel_processor->BuildSurfelAndFiniteDiffVertex(stream);
	m_appended_surfel_processor->SkinningFiniteDifferenceVertex(stream);
	m_appended_surfel_processor->FilterCandidateSurfels(stream);
}

/* Compact surfel to another buffer
 */
void SparseSurfelFusion::LiveGeometryUpdater::CompactSurfelToAnotherBufferSync(
	unsigned int& num_remaining_surfel,
	unsigned int& num_appended_surfel,
	cudaStream_t stream
) {
	//Construct the remaining surfel
	RemainingLiveSurfelKNN remaining_surfel_knn = GetRemainingLiveSurfelKNN();

	//Construct the appended surfel
	const AppendedObservationSurfelKNN appended_surfel = m_appended_surfel_processor->GetAppendedObservationSurfel();

	//The buffer that the compactor should write to
	const int compacted_to_idx = (m_updated_idx + 1) % 2;//处理完第零帧，在处理第一帧时，这里是1了！！

	//Ok, seems everything is ready
	m_surfel_compactor->SetFusionInputs(remaining_surfel_knn, appended_surfel, compacted_to_idx, m_surfel_geometry);
	m_surfel_compactor->PerformCompactionGeometryKNNSync(num_remaining_surfel, num_appended_surfel, stream);
}

void SparseSurfelFusion::LiveGeometryUpdater::TestCompactionKNNFirstIter(unsigned num_remaining_surfel, unsigned num_appended_surfel) {
	const auto compacted_to_idx = (m_updated_idx + 1) % 2;
	const auto geometry_to = m_surfel_geometry[0][compacted_to_idx]->SurfelFusionAccess();

	//Sanity check
	FUNCTION_CHECK(geometry_to.liveVertexConfidence.Size() == num_remaining_surfel + num_appended_surfel);

	//Check the appended surfel, they should be skinned using live nodes: seems correct
	{
		DeviceArrayView<float4> vertex = DeviceArrayView<float4>(geometry_to.liveVertexConfidence.RawPtr() + num_remaining_surfel, num_appended_surfel);
		DeviceArrayView<ushort4> knn = DeviceArrayView<ushort4>(geometry_to.surfelKnn.RawPtr() + num_remaining_surfel, num_appended_surfel);
		KNNSearch::CheckKNNSearch(m_warpfield_input.live_node_coords, vertex, knn);
	}

	//Check the remaining surfel, should use approximate knn search
	{
		//DeviceArrayView<float4> vertex = DeviceArrayView<float4>(geometry_to.live_vertex_confid.RawPtr(), num_remaining_surfel);
		//DeviceArrayView<ushort4> knn = DeviceArrayView<ushort4>(geometry_to.surfel_knn.RawPtr(), num_remaining_surfel);
		//KNNSearch::CheckApproximateKNNSearch(m_warpfield_input.live_node_coords, vertex, knn);
	}
}


/* The method for multi-stream processing
 */
void SparseSurfelFusion::LiveGeometryUpdater::initFusionStream() {
	CHECKCUDA(cudaStreamCreate(&m_fusion_stream[0]));
	CHECKCUDA(cudaStreamCreate(&m_fusion_stream[1]));
}

void SparseSurfelFusion::LiveGeometryUpdater::releaseFusionStream() {
	CHECKCUDA(cudaStreamDestroy(m_fusion_stream[0]));
	CHECKCUDA(cudaStreamDestroy(m_fusion_stream[1]));
	m_fusion_stream[0] = 0;
	m_fusion_stream[1] = 0;
}

void SparseSurfelFusion::LiveGeometryUpdater::ProcessFusionStreamed(unsigned& num_remaining_surfel, unsigned& num_appended_surfel) {

	const size_t num_surfels = m_surfel_geometry[0][m_updated_idx]->ValidSurfelsNum();
#ifdef DEBUG_RUNNING_INFO
	printf("当前Canincal中有效面元个数 = %zu \n", num_surfels);
#endif // DEBUG_RUNNING_INFO


	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, m_fusion_stream[1]);

	FuseCameraObservationSync(m_fusion_stream[0]);
	CHECKCUDA(cudaStreamSynchronize(m_fusion_stream[0]));
	CHECKCUDA(cudaStreamSynchronize(m_fusion_stream[1]));

	ProcessAppendedSurfels(m_fusion_stream[1]);
	MarkRemainingSurfels(m_fusion_stream[0]);

	CHECKCUDA(cudaStreamSynchronize(m_fusion_stream[0]));
	CHECKCUDA(cudaStreamSynchronize(m_fusion_stream[1]));

	// 添加的，要手动分离需要显示的添加面元
	const DeviceArrayView<ushort4> appended_pixel = m_surfel_fusion_handler->GetFusionIndicator().appended_pixels;
	DeviceArrayView<float4> refvertex = m_surfel_geometry[0][m_updated_idx]->GetCanonicalVertexConfidence();

	// 这个函数是传递surfelgemertory数据的地方，所以这里边需要改动
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, m_fusion_stream[1]);
}



