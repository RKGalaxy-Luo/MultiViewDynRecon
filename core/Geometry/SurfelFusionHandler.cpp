#include <base/Constants.h>
#include <core/NonRigidSolver/sanity_check.h>
#include <core/Geometry/SurfelFusionHandler.h>
#include <map>

SparseSurfelFusion::SurfelFusionHandler::SurfelFusionHandler(){
	memset(&m_fusion_maps, 0, sizeof(m_fusion_maps));
	memset(&m_fusion_geometry, 0, sizeof(m_fusion_geometry));
	
	//The surfel indicator is in the size of maximun surfels
	remainingSurfelIndicator.AllocateBuffer(Constants::maxSurfelsNum);

	fusion.AllocateBuffer(Constants::maxSurfelsNum);
	remain.AllocateBuffer(Constants::maxSurfelsNum);
	CHECKCUDA(cudaMalloc(&FusedDepthSurfelNum, sizeof(unsigned int)));
	CHECKCUDA(cudaMalloc(&RemainingLiveSurfelNum, sizeof(unsigned int)));
	
	//The append depth indicator is always in the same size as image pixels
	appendedObservedSurfelIndicator.create(devicesCount * clipedImageSize * 2);
	appendedObservedSurfelIndicatorPrefixSum.AllocateBuffer(devicesCount * clipedImageSize * 2);
	compactedAppendedPixel.AllocateBuffer(devicesCount * clipedImageSize);

	
	//The buffer for atomic appending
	CHECKCUDA(cudaMalloc(&atomicAppendedPixelIndex, sizeof(unsigned int)));
	atomicAppendedObservationPixel.AllocateBuffer(devicesCount * (uint64_t)clipedImageRows * (uint64_t)clipedImageCols);

	for (int i = 0; i < devicesCount; i++) {
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}
}

SparseSurfelFusion::SurfelFusionHandler::~SurfelFusionHandler() {
	remainingSurfelIndicator.ReleaseBuffer();
	//debug
	fusion.ReleaseBuffer();
	remain.ReleaseBuffer();

	appendedObservedSurfelIndicator.release();
	//The buffer for atomic appending
	CHECKCUDA(cudaFree(FusedDepthSurfelNum));
	CHECKCUDA(cudaFree(RemainingLiveSurfelNum));
	CHECKCUDA(cudaFree(atomicAppendedPixelIndex));
	atomicAppendedObservationPixel.ReleaseBuffer();
}

void SparseSurfelFusion::SurfelFusionHandler::SetInputs(
	const Renderer::FusionMaps* maps,
	const CameraObservation& observation,
	const SurfelGeometry::SurfelFusionInput& geometry,
	float current_time,
	const mat34* world2camera,
	bool use_atomic_append
) {
	for (int i = 0; i < devicesCount; i++) {
		m_fusion_maps[i] = maps[i];
		m_world2camera[i] = world2camera[i];
	}

	m_observation = observation;
	m_fusion_geometry = geometry;
	m_current_time = current_time;
	m_use_atomic_append = use_atomic_append;
}

SparseSurfelFusion::SurfelFusionHandler::FusionIndicator SparseSurfelFusion::SurfelFusionHandler::GetFusionIndicator() {
	FusionIndicator indicator;
	indicator.remaining_surfel_indicator = remainingSurfelIndicator.ArraySlice();
	//This also depends on whether using atomic append
	if(m_use_atomic_append) indicator.appended_pixels = atomicAppendedObservationPixel.ArrayView();
	else indicator.appended_pixels = compactedAppendedPixel.ArrayView();
	return indicator;
}


void SparseSurfelFusion::SurfelFusionHandler::ZeroInitializeRemainingIndicator(unsigned num_surfels, cudaStream_t stream) {
	CHECKCUDA(cudaMemsetAsync(remainingSurfelIndicator.Ptr(), 0, num_surfels * sizeof(unsigned), stream));
	remainingSurfelIndicator.ResizeArrayOrException(num_surfels);
}

SparseSurfelFusion::DeviceArrayHandle<unsigned> SparseSurfelFusion::SurfelFusionHandler::GetRemainingSurfelIndicator() {
	return remainingSurfelIndicator.ArraySlice();
}

//Only meaningful when using compaction append
SparseSurfelFusion::DeviceArrayView<unsigned> SparseSurfelFusion::SurfelFusionHandler::GetAppendedObservationCandidateIndicator() const {
	return DeviceArrayView<unsigned>(appendedObservedSurfelIndicator);
}

void SparseSurfelFusion::SurfelFusionHandler::ProcessFusion(cudaStream_t stream) {
	//Debug check
	//SURFELWARP_CHECK(!containsNaN(m_fusion_geometry.live_vertex_confid.ArrayView()));
	
	//Do fusion
	if(m_use_atomic_append) processFusionAppendAtomic(stream);
	else processFusionAppendCompaction(stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
	
	//Debug method
	//fusionStatistic(kUseAtomicAppend);
	//confidenceStatistic();
}

void SparseSurfelFusion::SurfelFusionHandler::ProcessFusionReinit(cudaStream_t stream) {
	processFusionReinit(stream);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::SurfelFusionHandler::BuildCandidateAppendedPixelsSync(cudaStream_t stream) {
	if (m_use_atomic_append) queryAtomicAppendedPixelSize(stream);
	else compactAppendedIndicator(stream);
}


/* These are debug methods
 */
void SparseSurfelFusion::SurfelFusionHandler::fusionStatistic(bool using_atomic) {
	LOGGING(INFO) << "The total number of surfel is " << remainingSurfelIndicator.ArraySize();
	LOGGING(INFO) << "The number of fused surfel is " << numNonZeroElement(remainingSurfelIndicator.ArrayView());
	if(using_atomic) {
		unsigned num_appended_surfels = 0;
		CHECKCUDA(cudaMemcpy(&num_appended_surfels, atomicAppendedPixelIndex, sizeof(unsigned), cudaMemcpyDeviceToHost));
		LOGGING(INFO) << "The number of appended observation surfel is " << num_appended_surfels;
	} else {
		LOGGING(INFO) << "The number of appended observation surfel is " << numNonZeroElement(DeviceArrayView<unsigned>(appendedObservedSurfelIndicator));
	}
}


void SparseSurfelFusion::SurfelFusionHandler::confidenceStatistic() {
	//Download the vertex confidence array
	std::vector<float4> h_vertex_confid;
	m_fusion_geometry.liveVertexConfidence.SyncDeviceToHost(h_vertex_confid);

	std::map<unsigned, unsigned> confid2number;
	for(auto i = 0; i < h_vertex_confid.size(); i++) {
		float confidence = h_vertex_confid[i].w;
		confidence += 0.01;
		if(confidence < 1.0f) {
			LOGGING(INFO) << "The vertex " << i;
		}
		const auto uint_confid = unsigned(confidence);
		auto ptr = confid2number.find(uint_confid);
		if(ptr == confid2number.end()) {
			confid2number[uint_confid] = 1;
		} else {
			const auto value = ptr->second;
			confid2number[uint_confid] = value + 1;
		}
	}

	LOGGING(INFO) << "The confidence of surfels";
	for(auto iter = confid2number.begin(); iter != confid2number.end(); iter++) {
		LOGGING(INFO) << "The confidence at " << iter->first << " has " << iter->second << " surfels";
	}
	LOGGING(INFO) << "End of the confidence of surfels";
}



