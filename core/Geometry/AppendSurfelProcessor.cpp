//
// Created by wei on 5/4/18.
//

#include <base/Constants.h>
#include "AppendSurfelProcessor.h"


SparseSurfelFusion::AppendSurfelProcessor::AppendSurfelProcessor() {
	memset(&m_observation, 0, sizeof(m_observation));
	
	//Init the warp field input
	m_warpfield_input.live_node_coords = DeviceArrayView<float4>();
	m_warpfield_input.reference_node_coords = DeviceArrayView<float4>();
	m_warpfield_input.node_se3 = DeviceArrayView<DualQuaternion>();
	m_live_node_skinner = nullptr;
	
	//The buffer for surfel and finite difference vertex
	m_surfel_vertex_confid.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_normal_radius.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_color_time.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_candidate_vertex_finite_diff.AllocateBuffer(Constants::kMaxNumSurfelCandidates * (kNumFiniteDiffVertex));
	
	//The buffer and array for skinning
	m_candidate_vertex_finitediff_knn.AllocateBuffer(m_candidate_vertex_finite_diff.Capacity());
	m_candidate_vertex_finitediff_knnweight.AllocateBuffer(m_candidate_vertex_finite_diff.Capacity());
	
	//The indicator for the array
	m_candidate_surfel_validity_indicator.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_knn.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_knn_weight.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_candidate_surfel_validity_prefixsum.AllocateBuffer(Constants::kMaxNumSurfelCandidates);

	for (int i = 0; i < devicesCount; i++) {
		m_observation.InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);//1-0
	}
}

SparseSurfelFusion::AppendSurfelProcessor::~AppendSurfelProcessor() {
	m_surfel_vertex_confid.ReleaseBuffer();
	m_surfel_normal_radius.ReleaseBuffer();
	m_surfel_color_time.ReleaseBuffer();
	m_candidate_vertex_finite_diff.ReleaseBuffer();
	
	m_candidate_vertex_finitediff_knn.ReleaseBuffer();
	m_candidate_vertex_finitediff_knnweight.ReleaseBuffer();
	
	m_candidate_surfel_validity_indicator.ReleaseBuffer();
	m_surfel_knn.ReleaseBuffer();
	m_surfel_knn_weight.ReleaseBuffer();
}

void SparseSurfelFusion::AppendSurfelProcessor::SetInputs(
	const CameraObservation &observation,
	const mat34* world2camera,
	const WarpField::LiveGeometryUpdaterInput& warpfield_input,
	const KNNSearch::Ptr& live_node_skinner,
	const DeviceArrayView<ushort4>& pixel_coordinate
) {
	for (int i = 0; i < devicesCount; i++) {
		m_observation.vertex_confid_map[i] = observation.vertexConfidenceMap[i];
		m_observation.normal_radius_map[i] = observation.normalRadiusMap[i];
		m_observation.color_time_map[i] = observation.colorTimeMap[i];
		m_observation.inter_vertex_map[i] = observation.interpolatedVertexMap[i];
		m_observation.inter_normal_map[i] = observation.interpolatedNormalMap[i];
		m_observation.inter_color_map[i] = observation.interpolatedColorMap[i];
		m_observation.m_camera2world[i] = world2camera[i].inverse();
	}
	m_warpfield_input = warpfield_input;
	m_live_node_skinner = live_node_skinner;
	m_surfel_candidate_pixel = pixel_coordinate;
}

SparseSurfelFusion::AppendedObservationSurfelKNN SparseSurfelFusion::AppendSurfelProcessor::GetAppendedObservationSurfel() const {
	AppendedObservationSurfelKNN observation_surfel_knn;
	observation_surfel_knn.validityIndicator = m_candidate_surfel_validity_indicator.ArrayView();
	observation_surfel_knn.validityIndicatorPrefixsum = m_candidate_surfel_validity_prefixsum.valid_prefixsum_array.ptr();
	observation_surfel_knn.surfelVertexConfidence = m_surfel_vertex_confid.Ptr();
	observation_surfel_knn.surfelNormalRadius = m_surfel_normal_radius.Ptr();
	observation_surfel_knn.surfelColorTime = m_surfel_color_time.Ptr();
	observation_surfel_knn.surfelKnn = m_surfel_knn.Ptr();
	observation_surfel_knn.surfelKnnWeight = m_surfel_knn_weight.Ptr();
	return observation_surfel_knn;
}
