//
// Created by wei on 5/11/18.
//

#include <base/Constants.h>
#include "WarpFieldExtender.h"
#include "WarpFieldUpdater.h"

SparseSurfelFusion::WarpFieldExtender::WarpFieldExtender() {
	//As the extender might be called on the whole surfel geometry
	//allcoate the maximun amount of buffer

	m_candidate_validity_indicator.AllocateBuffer(Constants::maxSurfelsNum);
	m_validity_indicator_prefixsum.AllocateBuffer(Constants::maxSurfelsNum);
	m_candidate_vertex_array.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
}

SparseSurfelFusion::WarpFieldExtender::~WarpFieldExtender() {
	m_candidate_validity_indicator.ReleaseBuffer();
}

void SparseSurfelFusion::WarpFieldExtender::ExtendReferenceNodesAndSE3Sync(
	const DeviceArrayView<float4>& reference_vertex,
	const DeviceArrayView<float4>& colorViewTime,
	const DeviceArrayView<ushort4>& vertex_knn,
	WarpField::Ptr& warp_field,
	cudaStream_t stream
) {
	FUNCTION_CHECK(reference_vertex.Size() == vertex_knn.Size());
	if(reference_vertex.Size() == 0) return;

	// 首先收集潜在的候选节点
	const DeviceArrayView<float4> node_coordinates = warp_field->getCanonicalNodesCoordinate();
	labelCollectUncoveredNodeCandidate(reference_vertex, colorViewTime, vertex_knn, node_coordinates, stream);
	syncQueryUncoveredNodeCandidateSize(stream);

	// 使用候选节点更新 warp field
	if(m_candidate_vertex_array.DeviceArraySize() > 0) {
		const std::vector<float4>& h_candidate = m_candidate_vertex_array.HostArray();
		WarpFieldUpdater::UpdateWarpFieldFromUncoveredCandidate(*warp_field, h_candidate, stream);
	}
}


