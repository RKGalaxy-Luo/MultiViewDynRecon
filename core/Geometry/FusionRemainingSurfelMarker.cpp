#include <base/Logging.h>
#include <base/Constants.h>
#include "FusionRemainingSurfelMarker.h"
#include <base/data_transfer.h>


SparseSurfelFusion::FusionRemainingSurfelMarker::FusionRemainingSurfelMarker(Intrinsic *rgbintrinsicclip){
	memset(&m_fusion_maps, 0, sizeof(m_fusion_maps));
	memset(&m_live_geometry, 0, sizeof(m_live_geometry));
	
	for (int i = 0; i < devicesCount; i++) {
		m_world2camera[i] = mat34::identity();
		m_intrinsic[i] = rgbintrinsicclip[i];
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}
	m_current_time = 0;

	//The buffer for prefixsum
	m_remaining_indicator_prefixsum.AllocateBuffer(Constants::maxSurfelsNum);
}


void SparseSurfelFusion::FusionRemainingSurfelMarker::SetInputs(
	const Renderer::FusionMaps* maps, 
	const mat34* world2camera
) {

	for (int i = 0; i < devicesCount; i++) {
		m_fusion_maps.vertex_confid_map[i] = maps[i].warp_vertex_map;
		m_fusion_maps.normal_radius_map[i] = maps[i].warp_normal_map;
		m_fusion_maps.index_map[i] = maps[i].index_map;
		m_fusion_maps.color_time_map[i] = maps[i].color_time_map;

		m_world2camera[i] = world2camera[i];
	}
	//debugIndexMap(m_fusion_maps.index_map[0], m_fusion_maps.index_map[1], m_fusion_maps.index_map[2], "fusionMapIndexMap");

}

void SparseSurfelFusion::FusionRemainingSurfelMarker::SetInputsNoNeedfor(
	const SurfelGeometry::SurfelFusionInput& geometry,
	float current_time,
	const DeviceArrayHandle<unsigned>& remaining_surfel_indicator
) {
	m_live_geometry.vertex_confid = geometry.liveVertexConfidence.ArrayView();
	m_live_geometry.normal_radius = geometry.liveNormalRadius.ArrayView();
	m_live_geometry.color_time = geometry.colorTime.ArrayView();

	m_current_time = current_time;

	m_remaining_surfel_indicator = remaining_surfel_indicator;

	//Do a sanity check?
	FUNCTION_CHECK_EQ(remaining_surfel_indicator.Size(), m_live_geometry.vertex_confid.Size());
	FUNCTION_CHECK_EQ(remaining_surfel_indicator.Size(), m_live_geometry.normal_radius.Size());
	FUNCTION_CHECK_EQ(remaining_surfel_indicator.Size(), m_live_geometry.color_time.Size());
}


