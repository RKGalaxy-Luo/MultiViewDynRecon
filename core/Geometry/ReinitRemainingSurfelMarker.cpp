//#include "common/ConfigParser.h"
#include "ReinitRemainingSurfelMarker.h"


SparseSurfelFusion::ReinitRemainingSurfelMarker::ReinitRemainingSurfelMarker(Intrinsic* clipedIntrinsicArray){
	memset(&m_surfel_geometry, 0, sizeof(m_surfel_geometry));
	memset(&m_observation, 0, sizeof(m_observation));
	memset(&m_fusion_maps, 0, sizeof(m_fusion_maps));
	for (int i = 0; i < devicesCount; i++) {
		m_world2camera[i] = mat34::identity();
		m_intrinsic[i] = clipedIntrinsicArray[i];
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}
}

void SparseSurfelFusion::ReinitRemainingSurfelMarker::SetInputs(
	const Renderer::FusionMaps* maps, 
	const SurfelGeometry::SurfelFusionInput& geometry,
	const CameraObservation& observation,
	float current_time, 
	const mat34* world2camera, 
	const DeviceArrayHandle<unsigned>& remaining_surfel_indicator
) {
	for (int i = 0; i < devicesCount; i++)
	{
		m_fusion_maps[i] = maps[i];
		m_world2camera[i] = world2camera[i];
	}

	m_surfel_geometry = geometry;
	m_observation = observation;

	m_remaining_surfel_indicator = remaining_surfel_indicator;

	//Sanity check
	FUNCTION_CHECK_EQ(remaining_surfel_indicator.Size(), geometry.liveVertexConfidence.Size());
	FUNCTION_CHECK_EQ(remaining_surfel_indicator.Size(), geometry.liveNormalRadius.Size());
	FUNCTION_CHECK_EQ(remaining_surfel_indicator.Size(), geometry.colorTime.Size());
}