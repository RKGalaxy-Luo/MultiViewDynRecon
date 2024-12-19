#include <base/Constants.h>
#include "core/geometry/DoubleBufferCompactor.h"

SparseSurfelFusion::DoubleBufferCompactor::DoubleBufferCompactor() {
	//The row and column of the image

	for (int i = 0; i < devicesCount; i++) {
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}
}

SparseSurfelFusion::DoubleBufferCompactor::~DoubleBufferCompactor()
{
}

void SparseSurfelFusion::DoubleBufferCompactor::SetFusionInputs(
	const RemainingLiveSurfelKNN& remaining_surfels,
	const AppendedObservationSurfelKNN& appended_surfels,
	const unsigned int compacted_to_idx,
	SurfelGeometry::Ptr compacted_geometry[MAX_CAMERA_COUNT][2]
) {
	m_appended_surfel_knn = appended_surfels;
	m_remaining_surfel = remaining_surfels.liveGeometry;
	m_remaining_knn = remaining_surfels.remainingKnn;
	updatedGeometryIndex = compacted_to_idx;
	for (int i = 0; i < devicesCount; i++) {
		m_compact_to_geometry[i][compacted_to_idx] = compacted_geometry[i][compacted_to_idx];
	}

}

void SparseSurfelFusion::DoubleBufferCompactor::SetReinitInputs(
	const RemainingLiveSurfel &remaining_surfels,
	const ReinitAppendedObservationSurfel &append_surfels,
	const unsigned int compacted_to_idx,
	SparseSurfelFusion::SurfelGeometry::Ptr compact_to_geometry[MAX_CAMERA_COUNT][2]
) {
	m_reinit_append_surfel = append_surfels;
	m_remaining_surfel = remaining_surfels;
	updatedGeometryIndex = compacted_to_idx;
	for (int i = 0; i < devicesCount; i++) {
		m_compact_to_geometry[i][compacted_to_idx] = compact_to_geometry[i][compacted_to_idx];
	}
}

