#include <base/Constants.h>
#include "SparseCorrespondenceHandler.h"

void SparseSurfelFusion::SparseCorrespondenceHandler::AllocateBuffer() {
	const auto max_features = Constants::kMaxMatchedSparseFeature;

	m_valid_pixel_indicator.AllocateBuffer(max_features);
	m_valid_pixel_prefixsum.AllocateBuffer(max_features);
	m_corrected_pixel_pairs.AllocateBuffer(max_features);

	m_valid_target_vertex.AllocateBuffer(max_features);
	m_valid_reference_vertex.AllocateBuffer(max_features);
	m_valid_vertex_knn.AllocateBuffer(max_features);
	m_valid_knn_weight.AllocateBuffer(max_features);
	m_valid_warped_vertex.AllocateBuffer(max_features);
	differentViewsCorrPairsOffset.AllocateBuffer(MAX_CAMERA_COUNT);

	CHECKCUDA(cudaMallocHost((void**)(&m_correspondence_array_size), sizeof(unsigned int)));
}

void SparseSurfelFusion::SparseCorrespondenceHandler::ReleaseBuffer() {
	
	m_valid_pixel_indicator.ReleaseBuffer();
	m_corrected_pixel_pairs.ReleaseBuffer();

	m_valid_target_vertex.ReleaseBuffer();
	m_valid_reference_vertex.ReleaseBuffer();
	m_valid_vertex_knn.ReleaseBuffer();
	m_valid_knn_weight.ReleaseBuffer();

	m_valid_warped_vertex.ReleaseBuffer();
	differentViewsCorrPairsOffset.ReleaseBuffer();

	CHECKCUDA(cudaFreeHost(m_correspondence_array_size));
}


/* The main processing interface
 */
void SparseSurfelFusion::SparseCorrespondenceHandler::SetInputs(
	DeviceArrayView<DualQuaternion> node_se3,
	DeviceArray2D<KNNAndWeight> *knn_map,
	cudaTextureObject_t *depth_vertex_map,
	cudaTextureObject_t* edgeMaskMap,
	DeviceArrayView<ushort4> *correspond_pixel_pairs,
	Renderer::SolverMaps *solvermaps,
	mat34* world2camera,
	mat34* InitialCameraSE3
) {
	m_node_se3 = node_se3;
	for (int i = 0; i < devicesCount; i++){
		observedSparseCorrInterface.depthVertexMap[i] = depth_vertex_map[i];
		observedSparseCorrInterface.correspondPixelPairs[i] = correspond_pixel_pairs[i];
		observedSparseCorrInterface.edgeMask[i] = edgeMaskMap[i];

		geometrySparseCorrInterface.referenceVertexMap[i] = solvermaps[i].reference_vertex_map;
		geometrySparseCorrInterface.liveVertexMap[i] = solvermaps[i].warp_vertex_map;
		geometrySparseCorrInterface.indexMap[i] = solvermaps[i].index_map;
		geometrySparseCorrInterface.knnMap[i] = knn_map[i];
		geometrySparseCorrInterface.camera2World[i] = world2camera[i].inverse();
		geometrySparseCorrInterface.initialCameraSE3[i] = InitialCameraSE3[i];
	}
}


void SparseSurfelFusion::SparseCorrespondenceHandler::UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3) {
	FUNCTION_CHECK(m_node_se3.Size() == node_se3.Size());
	m_node_se3 = node_se3;
}


/* Build the correspondence vertex pairs
 */
void SparseSurfelFusion::SparseCorrespondenceHandler::BuildCorrespondVertexKNN(cudaStream_t stream) {
	/*ChooseValidPixelPairs(stream);
	CompactQueryPixelPairs(stream);
	QueryCompactedArraySize(stream); */
	//This will sync host threads with stream
}



