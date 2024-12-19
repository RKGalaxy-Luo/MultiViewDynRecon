#include <base/Logging.h>
#include <base/Constants.h>
#include <base/ConfigParser.h>
#include "sanity_check.h"
#include <base/data_transfer.h>
#include <base/CommonUtils.h>
#include "DenseDepthHandler.h"

SparseSurfelFusion::DenseDepthHandler::DenseDepthHandler() {
	memset(&observedDenseDepthHandlerInterface, 0, sizeof(device::ObservationDenseDepthHandlerInterface));
	memset(&geometryDenseDepthHandlerInterface, 0, sizeof(device::GeometryMapDenseDepthHandlerInterface));
	for (int i = 0; i < devicesCount; i++) {
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
		InitialCameraSE3Inv[i] = InitialCameraSE3[i].inverse();
	}
}


void SparseSurfelFusion::DenseDepthHandler::AllocateBuffer() {
	const unsigned int num_pixels = devicesCount * m_image_height * m_image_width;

	//The buffer to match the pixel pairs
	m_pixel_match_indicator.create(num_pixels);
	m_pixel_pair_maps.create(num_pixels);
	
	//The buffer to compact the pixel pairs
	m_indicator_prefixsum.AllocateBuffer(num_pixels);
	m_valid_pixel_pairs.AllocateBuffer(num_pixels);
	m_dense_depth_knn.AllocateBuffer(num_pixels);
	m_dense_depth_knn_weight.AllocateBuffer(num_pixels);

	//The buffer for gradient
	m_term_residual.AllocateBuffer(num_pixels);
	m_term_twist_gradient.AllocateBuffer(num_pixels);

	//The buffer for alignment error
	for (int i = 0; i < devicesCount; i++) {
		createFloat1TextureSurface(m_image_height, m_image_width, nodeAccumulatedErrorAndWeight.alignmentErrorMap[i]);
	}

	nodeAccumulatedError.AllocateBuffer(Constants::maxNodesNum);
	nodeAccumulatedWeight.AllocateBuffer(Constants::maxNodesNum);
	nodeUnitedAlignmentError.AllocateBuffer(Constants::maxNodesNum);
	nodeLargeNodeErrorNum.AllocateBuffer(Constants::maxNodesNum);
}


void SparseSurfelFusion::DenseDepthHandler::ReleaseBuffer() {
	m_pixel_match_indicator.release();
	m_pixel_pair_maps.release();

	m_valid_pixel_pairs.ReleaseBuffer();
	m_dense_depth_knn.ReleaseBuffer();
	m_dense_depth_knn_weight.ReleaseBuffer();

	m_term_residual.ReleaseBuffer();
	m_term_twist_gradient.ReleaseBuffer();

	nodeAccumulatedError.ReleaseBuffer();
	nodeAccumulatedWeight.ReleaseBuffer();
	nodeUnitedAlignmentError.ReleaseBuffer();
	nodeLargeNodeErrorNum.ReleaseBuffer();
}

void SparseSurfelFusion::DenseDepthHandler::SetInputs(
	const DeviceArrayView<DualQuaternion>& node_se3,
	DeviceArray2D<KNNAndWeight>* knnMap,
	cudaTextureObject_t* depthVertexMap, 
	cudaTextureObject_t* depthNormalMap,
	cudaTextureObject_t* filteredForegroundMap,
	Renderer::SolverMaps* solverMap,
	mat34* world2camera,
	const ImageTermKNNFetcher::ImageTermPixelAndKNN& pixelsKnn
) {
	m_node_se3 = node_se3;
	m_potential_pixels_knn = pixelsKnn;
	for (int i = 0; i < devicesCount; i++) {
		observedDenseDepthHandlerInterface.vertexMap[i] = depthVertexMap[i];
		observedDenseDepthHandlerInterface.normalMap[i] = depthNormalMap[i];
		observedDenseDepthHandlerInterface.filteredForegroundMap[i] = filteredForegroundMap[i];

		geometryDenseDepthHandlerInterface.knnMap[i] = knnMap[i];
		geometryDenseDepthHandlerInterface.world2Camera[i] = world2camera[i];
		geometryDenseDepthHandlerInterface.camera2World[i] = world2camera[i].inverse();
		geometryDenseDepthHandlerInterface.referenceVertexMap[i] = solverMap[i].reference_vertex_map;
		geometryDenseDepthHandlerInterface.referenceNormalMap[i] = solverMap[i].reference_normal_map;
		geometryDenseDepthHandlerInterface.indexMap[i] = solverMap[i].index_map;
		geometryDenseDepthHandlerInterface.InitialCameraSE3[i] = InitialCameraSE3[i];
		geometryDenseDepthHandlerInterface.InitialCameraSE3Inverse[i] = InitialCameraSE3Inv[i];
	}
}

void SparseSurfelFusion::DenseDepthHandler::SetInitalData(Intrinsic* intr)
{
	for (int i = 0; i < devicesCount; i++) {
		geometryDenseDepthHandlerInterface.intrinsic[i] = intr[i];
	}
	
}

void SparseSurfelFusion::DenseDepthHandler::UpdateNodeSE3(SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DualQuaternion> node_se3) {
	FUNCTION_CHECK_EQ(node_se3.Size(), m_node_se3.Size());
	m_node_se3 = node_se3;
}

void SparseSurfelFusion::DenseDepthHandler::FindCorrespondenceSynced(cudaStream_t stream) {
	MarkMatchedPixelPairs(stream);
	CompactMatchedPixelPairs(stream);
	SyncQueryCompactedArraySize(stream);
}

void SparseSurfelFusion::DenseDepthHandler::compactedPairSanityCheck(DeviceArrayView<ushort4> surfel_knn_array)
{
	//LOGGING(INFO) << "Check of compacted pair sanity";

	//FUNCTION_CHECK_EQ(m_dense_depth_knn_weight.ArraySize(), m_valid_pixel_pairs.ArraySize());
	//FUNCTION_CHECK_EQ(m_dense_depth_knn.ArraySize(), m_valid_pixel_pairs.ArraySize());

	////Query the index maps
	//DeviceArray<unsigned> queried_index_array;
	//queried_index_array.create(m_valid_pixel_pairs.ArraySize());
	//queryIndexMapFromPixels(m_geometry_maps.index_map, m_valid_pixel_pairs.ArrayReadOnly(), queried_index_array);

	////Download the index
	//std::vector<unsigned> h_queried_index;
	//queried_index_array.download(h_queried_index);

	////Download the knn array and compacted_knn array
	//std::vector<ushort4> h_surfel_knn_array, h_compacted_knn_array;
	//surfel_knn_array.Download(h_surfel_knn_array);
	//m_dense_depth_knn.ArrayReadOnly().Download(h_compacted_knn_array);

	////Check it
	//for (auto i = 0; i < m_dense_depth_knn.ArraySize(); i++)
	//{
	//	ushort4 compacted_knn = h_compacted_knn_array[i];
	//	auto index = h_queried_index[i];
	//	FUNCTION_CHECK_NE(index, 0xFFFFFFFF);
	//	FUNCTION_CHECK(index < h_surfel_knn_array.size());
	//	ushort4 original_knn = h_surfel_knn_array[index];
	//	FUNCTION_CHECK_EQ(compacted_knn.x, original_knn.x);
	//	FUNCTION_CHECK_EQ(compacted_knn.y, original_knn.y);
	//	FUNCTION_CHECK_EQ(compacted_knn.z, original_knn.z);
	//	FUNCTION_CHECK_EQ(compacted_knn.w, original_knn.w);
	//}

	//LOGGING(INFO) << "Seems correct";
}
