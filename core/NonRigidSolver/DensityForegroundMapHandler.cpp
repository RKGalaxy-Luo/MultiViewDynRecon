#include "DensityForegroundMapHandler.h"
#include "visualization/Visualizer.h"
SparseSurfelFusion::DensityForegroundMapHandler::DensityForegroundMapHandler() {
	
	m_image_height = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
	m_image_width = FRAME_WIDTH - 2 * CLIP_BOUNDARY;

	memset(&depthObservationForegroundInterface, 0, sizeof(device::DepthObservationForegroundInterface));
	memset(&geometryMapForegroundInterface, 0, sizeof(device::GeometryMapForegroundInterface));

	for (int i = 0; i < devicesCount; i++) {
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
		InitialCameraSE3Inv[i] = InitialCameraSE3[i].inverse();
	}
}


void SparseSurfelFusion::DensityForegroundMapHandler::AllocateBuffer() {
	const auto num_pixels = MAX_CAMERA_COUNT * m_image_height * m_image_width;

	//The buffer of the marked pixel pairs
	m_color_pixel_indicator_map.create(num_pixels);
	m_mask_pixel_indicator_map.create(num_pixels);

	//The compaction maps
	m_color_pixel_indicator_prefixsum.AllocateBuffer(num_pixels);
	m_mask_pixel_indicator_prefixsum.AllocateBuffer(num_pixels);
	m_valid_color_pixel.AllocateBuffer(num_pixels);
	m_valid_mask_pixel.AllocateBuffer(num_pixels);
	m_valid_color_pixel_knn.AllocateBuffer(num_pixels);
	m_valid_mask_pixel_knn.AllocateBuffer(num_pixels);
	m_valid_color_pixel_knn_weight.AllocateBuffer(num_pixels);
	m_valid_mask_pixel_knn_weight.AllocateBuffer(num_pixels);
	differentViewsForegroundMapOffset.AllocateBuffer(MAX_CAMERA_COUNT);

	//The page-locked memory
	cudaSafeCall(cudaMallocHost((void**)(&m_num_mask_pixel), sizeof(unsigned)));

	//The twist gradient
	m_color_residual.AllocateBuffer(num_pixels);
	m_color_twist_gradient.AllocateBuffer(num_pixels);
	m_foreground_residual.AllocateBuffer(num_pixels);
	m_foreground_twist_gradient.AllocateBuffer(num_pixels);
}

void SparseSurfelFusion::DensityForegroundMapHandler::ReleaseBuffer() {

	m_color_pixel_indicator_map.release();
	m_mask_pixel_indicator_map.release();

	//The compaction maps
	m_valid_color_pixel.ReleaseBuffer();
	m_valid_mask_pixel.ReleaseBuffer();
	m_valid_color_pixel_knn.ReleaseBuffer();
	m_valid_mask_pixel_knn.ReleaseBuffer();
	m_valid_color_pixel_knn_weight.ReleaseBuffer();
	m_valid_mask_pixel_knn_weight.ReleaseBuffer();
	differentViewsForegroundMapOffset.ReleaseBuffer();

	//The page-lock memory
	CHECKCUDA(cudaFreeHost(m_num_mask_pixel));

	//The twist gradient
	m_color_residual.ReleaseBuffer();
	m_color_twist_gradient.ReleaseBuffer();
	m_foreground_residual.ReleaseBuffer();
	m_foreground_twist_gradient.ReleaseBuffer();
}

/* The processing interface
 */
void SparseSurfelFusion::DensityForegroundMapHandler::SetInputs(
	const DeviceArrayView<DualQuaternion>& node_se3,
	DeviceArray2D<KNNAndWeight>* knnMap,
	//The foreground mask terms
	cudaTextureObject_t* foregroundMask,
	cudaTextureObject_t* filteredForegroundMask,
	cudaTextureObject_t* foregroundGradientMap,
	//The color density terms
	cudaTextureObject_t* densityMap,
	cudaTextureObject_t* densityGradientMap,
	cudaTextureObject_t* normalizedRgbMap,
	mat34* world2camera,
	Intrinsic* Clipcolor,
	Renderer::SolverMaps* solver,
	//The potential pixels,
	const ImageTermKNNFetcher::ImageTermPixelAndKNN& potentialPixelsKnn
) {
	m_node_se3 = node_se3;
	m_potential_pixels_knn = potentialPixelsKnn;
	for (int i = 0; i < devicesCount; i++) {
		depthObservationForegroundInterface.foregroundMask[i] = foregroundMask[i];
		depthObservationForegroundInterface.filteredForegroundMask[i] = filteredForegroundMask[i];
		depthObservationForegroundInterface.foregroundMaskGradientMap[i] = foregroundGradientMap[i];
		depthObservationForegroundInterface.densityMap[i] = densityMap[i];
		depthObservationForegroundInterface.densityGradientMap[i] = densityGradientMap[i];

		geometryMapForegroundInterface.referenceVertexMap[i] = solver[i].reference_vertex_map;
		geometryMapForegroundInterface.referenceNormalMap[i] = solver[i].reference_normal_map;
		geometryMapForegroundInterface.indexMap[i] = solver[i].index_map;
		geometryMapForegroundInterface.normalizedRgbMap[i] = solver[i].normalized_rgb_map;
		geometryMapForegroundInterface.intrinsic[i] = Clipcolor[i];
		geometryMapForegroundInterface.world2Camera[i] = world2camera[i];
		geometryMapForegroundInterface.knnMap[i] = knnMap[i];
		geometryMapForegroundInterface.initialCameraSE3[i] = InitialCameraSE3[i];
		geometryMapForegroundInterface.initialCameraSE3Inverse[i] = InitialCameraSE3Inv[i];
	}
}


void SparseSurfelFusion::DensityForegroundMapHandler::UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3) {
	FUNCTION_CHECK_EQ(node_se3.Size(), m_node_se3.Size());
	m_node_se3 = node_se3;
}

void SparseSurfelFusion::DensityForegroundMapHandler::FindValidColorForegroundMaskPixels(
	cudaStream_t color_stream,
	cudaStream_t mask_stream
) {
	//Use color stream for marking the value
	//MarkValidColorForegroundMaskPixels(color_stream);

	//Sync before using more streams
	CHECKCUDA(cudaStreamSynchronize(color_stream));

	//Use two streams for compaction
	CompactValidColorPixel(color_stream);
	//CompactValidMaskPixel(mask_stream);

	//Query the size: this will sync the stream, these will sync
	QueryCompactedColorPixelArraySize(color_stream);
	QueryCompactedMaskPixelArraySize(mask_stream);
}

void SparseSurfelFusion::DensityForegroundMapHandler::FindPotentialForegroundMaskPixelSynced(cudaStream_t stream) {
	//Use color stream for marking the value
	//MarkValidColorForegroundMaskPixels(,stream);

	//Use two streams for compaction
	//CompactValidMaskPixel(stream);

	//Query the size: this will sync the stream, these will sync
	QueryCompactedMaskPixelArraySize(stream);
}

void SparseSurfelFusion::DensityForegroundMapHandler::getDiffoffsetDensityForegroundMapHandler(std::vector<unsigned> &diff)
{
	differentViewsForegroundMapOffset.Array().download(diff);
}