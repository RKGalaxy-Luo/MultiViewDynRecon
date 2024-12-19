#include "sanity_check.h"
#include "base/Logging.h"
#include "DenseDepthHandler.h"
#include "solver_constants.h"
#include "geometry_icp_jacobian.cuh"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		__global__ void markMatchedGeometryPixelPairsKernel(
			ObservationDenseDepthHandlerInterface observedDenseDepthHandlerInterface,
			GeometryMapDenseDepthHandlerInterface geometryDenseDepthHandlerInterface,
			const unsigned int knnMapCols, const unsigned int knnMapRows, const unsigned int devicesCount,
			const DualQuaternion* device_warp_field,
			unsigned* reference_pixel_matched_indicator,
			ushort2* pixel_pairs_array
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= knnMapCols || y >= knnMapRows) return;
			const unsigned int knnMapSize = knnMapRows * knnMapCols;
			for (int i = 0; i < devicesCount; i++) {
				//The indicator will must be written to pixel_occupied_array
				unsigned valid_indicator = 0;
				ushort2 pixel_pair = make_ushort2(0xFFFF, 0xFFFF);
				//Read the value on index map
				const unsigned int surfel_index = tex2D<unsigned int>(geometryDenseDepthHandlerInterface.indexMap[i], x, y);
				if (surfel_index != d_invalid_index) {
					//Get the vertex
					const float4 can_vertex4 = tex2D<float4>(geometryDenseDepthHandlerInterface.referenceVertexMap[i], x, y);
					const float4 can_normal4 = tex2D<float4>(geometryDenseDepthHandlerInterface.referenceNormalMap[i], x, y);
					const KNNAndWeight knn = geometryDenseDepthHandlerInterface.knnMap[i](y, x);
					DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn.knn, knn.weight);
					const mat34 se3 = dq_average.se3_matrix();

					//And warp it
					const float3 warped_vertex = se3.rot * can_vertex4 + se3.trans;
					const float3 warped_normal = se3.rot * can_normal4;

					//Transfer to the camera frame
					const float3 warped_vertex_camera = geometryDenseDepthHandlerInterface.world2Camera[i].rot * warped_vertex + geometryDenseDepthHandlerInterface.world2Camera[i].trans;
					const float3 warped_normal_camera = geometryDenseDepthHandlerInterface.world2Camera[i].rot * warped_normal;

					//Project the vertex into image
					const int2 img_coord = {
						__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * geometryDenseDepthHandlerInterface.intrinsic[i].focal_x) + geometryDenseDepthHandlerInterface.intrinsic[i].principal_x),
						__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * geometryDenseDepthHandlerInterface.intrinsic[i].focal_y) + geometryDenseDepthHandlerInterface.intrinsic[i].principal_y)
					};

					if (img_coord.x >= 0 && img_coord.x < knnMapCols && img_coord.y >= 0 && img_coord.y < knnMapRows) {
						const float4 depth_vertex4 = tex2D<float4>(observedDenseDepthHandlerInterface.vertexMap[i], img_coord.x, img_coord.y);
						const float4 depth_normal4 = tex2D<float4>(observedDenseDepthHandlerInterface.normalMap[i], img_coord.x, img_coord.y);
						const float3 depth_vertex = make_float3(depth_vertex4.x, depth_vertex4.y, depth_vertex4.z);
						const float3 depth_normal = make_float3(depth_normal4.x, depth_normal4.y, depth_normal4.z);

						//Check the matched
						bool valid_pair = true;

						//The depth pixel is not valid
						if (is_zero_vertex(depth_vertex4)) {
							valid_pair = false;
						}

						//The orientation is not matched
						if (dot(depth_normal, warped_normal_camera) < d_correspondence_normal_dot_threshold) {
							valid_pair = false;
						}

						//The distance is too far away
						if (squared_norm(depth_vertex - warped_vertex_camera) > d_correspondence_distance_threshold_square) {
							valid_pair = false;
						}

						//Update if required
						if (valid_pair) {
							valid_indicator = 1;
							pixel_pair.x = img_coord.x;
							pixel_pair.y = img_coord.y;
						}
					} // The vertex project to a valid depth pixel
				} // The reference vertex is valid

				//Write to output
				const int offset = y * knnMapCols + x;
				if (valid_indicator > 0) {
					reference_pixel_matched_indicator[i * knnMapSize + offset] = valid_indicator;
					pixel_pairs_array[i * knnMapSize + offset] = pixel_pair;
				}
				else {
					reference_pixel_matched_indicator[i * knnMapSize + offset] = 0;
				}
			}
		}


		//__global__ void markPotentialMatchedDepthPairKernel(
		//	cudaTextureObject_t index_map,
		//	unsigned img_rows, unsigned img_cols,
		//	unsigned* reference_pixel_matched_indicator
		//) {
		//	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		//	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		//	if (x >= img_cols || y >= img_rows) return;

		//	//The indicator will must be written to pixel_occupied_array
		//	const auto offset = y * img_cols + x;

		//	//Read the value on index map
		//	const auto surfel_index = tex2D<unsigned>(index_map, x, y);

		//	//Need other criterion?
		//	unsigned indicator = 0;
		//	if (surfel_index != d_invalid_index) {
		//		indicator = 1;
		//	}

		//	reference_pixel_matched_indicator[offset] = indicator;
		//}


		__global__ void compactMatchedPixelPairsKernel(
			GeometryMapDenseDepthHandlerInterface geometryDenseDepthHandlerInterface,
			const unsigned int knnMapCols,
			const unsigned int knnMapRows,
			const unsigned int devicesCount,
			const unsigned int* reference_pixel_matched_indicator,
			const unsigned int* prefixsum_matched_indicator,
			const ushort2* pixel_pairs_map,
			ushort4* valid_pixel_pairs_array,
			ushort4* valid_pixel_pairs_knn,
			float4* valid_pixel_pairs_knn_weight
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= knnMapCols || y >= knnMapRows) return;
			const unsigned int knnMapSize = knnMapCols * knnMapRows;
			const unsigned int flattenIdx = x + y * knnMapCols;
			for (int i = 0; i < devicesCount; i++) {
				if (reference_pixel_matched_indicator[flattenIdx + i * knnMapSize] > 0) {
					const unsigned int offset = prefixsum_matched_indicator[flattenIdx + i * knnMapSize] - 1;
					const KNNAndWeight knn = geometryDenseDepthHandlerInterface.knnMap[i](y, x);
					const ushort2 target_pixel = pixel_pairs_map[flattenIdx + i * knnMapSize];
					valid_pixel_pairs_array[flattenIdx + i * knnMapSize] = make_ushort4(x, y, target_pixel.x, target_pixel.y);
					valid_pixel_pairs_knn[flattenIdx + i * knnMapSize] = knn.knn;
					valid_pixel_pairs_knn_weight[flattenIdx + i * knnMapSize] = knn.weight;
				}
			}
		}

		//__global__ void compactPontentialMatchedPixelPairsKernel(
		//	const DeviceArrayView2D<KNNAndWeight> knn_map,
		//	const unsigned* reference_pixel_matched_indicator,
		//	const unsigned* prefixsum_matched_indicator,
		//	ushort2* potential_matched_pixels,
		//	ushort4* potential_matched_knn,
		//	float4* potential_matched_knn_weight
		//) {
		//	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		//	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		//	if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;
		//	const auto flatten_idx = x + y * knn_map.Cols();
		//	if (reference_pixel_matched_indicator[flatten_idx] > 0)
		//	{
		//		const auto offset = prefixsum_matched_indicator[flatten_idx] - 1;
		//		const KNNAndWeight knn = knn_map(y, x);
		//		potential_matched_pixels[offset] = make_ushort2(x, y);
		//		potential_matched_knn[offset] = knn.knn;
		//		potential_matched_knn_weight[offset] = knn.weight;
		//	}

		//}

	} // namespace device
} // namespace SparseSurfelFusion




/* The methods for mark pixel pairs
 */
void SparseSurfelFusion::DenseDepthHandler::MarkMatchedPixelPairs(cudaStream_t stream) {
	dim3 block(16, 16);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y));
	device::markMatchedGeometryPixelPairsKernel << <grid, block, 0, stream >> > (
		observedDenseDepthHandlerInterface,
		geometryDenseDepthHandlerInterface,
		m_image_width,
		m_image_height,
		devicesCount,
		m_node_se3,
		//The output
		m_pixel_match_indicator.ptr(), m_pixel_pair_maps.ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

/* The method to compact the matched pixel pairs
 */
void SparseSurfelFusion::DenseDepthHandler::CompactMatchedPixelPairs(cudaStream_t stream) {
	//Do a prefix sum
	m_indicator_prefixsum.InclusiveSum(m_pixel_match_indicator, stream);

	//Invoke the kernel
	dim3 block(16, 16);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y));
	device::compactMatchedPixelPairsKernel << <grid, block, 0, stream >> >(
		geometryDenseDepthHandlerInterface,
		m_image_width,
		m_image_height,
		devicesCount,
		m_pixel_match_indicator,
		m_indicator_prefixsum.valid_prefixsum_array,
		m_pixel_pair_maps,
		m_valid_pixel_pairs.Ptr(),
		m_dense_depth_knn.Ptr(),
		m_dense_depth_knn_weight.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


void SparseSurfelFusion::DenseDepthHandler::SyncQueryCompactedArraySize(cudaStream_t stream)
{
	//Sync the stream and query the size
	unsigned num_valid_pairs;
	CHECKCUDA(cudaMemcpyAsync(
		&num_valid_pairs,
		m_indicator_prefixsum.valid_prefixsum_array.ptr() + m_pixel_match_indicator.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost, stream
	));
	CHECKCUDA(cudaStreamSynchronize(stream));

	//Correct the size of array
	m_valid_pixel_pairs.ResizeArrayOrException(num_valid_pairs);
	m_dense_depth_knn.ResizeArrayOrException(num_valid_pairs);
	m_dense_depth_knn_weight.ResizeArrayOrException(num_valid_pairs);
}