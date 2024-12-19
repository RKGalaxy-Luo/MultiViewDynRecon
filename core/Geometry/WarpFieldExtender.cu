#include <base/Constants.h>
#include "WarpFieldExtender.h"

#include <device_launch_parameters.h>

namespace SparseSurfelFusion { 
	namespace device {
	
		/* Kernel and method for choosing node candidate from init knn array (not field)
		*/
		__global__ void labelVertexCandidateKernel(
			const DeviceArrayView<float4> vertex_confid_array,
			const ushort4* vertex_knn_array,
			const float4* node_coords_array,
			unsigned* vertex_candidate_label
		) {
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= vertex_confid_array.Size()) return;

			//Obtain vertex and its knn
			const float4 vertex_confid = vertex_confid_array[idx];
			const float4 vertex = make_float4(vertex_confid.x, vertex_confid.y, vertex_confid.z, 1.0);
			const ushort4 knn = vertex_knn_array[idx];
			//if (idx <30) {
			//	printf("(%f, %f, %f) \n", vertex_confid.x, vertex_confid.y, vertex_confid.z);
			//	//printf("(%u, %u, %u) \n", knn.x, knn.y, knn.z);
			//}
			//Check its distance to node 
			float4 node; float dist_square;
			bool covered = false;

			//knn-0
			node = node_coords_array[knn.x];
			dist_square = squared_norm_xyz(node - vertex);
			if (dist_square < NODE_RADIUS_SQUARE) {
				covered = true;
			}

			//knn-1
			node = node_coords_array[knn.y];
			dist_square = squared_norm_xyz(node - vertex);
			if (dist_square < NODE_RADIUS_SQUARE) {
				covered = true;
			}

			//knn-2
			node = node_coords_array[knn.z];
			dist_square = squared_norm_xyz(node - vertex);
			if (dist_square < NODE_RADIUS_SQUARE) {
				covered = true;
			}

			//knn-3
			node = node_coords_array[knn.w];
			dist_square = squared_norm_xyz(node - vertex);
			if (dist_square < NODE_RADIUS_SQUARE) {
				covered = true;
			}

			//Write it to output
			unsigned label = 1;
			if (covered) {
				label = 0;
			}
			vertex_candidate_label[idx] = label;
			//printf("vertex_candidate_label[%u] = %u\n", idx, label);
		}

		__global__ void compactCandidateKernel(
			const DeviceArrayView<unsigned> candidate_validity_label,
			const unsigned* prefixsum_validity_label,
			const float4* vertex_array,
			const float4* color_view_time_array,
			float4* valid_candidate_vertex
		) {
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= candidate_validity_label.Size()) return;
			if (candidate_validity_label[idx] > 0) {
				const float4 vertex = vertex_array[idx];
				const float cameraView = color_view_time_array[idx].y;
				valid_candidate_vertex[prefixsum_validity_label[idx] - 1] = make_float4(vertex.x, vertex.y, vertex.z, cameraView);
			}
		}

	}
}


void SparseSurfelFusion::WarpFieldExtender::labelCollectUncoveredNodeCandidate(
	const DeviceArrayView<float4>& vertexArray,
	const DeviceArrayView<float4>& colorViewTime,
	const DeviceArrayView<ushort4>& vertex_knn,
	const DeviceArrayView<float4>& node_coordinates,
	cudaStream_t stream
) {
	m_candidate_validity_indicator.ResizeArrayOrException(vertexArray.Size());
	
	dim3 blk(64);
	dim3 grid(divUp(vertexArray.Size(), blk.x));
	//如果label是1表示未覆盖的
	device::labelVertexCandidateKernel << <grid, blk, 0, stream >> > (
		vertexArray,
		vertex_knn.RawPtr(),
		node_coordinates.RawPtr(),
		m_candidate_validity_indicator.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
	
	//Do a prefix sum
	FUNCTION_CHECK(vertexArray.Size() == m_candidate_validity_indicator.ArraySize());
	m_validity_indicator_prefixsum.InclusiveSum(m_candidate_validity_indicator.ArrayView(), stream);
	
	//Do compaction
	device::compactCandidateKernel << <grid, blk, 0, stream >> > (
		m_candidate_validity_indicator.ArrayView(),
		m_validity_indicator_prefixsum.valid_prefixsum_array.ptr(),
		vertexArray.RawPtr(),
		colorViewTime.RawPtr(),
		m_candidate_vertex_array.DevicePtr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::WarpFieldExtender::syncQueryUncoveredNodeCandidateSize(
	cudaStream_t stream
) {
	//Check the size
	const auto& prefixsum_array = m_validity_indicator_prefixsum.valid_prefixsum_array;
	FUNCTION_CHECK(prefixsum_array.size() == m_candidate_validity_indicator.ArraySize());
	
	//The device ptr
	const unsigned* candidate_size_dev = prefixsum_array.ptr() + prefixsum_array.size() - 1;
	unsigned candidate_size;
	cudaSafeCall(cudaMemcpyAsync(
		&candidate_size,
		candidate_size_dev,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	
	//Sync and check the size
	CHECKCUDA(cudaStreamSynchronize(stream));
	m_candidate_vertex_array.ResizeArrayOrException(candidate_size);
	if(candidate_size != 0)
		m_candidate_vertex_array.SynchronizeToHost(stream, true);
	
	//printf("新添加的深度面元中，没有被当前节点图覆盖的深度面元的个数 = %u\n", candidate_size);
	//Debug method
	//LOGGING(INFO) << "The number of node candidates is " << candidate_size/*m_candidate_vertex_array.DeviceArraySize()*/<<std::endl;
}


