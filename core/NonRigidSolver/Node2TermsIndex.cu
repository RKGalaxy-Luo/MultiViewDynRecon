#include "Node2TermsIndex.h"
#include <base/term_offset_types.h>
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		__global__ void buildTermKeyValueKernel(
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph,
			//These terms might be empty
			DeviceArrayView<ushort4> foreground_mask_knn,
			DeviceArrayView<ushort4> sparse_feature_knn,
			DeviceArrayView<ushort4> cross_corr_knn,
			//The output
			unsigned short* node_keys,
			unsigned* term_values
		) {
			const auto term_idx = threadIdx.x + blockDim.x * blockIdx.x;

			//The offset value for term and kv
			unsigned term_offset = 0;
			unsigned kv_offset = 0;

			//This term is in the scope of dense depth
			if (term_idx < dense_image_knn.Size()) {
				const auto in_term_offset = term_idx - term_offset;
				const auto fill_offset = kv_offset + 4 * in_term_offset;
				const auto knn = dense_image_knn[in_term_offset];
				node_keys[fill_offset + 0] = knn.x;
				node_keys[fill_offset + 1] = knn.y;
				node_keys[fill_offset + 2] = knn.z;
				node_keys[fill_offset + 3] = knn.w;
				term_values[fill_offset + 0] = term_idx;
				term_values[fill_offset + 1] = term_idx;
				term_values[fill_offset + 2] = term_idx;
				term_values[fill_offset + 3] = term_idx;
				return;
			}

			//For smooth term
			term_offset += dense_image_knn.Size();
			kv_offset += 4 * dense_image_knn.Size();
			if (term_idx < term_offset + node_graph.Size()) {
				const auto in_term_offset = term_idx - term_offset;
				const auto fill_offset = kv_offset + 2 * in_term_offset;
				const auto node_pair = node_graph[in_term_offset];
				node_keys[fill_offset + 0] = node_pair.x;
				node_keys[fill_offset + 1] = node_pair.y;
				term_values[fill_offset + 0] = term_idx;
				term_values[fill_offset + 1] = term_idx;
				return;
			}

			//This is for foreground mask
			term_offset += node_graph.Size();
			kv_offset += 2 * node_graph.Size();
			if (term_idx < term_offset + foreground_mask_knn.Size()) {
				const auto in_term_offset = term_idx - term_offset;
				const auto fill_offset = kv_offset + 4 * in_term_offset;
				const auto knn = foreground_mask_knn[in_term_offset];
				node_keys[fill_offset + 0] = knn.x;
				node_keys[fill_offset + 1] = knn.y;
				node_keys[fill_offset + 2] = knn.z;
				node_keys[fill_offset + 3] = knn.w;
				term_values[fill_offset + 0] = term_idx;
				term_values[fill_offset + 1] = term_idx;
				term_values[fill_offset + 2] = term_idx;
				term_values[fill_offset + 3] = term_idx;
				return;
			}

			//For sparse feature
			term_offset += foreground_mask_knn.Size();
			kv_offset += 4 * foreground_mask_knn.Size();
			if (term_idx < term_offset + sparse_feature_knn.Size())
			{
				const auto in_term_offset = term_idx - term_offset;
				const auto fill_offset = kv_offset + 4 * in_term_offset;
				const auto knn = sparse_feature_knn[in_term_offset];
				node_keys[fill_offset + 0] = knn.x;
				node_keys[fill_offset + 1] = knn.y;
				node_keys[fill_offset + 2] = knn.z;
				node_keys[fill_offset + 3] = knn.w;
				term_values[fill_offset + 0] = term_idx;
				term_values[fill_offset + 1] = term_idx;
				term_values[fill_offset + 2] = term_idx;
				term_values[fill_offset + 3] = term_idx;
				return;
			}

			//For cross view correspondence
			term_offset += sparse_feature_knn.Size();
			kv_offset += 4 * sparse_feature_knn.Size();
			if (term_idx < term_offset + cross_corr_knn.Size())
			{
				const auto in_term_offset = term_idx - term_offset;
				const auto fill_offset = kv_offset + 4 * in_term_offset;
				const auto knn = cross_corr_knn[in_term_offset];
				node_keys[fill_offset + 0] = knn.x;
				node_keys[fill_offset + 1] = knn.y;
				node_keys[fill_offset + 2] = knn.z;
				node_keys[fill_offset + 3] = knn.w;
				term_values[fill_offset + 0] = term_idx;
				term_values[fill_offset + 1] = term_idx;
				term_values[fill_offset + 2] = term_idx;
				term_values[fill_offset + 3] = term_idx;
				return;
			}
		} // The kernel to fill the key-value pairs

		__global__ void computeTermOffsetKernel(
			const DeviceArrayView<unsigned short> sorted_term_key,
			DeviceArrayHandle<unsigned> node2term_offset
		) {
			const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= sorted_term_key.Size()) return;

			if (idx == 0) {
				node2term_offset[0] = 0;
			}
			else {
				//i0和i1是节点的编号
				const auto i_0 = sorted_term_key[idx - 1];
				const auto i_1 = sorted_term_key[idx];
				if (i_0 != i_1) {
					//这个里边序号是节点的编号，里边的值是在原来排序后的key里的偏移
					node2term_offset[i_1] = idx;
				}
				if (idx == sorted_term_key.Size() - 1) {
					node2term_offset[i_1 + 1] = sorted_term_key.Size();
				}
			}
		}

	}
}


void SparseSurfelFusion::Node2TermsIndex::buildTermKeyValue(cudaStream_t stream) {
	//Correct the size
	const auto num_kv_pairs = NumKeyValuePairs();
	m_node_keys.ResizeArrayOrException(num_kv_pairs);
	m_term_idx_values.ResizeArrayOrException(num_kv_pairs);

	const unsigned int num_terms = NumTerms();
	dim3 block(128);
	dim3 grid(divUp(num_terms, block.x));
	device::buildTermKeyValueKernel << <grid, block, 0, stream >> > (
		m_term2node.dense_image_knn,
		m_term2node.node_graph,
		m_term2node.foreground_mask_knn,
		m_term2node.sparse_feature_knn,
		m_term2node.cross_corr_knn,
		m_node_keys.Ptr(),
		m_term_idx_values.Ptr()
	);
	//printf("(denseImage, nodeGragh, foregroundMask, Feature) = (%d, %d, %d, %d)\n", m_term2node.dense_image_knn.Size(), m_term2node.node_graph.Size(), m_term2node.foreground_mask_knn.Size(), m_term2node.sparse_feature_knn.Size());


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif
}


void SparseSurfelFusion::Node2TermsIndex::sortCompactTermIndex(cudaStream_t stream) {
	// 将根据knn的数据排升序：
	m_node2term_sorter.Sort(m_node_keys.ArrayView(), m_term_idx_values.ArrayView(), stream);

	m_node2term_offset.ResizeArrayOrException(m_num_nodes + 1);
	const auto offset_slice = m_node2term_offset.ArraySlice();
	const DeviceArrayView<unsigned short> sorted_key_view(m_node2term_sorter.valid_sorted_key.ptr(), m_node2term_sorter.valid_sorted_key.size());

	//Compute the offset map
	dim3 blk(256);
	dim3 grid(divUp(m_node2term_sorter.valid_sorted_key.size(), blk.x));
	device::computeTermOffsetKernel << <grid, blk, 0, stream >> > (sorted_key_view, offset_slice);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif
}

