#include "NodeGraphSmoothHandler.h"
#include <base/Constants.h>
#include "solver_constants.h"

#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		__global__ void forwardWarpSmootherNodeKernel(
			DeviceArrayView<ushort2> node_graph,
			const float4* reference_node_array,
			const DualQuaternion* node_se3,
			float3* Ti_xj_array,
			float3* Tj_xj_array,
			unsigned char* validity_indicator_array
		) {
			const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx < node_graph.Size()) {
				const ushort2 node_ij = node_graph[idx];
				
				const float4 xi = reference_node_array[node_ij.x];	// xi是当前Reference节点
				const float4 xj = reference_node_array[node_ij.y];	// xj是xi的8个邻居中的一个

				DualQuaternion dq_i = node_se3[node_ij.x];			// 获得当前节点自身的dq
				DualQuaternion dq_j = node_se3[node_ij.y];			// 获得这个节点邻居的dq
				const mat34 Ti = dq_i.se3_matrix();					// 获得当前节点自身的Se3
				const mat34 Tj = dq_j.se3_matrix();					// 获得这个节点邻居的Se3

				const float3 Ti_xj = Ti.rot * xj + Ti.trans;		// 用当前节点的Se3扭曲其所有邻居
				const float3 Tj_xj = Tj.rot * xj + Tj.trans;		// 用当前节点的邻居自己的Se3扭曲他自己
				unsigned char validity_indicator = 1;
#if defined(CLIP_FARAWAY_NODEGRAPH_PAIR)
				if (squared_norm_xyz(xi - xj) > 64 * NODE_RADIUS_SQUARE) {
					validity_indicator = 0;
				}
#endif
				//Save all the data
				Ti_xj_array[idx] = Ti_xj;
				Tj_xj_array[idx] = Tj_xj;
				validity_indicator_array[idx] = validity_indicator;
			}
		}
	}
}


SparseSurfelFusion::NodeGraphSmoothHandler::NodeGraphSmoothHandler() {
	const auto num_smooth_terms = Constants::maxNodesNum * Constants::nodesGraphNeigboursNum;
	Ti_xj_.AllocateBuffer(num_smooth_terms);
	Tj_xj_.AllocateBuffer(num_smooth_terms);
	m_pair_validity_indicator.AllocateBuffer(num_smooth_terms);
}

SparseSurfelFusion::NodeGraphSmoothHandler::~NodeGraphSmoothHandler() {
	Ti_xj_.ReleaseBuffer();
	Tj_xj_.ReleaseBuffer();
	m_pair_validity_indicator.ReleaseBuffer();
}

void SparseSurfelFusion::NodeGraphSmoothHandler::SetInputs(
	const DeviceArrayView<DualQuaternion>& node_se3,
	const DeviceArrayView<ushort2>& node_graph,
	const DeviceArrayView<float4>& reference_nodes
) {
	m_node_se3 = node_se3;
	m_node_graph = node_graph;
	m_reference_node_coords = reference_nodes;
}



/* The method to build the term2jacobian
 */
void SparseSurfelFusion::NodeGraphSmoothHandler::forwardWarpSmootherNodes(cudaStream_t stream) {
	Ti_xj_.ResizeArrayOrException(m_node_graph.Size());//节点图的大小是8倍的节点数
	Tj_xj_.ResizeArrayOrException(m_node_graph.Size());
	m_pair_validity_indicator.ResizeArrayOrException(m_node_graph.Size());

	dim3 blk(128);
	dim3 grid(divUp(m_node_graph.Size(), blk.x));
	device::forwardWarpSmootherNodeKernel << <grid, blk, 0, stream >> > (
		m_node_graph,
		m_reference_node_coords.RawPtr(),
		m_node_se3.RawPtr(),
		Ti_xj_.Ptr(), 
		Tj_xj_.Ptr(),
		m_pair_validity_indicator.Ptr()
	);

	//std::vector<float3> Ti_xj_Host(Ti_xj_.ArraySize());
	//Ti_xj_.ArrayView().Download(Ti_xj_Host);
	//std::vector<float3> Tj_xj_Host(Tj_xj_.ArraySize());
	//Tj_xj_.ArrayView().Download(Tj_xj_Host);

	//for (int i = 0; i < m_node_graph.Size(); i++) {
	//	printf("idx = %d   Tixj = (%.5f,%.5f,%.5f)   Tjxj = (%.5f,%.5f,%.5f)\n", i, Ti_xj_Host[i].x, Ti_xj_Host[i].y, Ti_xj_Host[i].z, Tj_xj_Host[i].x, Tj_xj_Host[i].y, Tj_xj_Host[i].z);
	//}

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));

#endif
}

void SparseSurfelFusion::NodeGraphSmoothHandler::BuildTerm2Jacobian(cudaStream_t stream) {
	forwardWarpSmootherNodes(stream);
}

SparseSurfelFusion::NodeGraphSmoothTerm2Jacobian SparseSurfelFusion::NodeGraphSmoothHandler::Term2JacobianMap() const
{
	NodeGraphSmoothTerm2Jacobian map;
	map.node_se3 = m_node_se3;
	map.reference_node_coords = m_reference_node_coords;
	map.node_graph = m_node_graph;
	map.Ti_xj = Ti_xj_.ArrayView();
	map.Tj_xj = Tj_xj_.ArrayView();
	map.validity_indicator = m_pair_validity_indicator.ArrayView();
	return map;
}