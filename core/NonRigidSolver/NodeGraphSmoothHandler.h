#pragma once

#include <base/CommonUtils.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include "solver_types.h"

#include <memory>

namespace SparseSurfelFusion {

	class NodeGraphSmoothHandler {
	private:
		//The input data from solver
		DeviceArrayView<DualQuaternion> m_node_se3;
		DeviceArrayView<ushort2> m_node_graph;
		DeviceArrayView<float4> m_reference_node_coords;

	public:
		using Ptr = std::shared_ptr<NodeGraphSmoothHandler>;
		NodeGraphSmoothHandler();
		~NodeGraphSmoothHandler();
		NO_COPY_ASSIGN_MOVE(NodeGraphSmoothHandler);

		//The input interface from solver
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3,
			const DeviceArrayView<ushort2>& node_graph,
			const DeviceArrayView<float4>& reference_nodes
		);

		//Do a forward warp on nodes
	private:
		DeviceBufferArray<float3> Ti_xj_;
		DeviceBufferArray<float3> Tj_xj_;
		DeviceBufferArray<unsigned char> m_pair_validity_indicator;
		void forwardWarpSmootherNodes(cudaStream_t stream = 0);
	public:
		void BuildTerm2Jacobian(cudaStream_t stream = 0);
		NodeGraphSmoothTerm2Jacobian Term2JacobianMap() const;
	};


} // surfelwarp