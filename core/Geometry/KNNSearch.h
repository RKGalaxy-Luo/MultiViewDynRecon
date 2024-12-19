//
// Created by wei on 3/24/18.
//

#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>
#include <memory>

namespace SparseSurfelFusion {
	
	class KNNSearch {
	public:
		using Ptr = std::shared_ptr<KNNSearch>;
		KNNSearch() = default;
		virtual ~KNNSearch() = default;
		NO_COPY_ASSIGN_MOVE(KNNSearch);
		
		//Explicit allocate
		virtual void AllocateBuffer(unsigned max_num_points) = 0;
		virtual void ReleaseBuffer() = 0;
		
		//Explicit build search index
		virtual void BuildIndex(const DeviceArrayView<float4>& nodes, cudaStream_t stream = 0) = 0;
		virtual void BuildIndexHostNodes(const std::vector<float4>& nodes, cudaStream_t stream = 0) {
			LOGGING(FATAL) << "The index doesnt use host array, should use device array instread";
		}
		

		//Perform searching
		virtual void Skinning(
			const DeviceArrayView<float4>& vertex,
			DeviceArrayHandle<ushort4> knn,
			DeviceArrayHandle<float4> knn_weight,
			cudaStream_t stream = 0
		) = 0;
		virtual void Skinning(
			const DeviceArrayView<float4>& vertex, const DeviceArrayView<float4>& node,
			DeviceArrayHandle<ushort4> vertex_knn, DeviceArrayHandle<ushort4> node_knn,
			DeviceArrayHandle<float4> vertex_knn_weight, DeviceArrayHandle<float4> node_knn_weight,
			cudaStream_t stream = 0
		) = 0;

		//The checking function for KNN search
		static void CheckKNNSearch(
			const DeviceArrayView<float4>& nodes, 
			const DeviceArrayView<float4>& vertex,
			const DeviceArrayView<ushort4>& knn
		);
		
		//The result may not exactly correct, but
		//the distance should be almost the same
		static void CheckApproximateKNNSearch(
			const DeviceArrayView<float4>& nodes,
			const DeviceArrayView<float4>& vertex,
			const DeviceArrayView<ushort4>& knn
		);
	};
	
}
