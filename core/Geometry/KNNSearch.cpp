#include <base/Logging.h>
#include "core/Geometry/KNNSearch.h"
#include <math/VectorUtils.h>

void SparseSurfelFusion::KNNSearch::CheckKNNSearch(
	const DeviceArrayView<float4>& nodes, 
	const DeviceArrayView<float4>& vertex,
	const DeviceArrayView<ushort4>& knn
) {
	LOGGING(INFO) << "CPU check of skinning";
	FUNCTION_CHECK(vertex.Size() == knn.Size());

	//Download the required data
	std::vector<float4> vertices_cpu, nodes_cpu;
	std::vector<ushort4> nn_cpu;
	vertex.Download(vertices_cpu);
	nodes.Download(nodes_cpu);
	knn.Download(nn_cpu);

	//Do nearest neighbour computation on cpu
	std::vector<std::pair<float, unsigned short>> dist_key_value;
	for (size_t vert_idx = 0; vert_idx < vertices_cpu.size(); vert_idx++) {
		const float4 vertex = make_float4(vertices_cpu[vert_idx].x, vertices_cpu[vert_idx].y, vertices_cpu[vert_idx].z, 1.0f);
		dist_key_value.clear();
		for (size_t node_idx = 0; node_idx < nodes_cpu.size(); node_idx++) {
			float4 node = nodes_cpu[node_idx];
			float distance = norm(node - vertex);
			dist_key_value.push_back(std::make_pair(distance, node_idx));
		}
		std::sort(dist_key_value.begin(), dist_key_value.end());
		const ushort4 gpu_knn = nn_cpu[vert_idx];
		for (int i = 0; i < 4; i++) {
			if(gpu_knn.x == dist_key_value[i].second
			   || gpu_knn.y == dist_key_value[i].second
			   || gpu_knn.z == dist_key_value[i].second
			   || gpu_knn.w == dist_key_value[i].second) {
				
			} else {
				LOGGING(INFO) << "The " << vert_idx << " elements ";
			}
			
			FUNCTION_CHECK(
				gpu_knn.x == dist_key_value[i].second
				|| gpu_knn.y == dist_key_value[i].second
				|| gpu_knn.z == dist_key_value[i].second
				|| gpu_knn.w == dist_key_value[i].second
			) << "For i = " << vert_idx << " KNN is " << gpu_knn.x << " " << gpu_knn.y << " " << gpu_knn.z;
		}
	}

	LOGGING(INFO) << "Seems correct!";
}

void SparseSurfelFusion::KNNSearch::CheckApproximateKNNSearch(
	const DeviceArrayView<float4> &nodes,
	const DeviceArrayView<float4> &vertex,
	const DeviceArrayView<ushort4> &knn
) {
	LOGGING(INFO) << "CPU check of approximate skinning";
	FUNCTION_CHECK(vertex.Size() == knn.Size());
	
	//Download the required data
	std::vector<float4> vertices_cpu, nodes_cpu;
	std::vector<ushort4> nn_cpu;
	vertex.Download(vertices_cpu);
	nodes.Download(nodes_cpu);
	knn.Download(nn_cpu);
	
	//The exact knn margin
	const auto exact_knn_margin = 10;
	
	//Do nearest neighbour computation on cpu
	std::vector<std::pair<float, unsigned short>> dist_key_value;
	for (size_t vert_idx = 0; vert_idx < vertices_cpu.size(); vert_idx++) {
		const float4 vertex = vertices_cpu[vert_idx];
		dist_key_value.clear();
		for (size_t node_idx = 0; node_idx < nodes_cpu.size(); node_idx++) {
			float4 node = nodes_cpu[node_idx];
			float distance = norm(node - vertex);
			dist_key_value.push_back(std::make_pair(distance, node_idx));
		}
		std::sort(dist_key_value.begin(), dist_key_value.end());
		const ushort4 gpu_knn = nn_cpu[vert_idx];
		const unsigned short* gpu_knn_flat = (const unsigned short*)(&gpu_knn);
		bool gpu_knn_found[4] = {false};
		for(auto i = 0; i < exact_knn_margin; i++) {
			const auto cpu_value = dist_key_value[i].second;
			for(auto j = 0; j < 4; j++) {
				if(gpu_knn_flat[j] == cpu_value)
					gpu_knn_found[j] = true;
			}
		}
		
		//Check it
		for(auto i = 0; i < 4; i++) {
			if(!gpu_knn_found[i]) {
				LOGGING(INFO) << "The KNN is not found in first " << exact_knn_margin << " exact KNN";
			}
		}
	}
	
	LOGGING(INFO) << "Check done";
}
