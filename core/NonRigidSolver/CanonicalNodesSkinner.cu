/*****************************************************************//**
 * \file   CanonicalNodesSkinner.h
 * \brief  计算稀疏节点对稠密点的影响权重，对稀疏点进行蒙皮
 *
 * \author LUO
 * \date   March 8th 2024
 *********************************************************************/
#include "CanonicalNodesSkinner.h"
#include <core/Geometry/brute_foce_knn.cuh>

namespace SparseSurfelFusion {
	namespace device {
		__device__  float4 CanonicalNodes[MAX_NODE_COUNT];	// 记录Canonical域中的节点【全局内存】
		__device__ unsigned int devicesCount = MAX_CAMERA_COUNT;

		__global__ void updateVertexNodeKnnWeightKernel(
			const unsigned int denseVerticesNum,
			const unsigned int sparseNodesNum,
			const DeviceArrayView<float4> vertex_confid_array,
			SkinningKnnInterface interface,
			ushort4* node_knn_array, float4* node_knn_weight,
			// The offset and number of added nodes
			const int node_offset, const int padded_node_num
		) {
			// Outof bound: for both vertex and node knn are updated by this kernel
			const int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= denseVerticesNum + sparseNodesNum) return;

			// Collect information form global memory
			float3 v;
			ushort4 knn;
			if (idx < denseVerticesNum) {
				const float4 vertex_confid = vertex_confid_array[idx];
				v = make_float3(vertex_confid.x, vertex_confid.y, vertex_confid.z);
				knn = interface.denseVerticesKnn[0][idx];
			}
			else if (idx >= denseVerticesNum && idx < denseVerticesNum + sparseNodesNum) {
				const auto offset = idx - denseVerticesNum;
				const float4 node = CanonicalNodes[offset];
				v = make_float3(node.x, node.y, node.z);
				knn = node_knn_array[offset];
			}
			else {
				return;
			}

			// load knn for each thread
			const ushort4 knn_prev = knn;
			float4 n0 = CanonicalNodes[knn.x];
			float tmp0 = v.x - n0.x;
			float tmp1 = v.y - n0.y;
			float tmp2 = v.z - n0.z;

			float4 n1 = CanonicalNodes[knn.y];
			float tmp6 = v.x - n1.x;
			float tmp7 = v.y - n1.y;
			float tmp8 = v.z - n1.z;

			float4 n2 = CanonicalNodes[knn.z];
			float tmp12 = v.x - n2.x;
			float tmp13 = v.y - n2.y;
			float tmp14 = v.z - n2.z;

			float4 n3 = CanonicalNodes[knn.w];
			float tmp18 = v.x - n3.x;
			float tmp19 = v.y - n3.y;
			float tmp20 = v.z - n3.z;

			float tmp3 = __fmul_rn(tmp0, tmp0);
			float tmp9 = __fmul_rn(tmp6, tmp6);
			float tmp15 = __fmul_rn(tmp12, tmp12);
			float tmp21 = __fmul_rn(tmp18, tmp18);

			float tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
			float tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
			float tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
			float tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

			float tmp5 = __fmaf_rn(tmp2, tmp2, tmp4);
			float tmp11 = __fmaf_rn(tmp8, tmp8, tmp10);
			float tmp17 = __fmaf_rn(tmp14, tmp14, tmp16);
			float tmp23 = __fmaf_rn(tmp20, tmp20, tmp22);

			//keep priority queue using heap
			float4 distance = make_float4(tmp5, tmp11, tmp17, tmp23);
			KnnHeapDevice heap(distance, knn);

			//The update loop
			for (auto k = node_offset; k < padded_node_num + node_offset; k += 4) {
				n0 = CanonicalNodes[k + 0];
				tmp0 = v.x - n0.x;
				tmp1 = v.y - n0.y;
				tmp2 = v.z - n0.z;

				n1 = CanonicalNodes[k + 1];
				tmp6 = v.x - n1.x;
				tmp7 = v.y - n1.y;
				tmp8 = v.z - n1.z;

				n2 = CanonicalNodes[k + 2];
				tmp12 = v.x - n2.x;
				tmp13 = v.y - n2.y;
				tmp14 = v.z - n2.z;

				n3 = CanonicalNodes[k + 3];
				tmp18 = v.x - n3.x;
				tmp19 = v.y - n3.y;
				tmp20 = v.z - n3.z;

				tmp3 = __fmul_rn(tmp0, tmp0);
				tmp9 = __fmul_rn(tmp6, tmp6);
				tmp15 = __fmul_rn(tmp12, tmp12);
				tmp21 = __fmul_rn(tmp18, tmp18);

				tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
				tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
				tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
				tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

				tmp5 = __fmaf_rn(tmp2, tmp2, tmp4);
				tmp11 = __fmaf_rn(tmp8, tmp8, tmp10);
				tmp17 = __fmaf_rn(tmp14, tmp14, tmp16);
				tmp23 = __fmaf_rn(tmp20, tmp20, tmp22);

				//Update it
				heap.update(k + 0, tmp5);
				heap.update(k + 1, tmp11);
				heap.update(k + 2, tmp17);
				heap.update(k + 3, tmp23);
			}//End of the update loop

			 // If the knn doesn't change
			if (knn.x == knn_prev.x && knn.y == knn_prev.y && knn.z == knn_prev.z && knn.w == knn_prev.w) return;

			// If changed, update the weight
			const float4 node0_v4 = CanonicalNodes[knn.x];
			const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
			const float vn_dist0 = squared_norm(v - node0_v);

			const float4 node1_v4 = CanonicalNodes[knn.y];
			const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
			const float vn_dist1 = squared_norm(v - node1_v);

			const float4 node2_v4 = CanonicalNodes[knn.z];
			const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
			const float vn_dist2 = squared_norm(v - node2_v);

			const float4 node3_v4 = CanonicalNodes[knn.w];
			const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
			const float vn_dist3 = squared_norm(v - node3_v);

			// Compute the weight of this node
			float4 weight;
			weight.x = __expf(-vn_dist0 / (2 * NODE_RADIUS_SQUARE));
			weight.y = __expf(-vn_dist1 / (2 * NODE_RADIUS_SQUARE));
			weight.z = __expf(-vn_dist2 / (2 * NODE_RADIUS_SQUARE));
			weight.w = __expf(-vn_dist3 / (2 * NODE_RADIUS_SQUARE));

			//Do a normalization?
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
			const float weight_sum = weight.x + weight.y + weight.z + weight.w;
			const float inv_weight_sum = 1.0f / weight_sum;
			weight.x *= inv_weight_sum;
			weight.y *= inv_weight_sum;
			weight.z *= inv_weight_sum;
			weight.w *= inv_weight_sum;
#endif

			// Store the result to global memory
			if (idx < denseVerticesNum) {
				for (int i = 0; i < device::devicesCount; i++) {
					interface.denseVerticesKnn[i][idx] = knn;
					interface.denseVerticesKnnWeight[i][idx] = weight;
				}
			}
			else if (idx >= denseVerticesNum && idx < denseVerticesNum + sparseNodesNum) {
				const int offset = idx - denseVerticesNum;
				node_knn_array[offset] = knn;
				node_knn_weight[offset] = weight;
			}

		} // End of kernel


		__global__ void updateVertexNodeKnnWeightKernelindexmap(
			const DeviceArrayView<float4> vertex_confid_array,
			ushort4* vertex_knn_array, float4* vertex_knn_weight,
			ushort4* vertex_knn_array_indexmap, float4* vertex_knn_weight_indexmap,
			DeviceArrayHandle<ushort4> node_knn_array, float4* node_knn_weight,
			// The offset and number of added nodes
			const int node_offset, const int padded_node_num
		) {
			// Outof bound: for both vertex and node knn are updated by this kernel
			const int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= vertex_confid_array.Size() + node_knn_array.Size()) return;

			// Collect information form global memory
			float3 v;
			ushort4 knn;
			if (idx < vertex_confid_array.Size()) {
				const float4 vertex_confid = vertex_confid_array[idx];
				v = make_float3(vertex_confid.x, vertex_confid.y, vertex_confid.z);
				knn = vertex_knn_array[idx];
			}
			else if (idx >= vertex_confid_array.Size() && idx < vertex_confid_array.Size() + node_knn_array.Size()) {
				const auto offset = idx - vertex_confid_array.Size();
				const float4 node = CanonicalNodes[offset];
				v = make_float3(node.x, node.y, node.z);
				knn = node_knn_array[offset];
			}
			else {
				return;
			}

			// load knn for each thread
			const ushort4 knn_prev = knn;
			float4 n0 = CanonicalNodes[knn.x];
			float tmp0 = v.x - n0.x;
			float tmp1 = v.y - n0.y;
			float tmp2 = v.z - n0.z;

			float4 n1 = CanonicalNodes[knn.y];
			float tmp6 = v.x - n1.x;
			float tmp7 = v.y - n1.y;
			float tmp8 = v.z - n1.z;

			float4 n2 = CanonicalNodes[knn.z];
			float tmp12 = v.x - n2.x;
			float tmp13 = v.y - n2.y;
			float tmp14 = v.z - n2.z;

			float4 n3 = CanonicalNodes[knn.w];
			float tmp18 = v.x - n3.x;
			float tmp19 = v.y - n3.y;
			float tmp20 = v.z - n3.z;

			float tmp3 = __fmul_rn(tmp0, tmp0);
			float tmp9 = __fmul_rn(tmp6, tmp6);
			float tmp15 = __fmul_rn(tmp12, tmp12);
			float tmp21 = __fmul_rn(tmp18, tmp18);

			float tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
			float tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
			float tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
			float tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

			float tmp5 = __fmaf_rn(tmp2, tmp2, tmp4);
			float tmp11 = __fmaf_rn(tmp8, tmp8, tmp10);
			float tmp17 = __fmaf_rn(tmp14, tmp14, tmp16);
			float tmp23 = __fmaf_rn(tmp20, tmp20, tmp22);

			//keep priority queue using heap
			float4 distance = make_float4(tmp5, tmp11, tmp17, tmp23);
			/*SparseSurfelFusion::device::*/KnnHeapDevice heap(distance, knn);

			//The update loop
			for (auto k = node_offset; k < padded_node_num + node_offset; k += 4) {
				n0 = CanonicalNodes[k + 0];
				tmp0 = v.x - n0.x;
				tmp1 = v.y - n0.y;
				tmp2 = v.z - n0.z;

				n1 = CanonicalNodes[k + 1];
				tmp6 = v.x - n1.x;
				tmp7 = v.y - n1.y;
				tmp8 = v.z - n1.z;

				n2 = CanonicalNodes[k + 2];
				tmp12 = v.x - n2.x;
				tmp13 = v.y - n2.y;
				tmp14 = v.z - n2.z;

				n3 = CanonicalNodes[k + 3];
				tmp18 = v.x - n3.x;
				tmp19 = v.y - n3.y;
				tmp20 = v.z - n3.z;

				tmp3 = __fmul_rn(tmp0, tmp0);
				tmp9 = __fmul_rn(tmp6, tmp6);
				tmp15 = __fmul_rn(tmp12, tmp12);
				tmp21 = __fmul_rn(tmp18, tmp18);

				tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
				tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
				tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
				tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

				tmp5 = __fmaf_rn(tmp2, tmp2, tmp4);
				tmp11 = __fmaf_rn(tmp8, tmp8, tmp10);
				tmp17 = __fmaf_rn(tmp14, tmp14, tmp16);
				tmp23 = __fmaf_rn(tmp20, tmp20, tmp22);

				//Update it
				heap.update(k + 0, tmp5);
				heap.update(k + 1, tmp11);
				heap.update(k + 2, tmp17);
				heap.update(k + 3, tmp23);
			}//End of the update loop

			 // If the knn doesn't change
			if (knn.x == knn_prev.x && knn.y == knn_prev.y && knn.z == knn_prev.z && knn.w == knn_prev.w) return;

			// If changed, update the weight
			const float4 node0_v4 = CanonicalNodes[knn.x];
			const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
			const float vn_dist0 = squared_norm(v - node0_v);

			const float4 node1_v4 = CanonicalNodes[knn.y];
			const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
			const float vn_dist1 = squared_norm(v - node1_v);

			const float4 node2_v4 = CanonicalNodes[knn.z];
			const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
			const float vn_dist2 = squared_norm(v - node2_v);

			const float4 node3_v4 = CanonicalNodes[knn.w];
			const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
			const float vn_dist3 = squared_norm(v - node3_v);

			// Compute the weight of this node
			float4 weight;
			weight.x = __expf(-vn_dist0 / (2 * NODE_RADIUS_SQUARE));
			weight.y = __expf(-vn_dist1 / (2 * NODE_RADIUS_SQUARE));
			weight.z = __expf(-vn_dist2 / (2 * NODE_RADIUS_SQUARE));
			weight.w = __expf(-vn_dist3 / (2 * NODE_RADIUS_SQUARE));

			//Do a normalization?
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
			const float weight_sum = weight.x + weight.y + weight.z + weight.w;
			const float inv_weight_sum = 1.0f / weight_sum;
			weight.x *= inv_weight_sum;
			weight.y *= inv_weight_sum;
			weight.z *= inv_weight_sum;
			weight.w *= inv_weight_sum;
#endif

			// Store the result to global memory
			if (idx < vertex_confid_array.Size()) {
				vertex_knn_array[idx] = knn;
				vertex_knn_weight[idx] = weight;
				vertex_knn_array_indexmap[idx] = knn;
				vertex_knn_weight_indexmap[idx] = weight;
			}
			else if (idx >= vertex_confid_array.Size() && idx < vertex_confid_array.Size() + node_knn_array.Size()) {
				const int offset = idx - vertex_confid_array.Size();
				node_knn_array[offset] = knn;
				node_knn_weight[offset] = weight;
			}

		} // End of kernel

	}
}

__global__ void SparseSurfelFusion::device::skinningVertexAndNodeBruteForceKernel(
	const DeviceArrayView<float4> denseVertices, 
	const unsigned int denseVerticesNum, 
	const unsigned int nodesNum, 
	ushort4* denseVerticesKnn, 
	float4* denseVerticesKnnWeight, 
	ushort4* nodesKnn, 
	float4* nodesKnnWeight
)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= denseVerticesNum + nodesNum)	return;
	float4 point;
	if (idx < denseVerticesNum) {	// 查找的是稠密点对应的node邻居
		const float4 vertex = denseVertices[idx];
		point = make_float4(vertex.x, vertex.y, vertex.z, 1.0f);
	}
	else {	// 查找的是Nodes之间的邻居
		const int offset = idx - denseVerticesNum;
		const float4 node = CanonicalNodes[offset];
		point = make_float4(node.x, node.y, node.z, 1.0);
		//6.20这里没问题，也就是说CanonicalNodes内存没问题
		//if (offset%(unsigned)200==0) {
		//	printf("node=(%f,%f,%f)\n", node.x, node.y, node.z);
		//}
	}

	// 使用堆保持优先级队列
	float4 knnDistance = make_float4(1e6f, 1e6f, 1e6f, 1e6f);
	ushort4 knnIndex = make_ushort4(0, 0, 0, 0);

	//bruteForceSearch4NearestPoints(point, CanonicalNodes, nodesNum, knnDistance, knnIndex);
	bruteForceSearch4Padded(point, CanonicalNodes, nodesNum, knnDistance, knnIndex);
	// 顶点的w分量是置信度
	const float3 v = make_float3(point.x, point.y, point.z);

	//Compute the knn weight given knn
	const float4 node0_v4 = CanonicalNodes[knnIndex.x];
	const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
	const float vn_dist0 = squared_norm(v - node0_v);

	const float4 node1_v4 = CanonicalNodes[knnIndex.y];
	const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
	const float vn_dist1 = squared_norm(v - node1_v);

	const float4 node2_v4 = CanonicalNodes[knnIndex.z];
	const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
	const float vn_dist2 = squared_norm(v - node2_v);

	const float4 node3_v4 = CanonicalNodes[knnIndex.w];
	const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
	const float vn_dist3 = squared_norm(v - node3_v);

	// 计算该节点的权值
	float4 knnWeight;
	knnWeight.x = __expf(-vn_dist0 / (2 * NODE_RADIUS_SQUARE));
	knnWeight.y = __expf(-vn_dist1 / (2 * NODE_RADIUS_SQUARE));
	knnWeight.z = __expf(-vn_dist2 / (2 * NODE_RADIUS_SQUARE));
	knnWeight.w = __expf(-vn_dist3 / (2 * NODE_RADIUS_SQUARE));

	// 权重归一化
	const float weightSumInverse = 1.0f / fabsf_sum(knnWeight);
	knnWeight.x *= weightSumInverse;
	knnWeight.y *= weightSumInverse;
	knnWeight.z *= weightSumInverse;
	knnWeight.w *= weightSumInverse;

	// 赋值knn及weight
	if (idx < denseVerticesNum) {
		denseVerticesKnn[idx] = knnIndex;
		denseVerticesKnnWeight[idx] = knnWeight;
	}
	else if (idx >= denseVerticesNum && idx < denseVerticesNum + nodesNum) {
		nodesKnn[idx - denseVerticesNum] = knnIndex;
		nodesKnnWeight[idx - denseVerticesNum] = knnWeight;
		//6.20 没问题
		//if ((idx - denseVerticesNum )%(unsigned)200==0) {
		//	printf("node=(%u,%u,%u)\n", knnIndex.x, knnIndex.y, knnIndex.z);
		//}
	}
}

__global__ void SparseSurfelFusion::device::skinningVertexAndNodeBruteForceKernel(const DeviceArrayView<float4> denseVertices, const unsigned int denseVerticesNum, const unsigned int nodesNum, SkinningKnnInterface denseKnnInterface, ushort4* nodesKnn, float4* nodesKnnWeight)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= denseVerticesNum + nodesNum)	return;
	float4 point;
	if (idx < denseVerticesNum) {	// 查找的是稠密点对应的node邻居
		const float4 vertex = denseVertices[idx];
		point = make_float4(vertex.x, vertex.y, vertex.z, 1.0f);
	}
	else {	// 查找的是Nodes之间的邻居
		const int offset = idx - denseVerticesNum;
		const float4 node = CanonicalNodes[offset];
		point = make_float4(node.x, node.y, node.z, 1.0);
	}

	// 使用堆保持优先级队列
	float4 knnDistance = make_float4(1e6f, 1e6f, 1e6f, 1e6f);
	ushort4 knnIndex = make_ushort4(0, 0, 0, 0);

	//bruteForceSearch4NearestPoints(point, CanonicalNodes, nodesNum, knnDistance, knnIndex);
	bruteForceSearch4Padded(point, CanonicalNodes, nodesNum, knnDistance, knnIndex);
	// 顶点的w分量是置信度
	const float3 v = make_float3(point.x, point.y, point.z);

	//Compute the knn weight given knn
	const float4 node0_v4 = CanonicalNodes[knnIndex.x];
	const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
	const float vn_dist0 = squared_norm(v - node0_v);

	const float4 node1_v4 = CanonicalNodes[knnIndex.y];
	const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
	const float vn_dist1 = squared_norm(v - node1_v);

	const float4 node2_v4 = CanonicalNodes[knnIndex.z];
	const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
	const float vn_dist2 = squared_norm(v - node2_v);

	const float4 node3_v4 = CanonicalNodes[knnIndex.w];
	const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
	const float vn_dist3 = squared_norm(v - node3_v);

	// 计算该节点的权值
	float4 knnWeight;
	knnWeight.x = __expf(-vn_dist0 / (2 * NODE_RADIUS_SQUARE));
	knnWeight.y = __expf(-vn_dist1 / (2 * NODE_RADIUS_SQUARE));
	knnWeight.z = __expf(-vn_dist2 / (2 * NODE_RADIUS_SQUARE));
	knnWeight.w = __expf(-vn_dist3 / (2 * NODE_RADIUS_SQUARE));

	// 权重归一化
	const float weightSumInverse = 1.0f / fabsf_sum(knnWeight);
	knnWeight.x *= weightSumInverse;
	knnWeight.y *= weightSumInverse;
	knnWeight.z *= weightSumInverse;
	knnWeight.w *= weightSumInverse;

	// 赋值knn及weight
	if (idx < denseVerticesNum) {
		for (int i = 0; i < device::devicesCount; i++) {
			denseKnnInterface.denseVerticesKnn[i][idx] = knnIndex;
			denseKnnInterface.denseVerticesKnnWeight[i][idx] = knnWeight;
		}
	}
	else if (idx >= denseVerticesNum && idx < denseVerticesNum + nodesNum) {
		nodesKnn[idx - denseVerticesNum] = knnIndex;
		nodesKnnWeight[idx - denseVerticesNum] = knnWeight;
	}

}


void SparseSurfelFusion::CanonicalNodesSkinner::BuildInitialSkinningIndex(DeviceArrayView<float4>& canonicalNodes, cudaStream_t stream)
{
	if(canonicalNodes.Size() > Constants::maxNodesNum) LOGGING(FATAL) << "节点数超出最大节点限制";
	fillInvalidGlobalPoints(stream);	// 清空之前的Canonical域中节点信息
	// 给CanonicalNodes的Buffer赋值
	//这里是device给device传递数据，把节点数据传到常量内存中
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::CanonicalNodes, canonicalNodes.RawPtr(), canonicalNodes.Size() * sizeof(float4), 0, cudaMemcpyDeviceToDevice, stream));
	// 更新节点大小
	CanonicalNodesNum = canonicalNodes.Size();
}

void SparseSurfelFusion::CanonicalNodesSkinner::fillInvalidGlobalPoints(cudaStream_t stream)
{
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::CanonicalNodes, invalidNodes.ptr(), sizeof(float4) * Constants::maxNodesNum, 0, cudaMemcpyDeviceToDevice, stream));
}

void SparseSurfelFusion::CanonicalNodesSkinner::skinningVertexAndNodeBruteForce(
	const DeviceArrayView<float4>& denseVertices, 
	DeviceArrayHandle<ushort4> verticesKnn, 
	DeviceArrayHandle<float4> verticesKnnWeight, 
	const DeviceArrayView<float4>& canonicalNodes, 
	DeviceArrayHandle<ushort4> nodesKnn, 
	DeviceArrayHandle<float4> nodesKnnWeight, 
	cudaStream_t stream
)
{
	if (canonicalNodes.Size() != CanonicalNodesNum)	LOGGING(FATAL) << "传入WarpField中映射的Canonical域中的节点与蒙皮有效节点不相等";
	const unsigned int denseVerticesNum = denseVertices.Size();
	const unsigned int DenseVerticesAndValidNodesNum = denseVerticesNum + CanonicalNodesNum;
	dim3 block(256);
	dim3 grid(divUp(DenseVerticesAndValidNodesNum, block.x));
	device::skinningVertexAndNodeBruteForceKernel << <grid, block, 0, stream >> > (
		denseVertices,
		denseVerticesNum,
		CanonicalNodesNum,
		verticesKnn,
		verticesKnnWeight,
		nodesKnn,
		nodesKnnWeight
	);
}

void SparseSurfelFusion::CanonicalNodesSkinner::skinningVertexAndNodeBruteForce(
	const DeviceArrayView<float4>& denseVertices, 
	device::SkinningKnnInterface& skinningKnnInterface,
	const DeviceArrayView<float4>& canonicalNodes, 
	DeviceArrayHandle<ushort4> nodesKnn, 
	DeviceArrayHandle<float4> nodesKnnWeight, 
	cudaStream_t stream
)
{
	if (canonicalNodes.Size() != CanonicalNodesNum)	LOGGING(FATAL) << "传入WarpField中映射的Canonical域中的节点与蒙皮有效节点不相等";
	const unsigned int denseVerticesNum = denseVertices.Size();
	const unsigned int DenseVerticesAndValidNodesNum = denseVerticesNum + CanonicalNodesNum;
	dim3 block(256);
	dim3 grid(divUp(DenseVerticesAndValidNodesNum, block.x));
	device::skinningVertexAndNodeBruteForceKernel << <grid, block, 0, stream >> > (denseVertices, denseVerticesNum, CanonicalNodesNum, skinningKnnInterface, nodesKnn, nodesKnnWeight);
}

void SparseSurfelFusion::CanonicalNodesSkinner::updateSkinning(
	unsigned int newNodesOffset, 
	const DeviceArrayView<float4>& denseCanonicalVertices, 
	device::SkinningKnnInterface& skinningKnnInterface,
	const DeviceArrayView<float4>& canonicalNodes,
	DeviceArrayHandle<ushort4> nodesKnn, 
	DeviceArrayHandle<float4> nodesKnnWeight, 
	cudaStream_t stream
) const
{
	//Check the size
	const unsigned prev_nodesize = newNodesOffset;
	FUNCTION_CHECK(canonicalNodes.Size() >= prev_nodesize); //There should be more nodes now
	if (canonicalNodes.Size() == prev_nodesize) return;

	//The index should be updated
	FUNCTION_CHECK(m_num_bruteforce_nodes == canonicalNodes.Size()) << "The index is not updated";

	//The numer of append node
	const unsigned int num_appended_node = m_num_bruteforce_nodes - newNodesOffset;
	const unsigned int padded_newnode_num = divUp(num_appended_node, 4) * 4;

	//Let's to it 
	dim3 blk(256);
	dim3 grid(divUp(denseCanonicalVertices.Size() + canonicalNodes.Size(), blk.x));
	device::updateVertexNodeKnnWeightKernel << <grid, blk, 0, stream >> > (
		denseCanonicalVertices.Size(),
		canonicalNodes.Size(),
		denseCanonicalVertices,
		skinningKnnInterface,
		nodesKnn.RawPtr(),
		nodesKnnWeight.RawPtr(),
		newNodesOffset,
		padded_newnode_num
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::CanonicalNodesSkinner::updateSkinning(
	unsigned newnode_offset, 
	const DeviceArrayView<float4>& reference_vertex, 
	const DeviceArrayView<float4>& reference_node,
	DeviceArrayHandle<ushort4> vertex_knn, DeviceArrayHandle<ushort4> node_knn, 
	DeviceArrayHandle<float4> vertex_knn_weight, DeviceArrayHandle<float4> node_knn_weight, 
	DeviceArrayHandle<ushort4> vertex_knn_indexmap, 
	DeviceArrayHandle<float4> vertex_knn_weight_indexmap, 
	cudaStream_t stream
) const
{
	//Check the size
	const unsigned prev_nodesize = newnode_offset;
	FUNCTION_CHECK(reference_node.Size() >= prev_nodesize); //There should be more nodes now
	if (reference_node.Size() == prev_nodesize) return;

	//The index should be updated
	FUNCTION_CHECK(m_num_bruteforce_nodes == reference_node.Size()) << "The index is not updated";

	//The numer of append node
	const auto num_appended_node = m_num_bruteforce_nodes - newnode_offset;
	const auto padded_newnode_num = divUp(num_appended_node, 4) * 4;

	//Let's to it 
	dim3 blk(256);
	dim3 grid(divUp(reference_vertex.Size() + m_num_bruteforce_nodes, blk.x));
	device::updateVertexNodeKnnWeightKernelindexmap << <grid, blk, 0, stream >> > (
		reference_vertex,
		vertex_knn.RawPtr(), vertex_knn_weight.RawPtr(),
		vertex_knn_indexmap.RawPtr(), vertex_knn_weight_indexmap.RawPtr(),
		node_knn, node_knn_weight.RawPtr(),
		newnode_offset, padded_newnode_num
		);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::CanonicalNodesSkinner::UpdateBruteForceSkinningIndexWithNewNodes(
	const DeviceArrayView<float4>& nodes, 
	unsigned newnode_offset, 
	cudaStream_t stream)
{
	//Check the size
	const unsigned prev_nodesize = newnode_offset;
	FUNCTION_CHECK(nodes.Size() >= prev_nodesize); //There should be more nodes now
	FUNCTION_CHECK(nodes.Size() <= Constants::maxNodesNum);

	//There is no node to append
	if (nodes.Size() == prev_nodesize) return;

	//Everything seems to be correct, do it
	const auto new_node_size = nodes.Size() - newnode_offset;
	const float4* node_ptr = nodes.RawPtr() + newnode_offset;
	CHECKCUDA(cudaMemcpyToSymbolAsync(
		device::CanonicalNodes,
		node_ptr,
		new_node_size * sizeof(float4),
		newnode_offset * sizeof(float4),
		cudaMemcpyDeviceToDevice,
		stream
	));

	//Update the size
	m_num_bruteforce_nodes = nodes.Size();

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::CanonicalNodesSkinner::PerformSkinningUpdate(
	SurfelGeometry::SkinnerInput* geometry, 
	WarpField::SkinnerInput warpField, 
	unsigned newNodesOffset, 
	cudaStream_t stream)
{	
	//Check the size
	for (int i = 0; i < devicesCount; i++) {
		FUNCTION_CHECK(geometry[i].denseSurfelsKnn.Size() == geometry[i].denseSurfelsKnnWeight.Size());
		FUNCTION_CHECK(geometry[i].denseSurfelsKnn.Size() == geometry[i].canonicalVerticesConfidence.Size());
	}

	FUNCTION_CHECK(warpField.canonicalNodesCoordinate.Size() == warpField.sparseNodesKnn.Size());
	FUNCTION_CHECK(warpField.canonicalNodesCoordinate.Size() == warpField.sparseNodesKnnWeight.Size());

	device::SkinningKnnInterface skinningKnnInterface;
	for (int i = 0; i < devicesCount; i++) {
		skinningKnnInterface.denseVerticesKnn[i] = geometry[i].denseSurfelsKnn;
		skinningKnnInterface.denseVerticesKnnWeight[i] = geometry[i].denseSurfelsKnnWeight;
	}

	//Hand in to workforce
	updateSkinning(newNodesOffset, geometry[0].canonicalVerticesConfidence, skinningKnnInterface, warpField.canonicalNodesCoordinate, warpField.sparseNodesKnn, warpField.sparseNodesKnnWeight, stream);

	//Check it
	/*KNNSearch::CheckApproximateKNNSearch(
		warp_field.reference_node_coords,
		geometry.reference_vertex_confid,
		geometry.surfel_knn.ArrayView()
	);
	KNNSearch::CheckKNNSearch(
		warp_field.reference_node_coords,
		warp_field.reference_node_coords,
		warp_field.node_knn.ArrayView()
	);*/
}

void SparseSurfelFusion::CanonicalNodesSkinner::PerformSkinningUpdate(SurfelGeometry::SkinnerInput geometry, SurfelGeometry::SkinnerInput geometryindexmap, WarpField::SkinnerInput warp_field, unsigned newnode_offset, cudaStream_t stream)
{
	//Check the size
	FUNCTION_CHECK(geometry.denseSurfelsKnn.Size() == geometry.denseSurfelsKnnWeight.Size());
	FUNCTION_CHECK(geometry.denseSurfelsKnn.Size() == geometry.canonicalVerticesConfidence.Size());
	FUNCTION_CHECK(geometryindexmap.denseSurfelsKnn.Size() == geometryindexmap.denseSurfelsKnnWeight.Size());
	FUNCTION_CHECK(geometryindexmap.denseSurfelsKnn.Size() == geometryindexmap.canonicalVerticesConfidence.Size());
	FUNCTION_CHECK(warp_field.canonicalNodesCoordinate.Size() == warp_field.sparseNodesKnn.Size());
	FUNCTION_CHECK(warp_field.canonicalNodesCoordinate.Size() == warp_field.sparseNodesKnnWeight.Size());

	//Hand in to workforce
	updateSkinning(
		newnode_offset,
		geometry.canonicalVerticesConfidence,
		warp_field.canonicalNodesCoordinate,
		geometry.denseSurfelsKnn, warp_field.sparseNodesKnn,
		geometry.denseSurfelsKnnWeight, warp_field.sparseNodesKnnWeight,
		geometryindexmap.denseSurfelsKnn,
		geometryindexmap.denseSurfelsKnnWeight,
		stream
	);
}
