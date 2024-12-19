/*****************************************************************//**
 * \file   WarpField.cu
 * \brief  构建、更新扭曲场节点，执行向前或者向后扭曲节点
 * 
 * \author LUO
 * \date   March 8th 2024
 *********************************************************************/
#include "WarpField.h"

__device__ __forceinline__ float SparseSurfelFusion::device::distanceSquare(const float4& p1, const float4& p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

__global__ void SparseSurfelFusion::device::BuildNodesGraphKernel(DeviceArrayView<float4> canonicalNodes, ushort2* canonicalNodesGraph)
{
	const int node_num = canonicalNodes.Size();
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= node_num) return;
	float dist_vec[nodesNeighborsNum];
	int idx_vec[nodesNeighborsNum];
	//初始化值：每个node分8个邻居节点
	for (int k = 0; k < nodesNeighborsNum; k++) {
		idx_vec[k] = -1;
		dist_vec[k] = 1e5;
	}

	// 对这些节点执行暴力搜索
	// p_idx是节点
	const float4 p_idx = canonicalNodes[idx];
	int max_index = 0;	// 8个邻居点中的最大距离的index
	//遍历所有节点
	for (int k = 0; k < node_num; k++) {
		const float4 coord = canonicalNodes[k];	// 这样访问是只读访问，不存在访存冲突
		const float new_dist = distanceSquare(p_idx, coord);//节点与遍历所有节点的距离
		//当不是自己并且距离自己最近
		if (new_dist > 1e-6f && new_dist < dist_vec[max_index]) {
			dist_vec[max_index] = new_dist;
			idx_vec[max_index] = k;

			//重新选择距离最大的索引
			max_index = 0;
			float max_dist = 0;
			//遍历dist_vec
			for (int j = 0; j < nodesNeighborsNum; j++) {
				if (dist_vec[j] > max_dist) {
					max_index = j;
					max_dist = dist_vec[j];
				}
			}
		}
	}

	//计算这些邻域点在指针（地址）上的位置
	for (int k = 0; k < nodesNeighborsNum; k++) {
		const int offset = idx * nodesNeighborsNum + k;
		canonicalNodesGraph[offset] = make_ushort2(idx, idx_vec[k]);
	}
}


void SparseSurfelFusion::WarpField::BuildNodesGraph(cudaStream_t stream)
{
	DeviceArrayView<float4> canonicalNodesView = canonicalNodesCoordinate.DeviceArrayReadOnly();
	// 节点图以Constants::nodesGraphNeigboursNum为一个offset，存入对应Points ID = idx % offset点对应的8个邻居点的ID
	nodesGraph.ResizeArrayOrException(canonicalNodesView.Size() * Constants::nodesGraphNeigboursNum);
	DeviceArrayHandle<ushort2> nodesGraphHandle = nodesGraph.ArrayHandle();	// 暴露指针，在核函数中存入数据
	dim3 block(128);
	dim3 grid(divUp(canonicalNodesView.Size(), block.x));
	device::BuildNodesGraphKernel << <grid, block, 0, stream >> > (canonicalNodesView, nodesGraphHandle);	// 此处不用KNN算法，只简单遍历，避免消耗GPU显存
}

void SparseSurfelFusion::WarpField::AdjustNodeSourceFrom(cudaStream_t stream)
{

}