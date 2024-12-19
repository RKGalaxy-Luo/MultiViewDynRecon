/*****************************************************************//**
 * \file   WarpField.cu
 * \brief  ����������Ť�����ڵ㣬ִ����ǰ�������Ť���ڵ�
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
	//��ʼ��ֵ��ÿ��node��8���ھӽڵ�
	for (int k = 0; k < nodesNeighborsNum; k++) {
		idx_vec[k] = -1;
		dist_vec[k] = 1e5;
	}

	// ����Щ�ڵ�ִ�б�������
	// p_idx�ǽڵ�
	const float4 p_idx = canonicalNodes[idx];
	int max_index = 0;	// 8���ھӵ��е��������index
	//�������нڵ�
	for (int k = 0; k < node_num; k++) {
		const float4 coord = canonicalNodes[k];	// ����������ֻ�����ʣ������ڷô��ͻ
		const float new_dist = distanceSquare(p_idx, coord);//�ڵ���������нڵ�ľ���
		//�������Լ����Ҿ����Լ����
		if (new_dist > 1e-6f && new_dist < dist_vec[max_index]) {
			dist_vec[max_index] = new_dist;
			idx_vec[max_index] = k;

			//����ѡ�������������
			max_index = 0;
			float max_dist = 0;
			//����dist_vec
			for (int j = 0; j < nodesNeighborsNum; j++) {
				if (dist_vec[j] > max_dist) {
					max_index = j;
					max_dist = dist_vec[j];
				}
			}
		}
	}

	//������Щ�������ָ�루��ַ���ϵ�λ��
	for (int k = 0; k < nodesNeighborsNum; k++) {
		const int offset = idx * nodesNeighborsNum + k;
		canonicalNodesGraph[offset] = make_ushort2(idx, idx_vec[k]);
	}
}


void SparseSurfelFusion::WarpField::BuildNodesGraph(cudaStream_t stream)
{
	DeviceArrayView<float4> canonicalNodesView = canonicalNodesCoordinate.DeviceArrayReadOnly();
	// �ڵ�ͼ��Constants::nodesGraphNeigboursNumΪһ��offset�������ӦPoints ID = idx % offset���Ӧ��8���ھӵ��ID
	nodesGraph.ResizeArrayOrException(canonicalNodesView.Size() * Constants::nodesGraphNeigboursNum);
	DeviceArrayHandle<ushort2> nodesGraphHandle = nodesGraph.ArrayHandle();	// ��¶ָ�룬�ں˺����д�������
	dim3 block(128);
	dim3 grid(divUp(canonicalNodesView.Size(), block.x));
	device::BuildNodesGraphKernel << <grid, block, 0, stream >> > (canonicalNodesView, nodesGraphHandle);	// �˴�����KNN�㷨��ֻ�򵥱�������������GPU�Դ�
}

void SparseSurfelFusion::WarpField::AdjustNodeSourceFrom(cudaStream_t stream)
{

}