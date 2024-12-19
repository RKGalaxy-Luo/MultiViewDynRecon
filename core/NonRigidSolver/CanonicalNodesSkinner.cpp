/*****************************************************************//**
 * \file   CanonicalNodesSkinner.h
 * \brief  ����ϡ��ڵ�Գ��ܵ��Ӱ��Ȩ�أ���ϡ��������Ƥ
 *
 * \author LUO
 * \date   March 8th 2024
 *********************************************************************/
#include "CanonicalNodesSkinner.h"

SparseSurfelFusion::CanonicalNodesSkinner::CanonicalNodesSkinner(unsigned int devCount) : devicesCount(devCount)
{
	CanonicalNodesNum = 0;
	invalidNodes.create(Constants::maxNodesNum);
	std::vector<float4> hoseValidNodes;
	hoseValidNodes.resize(Constants::maxNodesNum);
	float* begin = (float*)hoseValidNodes.data();
	float* end = begin + 4 * Constants::maxNodesNum;
	std::fill(begin, end, 1e6f);
	invalidNodes.upload(hoseValidNodes);
	fillInvalidGlobalPoints();
}

SparseSurfelFusion::CanonicalNodesSkinner::~CanonicalNodesSkinner()
{
	invalidNodes.release();
}


void SparseSurfelFusion::CanonicalNodesSkinner::PerformSkinning(SurfelGeometry::SkinnerInput denseSurfels, WarpField::SkinnerInput sparseNodes, cudaStream_t stream)
{
	//auto start = std::chrono::high_resolution_clock::now();
	// ������������ֲ������������ͷš�
	skinningVertexAndNodeBruteForce(denseSurfels.canonicalVerticesConfidence, denseSurfels.denseSurfelsKnn, denseSurfels.denseSurfelsKnnWeight, 
									sparseNodes.canonicalNodesCoordinate, sparseNodes.sparseNodesKnn, sparseNodes.sparseNodesKnnWeight);
	//// ��ȡ�������ʱ���
	//CHECKCUDA(cudaDeviceSynchronize());	// CUDAͬ��
	//auto end = std::chrono::high_resolution_clock::now();
	//// �����������ʱ�䣨���룩
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	//std::cout << "Skinner����ʱ��: " << duration << " ����" << std::endl;
}

void SparseSurfelFusion::CanonicalNodesSkinner::PerformSkinning(
	SurfelGeometry::SkinnerInput* denseSurfels, 
	WarpField::SkinnerInput sparseNodes, 
	cudaStream_t stream
)
{
	device::SkinningKnnInterface skinningKnnInterface;
	for (int i = 0; i < devicesCount; i++) {
		skinningKnnInterface.denseVerticesKnn[i] = denseSurfels[i].denseSurfelsKnn;
		skinningKnnInterface.denseVerticesKnnWeight[i] = denseSurfels[i].denseSurfelsKnnWeight;
	}

	// ˢ��Canonical��
	skinningVertexAndNodeBruteForce(denseSurfels[0].canonicalVerticesConfidence, skinningKnnInterface, sparseNodes.canonicalNodesCoordinate, sparseNodes.sparseNodesKnn, sparseNodes.sparseNodesKnnWeight);

}
