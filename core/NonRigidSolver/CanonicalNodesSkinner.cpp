/*****************************************************************//**
 * \file   CanonicalNodesSkinner.h
 * \brief  计算稀疏节点对稠密点的影响权重，对稀疏点进行蒙皮
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
	// 构造求解器【局部变量，用完释放】
	skinningVertexAndNodeBruteForce(denseSurfels.canonicalVerticesConfidence, denseSurfels.denseSurfelsKnn, denseSurfels.denseSurfelsKnnWeight, 
									sparseNodes.canonicalNodesCoordinate, sparseNodes.sparseNodesKnn, sparseNodes.sparseNodesKnnWeight);
	//// 获取程序结束时间点
	//CHECKCUDA(cudaDeviceSynchronize());	// CUDA同步
	//auto end = std::chrono::high_resolution_clock::now();
	//// 计算程序运行时间（毫秒）
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	//std::cout << "Skinner运行时间: " << duration << " 毫秒" << std::endl;
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

	// 刷新Canonical域
	skinningVertexAndNodeBruteForce(denseSurfels[0].canonicalVerticesConfidence, skinningKnnInterface, sparseNodes.canonicalNodesCoordinate, sparseNodes.sparseNodesKnn, sparseNodes.sparseNodesKnnWeight);

}
