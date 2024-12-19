/*****************************************************************//**
 * \file   SurfelsProcessor.h
 * \brief  用于对于面元的一些基本处理，融合、下采样、提取分量等
 *
 * \author LUO
 * \date   March 7th 2024
 *********************************************************************/
#include "SurfelsProcessor.h"

void SparseSurfelFusion::SurfelsProcessor::MergeDenseSurfels(const unsigned int devCount, DeviceArrayView<DepthSurfel>* surfelsArray, const mat34* cameraPose, DeviceBufferArray<DepthSurfel>& mergedSurfels, cudaStream_t stream)
{
	size_t totalDenseSurfelNum = 0;								// 总共开辟preRigidAlignedSurfel多大的Array空间
	for (int i = 0; i < devCount; i++) {						// 计算总共需要开辟多大的空间
		totalDenseSurfelNum += surfelsArray[i].Size();
	}
	mergedSurfels.ResizeArrayOrException(totalDenseSurfelNum);	// 开辟空间

	for (int i = 0; i < devCount; i++) {						// 计算所有设备
		unsigned int offset = 0;								// 存入preRigidAlignedSurfel的偏差
		for (int j = 0; j < i; j++) {
			offset += surfelsArray[i - 1].Size();
		}
		MergeDenseSurfelToCanonicalField(mergedSurfels, surfelsArray[i], cameraPose[i], i, offset, stream);
	}
}

void SparseSurfelFusion::SurfelsProcessor::PerformVerticesSubsamplingSync(VoxelSubsampler::Ptr subsampler, const DeviceArrayView<float4>& canonicalVertices, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& candidateNodes, cudaStream_t stream)
{
	if (subsampler->isBufferEmpty() == false) LOGGING(FATAL) << "没有给下采样器分配算法运行的内存";
	subsampler->PerformSubsample(canonicalVertices, colorViewTime, candidateNodes, Constants::VoxelSize, stream);
}
