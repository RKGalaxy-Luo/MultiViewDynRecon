/*****************************************************************//**
 * \file   SurfelsProcessor.h
 * \brief  ���ڶ�����Ԫ��һЩ���������ںϡ��²�������ȡ������
 *
 * \author LUO
 * \date   March 7th 2024
 *********************************************************************/
#include "SurfelsProcessor.h"

void SparseSurfelFusion::SurfelsProcessor::MergeDenseSurfels(const unsigned int devCount, DeviceArrayView<DepthSurfel>* surfelsArray, const mat34* cameraPose, DeviceBufferArray<DepthSurfel>& mergedSurfels, cudaStream_t stream)
{
	size_t totalDenseSurfelNum = 0;								// �ܹ�����preRigidAlignedSurfel����Array�ռ�
	for (int i = 0; i < devCount; i++) {						// �����ܹ���Ҫ���ٶ��Ŀռ�
		totalDenseSurfelNum += surfelsArray[i].Size();
	}
	mergedSurfels.ResizeArrayOrException(totalDenseSurfelNum);	// ���ٿռ�

	for (int i = 0; i < devCount; i++) {						// ���������豸
		unsigned int offset = 0;								// ����preRigidAlignedSurfel��ƫ��
		for (int j = 0; j < i; j++) {
			offset += surfelsArray[i - 1].Size();
		}
		MergeDenseSurfelToCanonicalField(mergedSurfels, surfelsArray[i], cameraPose[i], i, offset, stream);
	}
}

void SparseSurfelFusion::SurfelsProcessor::PerformVerticesSubsamplingSync(VoxelSubsampler::Ptr subsampler, const DeviceArrayView<float4>& canonicalVertices, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& candidateNodes, cudaStream_t stream)
{
	if (subsampler->isBufferEmpty() == false) LOGGING(FATAL) << "û�и��²����������㷨���е��ڴ�";
	subsampler->PerformSubsample(canonicalVertices, colorViewTime, candidateNodes, Constants::VoxelSize, stream);
}
