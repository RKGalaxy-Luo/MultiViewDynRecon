/*****************************************************************//**
 * \file   SurfelsProcessor.h
 * \brief  ���ڶ�����Ԫ��һЩ���������ںϡ��²�������ȡ������
 *
 * \author LUO
 * \date   March 7th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>

#include <math/MatUtils.h>

#include <core/Geometry/VoxelSubsampler.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief ���������ͷ�������ںϵ�preAlignedSurfel.
		 *
		 * \param preAlignedSurfel �ں϶�����ͷ��Surfel���˴�
		 * \param depthSurfel ��Ҫת����preAlignedSurfel����Ԫ
		 * \param relativePose ��ʼ����ͷλ��
		 * \param pointsNum ��ǰ����ĵ������
		 * \param offset preAlignedSurfel��ƫ��������ʱӦ�ô��ĸ�λ�ÿ�ʼ�洢
		 */
		__global__ void MergeDenseSurfelToCanonicalFieldKernel(DeviceArrayHandle<DepthSurfel> mergedSurfels, DeviceArrayView<DepthSurfel> depthSurfel, mat34 relativePose, const unsigned int pointsNum, const unsigned int offset,const int i);
	}


	/**
	 * \brief ��Ԫ�Ļ�������Ԫ���ɶ�д������ӵ���ڴ�(�����ڴ�ķ�����ͷ�).
	 */
	class SurfelsProcessor
	{
	public:
		using Ptr = std::shared_ptr<SurfelsProcessor>;	// ����ָ�룬���ü�ɾ

		SurfelsProcessor() = default;
		~SurfelsProcessor() = default;
		NO_COPY_ASSIGN_MOVE(SurfelsProcessor);
		/**
		 * \brief ���캯���������ܵ����ںϵ�Canonical����(0��������������ϵ).
		 *
		 * \param devCount �������
		 * \param surfelsArray ��Ҫ�ںϵĵ�������
		 * \param cameraPose ���λ��
		 * \param mergedSurfels ��������ںϺ�ĵ���
		 * \param CUDA��ID�������޷��������У�ͬʱ��mergedSurfels���в��������ɲ���
		 */
		void MergeDenseSurfels(const unsigned int devCount, DeviceArrayView<DepthSurfel>* surfelsArray, const mat34* cameraPose, DeviceBufferArray<DepthSurfel>& mergedSurfels, cudaStream_t stream = 0);

		/**
		 * \brief �Գ��ܵ��ƽ����²���.
		 * 
		 * \param subsampler �����²��������÷������漰�����²������������ڴ�
		 * \param canonicalVertices �����׼��ĳ��ܶ���
		 * \param colorViewTime ����׷�ݽڵ���Դ���ĸ��ӽ�
		 * \param candidateNodes ��������²����ĺ�ѡ��
		 * \param stream CUDA��ID
		 */
		void PerformVerticesSubsamplingSync(VoxelSubsampler::Ptr subsampler, const DeviceArrayView<float4>& canonicalVertices, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& candidateNodes, cudaStream_t stream);
	private:
		/**
		 * \brief �����ܵ��������׼��.
		 * 
		 * \param mergedSurfels ����ͬCanonical���е���Ԫ���뵽preAlignedSurfel��
		 * \param currentValidDepthSurfel ��ǰ���ID�ĳ�����Ԫ
		 * \param CameraID ���ID
		 * \param offset ����preAlignedSurfel��λ��ƫ��
		 * \param stream CUDA��ID
		 */
		void MergeDenseSurfelToCanonicalField(DeviceBufferArray<DepthSurfel>& mergedSurfels, DeviceArrayView<DepthSurfel>& currentValidDepthSurfel, mat34 cameraPose, const unsigned int CameraID, const unsigned int offset, cudaStream_t stream = 0);
	};
}

