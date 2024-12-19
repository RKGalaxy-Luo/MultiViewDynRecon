/*****************************************************************//**
 * \file   VoxelSubsampler.h
 * \brief  �����Գ��ܶ�������²���
 * 
 * \author LUO
 * \date   February 26th 2024
 *********************************************************************/
#pragma once
#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/EncodeUtils.h>
#include <base/Constants.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>
#include <core/AlgorithmTypes.h>
#include <math/VectorUtils.h>
#include <memory>

/**
 * \brief ������ԭ��
 	���PerformSubsample�������������þ��ǣ���ÿһ��points������(x,y,z)������б���
 	�������points�ܶ࣬�����յ����ظ���(Volume)ֻ��1024��1024��1024��(�������أ�����x, y, z��(-512,512)��)
 	���Ȼ�ᵼ�²�ͬ��points���ܶ�Ӧ��ͬһ���������
 	��������кܶ��Points = (x0,y0,z0)��Ӧĳһ���������Code0����ô������������Code0ת����Std_Point = (std_x, std_y, std_z)
 	Ȼ�������Щ��ͬ�����Points��Std_Point�Ŀռ���룬��Pointsѡ������׼����������Ǹ�point����Ϊ�����ĵ�
 	��������ζ�ű���������������points���������
 */


namespace SparseSurfelFusion {

	namespace device {
		/**
		 * \brief ���������б���.
		 * 
		 * \param points ������ȵ�
		 * \param encodedVoxelKey �����ļ�ֵ
		 * \param voxelSize ���صĴ�С
		 */
		__global__ void createVoxelKeyKernel(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize);

		__global__ void createVoxelKeyKernelDebug(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize, float4* debugCanVertices,int * debugencodevoxel);
		/**
		 * \brief ������кõ����ؼ�ֵ(�����ǰֵ������ǰһ��ֵ��label = 1�� �����ǰֵ����ǰһ��ֵ��label = 0).
		 * 
		 * \param sortedVoxelKey 
		 * \param keyLabel ��VoxelLabel��m_array��GPU���飩��ֵ��
		 */
		__global__ void labelSortedVoxelKeyKernel(const PtrSize<const int> sortedVoxelKey, unsigned int* keyLabel);
		/**
		 * \brief ��compactedKey��compactedOffset��ֵ.
		 * 
		 * \param sortedVoxelKey ��Ч�����кõļ�ֵ
		 * \param voxelKeyLabel ���ص�label����
		 * \param prefixsumedLabel GPU��ǰ׺�͵ĵ�ַ
		 * \param compactedKey ��á���ǰһ�������ֵ��һ�����ı��루��ǰһ��һ������ȥ��
		 * \param compactedOffset ����������ǰһ�������ֵ��һ�����ı�����pointKeySort.valid_sorted_key�е�λ�ã�idx��
		 */
		__global__ void compactedVoxelKeyKernel(const PtrSize<const int> sortedVoxelKey, const unsigned int* voxelKeyLabel, const unsigned int* prefixsumedLabel, int* compactedKey, DeviceArrayHandle<int> compactedOffset);
		/**
		 * \brief �ҵ���ӽ���׼������Ǹ�������Ϊ����ѹ����Ķ��㣬���sampledPoints.
		 * 
		 * \param compactedKey �Զ���ѹ����ļ�
		 * \param compactedOffset ����ǰһ����һ���ı��롱��ԭʼ�ı��������е�λ�ã�idx��
		 * \param sortedPoints ԭʼ�����кõĵ�
		 * \param voxelSize ���ش�С
		 * \param sampledPoints ���յĲ�����
		 */
		__global__ void samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float voxelSize, float4* sampledPoints);

		/**
		 * \brief �ҵ���ӽ���׼������Ǹ�������Ϊ����ѹ����Ķ��㣬���sampledPoints.
		 *
		 * \param compactedKey �Զ���ѹ����ļ�
		 * \param compactedOffset ����ǰһ����һ���ı��롱��ԭʼ�ı��������е�λ�ã�idx��
		 * \param sortedPoints ԭʼ�����кõĵ�
		 * \param sortedColorViewTime �����кõĳ��ܵ�һһ��Ӧ���ӽ�׷������
		 * \param voxelSize ���ش�С
		 * \param sampledPoints ���յĲ�����
		 */
		__global__ void samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float4* sortedColorViewTime, const float voxelSize, float4* sampledPoints);
	}

	class VoxelSubsampler
	{
	public:
		using Ptr = std::shared_ptr<VoxelSubsampler>;
		VoxelSubsampler() = default;
		~VoxelSubsampler() = default;
		NO_COPY_ASSIGN_MOVE(VoxelSubsampler);

		/**
		 * \brief ���仺��.
		 * 
		 * \param maxInputPoints ����������Ķ������
		 */
		void AllocateBuffer(unsigned int maxInputPoints);
		/**
		 * \brief �ͷŻ���.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief ��ȡ�Ƿ�Ϊ�²������㷨���������ڴ�.
		 * 
		 * \return true��ʾ�Լ������ڴ�
		 */
		bool isBufferEmpty();

		/**
		 * \brief ִ�н�����(ֻ����ʽ).
		 * 
		 * \param points ���ܵ�
		 * \param voxelSize ���ش�С
		 * \param stream CUDA��ID
		 * \return ��������ĵ�(ֻ����ʽ)
		 */
		DeviceArrayView<float4> PerformSubsample(const DeviceArrayView<float4>& points, const float voxelSize, cudaStream_t stream = 0);

		/**
		 * \brief .
		 * 
		 * \param points ���ܵ�
		 * \param subsampledVertice �����Ĳ�����
		 * \param voxelSize ���ش�С
		 * \param stream CUDA��ID
		 * \return ��������ĵ�(ֻ����ʽ)
		 */
		void PerformSubsample(const DeviceArrayView<float4>& points, DeviceBufferArray<float4>& subsampledVertice,const float voxelSize, cudaStream_t stream = 0);

		/**
		 * \brief ִ�н�����(Device - Host ͬ��������ʽ).
		 * 
		 * \param points ���ܵ�
		 * \param colorViewTime ���ܵ��colorViewTime���飬��points���ܵ��Ӧ��y�����Ǹó��ܵ���Դ���ĸ��ӽ�
		 * \param subsampledPoints ��������ĵ�(Device - Host ͬ��������ʽ)
		 * \param voxelSize ���ش�С
		 * \param stream CUDA��ID
		 */
		void PerformSubsample(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& subsampledPoints, const float voxelSize, cudaStream_t stream = 0);
	
	private:

		bool allocatedBuffer = false;		// ����Ƿ�Ϊ�²���������������ڲ��㷨����������ڴ�

		DeviceBufferArray<int> pointKey;	// ÿ�����ع���һ������������������ĳ������
		/**
		 * \brief Ϊ������л�ȡ����ά�㹹���ֵ����ֵ�����Ǹ������꣬��˼�ֵ - ������һһӳ���ϵ��.
		 * 
		 * \param points �����
		 * \param voxel_size ���ش�С
		 * \param stream CUDA��ID
		 */
		void buildVoxelKeyForPoints(const DeviceArrayView<float4>& points, const float voxel_size, cudaStream_t stream = 0);


		KeyValueSort<int, float4> pointKeySort;		// ������ά����ӳ�䵽���أ��������ر��룬�ٽ��������򡿶����ؼ�ִ�������ѹ��
		KeyValueSort<int, float4> colorViewTimeSort;// ���������һ������������׷�ݳ��ܵ���������һ���ӽǵ����
		DeviceBufferArray<unsigned int> voxelLabel;	// ���������ı������ҵ�����ǰһ�����벻ͬ���������е�index����¼�����������ر��룬���m_voxel_label[idx] != m_voxel_label[idx-1]����label = 1�� ����label = 0
		PrefixSum voxelLabelPrefixsum;				// ������ǰ��һ���ж��ٸ����ر��롿����label��ǰ׺�ͣ���Ҫ��������ʾǰ���м�������ǰһ�����벻һ�����ı���
		DeviceBufferArray<int> compactedVoxelKey;	// ����һ�޶������ر���������顿����ǰһ�������ֵ��һ�����ı��루��ǰһ��һ������ȥ��
		DeviceBufferArray<int> compactedVoxelOffset;// �������һ�޶������ر�����pointKeySort���ĸ�λ�á��������ǰһ�������ֵ��һ�����ı�����m_point_key_sort.valid_sorted_key�е�λ�ã�idx��
	
		/**
		 * \brief �ѵ��xyzֵת�����������꣬������ͬʱ�ҵ����������ĸ�index��ʱ�򣬳��ܵ�ӳ����˲�ͬ��voxel�������ҵ�.
		 * 
		 * \param points �����
		 * \param stream CUDA��ID
		 */
		void sortCompactVoxelKeys(const DeviceArrayView<float4>& points, cudaStream_t stream = 0);


		/**
		 * \brief �ѵ��xyzֵת�����������꣬������ͬʱ�ҵ����������ĸ�index��ʱ�򣬳��ܵ�ӳ����˲�ͬ��voxel�������ҵ�.
		 *
		 * \param points �����
		 * \param colorViewTime ����׷���ӽ�
		 * \param stream CUDA��ID
		 */
		void sortCompactVoxelKeys(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, cudaStream_t stream = 0);

		/* �ռ�����ѹ��ƫ�����Ľ������� */
		DeviceBufferArray<float4> subsampledPoint; // �ռ����ս����������ά��
		/**
		 * \brief �ռ��²����㣺�ҵ���ӽ���׼������Ǹ�������Ϊ����ѹ����Ķ���.
		 * 
		 * \param subsampled_points �������㸳ֵ���������
		 * \param voxel_size ���ش�С
		 * \param stream CUDA��ID
		 */
		void collectSubsampledPoint(DeviceBufferArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream = 0);
		/**
		 * \brief �ռ������㲢ͬ��������.
		 * 
		 * \param subsampled_points �������㸳ֵ���������
		 * \param voxel_size ���ش�С
		 * \param stream CUDA��ID
		 */
		void collectSynchronizeSubsampledPoint(SynchronizeArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream = 0);

	};
}


