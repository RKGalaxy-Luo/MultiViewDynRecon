/*****************************************************************//**
 * \file   VoxelSubsampler.h
 * \brief  用作对稠密顶点进行下采样
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
 * \brief 降采样原理
 	这个PerformSubsample函数的最终作用就是，将每一个points根据其(x,y,z)坐标进行编码
 	这样如果points很多，而最终的体素格子(Volume)只有1024×1024×1024个(编码体素：假设x, y, z在(-512,512)中)
 	则必然会导致不同的points可能对应着同一个坐标编码
 	这里假设有很多个Points = (x0,y0,z0)对应某一个坐标编码Code0，那么将这个坐标编码Code0转化成Std_Point = (std_x, std_y, std_z)
 	然后计算这些相同编码的Points到Std_Point的空间距离，在Points选择距离标准编码最近的那个point，作为采样的点
 	这样就意味着编码的最大容量就是points的最大容量
 */


namespace SparseSurfelFusion {

	namespace device {
		/**
		 * \brief 将坐标点进行编码.
		 * 
		 * \param points 稠密深度点
		 * \param encodedVoxelKey 编码后的键值
		 * \param voxelSize 体素的大小
		 */
		__global__ void createVoxelKeyKernel(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize);

		__global__ void createVoxelKeyKernelDebug(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize, float4* debugCanVertices,int * debugencodevoxel);
		/**
		 * \brief 标记排列好的体素键值(如果当前值不等于前一个值则label = 1， 如果当前值等于前一个值则label = 0).
		 * 
		 * \param sortedVoxelKey 
		 * \param keyLabel 给VoxelLabel的m_array（GPU数组）赋值了
		 */
		__global__ void labelSortedVoxelKeyKernel(const PtrSize<const int> sortedVoxelKey, unsigned int* keyLabel);
		/**
		 * \brief 给compactedKey和compactedOffset赋值.
		 * 
		 * \param sortedVoxelKey 有效且排列好的键值
		 * \param voxelKeyLabel 体素的label数组
		 * \param prefixsumedLabel GPU中前缀和的地址
		 * \param compactedKey 获得“与前一个编码键值不一样”的编码（与前一个一样的舍去）
		 * \param compactedOffset 获得这个“与前一个编码键值不一样”的编码在pointKeySort.valid_sorted_key中的位置（idx）
		 */
		__global__ void compactedVoxelKeyKernel(const PtrSize<const int> sortedVoxelKey, const unsigned int* voxelKeyLabel, const unsigned int* prefixsumedLabel, int* compactedKey, DeviceArrayHandle<int> compactedOffset);
		/**
		 * \brief 找到最接近标准编码的那个顶点作为最终压缩后的顶点，存给sampledPoints.
		 * 
		 * \param compactedKey 对顶点压缩后的键
		 * \param compactedOffset “与前一个不一样的编码”在原始的编码数组中的位置（idx）
		 * \param sortedPoints 原始的排列好的点
		 * \param voxelSize 体素大小
		 * \param sampledPoints 最终的采样点
		 */
		__global__ void samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float voxelSize, float4* sampledPoints);

		/**
		 * \brief 找到最接近标准编码的那个顶点作为最终压缩后的顶点，存给sampledPoints.
		 *
		 * \param compactedKey 对顶点压缩后的键
		 * \param compactedOffset “与前一个不一样的编码”在原始的编码数组中的位置（idx）
		 * \param sortedPoints 原始的排列好的点
		 * \param sortedColorViewTime 与排列好的稠密点一一对应的视角追溯数组
		 * \param voxelSize 体素大小
		 * \param sampledPoints 最终的采样点
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
		 * \brief 分配缓存.
		 * 
		 * \param maxInputPoints 最大可能输入的顶点个数
		 */
		void AllocateBuffer(unsigned int maxInputPoints);
		/**
		 * \brief 释放缓存.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief 获取是否为下采样器算法分配所需内存.
		 * 
		 * \return true表示以及分配内存
		 */
		bool isBufferEmpty();

		/**
		 * \brief 执行降采样(只读形式).
		 * 
		 * \param points 稠密点
		 * \param voxelSize 体素大小
		 * \param stream CUDA流ID
		 * \return 降采样后的点(只读形式)
		 */
		DeviceArrayView<float4> PerformSubsample(const DeviceArrayView<float4>& points, const float voxelSize, cudaStream_t stream = 0);

		/**
		 * \brief .
		 * 
		 * \param points 稠密点
		 * \param subsampledVertice 传出的采样点
		 * \param voxelSize 体素大小
		 * \param stream CUDA流ID
		 * \return 降采样后的点(只读形式)
		 */
		void PerformSubsample(const DeviceArrayView<float4>& points, DeviceBufferArray<float4>& subsampledVertice,const float voxelSize, cudaStream_t stream = 0);

		/**
		 * \brief 执行降采样(Device - Host 同步缓存形式).
		 * 
		 * \param points 稠密点
		 * \param colorViewTime 稠密点的colorViewTime数组，与points稠密点对应，y分量是该稠密点来源于哪个视角
		 * \param subsampledPoints 降采样后的点(Device - Host 同步缓存形式)
		 * \param voxelSize 体素大小
		 * \param stream CUDA流ID
		 */
		void PerformSubsample(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& subsampledPoints, const float voxelSize, cudaStream_t stream = 0);
	
	private:

		bool allocatedBuffer = false;		// 检查是否为下采样器分配采样器内部算法所需的运行内存

		DeviceBufferArray<int> pointKey;	// 每个体素构建一个键，可以用来索引某个体素
		/**
		 * \brief 为从相机中获取的三维点构造键值【键值构造是根据坐标，因此键值 - 坐标是一一映射关系】.
		 * 
		 * \param points 传入点
		 * \param voxel_size 体素大小
		 * \param stream CUDA流ID
		 */
		void buildVoxelKeyForPoints(const DeviceArrayView<float4>& points, const float voxel_size, cudaStream_t stream = 0);


		KeyValueSort<int, float4> pointKeySort;		// 【将三维坐标映射到体素，并将体素编码，再将编码排序】对体素键执行排序和压缩
		KeyValueSort<int, float4> colorViewTimeSort;// 【和坐标点一起排序】这里是追溯稠密点是来自哪一个视角的相机
		DeviceBufferArray<unsigned int> voxelLabel;	// 【在排序后的编码中找到，与前一个编码不同的在数组中的index】记录着排序后的体素编码，如果m_voxel_label[idx] != m_voxel_label[idx-1]，则label = 1， 否则label = 0
		PrefixSum voxelLabelPrefixsum;				// 【计算前面一共有多少个体素编码】体素label的前缀和，主要作用是显示前面有几个“与前一个编码不一样”的编码
		DeviceBufferArray<int> compactedVoxelKey;	// 【独一无二的体素编码放入数组】“与前一个编码键值不一样”的编码（与前一个一样的舍去）
		DeviceBufferArray<int> compactedVoxelOffset;// 【这个独一无二的体素编码在pointKeySort的哪个位置】这个“与前一个编码键值不一样”的编码在m_point_key_sort.valid_sorted_key中的位置（idx）
	
		/**
		 * \brief 把点的xyz值转换成体素坐标，并排序，同时找到在数组中哪个index的时候，稠密点映射成了不同的voxel，并且找到.
		 * 
		 * \param points 输入点
		 * \param stream CUDA流ID
		 */
		void sortCompactVoxelKeys(const DeviceArrayView<float4>& points, cudaStream_t stream = 0);


		/**
		 * \brief 把点的xyz值转换成体素坐标，并排序，同时找到在数组中哪个index的时候，稠密点映射成了不同的voxel，并且找到.
		 *
		 * \param points 输入点
		 * \param colorViewTime 用于追溯视角
		 * \param stream CUDA流ID
		 */
		void sortCompactVoxelKeys(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, cudaStream_t stream = 0);

		/* 收集给定压缩偏移量的降采样点 */
		DeviceBufferArray<float4> subsampledPoint; // 收集最终降采样完的三维点
		/**
		 * \brief 收集下采样点：找到最接近标准编码的那个顶点作为最终压缩后的顶点.
		 * 
		 * \param subsampled_points 将采样点赋值给这个数组
		 * \param voxel_size 体素大小
		 * \param stream CUDA流ID
		 */
		void collectSubsampledPoint(DeviceBufferArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream = 0);
		/**
		 * \brief 收集采样点并同步到主机.
		 * 
		 * \param subsampled_points 将采样点赋值给这个数组
		 * \param voxel_size 体素大小
		 * \param stream CUDA流ID
		 */
		void collectSynchronizeSubsampledPoint(SynchronizeArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream = 0);

	};
}


