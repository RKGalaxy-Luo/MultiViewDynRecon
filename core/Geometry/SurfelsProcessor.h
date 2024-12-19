/*****************************************************************//**
 * \file   SurfelsProcessor.h
 * \brief  用于对于面元的一些基本处理，融合、下采样、提取分量等
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
		 * \brief 将多个摄像头的数据融合到preAlignedSurfel.
		 *
		 * \param preAlignedSurfel 融合多摄像头的Surfel到此处
		 * \param depthSurfel 需要转换到preAlignedSurfel的面元
		 * \param relativePose 初始摄像头位姿
		 * \param pointsNum 当前相机的点的数量
		 * \param offset preAlignedSurfel的偏移量，此时应该从哪个位置开始存储
		 */
		__global__ void MergeDenseSurfelToCanonicalFieldKernel(DeviceArrayHandle<DepthSurfel> mergedSurfels, DeviceArrayView<DepthSurfel> depthSurfel, mat34 relativePose, const unsigned int pointsNum, const unsigned int offset,const int i);
	}


	/**
	 * \brief 面元的基本处理单元，可读写，但不拥有内存(不管内存的分配和释放).
	 */
	class SurfelsProcessor
	{
	public:
		using Ptr = std::shared_ptr<SurfelsProcessor>;	// 智能指针，即用即删

		SurfelsProcessor() = default;
		~SurfelsProcessor() = default;
		NO_COPY_ASSIGN_MOVE(SurfelsProcessor);
		/**
		 * \brief 构造函数，将稠密点云融合到Canonical域中(0号相机的相机坐标系).
		 *
		 * \param devCount 相机数量
		 * \param surfelsArray 需要融合的点云数组
		 * \param cameraPose 相机位姿
		 * \param mergedSurfels 【输出】融合后的点云
		 * \param CUDA流ID，这里无法多流进行，同时对mergedSurfels进行操作，不可并行
		 */
		void MergeDenseSurfels(const unsigned int devCount, DeviceArrayView<DepthSurfel>* surfelsArray, const mat34* cameraPose, DeviceBufferArray<DepthSurfel>& mergedSurfels, cudaStream_t stream = 0);

		/**
		 * \brief 对稠密点云进行下采样.
		 * 
		 * \param subsampler 传入下采样器，该方法不涉及分配下采样器的运行内存
		 * \param canonicalVertices 输入标准域的稠密顶点
		 * \param colorViewTime 用于追溯节点来源于哪个视角
		 * \param candidateNodes 【输出】下采样的候选点
		 * \param stream CUDA流ID
		 */
		void PerformVerticesSubsamplingSync(VoxelSubsampler::Ptr subsampler, const DeviceArrayView<float4>& canonicalVertices, const DeviceArrayView<float4>& colorViewTime, SynchronizeArray<float4>& candidateNodes, cudaStream_t stream);
	private:
		/**
		 * \brief 将稠密点云融入标准域.
		 * 
		 * \param mergedSurfels 将不同Canonical域中的面元对齐到preAlignedSurfel中
		 * \param currentValidDepthSurfel 当前相机ID的稠密面元
		 * \param CameraID 相机ID
		 * \param offset 存入preAlignedSurfel的位置偏移
		 * \param stream CUDA流ID
		 */
		void MergeDenseSurfelToCanonicalField(DeviceBufferArray<DepthSurfel>& mergedSurfels, DeviceArrayView<DepthSurfel>& currentValidDepthSurfel, mat34 cameraPose, const unsigned int CameraID, const unsigned int offset, cudaStream_t stream = 0);
	};
}

