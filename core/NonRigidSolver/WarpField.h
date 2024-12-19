/*****************************************************************//**
 * \file   WarpField.h
 * \brief  构建、更新扭曲场节点，执行向前或者向后扭曲节点
 * 
 * \author LUO
 * \date   March 8th 2024
 *********************************************************************/
#pragma once
#include <chrono>

#include <base/Constants.h>
#include <base/CommonUtils.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>

#include <core/Geometry/SurfelGeometry.h>
#include <core/Geometry/VoxelSubsampler.h>
#include <core/Geometry/SurfelsProcessor.h>

#include <math/DualQuaternion/DualQuaternion.h>

namespace SparseSurfelFusion {

	namespace device {

		enum {
			nodesNeighborsNum = 8	// 邻居节点数量
		};

		/**
		 * \param 计算两个float4点的距离(只考虑x,y,z分量).
		 * 
		 * \param p1 点p1
		 * \param p2 点p2
		 */
		__device__ __forceinline__ float distanceSquare(const float4& p1, const float4& p2);

		/**
		 * \brief 构建节点图的核函数，【K邻域搜索】节点图是控制节点的邻居，每个节点有8个邻居，作为结果元素存储在节点图中。节点的数量都在千个级别，因此只遍历节点并不会非常昂贵.
		 * 
		 * \param canonical域中的节点，计算得到的节点图
		 */
		__global__ void BuildNodesGraphKernel(DeviceArrayView<float4> canonicalNodes, ushort2* canonicalNodesGraph);
	}

	/**
	 * \brief 扭曲场构建方法及非刚性形变参数，包括：获得初始化节点，构造节点图等.
	 */
	class WarpField
	{
	public:
		using Ptr = std::shared_ptr<WarpField>;
		/**
		 * \brief 构造函数，分配扭曲场需要的内存.
		 * 
		 */
		WarpField();
		/**
		 * \brief 释放分配的内存.
		 * 
		 */
		~WarpField();
		/**
		 * \brief 调整设备上nodesKNN、nodesKNNWeight以及liveNodesCoordinate的Array大小.
		 * 
		 * \param nodesSize 节点数量
		 */
		void ResizeDeviceArrayToNodeSize(const unsigned int nodesNum);

		/**
		 * \brief 从多视角深度相机融合的深度面元中，初始化标准域中的节点以及每个节点的SE3参数.
		 * 
		 * \param canonicalVertices 标准域中稠密顶点
		 * \param colorViewTime 主要是为了追溯Node来自于哪个相机
		 * \param CUDA流ID
		 */
		void InitializeCanonicalNodesAndSE3FromMergedVertices(DeviceArrayView<float4>& canonicalVertices, DeviceArrayView<float4>& colorViewTime, cudaStream_t stream = 0);

		/**
		 * \brief 从候选节点中，初始化标准域中的节点以及对应的SE3参数.
		 * 
		 * \param nodesCandidates 候选节点
		 * \param stream CUDA流ID
		 */
		void InitializeCanonicalNodesAndSE3FromCandidates(const std::vector<float4>& nodesCandidates, cudaStream_t stream = 0);
	
		/**
		 * \brief 根据下采样的节点，构建节点图及权重.
		 * 
		 * \param stream CUDA流ID
		 */
		void BuildNodesGraph(cudaStream_t stream = 0);

		/**
		 * \brief 在这里调整一下节点的归属问题，有些节点在warp之后，已经处于当前帧观察不到，或者是与当前帧夹角很大的情况，需要根据实际情况调整.
		 * 
		 * \param stream cuda流
		 */
		void AdjustNodeSourceFrom(cudaStream_t stream = 0);

		unsigned CheckAndGetNodeSize() const;


	private:

		// 可以在主机上访问的同步成员
		SynchronizeArray<float4> canonicalNodesCoordinate;	// Canonical域节点坐标
		SynchronizeArray<DualQuaternion> nodesSE3;			// 每个节点的SE3(位姿变换)
		SynchronizeArray<float4> candidateNodes;			// 降采样后的点

		// 这些属性将从主机上传到设备
		DeviceBufferArray<ushort4> nodesKNN;				// 与节点邻近的点的index
		DeviceBufferArray<float4> nodesKNNWeight;			// 与节点邻近的点的Weight(权重)

		// 只能在设备上访问的成员
		DeviceBufferArray<float4> liveNodesCoordinate;		// Live域节点坐标
		DeviceBufferArray<ushort2> nodesGraph;				// nodesGraph  -->  <当前节点的idx, 与当前节点相邻的8个节点的idx(数组的首地址)>
		
		VoxelSubsampler::Ptr voxelSubsampler;				// 声明下采样器


		friend class WarpFieldUpdater;
		friend class SurfelNodeDeformer;
		/**
		 * \brief 开辟WarpField相关内存参数的Buffer，一旦开辟不能拓展.
		 * 
		 * \param maxNodesNum 最大的节点个数
		 */
		void allocateBuffer(size_t maxNodesNum);

		/**
		 * \brief 释放WarpField相关内存参数的Buffer.
		 * 
		 */
		void releaseBuffer();

	public:

		/**
		 * \brief 数据打包结构体：发送WarpField节点数据给蒙皮器【暴露地址，可供修改】.
		 */
		struct SkinnerInput {
			DeviceArrayView<float4> canonicalNodesCoordinate;		// 可读稀疏节点位置信息
			DeviceArrayHandle<ushort4> sparseNodesKnn;				// 可写入稀疏节点KNN点索引信息
			DeviceArrayHandle<float4> sparseNodesKnnWeight;			// 可写入稀疏节点KNN点权重信息
		};
		/**
		 * \brief 绑定SkinnerInput与WarpField中，与蒙皮有关的KNN Index和Weight数据，后续对SkinnerInput操作可直接映射到WarpField相应变量.
		 *
		 * \return 返回Skinner数据包
		 */
		SkinnerInput BindWarpFieldSkinnerInfo();

		/**
		 * \brief 非刚性求解器的输入，传入只读数据以供变换，不暴露地址.
		 */
		struct NonRigidSolverInput {
			DeviceArrayView<DualQuaternion> nodesSE3;
			DeviceArrayView<float4> canonicalNodesCoordinate;
			DeviceArrayView<ushort2> nodesGraph;
		};
		/**
		 * \brief 绑定非刚性变换的输入参数.
		 *
		 * \return 返回打包好的非刚性变换的参数
		 */
		NonRigidSolverInput BindNonRigidSolverInfo() const;

		struct OpticalFlowGuideInput {
			DeviceArray<DualQuaternion> nodesSE3;			// 节点SE3，Canonical转Live

			DeviceArray<ushort4> nodesKNN;					// 与节点邻近的点的index
			DeviceArray<float4> nodesKNNWeight;				// 与节点邻近的点的Weight(权重)

			// 只能在设备上访问的成员
			DeviceArray<float4> liveNodesCoordinate;		// Live域节点坐标
		};

		OpticalFlowGuideInput BindOpticalFlowGuideInfo();

		/* The input accessed by geometry updater
		 */
		struct LiveGeometryUpdaterInput {
			DeviceArrayView<float4> live_node_coords;
			DeviceArrayView<float4> reference_node_coords;
			DeviceArrayView<DualQuaternion> node_se3;
		};
		LiveGeometryUpdaterInput GeometryUpdaterAccess() const;

	public:
		/**
		 * \brief 获得Canonical域中的节点坐标.
		 * 
		 * \return Canonical域中的节点坐标
		 */
		DeviceArrayView<float4> getCanonicalNodesCoordinate();
		/**
		 * \brief 获得Live域中的节点坐标.
		 * 
		 * \return Live域中的节点坐标
		 */
		DeviceArrayView<float4> getLiveNodesCoordinate();

//************************************ 这些是更新完非刚性SE后用到的函数*****************************************
		/* Update my warp field from the solved value
		 */
	public:
		void UpdateHostDeviceNodeSE3NoSync(DeviceArrayView<DualQuaternion> node_se3, cudaStream_t stream = 0);
		void SyncNodeSE3ToHost() { nodesSE3.SynchronizeToHost(); };

	};
}



