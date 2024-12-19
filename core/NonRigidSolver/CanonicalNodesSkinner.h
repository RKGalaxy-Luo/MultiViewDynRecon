/*****************************************************************//**
 * \file   CanonicalNodesSkinner.h
 * \brief  计算稀疏节点对稠密点的影响权重，对稀疏点进行蒙皮
 * 
 * \author LUO
 * \date   March 8th 2024
 *********************************************************************/
#pragma once
#include <base/Logging.h>
#include <base/Constants.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/SynchronizeArray.h>

//#include <core/KNNSearchFunction.h>
#include <core/Geometry/SurfelGeometry.h>
#include <core/NonRigidSolver/WarpField.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <memory>

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief 打包所有视角的蒙皮接口.
		 */
		struct SkinningKnnInterface {
			ushort4* denseVerticesKnn[MAX_CAMERA_COUNT];
			float4* denseVerticesKnnWeight[MAX_CAMERA_COUNT];
		};

		/**
		 * \brief 计算Canonical节点之间的权重，DenseVertices与节点之间的权重.
		 * 
		 * \param denseVertices 稠密顶点
		 * \param denseVerticesNum 稠密顶点的数量
		 * \param nodesNum CanonicalNodes中有效节点的数量
		 * \param denseVerticesKnn 【输出】稠密顶点的Nodes邻居
		 * \param denseVerticesKnnWeight 【输出】稠密顶点的Nodes邻居的权重
		 * \param nodesKnn 【输出】Nodes顶点之间的邻居
		 * \param nodesKnnWeight 【输出】Nodes顶点之间的权重
		 */
		__global__ void skinningVertexAndNodeBruteForceKernel(const DeviceArrayView<float4> denseVertices, const unsigned int denseVerticesNum, const unsigned int nodesNum, ushort4* denseVerticesKnn, float4* denseVerticesKnnWeight, ushort4* nodesKnn, float4* nodesKnnWeight);
		
		/**
		 * \brief 给节点和顶点蒙皮.
		 */
		__global__ void skinningVertexAndNodeBruteForceKernel(const DeviceArrayView<float4> denseVertices, const unsigned int denseVerticesNum, const unsigned int nodesNum, SkinningKnnInterface denseKnnInterface, ushort4* nodesKnn, float4* nodesKnnWeight);
		
	}


	/**
	 * \brief Canonical域中的蒙皮器，自带KNN算法并合理分配内存.
	 */
	class CanonicalNodesSkinner
	{
	public:
		using Ptr = std::shared_ptr<CanonicalNodesSkinner>;

		NO_COPY_ASSIGN_MOVE(CanonicalNodesSkinner);

		/**
		 * \brief 构造函数，分配算法运行内存.
		 * 
		 */
		CanonicalNodesSkinner(unsigned int devCount);

		/**
		 * 析构函数，释放内存.
		 * 
		 */
		~CanonicalNodesSkinner();



		/**
		 * \brief 第0帧构建(或者定期清除重建)蒙皮索引.
		 * 
		 * \param canonicalNodes canonical域的节点
		 * \param stream CUDA流ID
		 */
		void BuildInitialSkinningIndex(DeviceArrayView<float4>& canonicalNodes, cudaStream_t stream = 0);

		/**
		 * \brief 执行蒙皮操作，求解KNN及Weight，使用Skinner内部自带的cudaStream去并行求解KNN及Weight，无需传入额外Stream.
		 * 
		 * \param denseSurfels 稠密面元
		 * \param sparseNodes Canonical域稀疏节点
		 */
		void PerformSkinning(SurfelGeometry::SkinnerInput denseSurfels, WarpField::SkinnerInput sparseNodes, cudaStream_t stream = 0);

		//这个函数是用在更新can域时，给indexmap传递新值的函数
		void PerformSkinning(SurfelGeometry::SkinnerInput* denseSurfels, WarpField::SkinnerInput sparseNodes, cudaStream_t stream = 0);

	private:
		DeviceArray<float4> invalidNodes;				// 无效节点
		unsigned m_num_bruteforce_nodes; //The number of nodes recorded in the brute force skinning index
		unsigned int CanonicalNodesNum = 0;				// 记录当前有效的Canonical域中的节点数

		unsigned int devicesCount = 0;

		/**
		 * \brief 用无效值填充CanonicalNodes内存.
		 * 
		 * \param stream CUDA流ID
		 */
		void fillInvalidGlobalPoints(cudaStream_t stream = 0);

		/**
		 * \brief 暴力蒙皮，遍历的方式寻找KNN.
		 * 
		 * \param denseVertices 稠密面元
		 * \param verticesKnn 稠密面元的KNN索引
		 * \param verticesKnnWeight 稠密面元的KNN权重
		 * \param canonicalNodes canonical域节点
		 * \param nodesKnn canonical域节点KNN索引
		 * \param nodesKnnWeight canonical域节点KNN权重
		 * \param stream CUDA流ID
		 */
		void skinningVertexAndNodeBruteForce(const DeviceArrayView<float4>& denseVertices, DeviceArrayHandle<ushort4> verticesKnn, DeviceArrayHandle<float4> verticesKnnWeight, const DeviceArrayView<float4>& canonicalNodes, DeviceArrayHandle<ushort4> nodesKnn, DeviceArrayHandle<float4> nodesKnnWeight, cudaStream_t stream = 0);
		
		/**
		 * \brief 刷新所有视角的Canonical域.
		 * 
		 * \param denseVertices 稠密顶点
		 * \param skinningKnnInterface 所有视角的Sknning接口
		 * \param canonicalNodes Canonical域的节点
		 * \param nodesKnn 节点Knn
		 * \param nodesKnnWeight 节点权重
		 * \param stream cuda流
		 */
		void skinningVertexAndNodeBruteForce(
			const DeviceArrayView<float4>& denseVertices, 
			device::SkinningKnnInterface& skinningKnnInterface,
			const DeviceArrayView<float4>& canonicalNodes, 
			DeviceArrayHandle<ushort4> nodesKnn, 
			DeviceArrayHandle<float4> nodesKnnWeight, 
			cudaStream_t stream = 0
		);

		//********************************hsg fusion part
		/* The method for index and skinning update. Only perform for brute-force index.
		 * The init_skinner does not need to be updated.
		 */
	private:
		//The workforce function
		void updateSkinning(
			unsigned int newNodesOffset,
			const DeviceArrayView<float4>& denseCanonicalVertices,
			device::SkinningKnnInterface& skinningKnnInterface,
			const DeviceArrayView<float4>& canonicalNodes,
			DeviceArrayHandle<ushort4> nodesKnn,
			DeviceArrayHandle<float4> nodesKnnWeight,
			cudaStream_t stream = 0
		) const;
		void updateSkinning(
			unsigned newnode_offset,
			const DeviceArrayView<float4>& reference_vertex,
			const DeviceArrayView<float4>& reference_node,
			DeviceArrayHandle<ushort4> vertex_knn,
			DeviceArrayHandle<ushort4> node_knn,
			DeviceArrayHandle<float4> vertex_knn_weight,
			DeviceArrayHandle<float4> node_knn_weight,
			DeviceArrayHandle<ushort4> vertex_knn_indexmap,
			DeviceArrayHandle<float4> vertex_knn_weight_indexmap,
			cudaStream_t stream = 0
		) const;

	public:
		//nodes[newnode_offset] should be the first new node
		void UpdateBruteForceSkinningIndexWithNewNodes(const DeviceArrayView<float4>& nodes, unsigned newnode_offset, cudaStream_t stream = 0);
		void PerformSkinningUpdate(
			SurfelGeometry::SkinnerInput* geometry,
			WarpField::SkinnerInput warp_field,
			unsigned newnode_offset,
			cudaStream_t stream = 0
		);
		void PerformSkinningUpdate(
			SurfelGeometry::SkinnerInput geometry,
			SurfelGeometry::SkinnerInput geometryindexmap,
			WarpField::SkinnerInput warp_field,
			unsigned newnode_offset,
			cudaStream_t stream = 0
		);
	};
}


