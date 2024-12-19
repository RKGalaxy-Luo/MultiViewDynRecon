/*****************************************************************//**
 * \file   CanonicalNodesSkinner.h
 * \brief  ����ϡ��ڵ�Գ��ܵ��Ӱ��Ȩ�أ���ϡ��������Ƥ
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
		 * \brief ��������ӽǵ���Ƥ�ӿ�.
		 */
		struct SkinningKnnInterface {
			ushort4* denseVerticesKnn[MAX_CAMERA_COUNT];
			float4* denseVerticesKnnWeight[MAX_CAMERA_COUNT];
		};

		/**
		 * \brief ����Canonical�ڵ�֮���Ȩ�أ�DenseVertices��ڵ�֮���Ȩ��.
		 * 
		 * \param denseVertices ���ܶ���
		 * \param denseVerticesNum ���ܶ��������
		 * \param nodesNum CanonicalNodes����Ч�ڵ������
		 * \param denseVerticesKnn ����������ܶ����Nodes�ھ�
		 * \param denseVerticesKnnWeight ����������ܶ����Nodes�ھӵ�Ȩ��
		 * \param nodesKnn �������Nodes����֮����ھ�
		 * \param nodesKnnWeight �������Nodes����֮���Ȩ��
		 */
		__global__ void skinningVertexAndNodeBruteForceKernel(const DeviceArrayView<float4> denseVertices, const unsigned int denseVerticesNum, const unsigned int nodesNum, ushort4* denseVerticesKnn, float4* denseVerticesKnnWeight, ushort4* nodesKnn, float4* nodesKnnWeight);
		
		/**
		 * \brief ���ڵ�Ͷ�����Ƥ.
		 */
		__global__ void skinningVertexAndNodeBruteForceKernel(const DeviceArrayView<float4> denseVertices, const unsigned int denseVerticesNum, const unsigned int nodesNum, SkinningKnnInterface denseKnnInterface, ushort4* nodesKnn, float4* nodesKnnWeight);
		
	}


	/**
	 * \brief Canonical���е���Ƥ�����Դ�KNN�㷨����������ڴ�.
	 */
	class CanonicalNodesSkinner
	{
	public:
		using Ptr = std::shared_ptr<CanonicalNodesSkinner>;

		NO_COPY_ASSIGN_MOVE(CanonicalNodesSkinner);

		/**
		 * \brief ���캯���������㷨�����ڴ�.
		 * 
		 */
		CanonicalNodesSkinner(unsigned int devCount);

		/**
		 * �����������ͷ��ڴ�.
		 * 
		 */
		~CanonicalNodesSkinner();



		/**
		 * \brief ��0֡����(���߶�������ؽ�)��Ƥ����.
		 * 
		 * \param canonicalNodes canonical��Ľڵ�
		 * \param stream CUDA��ID
		 */
		void BuildInitialSkinningIndex(DeviceArrayView<float4>& canonicalNodes, cudaStream_t stream = 0);

		/**
		 * \brief ִ����Ƥ���������KNN��Weight��ʹ��Skinner�ڲ��Դ���cudaStreamȥ�������KNN��Weight�����贫�����Stream.
		 * 
		 * \param denseSurfels ������Ԫ
		 * \param sparseNodes Canonical��ϡ��ڵ�
		 */
		void PerformSkinning(SurfelGeometry::SkinnerInput denseSurfels, WarpField::SkinnerInput sparseNodes, cudaStream_t stream = 0);

		//������������ڸ���can��ʱ����indexmap������ֵ�ĺ���
		void PerformSkinning(SurfelGeometry::SkinnerInput* denseSurfels, WarpField::SkinnerInput sparseNodes, cudaStream_t stream = 0);

	private:
		DeviceArray<float4> invalidNodes;				// ��Ч�ڵ�
		unsigned m_num_bruteforce_nodes; //The number of nodes recorded in the brute force skinning index
		unsigned int CanonicalNodesNum = 0;				// ��¼��ǰ��Ч��Canonical���еĽڵ���

		unsigned int devicesCount = 0;

		/**
		 * \brief ����Чֵ���CanonicalNodes�ڴ�.
		 * 
		 * \param stream CUDA��ID
		 */
		void fillInvalidGlobalPoints(cudaStream_t stream = 0);

		/**
		 * \brief ������Ƥ�������ķ�ʽѰ��KNN.
		 * 
		 * \param denseVertices ������Ԫ
		 * \param verticesKnn ������Ԫ��KNN����
		 * \param verticesKnnWeight ������Ԫ��KNNȨ��
		 * \param canonicalNodes canonical��ڵ�
		 * \param nodesKnn canonical��ڵ�KNN����
		 * \param nodesKnnWeight canonical��ڵ�KNNȨ��
		 * \param stream CUDA��ID
		 */
		void skinningVertexAndNodeBruteForce(const DeviceArrayView<float4>& denseVertices, DeviceArrayHandle<ushort4> verticesKnn, DeviceArrayHandle<float4> verticesKnnWeight, const DeviceArrayView<float4>& canonicalNodes, DeviceArrayHandle<ushort4> nodesKnn, DeviceArrayHandle<float4> nodesKnnWeight, cudaStream_t stream = 0);
		
		/**
		 * \brief ˢ�������ӽǵ�Canonical��.
		 * 
		 * \param denseVertices ���ܶ���
		 * \param skinningKnnInterface �����ӽǵ�Sknning�ӿ�
		 * \param canonicalNodes Canonical��Ľڵ�
		 * \param nodesKnn �ڵ�Knn
		 * \param nodesKnnWeight �ڵ�Ȩ��
		 * \param stream cuda��
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


