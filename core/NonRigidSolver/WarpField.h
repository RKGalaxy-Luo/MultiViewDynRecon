/*****************************************************************//**
 * \file   WarpField.h
 * \brief  ����������Ť�����ڵ㣬ִ����ǰ�������Ť���ڵ�
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
			nodesNeighborsNum = 8	// �ھӽڵ�����
		};

		/**
		 * \param ��������float4��ľ���(ֻ����x,y,z����).
		 * 
		 * \param p1 ��p1
		 * \param p2 ��p2
		 */
		__device__ __forceinline__ float distanceSquare(const float4& p1, const float4& p2);

		/**
		 * \brief �����ڵ�ͼ�ĺ˺�������K�����������ڵ�ͼ�ǿ��ƽڵ���ھӣ�ÿ���ڵ���8���ھӣ���Ϊ���Ԫ�ش洢�ڽڵ�ͼ�С��ڵ����������ǧ���������ֻ�����ڵ㲢����ǳ�����.
		 * 
		 * \param canonical���еĽڵ㣬����õ��Ľڵ�ͼ
		 */
		__global__ void BuildNodesGraphKernel(DeviceArrayView<float4> canonicalNodes, ushort2* canonicalNodesGraph);
	}

	/**
	 * \brief Ť���������������Ǹ����α��������������ó�ʼ���ڵ㣬����ڵ�ͼ��.
	 */
	class WarpField
	{
	public:
		using Ptr = std::shared_ptr<WarpField>;
		/**
		 * \brief ���캯��������Ť������Ҫ���ڴ�.
		 * 
		 */
		WarpField();
		/**
		 * \brief �ͷŷ�����ڴ�.
		 * 
		 */
		~WarpField();
		/**
		 * \brief �����豸��nodesKNN��nodesKNNWeight�Լ�liveNodesCoordinate��Array��С.
		 * 
		 * \param nodesSize �ڵ�����
		 */
		void ResizeDeviceArrayToNodeSize(const unsigned int nodesNum);

		/**
		 * \brief �Ӷ��ӽ��������ںϵ������Ԫ�У���ʼ����׼���еĽڵ��Լ�ÿ���ڵ��SE3����.
		 * 
		 * \param canonicalVertices ��׼���г��ܶ���
		 * \param colorViewTime ��Ҫ��Ϊ��׷��Node�������ĸ����
		 * \param CUDA��ID
		 */
		void InitializeCanonicalNodesAndSE3FromMergedVertices(DeviceArrayView<float4>& canonicalVertices, DeviceArrayView<float4>& colorViewTime, cudaStream_t stream = 0);

		/**
		 * \brief �Ӻ�ѡ�ڵ��У���ʼ����׼���еĽڵ��Լ���Ӧ��SE3����.
		 * 
		 * \param nodesCandidates ��ѡ�ڵ�
		 * \param stream CUDA��ID
		 */
		void InitializeCanonicalNodesAndSE3FromCandidates(const std::vector<float4>& nodesCandidates, cudaStream_t stream = 0);
	
		/**
		 * \brief �����²����Ľڵ㣬�����ڵ�ͼ��Ȩ��.
		 * 
		 * \param stream CUDA��ID
		 */
		void BuildNodesGraph(cudaStream_t stream = 0);

		/**
		 * \brief ���������һ�½ڵ�Ĺ������⣬��Щ�ڵ���warp֮���Ѿ����ڵ�ǰ֡�۲첻�����������뵱ǰ֡�нǺܴ���������Ҫ����ʵ���������.
		 * 
		 * \param stream cuda��
		 */
		void AdjustNodeSourceFrom(cudaStream_t stream = 0);

		unsigned CheckAndGetNodeSize() const;


	private:

		// �����������Ϸ��ʵ�ͬ����Ա
		SynchronizeArray<float4> canonicalNodesCoordinate;	// Canonical��ڵ�����
		SynchronizeArray<DualQuaternion> nodesSE3;			// ÿ���ڵ��SE3(λ�˱任)
		SynchronizeArray<float4> candidateNodes;			// ��������ĵ�

		// ��Щ���Խ��������ϴ����豸
		DeviceBufferArray<ushort4> nodesKNN;				// ��ڵ��ڽ��ĵ��index
		DeviceBufferArray<float4> nodesKNNWeight;			// ��ڵ��ڽ��ĵ��Weight(Ȩ��)

		// ֻ�����豸�Ϸ��ʵĳ�Ա
		DeviceBufferArray<float4> liveNodesCoordinate;		// Live��ڵ�����
		DeviceBufferArray<ushort2> nodesGraph;				// nodesGraph  -->  <��ǰ�ڵ��idx, �뵱ǰ�ڵ����ڵ�8���ڵ��idx(������׵�ַ)>
		
		VoxelSubsampler::Ptr voxelSubsampler;				// �����²�����


		friend class WarpFieldUpdater;
		friend class SurfelNodeDeformer;
		/**
		 * \brief ����WarpField����ڴ������Buffer��һ�����ٲ�����չ.
		 * 
		 * \param maxNodesNum ���Ľڵ����
		 */
		void allocateBuffer(size_t maxNodesNum);

		/**
		 * \brief �ͷ�WarpField����ڴ������Buffer.
		 * 
		 */
		void releaseBuffer();

	public:

		/**
		 * \brief ���ݴ���ṹ�壺����WarpField�ڵ����ݸ���Ƥ������¶��ַ���ɹ��޸ġ�.
		 */
		struct SkinnerInput {
			DeviceArrayView<float4> canonicalNodesCoordinate;		// �ɶ�ϡ��ڵ�λ����Ϣ
			DeviceArrayHandle<ushort4> sparseNodesKnn;				// ��д��ϡ��ڵ�KNN��������Ϣ
			DeviceArrayHandle<float4> sparseNodesKnnWeight;			// ��д��ϡ��ڵ�KNN��Ȩ����Ϣ
		};
		/**
		 * \brief ��SkinnerInput��WarpField�У�����Ƥ�йص�KNN Index��Weight���ݣ�������SkinnerInput������ֱ��ӳ�䵽WarpField��Ӧ����.
		 *
		 * \return ����Skinner���ݰ�
		 */
		SkinnerInput BindWarpFieldSkinnerInfo();

		/**
		 * \brief �Ǹ�������������룬����ֻ�������Թ��任������¶��ַ.
		 */
		struct NonRigidSolverInput {
			DeviceArrayView<DualQuaternion> nodesSE3;
			DeviceArrayView<float4> canonicalNodesCoordinate;
			DeviceArrayView<ushort2> nodesGraph;
		};
		/**
		 * \brief �󶨷Ǹ��Ա任���������.
		 *
		 * \return ���ش���õķǸ��Ա任�Ĳ���
		 */
		NonRigidSolverInput BindNonRigidSolverInfo() const;

		struct OpticalFlowGuideInput {
			DeviceArray<DualQuaternion> nodesSE3;			// �ڵ�SE3��CanonicalתLive

			DeviceArray<ushort4> nodesKNN;					// ��ڵ��ڽ��ĵ��index
			DeviceArray<float4> nodesKNNWeight;				// ��ڵ��ڽ��ĵ��Weight(Ȩ��)

			// ֻ�����豸�Ϸ��ʵĳ�Ա
			DeviceArray<float4> liveNodesCoordinate;		// Live��ڵ�����
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
		 * \brief ���Canonical���еĽڵ�����.
		 * 
		 * \return Canonical���еĽڵ�����
		 */
		DeviceArrayView<float4> getCanonicalNodesCoordinate();
		/**
		 * \brief ���Live���еĽڵ�����.
		 * 
		 * \return Live���еĽڵ�����
		 */
		DeviceArrayView<float4> getLiveNodesCoordinate();

//************************************ ��Щ�Ǹ�����Ǹ���SE���õ��ĺ���*****************************************
		/* Update my warp field from the solved value
		 */
	public:
		void UpdateHostDeviceNodeSE3NoSync(DeviceArrayView<DualQuaternion> node_se3, cudaStream_t stream = 0);
		void SyncNodeSE3ToHost() { nodesSE3.SynchronizeToHost(); };

	};
}



