#pragma once

#include "solver_constants.h"

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

#include <core/AlgorithmTypes.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include "solver_types.h"
#include <memory>
#include <base/GlobalConfigs.h>
#include <render/Renderer.h>
#include <base/Constants.h>
#include <base/data_transfer.h> 

#include <visualization/Visualizer.h>


namespace SparseSurfelFusion {
	namespace device {

		enum {
			CrossGeometrySearchHalfSize = 2,
			CrossGeometrySearchStep = 2
		};

		struct ObservedCrossViewCorrespondenceInterface {
			DeviceArrayView<CrossViewCorrPairs> crossCorrPairs;			// �羵ƥ���
			DeviceArrayView2D<ushort4> corrMap[MAX_CAMERA_COUNT];		// ƥ��ͼ
			cudaTextureObject_t depthVertexMap[MAX_CAMERA_COUNT];		// �۲쵽�ĵ�ǰ֡vertex
		};

		struct GeometryMapCrossViewCorrespondenceInterface {
			cudaTextureObject_t canonicalVertexMap[MAX_CAMERA_COUNT];	// Canonical��vertexMap
			cudaTextureObject_t liveVertexMap[MAX_CAMERA_COUNT];		// Live��VertexMap
			cudaTextureObject_t indexMap[MAX_CAMERA_COUNT];				// IndexMap
			DeviceArrayView2D<KNNAndWeight> knnMap[MAX_CAMERA_COUNT];	// KnnMap
			mat34 camera2World[MAX_CAMERA_COUNT];						// ����������ת��world����ϵ
			mat34 initialCameraSE3[MAX_CAMERA_COUNT];					// ���λ��
		};

		/**
		 * \brief �羵ƥ��ĵ���Geometry���Ƿ�����Ч��(�����ĵ�Ϊ׼).
		 */
		__device__ ushort2 ValidCrossGeometryPixelInWindow(cudaTextureObject_t indexMap, unsigned short center_x, unsigned short center_y);
		
		/**
		 * \brief �羵ƥ��ĵ���Geometry���Ƿ�����Ч��(������ĵ�Ϊ׼).
		 */
		__device__ ushort2 ValidCrossGeometryPixelInWindow(cudaTextureObject_t indexMap, cudaTextureObject_t liveVertexMap, float4 observedVertex, unsigned short center_x, unsigned short center_y);

		/**
		 * \brief �羵ƥ��ĵ���Geometry���Ƿ�����Ч��(������ĵ�Ϊ׼).
		 */
		__device__ ushort4 ValidCrossGeometryPixelInWindow(const GeometryMapCrossViewCorrespondenceInterface& geoemtry, const ObservedCrossViewCorrespondenceInterface& observed, const CrossViewCorrPairs& crossPairsCenter, const float3& observed_1, const float3& observed_2, const float& observed_diff_xyz);


		/**
		 * \brief �羵ƥ��ĵ��ڹ۲�֡VertexMap���Ƿ���Ч(�����ĵ�Ϊ׼).
		 */
		__device__ ushort2 ValidDepthPixelInWindow(cudaTextureObject_t depthVertexMap, unsigned short center_x, unsigned short center_y);

		/**
		 * \brief ��ǿ羵ƥ����У���Geometry��Ч��Pairs.
		 */
		__global__ void ChooseValidCrossCorrPairsKernel(ObservedCrossViewCorrespondenceInterface observed, GeometryMapCrossViewCorrespondenceInterface geometry, const unsigned int rows, const unsigned int cols, const unsigned int pairsNum, unsigned int* indicator, CrossViewCorrPairs* ObCrossCorrPairs, CrossViewCorrPairs* GeoCrossCorrPairs);

		/**
		 * \brief ѹ���������Ч�Ŀ羵ƥ���.
		 */
		__global__ void CompactValidCrossCorrPairsKernel(ObservedCrossViewCorrespondenceInterface observed, GeometryMapCrossViewCorrespondenceInterface geometry, const unsigned int totalCrossCorrPairs, const DeviceArrayView<unsigned int> validIndicator, const unsigned int* prefixsumIndicator, DeviceArrayView<CrossViewCorrPairs> ObCrossCorrPairs, DeviceArrayView<CrossViewCorrPairs> GeoCrossCorrPairs, float4* targetVertexArray, float4* canonicalVertexArray, ushort4* knnArray, float4* knnWeightArray);

		/**
		 * \brief ���羵ƥ���ͨ��nodeSe3Ť��.
		 */
		__global__ void ForwardWarpCrossViewFeatureVertexKernel(DeviceArrayView<float4> canonicalVertexArray, const ushort4* vertexKnnArray, const float4* vertexKnnWeightArray, const DualQuaternion* nodeSe3, const unsigned int canonicalVertexNum, float4* warpedVertexArray);
	}

	class CrossViewCorrespondenceHandler
	{
	private:
		DeviceArrayView<DualQuaternion> NodeSe3;

		device::ObservedCrossViewCorrespondenceInterface observedCrossViewCorrInterface;
		device::GeometryMapCrossViewCorrespondenceInterface geometryCrossViewCorrInterface;
		unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int knnMapCols = CLIP_WIDTH;
		const unsigned int knnMapRows = CLIP_HEIGHT;

		unsigned int crossCorrPairsNum = 0;			// δɸѡƥ��������
		unsigned int validCrossCorrPairsNum = 0;	// ��¼ͨ��Geometryɸѡ��Pairs����
	public:
		using Ptr = std::shared_ptr<CrossViewCorrespondenceHandler>;
		DEFAULT_CONSTRUCT_DESTRUCT(CrossViewCorrespondenceHandler);
		NO_COPY_ASSIGN_MOVE(CrossViewCorrespondenceHandler);


		// ��ʽ�����ڴ�
		void AllocateBuffer();
		void ReleaseBuffer();

		unsigned int frameIdx = 0;

		/**
		 * \brief �羵ƥ��Non-Rigid����.
		 * 
		 * \param nodeSe3 �ڵ�Se3
		 * \param knnMap knnMap
		 * \param vertexMap ��ǰ֡�۲쵽��vertexMap
		 * \param crossCorrPairs �羵ƥ��ĵ��
		 * \param solverMaps ���SolverMap
		 * \param world2camera ���Զ���
		 * \param InitialCameraSE3 ��ʼ��λ��
		 */
		void SetInputs(DeviceArrayView<DualQuaternion> nodeSe3, DeviceArray2D<KNNAndWeight>* knnMap, cudaTextureObject_t* vertexMap, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, DeviceArrayView2D<ushort4>* corrMap, Renderer::SolverMaps* solverMaps, mat34* world2camera, mat34* InitialCameraSE3);

		/**
		 * \brief ����nodeSE3.
		 * 
		 * \param nodeSe3 �ڵ�Se3
		 */
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> nodeSe3);

	private:
		DeviceBufferArray<unsigned int> validCrossCorrIndicator;		// ��ʶindexMap��Ч��Pairs
		PrefixSum validCorrPrefixSum;
		DeviceBufferArray<CrossViewCorrPairs> GeometryCrossCorrPairs;	// ��Geometry�ϵĿ羵ƥ���
		DeviceBufferArray<CrossViewCorrPairs> ObservedCrossPairs;		// �ڹ۲�֡����Ч�Ŀ羵ƥ���

	public:
		void ChooseValidCrossCorrPairs(cudaStream_t stream = 0);


	private:
		DeviceBufferArray<float4> validTargetVertex;					// ��ǰ֡vertex
		DeviceBufferArray<float4> validCanonicalVertex;					// �뵱ǰ֡vertex��Ӧ��canonical���е�vertex
		DeviceBufferArray<ushort4> validVertexKnn;						// �羵ƥ����Knn
		DeviceBufferArray<float4> validKnnWeight;						// �羵ƥ����KnnWeight
	public:
		void CompactCrossViewCorrPairs(cudaStream_t stream = 0);		// ѹ���羵ƥ���
		void QueryCompactedCrossViewArraySize(cudaStream_t stream = 0);	// ���������ArraySize

		DeviceArrayView<float4> GetTargetVertex() { return validTargetVertex.ArrayView(); }
		DeviceArrayView<float4> GetCanVertex() { return validCanonicalVertex.ArrayView(); }

	private:
		DeviceBufferArray<float4> validWarpedVertex;					// ��һ֡live����Ч��vertex
		void forwardWarpFeatureVertex(cudaStream_t stream = 0);			// �����һ֡Live���vertex
	public:
		void BuildTerm2Jacobian(cudaStream_t stream = 0);				// ����Jacobian��

	public:
		Point2PointICPTerm2Jacobian Term2JacobianMap() const;
		DeviceArrayView<ushort4> CrossViewCorrKNN() const { return validVertexKnn.ArrayView(); }
	};
}


