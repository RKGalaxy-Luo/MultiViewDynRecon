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
			DeviceArrayView<CrossViewCorrPairs> crossCorrPairs;			// 跨镜匹配点
			DeviceArrayView2D<ushort4> corrMap[MAX_CAMERA_COUNT];		// 匹配图
			cudaTextureObject_t depthVertexMap[MAX_CAMERA_COUNT];		// 观察到的当前帧vertex
		};

		struct GeometryMapCrossViewCorrespondenceInterface {
			cudaTextureObject_t canonicalVertexMap[MAX_CAMERA_COUNT];	// Canonical域vertexMap
			cudaTextureObject_t liveVertexMap[MAX_CAMERA_COUNT];		// Live域VertexMap
			cudaTextureObject_t indexMap[MAX_CAMERA_COUNT];				// IndexMap
			DeviceArrayView2D<KNNAndWeight> knnMap[MAX_CAMERA_COUNT];	// KnnMap
			mat34 camera2World[MAX_CAMERA_COUNT];						// 将点从相机域转到world坐标系
			mat34 initialCameraSE3[MAX_CAMERA_COUNT];					// 相机位姿
		};

		/**
		 * \brief 跨镜匹配的点在Geometry上是否是有效的(以中心点为准).
		 */
		__device__ ushort2 ValidCrossGeometryPixelInWindow(cudaTextureObject_t indexMap, unsigned short center_x, unsigned short center_y);
		
		/**
		 * \brief 跨镜匹配的点在Geometry上是否是有效的(以最近的点为准).
		 */
		__device__ ushort2 ValidCrossGeometryPixelInWindow(cudaTextureObject_t indexMap, cudaTextureObject_t liveVertexMap, float4 observedVertex, unsigned short center_x, unsigned short center_y);

		/**
		 * \brief 跨镜匹配的点在Geometry上是否是有效的(以最近的点为准).
		 */
		__device__ ushort4 ValidCrossGeometryPixelInWindow(const GeometryMapCrossViewCorrespondenceInterface& geoemtry, const ObservedCrossViewCorrespondenceInterface& observed, const CrossViewCorrPairs& crossPairsCenter, const float3& observed_1, const float3& observed_2, const float& observed_diff_xyz);


		/**
		 * \brief 跨镜匹配的点在观察帧VertexMap上是否有效(以中心点为准).
		 */
		__device__ ushort2 ValidDepthPixelInWindow(cudaTextureObject_t depthVertexMap, unsigned short center_x, unsigned short center_y);

		/**
		 * \brief 标记跨镜匹配点中，在Geometry有效的Pairs.
		 */
		__global__ void ChooseValidCrossCorrPairsKernel(ObservedCrossViewCorrespondenceInterface observed, GeometryMapCrossViewCorrespondenceInterface geometry, const unsigned int rows, const unsigned int cols, const unsigned int pairsNum, unsigned int* indicator, CrossViewCorrPairs* ObCrossCorrPairs, CrossViewCorrPairs* GeoCrossCorrPairs);

		/**
		 * \brief 压缩并获得有效的跨镜匹配点.
		 */
		__global__ void CompactValidCrossCorrPairsKernel(ObservedCrossViewCorrespondenceInterface observed, GeometryMapCrossViewCorrespondenceInterface geometry, const unsigned int totalCrossCorrPairs, const DeviceArrayView<unsigned int> validIndicator, const unsigned int* prefixsumIndicator, DeviceArrayView<CrossViewCorrPairs> ObCrossCorrPairs, DeviceArrayView<CrossViewCorrPairs> GeoCrossCorrPairs, float4* targetVertexArray, float4* canonicalVertexArray, ushort4* knnArray, float4* knnWeightArray);

		/**
		 * \brief 将跨镜匹配点通过nodeSe3扭曲.
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

		unsigned int crossCorrPairsNum = 0;			// 未筛选匹配点的数量
		unsigned int validCrossCorrPairsNum = 0;	// 记录通过Geometry筛选的Pairs数量
	public:
		using Ptr = std::shared_ptr<CrossViewCorrespondenceHandler>;
		DEFAULT_CONSTRUCT_DESTRUCT(CrossViewCorrespondenceHandler);
		NO_COPY_ASSIGN_MOVE(CrossViewCorrespondenceHandler);


		// 显式分配内存
		void AllocateBuffer();
		void ReleaseBuffer();

		unsigned int frameIdx = 0;

		/**
		 * \brief 跨镜匹配Non-Rigid输入.
		 * 
		 * \param nodeSe3 节点Se3
		 * \param knnMap knnMap
		 * \param vertexMap 当前帧观察到的vertexMap
		 * \param crossCorrPairs 跨镜匹配的点对
		 * \param solverMaps 求解SolverMap
		 * \param world2camera 刚性对齐
		 * \param InitialCameraSE3 初始化位姿
		 */
		void SetInputs(DeviceArrayView<DualQuaternion> nodeSe3, DeviceArray2D<KNNAndWeight>* knnMap, cudaTextureObject_t* vertexMap, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, DeviceArrayView2D<ushort4>* corrMap, Renderer::SolverMaps* solverMaps, mat34* world2camera, mat34* InitialCameraSE3);

		/**
		 * \brief 更新nodeSE3.
		 * 
		 * \param nodeSe3 节点Se3
		 */
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> nodeSe3);

	private:
		DeviceBufferArray<unsigned int> validCrossCorrIndicator;		// 标识indexMap有效的Pairs
		PrefixSum validCorrPrefixSum;
		DeviceBufferArray<CrossViewCorrPairs> GeometryCrossCorrPairs;	// 在Geometry上的跨镜匹配点
		DeviceBufferArray<CrossViewCorrPairs> ObservedCrossPairs;		// 在观察帧上有效的跨镜匹配点

	public:
		void ChooseValidCrossCorrPairs(cudaStream_t stream = 0);


	private:
		DeviceBufferArray<float4> validTargetVertex;					// 当前帧vertex
		DeviceBufferArray<float4> validCanonicalVertex;					// 与当前帧vertex对应的canonical域中的vertex
		DeviceBufferArray<ushort4> validVertexKnn;						// 跨镜匹配点的Knn
		DeviceBufferArray<float4> validKnnWeight;						// 跨镜匹配点的KnnWeight
	public:
		void CompactCrossViewCorrPairs(cudaStream_t stream = 0);		// 压缩跨镜匹配点
		void QueryCompactedCrossViewArraySize(cudaStream_t stream = 0);	// 调整数组的ArraySize

		DeviceArrayView<float4> GetTargetVertex() { return validTargetVertex.ArrayView(); }
		DeviceArrayView<float4> GetCanVertex() { return validCanonicalVertex.ArrayView(); }

	private:
		DeviceBufferArray<float4> validWarpedVertex;					// 上一帧live域有效的vertex
		void forwardWarpFeatureVertex(cudaStream_t stream = 0);			// 获得上一帧Live域的vertex
	public:
		void BuildTerm2Jacobian(cudaStream_t stream = 0);				// 构建Jacobian项

	public:
		Point2PointICPTerm2Jacobian Term2JacobianMap() const;
		DeviceArrayView<ushort4> CrossViewCorrKNN() const { return validVertexKnn.ArrayView(); }
	};
}


