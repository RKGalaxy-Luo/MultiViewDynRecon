#pragma once

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

#include "solver_constants.h"

#include <visualization/Visualizer.h>

namespace SparseSurfelFusion {
	namespace device {
		struct ObservedSparseCorrespondenceInterface {
			DeviceArrayView<ushort4> correspondPixelPairs[MAX_CAMERA_COUNT];
			cudaTextureObject_t edgeMask[MAX_CAMERA_COUNT];
			cudaTextureObject_t depthVertexMap[MAX_CAMERA_COUNT];
		};

		struct GeometryMapSparseCorrespondenceInterface {
			cudaTextureObject_t referenceVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t liveVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t indexMap[MAX_CAMERA_COUNT];
			DeviceArrayView2D<KNNAndWeight> knnMap[MAX_CAMERA_COUNT];
			mat34 camera2World[MAX_CAMERA_COUNT];
			mat34 initialCameraSE3[MAX_CAMERA_COUNT];
		};
	};
	class SparseCorrespondenceHandler {
	private:

		//The input from warp field
		DeviceArrayView<DualQuaternion> m_node_se3;

		unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int knnMapCols = CLIP_WIDTH;
		const unsigned int knnMapRows = CLIP_HEIGHT;
		unsigned int CorrPairsOffsetArray[MAX_CAMERA_COUNT + 1];

		device::ObservedSparseCorrespondenceInterface observedSparseCorrInterface;
		device::GeometryMapSparseCorrespondenceInterface geometrySparseCorrInterface;

	public:
		using Ptr = std::shared_ptr<SparseCorrespondenceHandler>;
		DEFAULT_CONSTRUCT_DESTRUCT(SparseCorrespondenceHandler);
		NO_COPY_ASSIGN_MOVE(SparseCorrespondenceHandler);

		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();

		unsigned int frameIdx = 0;

		//The processing interface
		void SetInputs(
			DeviceArrayView<DualQuaternion> node_se3,
			DeviceArray2D<KNNAndWeight> *knn_map,
			cudaTextureObject_t *depth_vertex_map,
			cudaTextureObject_t* edgeMaskMap,
			DeviceArrayView<ushort4> *correspond_pixel_pairs,
			//The rendered maps
			Renderer::SolverMaps * solvermaps,
			/*cudaTextureObject_t *reference_vertex_map,
			cudaTextureObject_t *index_map,*/
			mat34* world2camera,
			mat34* InitialCameraSE3
		);

		//Update the node se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);


		/* Some pixels from the correspondence finder might
		 * not has a reference vertex on that pixel
		 */
	private:
		DeviceBufferArray<unsigned> m_valid_pixel_indicator;
		PrefixSum m_valid_pixel_prefixsum;
		DeviceBufferArray<ushort4> m_corrected_pixel_pairs;
	public:
		void ChooseValidPixelPairs(cudaStream_t stream);


		/* Collect the valid depth/reference vertex and their knn
		 */
	private:
		DeviceBufferArray<float4> m_valid_target_vertex;		// 当前帧vertex
		DeviceBufferArray<float4> m_valid_reference_vertex;		// 与当前帧vertex对应的canonical域中的vertex
		DeviceBufferArray<ushort4> m_valid_vertex_knn;
		DeviceBufferArray<float4> m_valid_knn_weight;
		DeviceBufferArray<unsigned int> differentViewsCorrPairsOffset;

		//The page-locked memory
		unsigned* m_correspondence_array_size;
	public:
		void CompactQueryPixelPairs(cudaStream_t stream);
		void QueryCompactedArraySize(cudaStream_t stream = 0);
		void BuildCorrespondVertexKNN(cudaStream_t stream = 0);//4.20 这函数没用，暂时里边注释了

		DeviceArrayView<float4> GetTargetVertex() { return m_valid_target_vertex.ArrayView(); }
		DeviceArrayView<float4> GetCanVertex() { return m_valid_reference_vertex.ArrayView(); }

	private:
		DeviceBufferArray<float4> m_valid_warped_vertex;	// 将reference的点，左乘以上一帧SE3，得到warpVertex(即上一帧Live)
		void forwardWarpFeatureVertex(cudaStream_t stream = 0);
	public:
		void BuildTerm2Jacobian(cudaStream_t stream = 0);

	public:
		Point2PointICPTerm2Jacobian Term2JacobianMap() const;
		DeviceArrayView<ushort4> SparseFeatureKNN() const { return m_valid_vertex_knn.ArrayView(); }
	};

}