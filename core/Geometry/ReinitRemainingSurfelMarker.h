#pragma once

#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceArrayHandle.h>
#include <base/CommonUtils.h>
#include <core/AlgorithmTypes.h>
#include <base/CameraObservation.h>
#include <math/MatUtils.h>
#include <render/Renderer.h>
#include <core/NonRigidSolver/solver_types.h>
#include <core/Geometry/fusion_types.h>

#include <memory>

namespace SparseSurfelFusion {
	namespace device {
		struct ReinitRemainingMarkerDevice {
			enum {
				window_halfsize = 2,
			};

			//The geometry model input
			struct {
				DeviceArrayView<float4> vertexConfidence;
				const float4* normalRadius;
				const float4* colorTime;
				const ushort4* surfelKnn;
			} liveGeometry;

			//The observation from camera
			struct {
				cudaTextureObject_t vertexMap;
				cudaTextureObject_t normalMap;
				cudaTextureObject_t foregroundMask;
			} cameraObservation[MAX_CAMERA_COUNT];

			//The information on camera
			mat34 InitialCameraSE3Inverse[MAX_CAMERA_COUNT];
			mat34 world2camera[MAX_CAMERA_COUNT];
			Intrinsic intrinsic[MAX_CAMERA_COUNT];

			__device__ __forceinline__ void processMarkingObservedOnly(const unsigned int remainingSurfelsCount, const unsigned int mapCols, const unsigned int mapRows, unsigned* remaining_indicator) const;

		};
		__global__ void markReinitRemainingSurfelObservedOnlyKernel(const ReinitRemainingMarkerDevice marker, const unsigned int remainingSurfelsCount, const unsigned int mapCols, const unsigned int mapRows, unsigned* remaining_indicator);
	}

	class ReinitRemainingSurfelMarker {
	private:
		//The input from outside
		Renderer::FusionMaps m_fusion_maps[MAX_CAMERA_COUNT];
		const unsigned int mapCols = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		const unsigned int mapRows = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int devicesCount = MAX_CAMERA_COUNT;

		//With the transform form observation to world
		CameraObservation m_observation;
		mat34 m_world2camera[MAX_CAMERA_COUNT];

		//The geometry as array
		SurfelGeometry::SurfelFusionInput m_surfel_geometry;

		//The fused remaining indicator, where 1 indicates the surfel is fused with some depth image
		DeviceArrayHandle<unsigned> m_remaining_surfel_indicator;

		//To project the surfel into image
		Intrinsic m_intrinsic[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];

	public:
		using Ptr = std::shared_ptr<ReinitRemainingSurfelMarker>;
		ReinitRemainingSurfelMarker(Intrinsic* clipedIntrinsicArray);
		~ReinitRemainingSurfelMarker() = default;
		NO_COPY_ASSIGN_MOVE(ReinitRemainingSurfelMarker);

		void SetInputs(
			const Renderer::FusionMaps* maps,
			const SurfelGeometry::SurfelFusionInput& geometry,
			const CameraObservation& observation,
			float current_time,
			const mat34* world2camera,
			const DeviceArrayHandle<unsigned>& remaining_surfel_indicator
		);


		//The processing interface
	private:
		void prepareMarkerArguments(device::ReinitRemainingMarkerDevice& marker);
		void prepareMarkerAndReplaceArguments(void* raw_marker, unsigned int cameraID);
	public:
		void MarkRemainingSurfelObservedOnly(cudaStream_t stream = 0);
		void MarkRemainingSurfelNodeError(const NodeAlignmentError& node_error, float threshold = 0.06f, cudaStream_t stream = 0);
		DeviceArrayHandle<unsigned> GetRemainingSurfelIndicator() { return m_remaining_surfel_indicator; }
	};

}