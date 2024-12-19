#pragma once

#include <base/CommonUtils.h>
#include <visualization/Visualizer.h>
#include "SurfelGeometry.h"
#include <core/NonRigidSolver/WarpField.h>
#include <memory>


namespace SparseSurfelFusion {
	namespace device {
		struct SurfelGeometryInterface {
			float4* liveVertexArray[MAX_CAMERA_COUNT];
			float4* liveNormalArray[MAX_CAMERA_COUNT];
			float4* referenceVertexArray[MAX_CAMERA_COUNT];
			float4* referenceNormalArray[MAX_CAMERA_COUNT];
		};

	}
	/* The deformer will perform forward and inverse
	 * warping given the geometry and warp field. It has
	 * full accessed to the SurfelGeometry instance.
	 */
	class SurfelNodeDeformer {
	public:
		//The processing of forward warp, may
		//use a node se3 different from the on in warp field
		static void ForwardWarpSurfelsAndNodes(
			bool showRender,
			WarpField& warp_field,
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], 
			const unsigned int devicesCount,
			const unsigned int updatedGeometryIndex,
			const DeviceArrayView<DualQuaternion>& node_se3,
			cudaStream_t stream = 0
		);
		static void ForwardWarpSurfelsAndNodes(
			bool showRender,
			WarpField& warp_field,
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
			const unsigned int devicesCount,
			const unsigned int updatedGeometryIndex,
			cudaStream_t stream = 0
		);


		//The processing interface, may use a node se3
		//different from the one in warp field
		static void InverseWarpSurfels(
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
			const unsigned int devicesCount,
			const unsigned int updatedGeometryIndex,
			const DeviceArrayView<DualQuaternion>& node_se3,
			const DeviceArrayView<mat34>& correct_se3,
			cudaStream_t stream = 0
		);
		static void InverseWarpSurfels(
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
			const unsigned int devicesCount,
			const unsigned int updatedGeometryIndex,
			const DeviceArrayView<DualQuaternion>& node_se3,
			cudaStream_t stream = 0
		);

		static void InverseWarpSurfels(
			const WarpField& warp_field,
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
			const unsigned int devicesCount,
			const unsigned int updatedGeometryIndex,
			const DeviceArrayView<DualQuaternion>& node_se3,
			const DeviceArrayView<mat34>& correct_se3,
			cudaStream_t stream = 0
		);
		static void InverseWarpSurfels(
			const WarpField& warp_field,
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
			const unsigned int devicesCount,
			const unsigned int updatedGeometryIndex,
			const DeviceArrayView<DualQuaternion>& node_se3,
			cudaStream_t stream = 0
		);
		static void InverseWarpSurfels(
			const WarpField& warp_field,
			SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
			cudaStream_t stream = 0
		);

		//Check the size of the geometry
		static void CheckSurfelGeometySize(const SurfelGeometry& geometry);
	};

}
