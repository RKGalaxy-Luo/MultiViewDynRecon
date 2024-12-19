//
// Created by wei on 5/4/18.
//

#pragma once

#include <core/AlgorithmTypes.h>
#include <base/CameraObservation.h>
#include <core/NonRigidSolver/WarpField.h>
#include <core/Geometry/fusion_types.h>
#include <memory>
#include <core/Geometry/KNNSearch.h>
#include<visualization/Visualizer.h>

namespace SparseSurfelFusion {
	namespace device {
		struct AppendSurfelInput {
			cudaTextureObject_t vertex_confid_map[MAX_CAMERA_COUNT];
			cudaTextureObject_t normal_radius_map[MAX_CAMERA_COUNT];
			cudaTextureObject_t color_time_map[MAX_CAMERA_COUNT];
			DeviceArrayView2D<float4> inter_vertex_map[MAX_CAMERA_COUNT];
			DeviceArrayView2D<float4> inter_normal_map[MAX_CAMERA_COUNT];
			DeviceArrayView2D<float4> inter_color_map[MAX_CAMERA_COUNT];
			mat34 m_camera2world[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		};
	}

	
	class AppendSurfelProcessor {
	private:
		device::AppendSurfelInput m_observation;
		
		//The input from warp field
		WarpField::LiveGeometryUpdaterInput m_warpfield_input;
		KNNSearch::Ptr m_live_node_skinner;
		
		//The input indicator for pixel or binary indicator?
		DeviceArrayView<ushort4> m_surfel_candidate_pixel;	// 标记候选面元的Map坐标，(x, y)是坐标, z是视角来源, w标记是否是插值, 0是原始值, 1是插值
		
		unsigned int devicesCount = MAX_CAMERA_COUNT;

	public:
		using Ptr = std::shared_ptr<AppendSurfelProcessor>;
		AppendSurfelProcessor();
		~AppendSurfelProcessor();
		NO_COPY_ASSIGN_MOVE(AppendSurfelProcessor);
		
		//The input interface
		void SetInputs(
			const CameraObservation& observation,
			const mat34* world2camera,
			const WarpField::LiveGeometryUpdaterInput& warpfield_input,
			const KNNSearch::Ptr& live_node_skinner,
			const DeviceArrayView<ushort4>& pixel_coordinate
		);


		/* The surfel used for compute finite difference. This
		 * version use only xyz component
		 */
	private:
		DeviceBufferArray<float4> m_surfel_vertex_confid;
		DeviceBufferArray<float4> m_surfel_normal_radius;
		DeviceBufferArray<float4> m_surfel_color_time;
		DeviceBufferArray<float4> m_candidate_vertex_finite_diff;
		static constexpr const int kNumFiniteDiffVertex = 4;
		static constexpr const float kFiniteDiffStep = 5e-3f; // 5 [mm]

	public:
		void BuildSurfelAndFiniteDiffVertex(cudaStream_t stream = 0);
		
		
		/* Perform skinning of the vertex using live vertex
		 */
	private:
		DeviceBufferArray<ushort4> m_candidate_vertex_finitediff_knn;
		DeviceBufferArray<float4> m_candidate_vertex_finitediff_knnweight;
	public:
		void SkinningFiniteDifferenceVertex(cudaStream_t stream = 0);
		
		
		/* The buffer and method to perform filtering
		 */
	private:
		DeviceBufferArray<unsigned> m_candidate_surfel_validity_indicator;
		DeviceBufferArray<ushort4> m_surfel_knn;
		DeviceBufferArray<float4> m_surfel_knn_weight;
		
		//Do a prefix sum for the validity indicator
		PrefixSum m_candidate_surfel_validity_prefixsum;
	public:
		void FilterCandidateSurfels(cudaStream_t stream = 0);
		
		
		/* The accessing interface
		 */
		AppendedObservationSurfelKNN GetAppendedObservationSurfel() const;
	};
	
	
}
