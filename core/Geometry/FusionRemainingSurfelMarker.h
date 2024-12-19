#pragma once

#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceArrayHandle.h>
#include <base/CommonUtils.h>
#include <core/AlgorithmTypes.h>
#include <math/MatUtils.h>
#include <render/Renderer.h>
#include <core/Geometry/fusion_types.h>


namespace SparseSurfelFusion {
	
	
	class FusionRemainingSurfelMarker {
	private:
		//The rendered fusion maps
		struct {
			cudaTextureObject_t vertex_confid_map[MAX_CAMERA_COUNT];
			cudaTextureObject_t normal_radius_map[MAX_CAMERA_COUNT];
			cudaTextureObject_t index_map[MAX_CAMERA_COUNT];
			cudaTextureObject_t color_time_map[MAX_CAMERA_COUNT];
		} m_fusion_maps;
		
		//The geometry model input
		struct {
			DeviceArrayView<float4> vertex_confid;
			DeviceArrayView<float4> normal_radius;
			DeviceArrayView<float4> color_time;
		} m_live_geometry;

		//The remainin surfel indicator from the fuser
		DeviceArrayHandle<unsigned> m_remaining_surfel_indicator;
		//DeviceArrayHandle<unsigned> m_remaining_surfel_indicator_temp[MAX_CAMERA_COUNT];
		//the camera and time information
		mat34 m_world2camera[MAX_CAMERA_COUNT];
		float m_current_time;

		//The global information
		Intrinsic m_intrinsic[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		const unsigned int devicesCount = MAX_CAMERA_COUNT;
	public:
		using Ptr = std::shared_ptr<FusionRemainingSurfelMarker>;
		FusionRemainingSurfelMarker(Intrinsic * rgbintrinsicclip);//需要传递内参
		~FusionRemainingSurfelMarker() = default;
		NO_COPY_ASSIGN_MOVE(FusionRemainingSurfelMarker);

		void SetInputs(const Renderer::FusionMaps* maps,
			const mat34* world2camera
		);
		void SetInputsNoNeedfor(
			const SurfelGeometry::SurfelFusionInput& geometry,
			float current_time,
			const DeviceArrayHandle<unsigned>& remaining_surfel_indicator
		);
	
		//The processing interface
		void UpdateRemainingSurfelIndicator(cudaStream_t stream = 0);
		DeviceArrayView<unsigned> GetRemainingSurfelIndicator() const { return m_remaining_surfel_indicator.ArrayView(); }
		
	private:
		PrefixSum m_remaining_indicator_prefixsum;
	public:
		void RemainingSurfelIndicatorPrefixSum(cudaStream_t stream = 0);
		DeviceArrayView<unsigned> GetRemainingSurfelIndicatorPrefixsum() const;
	};
	
	
}
