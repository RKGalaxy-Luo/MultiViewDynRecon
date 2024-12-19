#include <base/GlobalConfigs.h>
#include "FusionRemainingSurfelMarker.h"
#include "surfel_format.h"

#include <device_launch_parameters.h>
#include <base/data_transfer.h>

namespace SparseSurfelFusion { 
	namespace device {

		struct RemainingSurfelMarkerDevice {
			// Some constants defined as enum
			enum {
				scale_factor = d_fusion_map_scale,
				window_halfsize = scale_factor * 2,
				front_threshold = scale_factor * scale_factor * 3,
				surfel_keep_exist = REINITIALIZATION_TIME	// ��Ԫ������������ʱ��
			};

			//The rendered fusion maps
			struct {
				cudaTextureObject_t vertexConfidenceMap;
				cudaTextureObject_t normalRadiusMap;
				cudaTextureObject_t indexMap;
				cudaTextureObject_t colorTimeMap;
			} fusionMaps;
		
			//The geometry model input
			struct {
				DeviceArrayView<float4> vertexConfidence;
				const float4* normalRadius;
				const float4* colorTime;
			} liveGeometry;

			/**
			 * mutable �ؼ��������������ĳ����Ա������ const ��Ա�����б��޸ġ�
			 * ͨ����const ��Ա���������޸���ĳ�Ա�����������ĳ����Ա����������Ϊ mutable��
			 * ������� const ��Ա�������޸���.
			 */
			mutable unsigned* remainingSurfel;	// ���fuser�б�������Ԫ��ÿһ֡�����������ڴ�����

			//the camera and time information
			mat34 world2camera;
			float currentTime;
			mat34 InitialCameraSE3Inverse;

			//The global information
			Intrinsic intrinsic;

			__device__ __forceinline__ void processMarking() const {
				const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
				if (idx >= liveGeometry.vertexConfidence.Size()) return;
				// ע������Ķ���ģ�ͣ��Ǳ����Ķ�����Ԫ+�ںϵĶ�����Ԫ
				const float4 surfel_vertex_confid = liveGeometry.vertexConfidence[idx];	// Live��ĳ�����Ԫ
				const float4 surfel_normal_radius = liveGeometry.normalRadius[idx];	// Live��ĳ�����Ԫ����
				const float4 surfel_color_time = liveGeometry.colorTime[idx];			// Live��ĳ�����Ԫ��ɫ
				// ��Live��ĵ���ת����0���������ϵ���ٽ���ת������ǰCamera������ϵ
				float3 vertexView = InitialCameraSE3Inverse.rot * surfel_vertex_confid + InitialCameraSE3Inverse.trans;
				float3 vertex = world2camera.rot * vertexView + world2camera.trans;


				// �������ͶӰ����ǰ�������������ϵ
				const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
				const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);

				// ����Live���RenderMap������Щ���λ��
				const int map_x_center = scale_factor * x; // ��Щ�����surfel��չ��RenderMap�е�����λ�ã�4���ϲ�����
				const int map_y_center = scale_factor * y; // ��Щ�����surfel��չ��RenderMap�е�����λ�ã�4���ϲ�����
				int front_counter = 0;					   // RenderMap�б�ʵ��Surfel��Z�����Ͻ��ĵ�ĸ���

				// ����Щ��Ϊ���ı��������ڴ�СΪ16
				for (int map_y = map_y_center - window_halfsize; map_y < map_y_center + window_halfsize; map_y++) {
					for (int map_x = map_x_center - window_halfsize; map_x < map_x_center + window_halfsize; map_x++) {
						// ��Ӧ�������ϵ
						const float4 map_vertex_confid = tex2D<float4>(fusionMaps.vertexConfidenceMap, map_x, map_y);
						const float4 map_normal_radius = tex2D<float4>(fusionMaps.normalRadiusMap, map_x, map_y);
						const unsigned int index = tex2D<unsigned int>(fusionMaps.indexMap, map_x, map_y);
						if (index != 0xFFFFFFFF) {
							const float dot_value = dotxyz(surfel_normal_radius, map_normal_radius);
							const float3 diff_camera = world2camera.rot * (vertexView - map_vertex_confid);
							// Live��RenderMap�ϵ���Щ�� �� Live���д洢��ʵ�ʵ㣺�������С��37�㣬surfel��RenderMap�ϵĶ���z������Զ[0mm, 3mm]
							if (diff_camera.z >= 0 && diff_camera.z <= 3e-2f && dot_value >= 0.8f) {
								front_counter++;
							}
						}
					}
				}

				// ��Ǹ�Surfel�Ƿ�Ӧ������
				unsigned int keep_indicator = 1;

				// ���Surfel����RenderMap�϶�Ӧ�����еĵ㣬��Surfel�����Զ����ͳ����Щ���������Զ�ĵ㳬����ֵ
				if (front_counter > front_threshold) keep_indicator = 0;

				// �����Ԫ���Ŷ�С��10 ���� ��Ԫ���ڵ�ʱ�䳬��30֡
				if (surfel_vertex_confid.w < 10.0f && (currentTime - initialization_time(surfel_color_time)) > surfel_keep_exist) keep_indicator = 0;

				// ��������������������������δ����������¼��Щ��
				if (keep_indicator == 1 && remainingSurfel[idx] == 0) {
					remainingSurfel[idx] = 1;
				}
			}
		};

		__global__ void markRemainingSurfelKernel(const RemainingSurfelMarkerDevice marker) {
			marker.processMarking();
		}
	}
}


void SparseSurfelFusion::FusionRemainingSurfelMarker::UpdateRemainingSurfelIndicator(cudaStream_t stream) {
	//Construct the marker
	device::RemainingSurfelMarkerDevice marker[MAX_CAMERA_COUNT];

	for (int i = 0; i < devicesCount; i++) {
		marker[i].fusionMaps.vertexConfidenceMap = m_fusion_maps.vertex_confid_map[i];
		marker[i].fusionMaps.normalRadiusMap = m_fusion_maps.normal_radius_map[i];
		marker[i].fusionMaps.indexMap = m_fusion_maps.index_map[i];
		marker[i].fusionMaps.colorTimeMap = m_fusion_maps.color_time_map[i];
		
		marker[i].liveGeometry.vertexConfidence = m_live_geometry.vertex_confid;
		marker[i].liveGeometry.normalRadius = m_live_geometry.normal_radius;
		marker[i].liveGeometry.colorTime = m_live_geometry.color_time;
		
		marker[i].remainingSurfel = m_remaining_surfel_indicator.RawPtr();
		marker[i].world2camera = m_world2camera[i];
		marker[i].currentTime = m_current_time;
		marker[i].intrinsic = m_intrinsic[i];
		marker[i].InitialCameraSE3Inverse = InitialCameraSE3[i].inverse();

		dim3 block(256);
		dim3 grid(divUp(m_live_geometry.vertex_confid.Size(), block.x));
		device::markRemainingSurfelKernel << <grid, block, 0, stream >> > (marker[i]);
	}
	CHECKCUDA(cudaStreamSynchronize(stream));

#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::FusionRemainingSurfelMarker::RemainingSurfelIndicatorPrefixSum(cudaStream_t stream) {
	m_remaining_indicator_prefixsum.InclusiveSum(m_remaining_surfel_indicator.ArrayView(), stream);
}

SparseSurfelFusion::DeviceArrayView<unsigned int> SparseSurfelFusion::FusionRemainingSurfelMarker::GetRemainingSurfelIndicatorPrefixsum() const {
	const DeviceArray<unsigned int>& prefixsum_array = m_remaining_indicator_prefixsum.valid_prefixsum_array;
	FUNCTION_CHECK(m_remaining_surfel_indicator.Size() == prefixsum_array.size());
	return DeviceArrayView<unsigned>(prefixsum_array);
}

