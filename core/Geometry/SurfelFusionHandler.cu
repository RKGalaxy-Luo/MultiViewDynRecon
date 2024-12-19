#include <base/GlobalConfigs.h>
#include "SurfelFusionHandler.h"
#include "surfel_fusion_dev.cuh"
#include <math/VectorUtils.h>
#include <math/MatUtils.h>
#include <base/data_transfer.h>

#include <device_launch_parameters.h>

namespace SparseSurfelFusion { 
	namespace device {

		struct FusionAndMarkAppendedObservationSurfelDevice {
			// Some constants defined as enum
			enum {
				scale_factor = d_fusion_map_scale,//4
				fuse_window_halfsize = scale_factor >> 1,//2
				count_model_halfsize = 2 * scale_factor /*>> 1 */,//8
				append_window_halfsize = scale_factor,//4
				search_window_halfsize = scale_factor,//4
				inter_surfel_threshold = 4,
			};


			//The observation
			struct {
				cudaTextureObject_t vertexTimeMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t normalRadiusMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t colorTimeMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t foregroundMask[MAX_CAMERA_COUNT];

				DeviceArrayView2D<unsigned char> interMarkValidMap[MAX_CAMERA_COUNT];
				DeviceArrayView2D<float4> interVertexMap[MAX_CAMERA_COUNT];
				DeviceArrayView2D<float4> interNormalMap[MAX_CAMERA_COUNT];
				DeviceArrayView2D<float4> interColorMap[MAX_CAMERA_COUNT];

			} observation_maps;

			struct {
				mat34 initialCameraSE3[MAX_CAMERA_COUNT];
				mat34 world2camera[MAX_CAMERA_COUNT];
			} basicInfo;

			//The rendered maps
			struct {
				cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t colorTimeMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t indexMap[MAX_CAMERA_COUNT];
			} render_maps;

			//The written array
			struct {
				float4* vertexConfidence;
				float4* normalRadius;
				float4* colorTime;
				unsigned* fusedIndicator;
			} geometry_arrays;

			//The shared datas
			unsigned short imageRows, imageCols;
			float currentTime;
		
			int CameraNum;	// ���������¼���ĸ��ӽǵ�����
		
			__host__ __device__ __forceinline__ bool checkViewDirection(const float4& depth_vertex, const float4& depth_normal, float threshold = 0.4f) const {
				const float3 view_direction = - normalized(make_float3(depth_vertex.x, depth_vertex.y, depth_vertex.z));
				const float3 normal = normalized(make_float3(depth_normal.x, depth_normal.y, depth_normal.z));
				return dot(view_direction, normal) > threshold;
			}
		

			//The actual processing interface
			__device__ __forceinline__ void processIndicator(unsigned* appendingIndicator) const {
				const int x = threadIdx.x + blockDim.x * blockIdx.x;
				const int y = threadIdx.y + blockDim.y * blockIdx.y;
				const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
				const int offset = y * imageCols + x;
				if (CameraID >= CameraNum) return;
				if (x < search_window_halfsize || x >= imageCols - search_window_halfsize || y < search_window_halfsize || y >= imageRows - search_window_halfsize) {
					//Write to indicator before exit
					appendingIndicator[offset] = 0;
					return;
				}
				mat34 world2camera = basicInfo.world2camera[CameraID];
				//Load the data
				const float4 depth_vertex_confid = tex2D<float4>(observation_maps.vertexTimeMap[CameraID], x, y);
				const float4 depth_normal_radius = tex2D<float4>(observation_maps.normalRadiusMap[CameraID], x, y);
				const float4 image_color_time = tex2D<float4>(observation_maps.colorTimeMap[CameraID], x, y);
				if (is_zero_vertex(depth_vertex_confid)) return;

				//The windows search state
				const int map_x_center = scale_factor * x;
				const int map_y_center = scale_factor * y;

				//The window search iteration variables
				SurfelFusionWindowState fusion_state;
				unsigned model_count = 0;
				SurfelAppendingWindowState append_state;

				//The row search loop
				for(int dy = - search_window_halfsize; dy < search_window_halfsize; dy++) {
					for(int dx = - search_window_halfsize; dx < search_window_halfsize; dx++) {
						//The actual position of in the rendered map
						const int map_y = dy + map_y_center;
						const int map_x = dx + map_x_center;

						const auto index = tex2D<unsigned>(render_maps.indexMap[CameraID], map_x, map_y);
						if (index != 0xFFFFFFFF) {
							//Load the model vertex
							const float4 model_world_v4 = tex2D<float4>(render_maps.vertexMap[CameraID], map_x, map_y);
							const float4 model_world_n4 = tex2D<float4>(render_maps.normalMap[CameraID], map_x, map_y);
						
							//Transform it
							const float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
							const float3 model_camera_n3 = world2camera.rot * model_world_n4;
						
							//Some attributes commonly used for checking
							const float dot_value = dotxyz(model_camera_n3, depth_normal_radius);
							const float diff_z = fabsf(model_camera_v3.z - depth_vertex_confid.z);
							const float confidence = model_world_v4.w;
							const float z_dist = model_camera_v3.z;
							const float dist_square = squared_distance(model_camera_v3, depth_vertex_confid);

							//First check for fusion
							if (dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize) {
								if(dot_value >= 0.8f && diff_z <= 3 * 0.001f) { // Update it
									fusion_state.Update(confidence, z_dist, map_x, map_y);
								}
							}

							//Next check for count the model
							if(dx >= -count_model_halfsize && dy >= -count_model_halfsize && dx < count_model_halfsize && dy < count_model_halfsize) {
								if (dot_value > 0.3f) model_count++;
							}

							//Finally for appending
							{
								//if(dot_value >= 0.8f && diff_z <= 3 * 0.001f) { // Update it
								//	append_state.Update(confidence, z_dist);
								//}
								if(dot_value >= 0.8f && dist_square <= (2 * 0.001f) * (2 * 0.001f)) { // Update it
									append_state.Update(confidence, z_dist);
								}
							}

						} // There is a surfel here
					} // x iteration loop
				} // y iteration loop
			
				//For appending, as in reinit should mark all depth surfels
				unsigned pixel_indicator = 0;
				if(append_state.best_confid < -0.01
				   && model_count == 0
				   && checkViewDirection(depth_vertex_confid, depth_normal_radius)
					) {
					pixel_indicator = 1;
				}
				appendingIndicator[offset] = pixel_indicator;

				//For fusion
				if(fusion_state.best_confid > 0) {
					float4 model_vertex_confid = tex2D<float4>(render_maps.vertexMap[CameraID], fusion_state.best_map_x, fusion_state.best_map_y);
					float4 model_normal_radius = tex2D<float4>(render_maps.normalMap[CameraID], fusion_state.best_map_x, fusion_state.best_map_y);
					float4 model_color_time = tex2D<float4>(render_maps.colorTimeMap[CameraID], fusion_state.best_map_x, fusion_state.best_map_y);
					const unsigned index = tex2D<unsigned>(render_maps.indexMap[CameraID], fusion_state.best_map_x, fusion_state.best_map_y);
					fuse_surfel(
						depth_vertex_confid, depth_normal_radius, image_color_time, 
						world2camera, currentTime,
						model_vertex_confid, model_normal_radius, model_color_time
					);

					//Write it
					geometry_arrays.vertexConfidence[index] = model_vertex_confid;
					geometry_arrays.normalRadius[index] = model_normal_radius;
					geometry_arrays.colorTime[index] = model_color_time;
					geometry_arrays.fusedIndicator[index] = 1;
				}
			}
			
			/**
			 * \brief ԭ�Ӳ�������ںϺ����Ԫ.
			 *
			 * \param world2camera ��������ϵת�������ϵ
			 * \param appendingOffset ԭ�Ӳ���ƫ���ۼӣ���Ϊindex�����鸳ֵʹ��
			 * \param appended_pixels ������ӵ���Ԫ�洢
			 */
			__device__ __forceinline__ void processAtomic(
				unsigned* appendingOffset, 
				//debug
				unsigned int* fusionDepthLiveSurfelNum,
				float4* fusionLiveSurfel,
				ushort4* appendedPixels
			) const {
				const int x = threadIdx.x + blockDim.x * blockIdx.x;
				const int y = threadIdx.y + blockDim.y * blockIdx.y;
				const int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
				if (CameraID >= CameraNum || x < search_window_halfsize || x >= imageCols - search_window_halfsize || y < search_window_halfsize || y >= imageRows - search_window_halfsize) return;

				mat34 InitialCameraSE3 = basicInfo.initialCameraSE3[CameraID];
				mat34 world2camera = basicInfo.world2camera[CameraID];

				// ���ص�ǰ�ӽ��µ�����
				const float4 observedVertexConfidence = tex2D<float4>(observation_maps.vertexTimeMap[CameraID], x, y);
				const float4 observedNormalRadius = tex2D<float4>(observation_maps.normalRadiusMap[CameraID], x, y);
				const float4 observedColorTime = tex2D<float4>(observation_maps.colorTimeMap[CameraID], x, y);
				// ���ص�ǰ�ӽ��²�ֵ����
				const unsigned char interValidValue = observation_maps.interMarkValidMap[CameraID](y, x);
				const float4 interVertexConfidence = observation_maps.interVertexMap[CameraID](y, x);
				const float4 interNormalRadius = observation_maps.interNormalMap[CameraID](y, x);
				const float4 interColorTime = observation_maps.interColorMap[CameraID](y, x);

				bool  CheckObservation = false, CheckInterpolation = false;	// check�Ƿ������ֵ��ֵ
				if (!is_zero_vertex(interVertexConfidence)) CheckInterpolation = true;
				// ԭʼ�۲�Ͳ�ֵ��û��ֵ(֮ǰ�Ѿ�����ֵ������۲�vertex�ľ������ˣ�ʣ�µľ���vertex��ֵ����ֵ�����������)
				if (!is_zero_vertex(observedVertexConfidence)) CheckObservation = true;

				if (!CheckObservation && !CheckInterpolation) return;

				const int map_x_center = scale_factor * x;
				const int map_y_center = scale_factor * y;

				//The window search iteration variables
				unsigned int model_count = 0;				// ͳ�� 4 �� 4 �������ж��ٸ���Ч��Ԫ
				unsigned int model_inter_count = 0;			// ͳ�� 4 �� 4 �������ж��ٸ������ֵ��Ԫ�ӽ�����Ԫ
				SurfelFusionWindowState fusionState;		// �� 2 �� 2 ������Ѱ��������۲���Ԫ����(�н� < 37��, z_dist_diff < 3cm)��Live��ĵ㣬����¼��map_x��map_y��best_confidence��z_dist
				SurfelFusionWindowState fusionInterState;	// ��ֵSurfel���ں�
				SurfelAppendingWindowState appendState;		// �� 4 �� 4 ������Ѱ��������۲���Ԫ����(�н� < 37��, points_dist < 3cm)��Live��ĵ㣬����¼��best_confidence��z_dist
				SurfelAppendingWindowState appendInterState;// ��ֵSurfel���
				// �������ڣ���[-4, 3]�������ص�
				for (int dy = -search_window_halfsize; dy < search_window_halfsize; dy++) {
					for (int dx = -search_window_halfsize; dx < search_window_halfsize; dx++) {
						//The actual position of in the rendered map
						const int map_y = dy + map_y_center;
						const int map_x = dx + map_x_center;

						const unsigned int index = tex2D<unsigned>(render_maps.indexMap[CameraID], map_x, map_y);
						if (index != 0xFFFFFFFF) {	// ����FusionMap�ϵ�Geometry����
							model_count++;	// ͳ��FusionMap��������Ч��Geometry Surfels

							const float4 model_world_v4 = tex2D<float4>(render_maps.vertexMap[CameraID], map_x, map_y);
							const float4 model_world_n4 = tex2D<float4>(render_maps.normalMap[CameraID], map_x, map_y);
						
							// ת�����������ϵ
							float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
							float3 model_camera_n3 = world2camera.rot * model_world_n4;

							// һЩͨ�����ڼ������ԣ����render�е�Map��ʵ�ʹ۲���Ԫ�����ƶ�
							float dot_value, diff_z, confidence, z_dist, dist_square;

							if (CheckObservation) {
								dot_value = dotxyz(model_camera_n3, observedNormalRadius);
								diff_z = fabsf(model_camera_v3.z - observedVertexConfidence.z);
								confidence = model_world_v4.w;
								z_dist = model_camera_v3.z;
								dist_square = squared_distance(model_camera_v3, observedVertexConfidence);
								// �ںϵĴ��ڰ뾶ֻ��2pix�������������������ں������Ż�����ں�
								if (dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize) {
									if (dot_value >= 0.8f && diff_z <= 1e-2f) { // ��1cm�� ԭʼ������0.8 0.003
										fusionState.Update(confidence, z_dist, map_x, map_y);
									}
								}

								// �۲�Surfel���
								{
									if (dot_value >= 0.8f && dist_square <= 9e-4f) { //  ��3cm�� ԭʼ������0.8 0.003 * 0.003
										appendState.Update(confidence, z_dist);
									}
								}
							}

							if (CheckInterpolation) {
								// ����һֱ��㣬�������Ƶľ��ں�
								if (dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize) {
									dot_value = dotxyz(model_camera_n3, interNormalRadius);
									diff_z = fabsf(model_camera_v3.z - interVertexConfidence.z);
									dist_square = squared_distance(model_camera_v3, interVertexConfidence);
									if (dot_value >= 0.8f && diff_z <= 3e-3f) { // ��1cm�� ԭʼ������0.8 0.003
										fusionInterState.Update(confidence, z_dist, map_x, map_y);
									}
								}
								// ��ֵSurfel���
								{
									if (dot_value >= 0.8f && diff_z <= 3e-3f) { //  ��3cm�� ԭʼ������0.8 0.003 * 0.003
										appendInterState.Update(confidence, z_dist);
									}
								}
							}
						}
					}
				}

				//For fusion
				if (CheckObservation && fusionState.best_confid > 0) {
					float4 model_vertex_confid = tex2D<float4>(render_maps.vertexMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					float4 model_normal_radius = tex2D<float4>(render_maps.normalMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					float4 model_color_time = tex2D<float4>(render_maps.colorTimeMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					const unsigned index = tex2D<unsigned>(render_maps.indexMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					fuse_surfel(
						observedVertexConfidence, observedNormalRadius, observedColorTime,
						world2camera, currentTime,
						model_vertex_confid, model_normal_radius, model_color_time
					);

					float3 model_vertex_confid_0 = InitialCameraSE3.rot * model_vertex_confid + InitialCameraSE3.trans;
					float3 model_normal_radius_0 = InitialCameraSE3.rot * model_normal_radius;
					// ������������������ϵ
					geometry_arrays.vertexConfidence[index] = make_float4(model_vertex_confid_0.x, model_vertex_confid_0.y, model_vertex_confid_0.z, model_vertex_confid.w);
					geometry_arrays.normalRadius[index] = make_float4(model_normal_radius_0.x, model_normal_radius_0.y, model_normal_radius_0.z, model_normal_radius.w);
					geometry_arrays.colorTime[index] = model_color_time;
					geometry_arrays.fusedIndicator[index] = 1;
					//debug
					const unsigned int offset = atomicAdd(fusionDepthLiveSurfelNum, 1);
					fusionLiveSurfel[offset] = geometry_arrays.vertexConfidence[index];
				}

				// �����ֵSurfel���ں�
				if (CheckInterpolation && fusionInterState.best_confid > 0) {
					float4 model_vertex_confid = tex2D<float4>(render_maps.vertexMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					float4 model_normal_radius = tex2D<float4>(render_maps.normalMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					float4 model_color_time = tex2D<float4>(render_maps.colorTimeMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					const unsigned index = tex2D<unsigned>(render_maps.indexMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					fuse_surfel(
						interVertexConfidence, interNormalRadius, interColorTime,
						world2camera, currentTime,
						model_vertex_confid, model_normal_radius, model_color_time
					);

					float3 model_vertex_confid_0 = InitialCameraSE3.rot * model_vertex_confid + InitialCameraSE3.trans;
					float3 model_normal_radius_0 = InitialCameraSE3.rot * model_normal_radius;
					// ������������������ϵ
					geometry_arrays.vertexConfidence[index] = make_float4(model_vertex_confid_0.x, model_vertex_confid_0.y, model_vertex_confid_0.z, model_vertex_confid.w);
					geometry_arrays.normalRadius[index] = make_float4(model_normal_radius_0.x, model_normal_radius_0.y, model_normal_radius_0.z, model_normal_radius.w);
					geometry_arrays.colorTime[index] = model_color_time;
					geometry_arrays.fusedIndicator[index] = 1;
					//debug
					const unsigned int offset = atomicAdd(fusionDepthLiveSurfelNum, 1);
					fusionLiveSurfel[offset] = geometry_arrays.vertexConfidence[index];
				}

				//Check the view direction, and using atomic operation for appending
				// ˵�������Live���Ϊ���� 4 �� 4 ���������㣺
				// 1�����������еĵ㲻����(�н�С��37�� && �ռ���� <= 3cm)   
				// 2��IndexMap�������һ����Ч�ĵ㶼û��   
				// ����۲쵽����ȳ��ܵ����㣺3��������������Ԫ����:���������ϵԭ��O��������ȶ���P������Ϊn������н�<PO, n> < 66������
				if (CheckObservation && appendState.best_confid < -0.01 && model_count == 0 && checkViewDirection(observedVertexConfidence, observedNormalRadius)) {
					const unsigned char mask_value = tex2D<unsigned char>(observation_maps.foregroundMask[CameraID], x, y);
					bool checkMask = (mask_value == (unsigned char)1);
#ifndef REBUILD_WITHOUT_BACKGROUND
					checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

					if (checkMask) {
						// �����appendingOffset���ֵ���س�����Ȼ���ټ�1
						const unsigned int offset = atomicAdd(appendingOffset, 1);
						appendedPixels[offset] = make_ushort4(x, y, CameraID, 0);
					}
				}
				//if (model_inter_count != 0) printf("model_inter_count = %d\n", model_inter_count);

				// �����û�б��ںϣ�������ֻҪ�ҵ����ĵ㼸�����ᱻ�ںϣ�����Ҫ��֤����û��Geometry��Surfel
				if (CheckInterpolation/* && appendInterState.best_confid < -0.01&& checkViewDirection(interVertexConfidence, interNormalRadius) */) {
					const unsigned char mask_value = tex2D<unsigned char>(observation_maps.foregroundMask[CameraID], x, y);
					bool checkMask = (mask_value == (unsigned char)1);
#ifndef REBUILD_WITHOUT_BACKGROUND
					checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

					if (checkMask) {
						// ��ֵ����Ҳ���뵽�����������
						const unsigned int offset = atomicAdd(appendingOffset, 1);
						appendedPixels[offset] = make_ushort4(x, y, CameraID, 1);
					}
				}
			}


			//The fusion processor for re-initialize
			__device__ __forceinline__ void processFusionReinit(unsigned int* FusedSurfelNum, unsigned int* AppendedSurfelNum, unsigned int* appendingIndicator) const {
				const int x = threadIdx.x + blockDim.x * blockIdx.x;
				const int y = threadIdx.y + blockDim.y * blockIdx.y;
				const int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
				const unsigned int offset = y * imageCols + x;
				const unsigned int clipedImageSize = imageCols * imageRows;
				if (CameraID >= CameraNum) return;
				if (x < search_window_halfsize || x >= imageCols - search_window_halfsize || y < search_window_halfsize || y >= imageRows - search_window_halfsize) {
					//Write to indicator before exit
					appendingIndicator[CameraID * clipedImageSize + offset] = 0;
					appendingIndicator[(CameraNum + CameraID) * clipedImageSize + offset] = 0;
					return;
				}

				mat34 world2camera = basicInfo.world2camera[CameraID];
				mat34 InitialCameraSE3 = basicInfo.initialCameraSE3[CameraID];

				bool CheckObservation = false, CheckInterpolation = false;

				// ��Ϊ��������������������Լ��������ϵ�µ�, FusionMapҲ���ڶ�Ӧ�ӽ��µ����ݣ��������ﲻ��ҪSE3
				const float4 observedVertexConfidence = tex2D<float4>(observation_maps.vertexTimeMap[CameraID], x, y);
				const float4 observedNormalRadius = tex2D<float4>(observation_maps.normalRadiusMap[CameraID], x, y);
				const float4 observedColorTime = tex2D<float4>(observation_maps.colorTimeMap[CameraID], x, y);

				// ���ص�ǰ�ӽ��²�ֵ����
				const unsigned char interValidValue = observation_maps.interMarkValidMap[CameraID](y, x);
				const float4 interVertexConfidence = observation_maps.interVertexMap[CameraID](y, x);
				const float4 interNormalRadius = observation_maps.interNormalMap[CameraID](y, x);
				const float4 interColorTime = observation_maps.interColorMap[CameraID](y, x);

				if (!is_zero_vertex(observedVertexConfidence)) CheckObservation = true;
				else appendingIndicator[CameraID * clipedImageSize + offset] = 0;

				if (!is_zero_vertex(interVertexConfidence)) CheckInterpolation = true;
				else appendingIndicator[(CameraNum + CameraID) * clipedImageSize + offset] = 0;


				if (!CheckObservation && !CheckInterpolation) return;

				//The windows search state
				const int map_x_center = scale_factor * x;
				const int map_y_center = scale_factor * y;

				//The window search iteration variables
				SurfelFusionWindowState fusionState;
				SurfelFusionWindowState fusionInterState;

				//The row search loop
				for (int dy = -fuse_window_halfsize; dy < fuse_window_halfsize; dy++) {
					for (int dx = -fuse_window_halfsize; dx < fuse_window_halfsize; dx++) {
						//The actual position of in the rendered map
						const int map_y = dy + map_y_center;
						const int map_x = dx + map_x_center;

						const unsigned int index = tex2D<unsigned>(render_maps.indexMap[CameraID], map_x, map_y);
						if (index != 0xFFFFFFFF) {

							//Load the model vertex (0)
							const float4 model_world_v4 = tex2D<float4>(render_maps.vertexMap[CameraID], map_x, map_y);
							const float4 model_world_n4 = tex2D<float4>(render_maps.normalMap[CameraID], map_x, map_y);
						
							// �任����ǰ�������ϵ����۲�����ݽ��ж���
							float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
							float3 model_camera_n3 = world2camera.rot * model_world_n4;

							//Some attributes commonly used for checking
							float dot_value, diff_z, confidence, z_dist;
							if (CheckObservation) {
								 dot_value = dotxyz(model_camera_n3, observedNormalRadius);
								 diff_z = fabsf(model_camera_v3.z - observedVertexConfidence.z);
								 confidence = model_world_v4.w;
								 z_dist = model_camera_v3.z;
								 //First check for fusion
								 if (dot_value >= 0.9f && diff_z <= 2e-3f) { // 0.9f    0.002f
									 fusionState.Update(confidence, z_dist, map_x, map_y);
								 }
							}

							if (CheckInterpolation) {
								// ��ֵSurfels
								dot_value = dotxyz(model_camera_n3, interVertexConfidence);
								diff_z = fabsf(model_camera_v3.z - interVertexConfidence.z);
								if (dot_value >= 0.9f && diff_z <= 2e-3f) {
									fusionInterState.Update(confidence, z_dist, map_x, map_y);
								}
							}
						} // There is a surfel here
					} // x iteration loop
				} // y iteration loop
			
				//For appending, as in reinit should mark all depth surfels
				//�������˼�ǣ�û���ںϵ���Ԫ����Ҫֱ����ӽ�����
				unsigned int pixelIndicator = 0;
				if (CheckObservation && fusionState.best_confid < -1e-2f) {
					unsigned char mask_value = tex2D<unsigned char>(observation_maps.foregroundMask[CameraID], x, y);
					bool checkMask = (mask_value == 1);
#ifndef REBUILD_WITHOUT_BACKGROUND
					checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

					if (checkMask) {
						pixelIndicator = 1;
						atomicAdd(AppendedSurfelNum, 1);
					}
				}
				appendingIndicator[CameraID * clipedImageSize + offset] = pixelIndicator;

				unsigned int interPixelIndicator = 0;
				if (CheckInterpolation && fusionInterState.best_confid < -1e-2f) {	// ����ǲ�ֵ�����أ����������ֵ������Χ�Ǵ���Geometry��Surfels��
					unsigned char mask_value = tex2D<unsigned char>(observation_maps.foregroundMask[CameraID], x, y);
					bool checkMask = (mask_value == 1);
#ifndef REBUILD_WITHOUT_BACKGROUND
					checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

					if (checkMask) {
						interPixelIndicator = 1;
						atomicAdd(AppendedSurfelNum, 1);
					}
				}
				// �浽ͬһ������
				appendingIndicator[(CameraNum + CameraID) * clipedImageSize + offset] = interPixelIndicator;


				// �ں�
				if (CheckObservation && fusionState.best_confid > 0) {
					atomicAdd(FusedSurfelNum, 1);
					// Live���Ӧ�ӽ�����ϵ
					float4 model_vertex_confid = tex2D<float4>(render_maps.vertexMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					float4 model_normal_radius = tex2D<float4>(render_maps.normalMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					float4 model_color_time = tex2D<float4>(render_maps.colorTimeMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					const unsigned int index = tex2D<unsigned>(render_maps.indexMap[CameraID], fusionState.best_map_x, fusionState.best_map_y);
					fuse_surfel_replace_color(
						observedVertexConfidence, observedNormalRadius, observedColorTime,
						world2camera, currentTime,
						model_vertex_confid, model_normal_radius, model_color_time
					);

					float3 model_vertex_confid_0 = InitialCameraSE3.rot * model_vertex_confid + InitialCameraSE3.trans;
					float3 model_normal_radius_0 = InitialCameraSE3.rot * model_normal_radius;

					//Write it
					geometry_arrays.vertexConfidence[index] = make_float4(model_vertex_confid_0.x, model_vertex_confid_0.y, model_vertex_confid_0.z, model_vertex_confid.w);;
					geometry_arrays.normalRadius[index] = make_float4(model_normal_radius_0.x, model_normal_radius_0.y, model_normal_radius_0.z, model_normal_radius.w);;
					geometry_arrays.colorTime[index] = model_color_time;
					//����󶨵ľ����Ǹ�remainingbuffer�����buffer�Ĵ�С�Ǻ���һ֡��live�вο���Ԫ�ĸ���
					geometry_arrays.fusedIndicator[index] = 1;
				}

				// �ںϲ�ֵSurfels
				if (CheckInterpolation && fusionInterState.best_confid > 0) {
					atomicAdd(FusedSurfelNum, 1);
					// Live���Ӧ�ӽ�����ϵ
					float4 model_vertex_confid = tex2D<float4>(render_maps.vertexMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					float4 model_normal_radius = tex2D<float4>(render_maps.normalMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					float4 model_color_time = tex2D<float4>(render_maps.colorTimeMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					const unsigned int index = tex2D<unsigned>(render_maps.indexMap[CameraID], fusionInterState.best_map_x, fusionInterState.best_map_y);
					fuse_surfel_replace_color(
						interVertexConfidence, interNormalRadius, interColorTime,
						world2camera, currentTime,
						model_vertex_confid, model_normal_radius, model_color_time
					);

					float3 model_vertex_confid_0 = InitialCameraSE3.rot * model_vertex_confid + InitialCameraSE3.trans;
					float3 model_normal_radius_0 = InitialCameraSE3.rot * model_normal_radius;

					//Write it
					geometry_arrays.vertexConfidence[index] = make_float4(model_vertex_confid_0.x, model_vertex_confid_0.y, model_vertex_confid_0.z, model_vertex_confid.w);;
					geometry_arrays.normalRadius[index] = make_float4(model_normal_radius_0.x, model_normal_radius_0.y, model_normal_radius_0.z, model_normal_radius.w);;
					geometry_arrays.colorTime[index] = model_color_time;
					//����󶨵ľ����Ǹ�remainingbuffer�����buffer�Ĵ�С�Ǻ���һ֡��live�вο���Ԫ�ĸ���
					geometry_arrays.fusedIndicator[index] = 1;
				}
			}
		};

		__global__ void debuggetremainlivesurfel(
			DeviceArrayView<unsigned> fusiondepthlivesurfelindex,
			DeviceArrayView<float4> livesurfel,
			unsigned* remainlivesurfelnumber,
			float4* remainlivesurfel,
			unsigned surfelnumber
		) {
			const int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= surfelnumber) return;
			const unsigned index = fusiondepthlivesurfelindex[idx];
			if (index == 0) {
				//˵��û�ںϣ���Ҫֱ�ӱ���
				const auto offset = atomicAdd(remainlivesurfelnumber, 1);
				remainlivesurfel[offset] = livesurfel[idx];
			}
		}


		__global__ void fuseAndMarkAppendedObservationSurfelsKernel(
			const FusionAndMarkAppendedObservationSurfelDevice fuser,
			unsigned* appendedPixel
		) {
			fuser.processIndicator(appendedPixel);
		}

		__global__ void fusionAndMarkAppendObservationAtomicKernel(
			const FusionAndMarkAppendedObservationSurfelDevice fuser,
			unsigned* appendOffset,
			//debug
			unsigned int* FusedSurfelNum,
			float4* fusionDepthLiveSurfel,
			ushort4* appendedPixel
		) { 
			fuser.processAtomic(appendOffset, FusedSurfelNum, fusionDepthLiveSurfel, appendedPixel);
		}

		__global__ void fuseAndMarkAppendedObservationSurfelReinitKernel(
			unsigned int* FusedSurfelNum,
			unsigned int* AppendedSurfelNum,
			const FusionAndMarkAppendedObservationSurfelDevice fuser,
			unsigned int* appendedPixel
		) {
			fuser.processFusionReinit(FusedSurfelNum, AppendedSurfelNum, appendedPixel);
		}


		__global__ void compactIndicatorToPixelKernel(
			const unsigned* candidate_pixel_indicator,
			const unsigned* prefixsum_indicator,
			unsigned img_cols,
			ushort2* compacted_pixels
		) {
			const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
			if(candidate_pixel_indicator[idx] > 0) {
				const auto offset = prefixsum_indicator[idx] - 1;
				const unsigned short x = idx % img_cols;
				const unsigned short y = idx / img_cols;
				compacted_pixels[offset] = make_ushort2(x, y);
			}
		}

	}
}


void SparseSurfelFusion::SurfelFusionHandler::prepareFuserArguments(void* fuser_ptr) {
	//Recovery the fuser arguments
	auto& fuser = *((device::FusionAndMarkAppendedObservationSurfelDevice*)fuser_ptr);
	for (int i = 0; i < devicesCount; i++) {
		// ��ǰ֡�۲�����
		fuser.observation_maps.vertexTimeMap[i] = m_observation.vertexConfidenceMap[i];
		fuser.observation_maps.normalRadiusMap[i] = m_observation.normalRadiusMap[i];
		fuser.observation_maps.colorTimeMap[i] = m_observation.colorTimeMap[i];
		fuser.observation_maps.foregroundMask[i] = m_observation.foregroundMask[i];
		// ��ǰ֡�۲��ֵ����
		fuser.observation_maps.interMarkValidMap[i] = m_observation.interpolatedValidValue[i];
		fuser.observation_maps.interVertexMap[i] = m_observation.interpolatedVertexMap[i];
		fuser.observation_maps.interNormalMap[i] = m_observation.interpolatedNormalMap[i];
		fuser.observation_maps.interColorMap[i] = m_observation.interpolatedColorMap[i];

		fuser.basicInfo.initialCameraSE3[i] = InitialCameraSE3[i];
		fuser.basicInfo.world2camera[i] = m_world2camera[i];

		// 4 �� 4��FusionMap
		fuser.render_maps.vertexMap[i] = m_fusion_maps[i].warp_vertex_map;
		fuser.render_maps.normalMap[i] = m_fusion_maps[i].warp_normal_map;
		fuser.render_maps.indexMap[i] = m_fusion_maps[i].index_map;
		fuser.render_maps.colorTimeMap[i] = m_fusion_maps[i].color_time_map;
	}

	// ��Ҫ��FusionMap�Ĵ洢Array��д�������
	fuser.geometry_arrays.vertexConfidence = m_fusion_geometry.liveVertexConfidence.RawPtr();
	fuser.geometry_arrays.normalRadius = m_fusion_geometry.liveNormalRadius.RawPtr();
	fuser.geometry_arrays.colorTime = m_fusion_geometry.colorTime.RawPtr();
	fuser.geometry_arrays.fusedIndicator = remainingSurfelIndicator.Ptr();	// �����ںϵ�һ������Ҫ������
	
	// ��������
	fuser.currentTime = m_current_time;
	fuser.imageCols = clipedImageCols;
	fuser.imageRows = clipedImageRows;
	fuser.CameraNum = devicesCount;
}

void SparseSurfelFusion::SurfelFusionHandler::processFusionAppendCompaction(cudaStream_t stream)
{
	////Resize the array
	//const auto num_surfels = m_fusion_geometry.live_vertex_confid.Size();
	//m_remaining_surfel_indicator.ResizeArrayOrException(num_surfels);
	//
	////Construct the fuser
	//device::FusionAndMarkAppendedObservationSurfelDevice fuser;
	//prepareFuserArguments((void*)&fuser);
	//
	//dim3 blk(16, 16);
	//dim3 grid(divUp(m_image_cols, blk.x), divUp(m_image_rows, blk.y));
	//device::fuseAndMarkAppendedObservationSurfelsKernel<<<grid, blk, 0, stream>>>(
	//	fuser, 
	//	m_world2camera, 
	//	appendedObservedSurfelIndicator.ptr()
	//);
}

void SparseSurfelFusion::SurfelFusionHandler::processFusionReinit(cudaStream_t stream)
{
	// Live��ĳ��ܵ�Array��Size
	const size_t num_surfels = m_fusion_geometry.liveVertexConfidence.Size();
	remainingSurfelIndicator.ResizeArrayOrException(num_surfels);

	// ������¼�ں��������Ԫ�Ķ�����Դ����
	CHECKCUDA(cudaMemsetAsync(FusedDepthSurfelNum, 0, sizeof(unsigned int), stream));
	// ������¼�����Ķ�����Ԫ����
	CHECKCUDA(cudaMemsetAsync(RemainingLiveSurfelNum, 0, sizeof(unsigned int), stream));
	device::FusionAndMarkAppendedObservationSurfelDevice fuser;
	prepareFuserArguments((void*)&fuser);
	dim3 block(16, 16, 1);
	dim3 grid(divUp(clipedImageCols, block.x), divUp(clipedImageRows, block.y), divUp(devicesCount, block.z));//
	device::fuseAndMarkAppendedObservationSurfelReinitKernel << <grid, block, 0, stream >> > (
		FusedDepthSurfelNum,
		RemainingLiveSurfelNum,
		fuser,
		appendedObservedSurfelIndicator.ptr()
	);
	

#if defined(CUDA_DEBUG_SYNC_CHECK)
	unsigned int fusion, append;
	CHECKCUDA(cudaMemcpyAsync(&fusion, FusedDepthSurfelNum, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaMemcpyAsync(&append, RemainingLiveSurfelNum, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	int currentTimeInt = m_current_time;
	printf("��%d֡��ˢ��ֱ����ӵ���Ԫ = %d  �������ں���Ԫ = %d\n", currentTimeInt, append, fusion);
#endif
}

void SparseSurfelFusion::SurfelFusionHandler::processFusionAppendAtomic(cudaStream_t stream)
{
	//Clear the attributes
	CHECKCUDA(cudaMemsetAsync(atomicAppendedPixelIndex, 0, sizeof(unsigned int), stream));
	//debug ������¼�ں��������Ԫ�Ķ�����Դ����
	CHECKCUDA(cudaMemsetAsync(FusedDepthSurfelNum, 0, sizeof(unsigned int), stream));
	//debug������¼�����Ķ�����Ԫ����
	CHECKCUDA(cudaMemsetAsync(RemainingLiveSurfelNum, 0, sizeof(unsigned int), stream));

	const size_t num_surfels = m_fusion_geometry.liveVertexConfidence.Size();
	remainingSurfelIndicator.ResizeArrayOrException(num_surfels);

	device::FusionAndMarkAppendedObservationSurfelDevice fuser;// �����ں���

	prepareFuserArguments((void*)&(fuser));
	dim3 block(16, 16, 1);
	dim3 grid(divUp(clipedImageCols, block.x), divUp(clipedImageRows, block.y), divUp(devicesCount, block.z));//
	device::fusionAndMarkAppendObservationAtomicKernel << <grid, block, 0, stream >> > (
		fuser,
		atomicAppendedPixelIndex,
		//debug
		FusedDepthSurfelNum,
		fusion.Ptr(),
		atomicAppendedObservationPixel.Ptr()
	);


	dim3 block_2(128);
	dim3 grid_2(divUp(num_surfels, block_2.x));
	//debug ���ֱ�ӱ����Ķ�����Ԫ��
	device::debuggetremainlivesurfel << <grid_2, block_2, 0, stream >> > (
		remainingSurfelIndicator.ArrayView(),
		m_fusion_geometry.liveVertexConfidence.ArrayView(),
		RemainingLiveSurfelNum,
		remain.Ptr(),
		num_surfels
	);

#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	//unsigned int FusedDepthSurfelNumHost = 0;
	//CHECKCUDA(cudaMemcpyAsync(&FusedDepthSurfelNumHost, FusedDepthSurfelNum, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	//printf("FusedDepthSurfelNumHost = %d\n", FusedDepthSurfelNumHost);
#endif

}


void SparseSurfelFusion::SurfelFusionHandler::compactAppendedIndicator(cudaStream_t stream) {
//	m_appended_surfel_indicator_prefixsum.InclusiveSum(appendedObservedSurfelIndicator, stream);
//	
//	//Invoke the kernel
//	dim3 blk(128);
//	dim3 grid(divUp(m_image_cols * m_image_rows, blk.x));
//	device::compactIndicatorToPixelKernel<<<grid, blk, 0, stream>>>(
//		appendedObservedSurfelIndicator.ptr(),
//		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.ptr(),
//		m_image_cols,
//		m_compacted_appended_pixel.Ptr()
//	);
//	
//	//Sync and check error
//#if defined(CUDA_DEBUG_SYNC_CHECK)
//	cudaSafeCall(cudaStreamSynchronize(stream));
//	cudaSafeCall(cudaGetLastError());
//#endif
//	
//	//Query the size
//	unsigned num_appended_surfel;
//	cudaSafeCall(cudaMemcpyAsync(
//		&num_appended_surfel,
//		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.size() - 1,
//		sizeof(unsigned),
//		cudaMemcpyDeviceToHost,
//		stream
//	));
//	
//	cudaSafeCall(cudaStreamSynchronize(stream));
//	m_compacted_appended_pixel.ResizeArrayOrException(num_appended_surfel);
}

void SparseSurfelFusion::SurfelFusionHandler::queryAtomicAppendedPixelSize(cudaStream_t stream) {
	unsigned numCandidatePixels;
	CHECKCUDA(cudaMemcpyAsync(&numCandidatePixels, atomicAppendedPixelIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	atomicAppendedObservationPixel.ResizeArrayOrException(numCandidatePixels);

	//debug
	unsigned fusionnumber, remainnumber;
	CHECKCUDA(cudaMemcpyAsync(
		&fusionnumber,
		FusedDepthSurfelNum,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream)
	);
	CHECKCUDA(cudaMemcpyAsync(
		&remainnumber,
		RemainingLiveSurfelNum,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream)
	);
	CHECKCUDA(cudaStreamSynchronize(stream));
	fusion.ResizeArrayOrException(fusionnumber);
	remain.ResizeArrayOrException(remainnumber);
	//// fusion�Ǻ죬remain����
	//Visualizer::DrawFusedProcessInCanonicalField(fusion.ArrayView(), remain.ArrayView());
}