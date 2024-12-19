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
		
			int CameraNum;	// 这个用来记录是哪个视角的数据
		
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
			 * \brief 原子操作添加融合后的面元.
			 *
			 * \param world2camera 世界坐标系转相机坐标系
			 * \param appendingOffset 原子操作偏移累加，作为index给数组赋值使用
			 * \param appended_pixels 将新添加的面元存储
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

				// 加载当前视角下的数据
				const float4 observedVertexConfidence = tex2D<float4>(observation_maps.vertexTimeMap[CameraID], x, y);
				const float4 observedNormalRadius = tex2D<float4>(observation_maps.normalRadiusMap[CameraID], x, y);
				const float4 observedColorTime = tex2D<float4>(observation_maps.colorTimeMap[CameraID], x, y);
				// 加载当前视角下插值数据
				const unsigned char interValidValue = observation_maps.interMarkValidMap[CameraID](y, x);
				const float4 interVertexConfidence = observation_maps.interVertexMap[CameraID](y, x);
				const float4 interNormalRadius = observation_maps.interNormalMap[CameraID](y, x);
				const float4 interColorTime = observation_maps.interColorMap[CameraID](y, x);

				bool  CheckObservation = false, CheckInterpolation = false;	// check是否有需插值的值
				if (!is_zero_vertex(interVertexConfidence)) CheckInterpolation = true;
				// 原始观测和插值都没有值(之前已经将插值能填入观测vertex的均填入了，剩下的就是vertex有值，插值不能填入的了)
				if (!is_zero_vertex(observedVertexConfidence)) CheckObservation = true;

				if (!CheckObservation && !CheckInterpolation) return;

				const int map_x_center = scale_factor * x;
				const int map_y_center = scale_factor * y;

				//The window search iteration variables
				unsigned int model_count = 0;				// 统计 4 × 4 窗口中有多少个有效面元
				unsigned int model_inter_count = 0;			// 统计 4 × 4 窗口中有多少个能与插值面元接近的面元
				SurfelFusionWindowState fusionState;		// 在 2 × 2 窗口中寻找满足跟观测面元相似(夹角 < 37°, z_dist_diff < 3cm)的Live域的点，并记录其map_x，map_y，best_confidence，z_dist
				SurfelFusionWindowState fusionInterState;	// 插值Surfel的融合
				SurfelAppendingWindowState appendState;		// 在 4 × 4 窗口中寻找满足跟观测面元相似(夹角 < 37°, points_dist < 3cm)的Live域的点，并记录其best_confidence和z_dist
				SurfelAppendingWindowState appendInterState;// 插值Surfel添加
				// 遍历窗口，以[-4, 3]遍历像素点
				for (int dy = -search_window_halfsize; dy < search_window_halfsize; dy++) {
					for (int dx = -search_window_halfsize; dx < search_window_halfsize; dx++) {
						//The actual position of in the rendered map
						const int map_y = dy + map_y_center;
						const int map_x = dx + map_x_center;

						const unsigned int index = tex2D<unsigned>(render_maps.indexMap[CameraID], map_x, map_y);
						if (index != 0xFFFFFFFF) {	// 加载FusionMap上的Geometry数据
							model_count++;	// 统计FusionMap窗口内有效的Geometry Surfels

							const float4 model_world_v4 = tex2D<float4>(render_maps.vertexMap[CameraID], map_x, map_y);
							const float4 model_world_n4 = tex2D<float4>(render_maps.normalMap[CameraID], map_x, map_y);
						
							// 转换到相机坐标系
							float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
							float3 model_camera_n3 = world2camera.rot * model_world_n4;

							// 一些通常用于检查的属性，检查render中的Map与实际观测面元的相似度
							float dot_value, diff_z, confidence, z_dist, dist_square;

							if (CheckObservation) {
								dot_value = dotxyz(model_camera_n3, observedNormalRadius);
								diff_z = fabsf(model_camera_v3.z - observedVertexConfidence.z);
								confidence = model_world_v4.w;
								z_dist = model_camera_v3.z;
								dist_square = squared_distance(model_camera_v3, observedVertexConfidence);
								// 融合的窗口半径只有2pix，如果在这个区间满足融合条件才会进行融合
								if (dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize) {
									if (dot_value >= 0.8f && diff_z <= 1e-2f) { // 【1cm】 原始参数：0.8 0.003
										fusionState.Update(confidence, z_dist, map_x, map_y);
									}
								}

								// 观察Surfel添加
								{
									if (dot_value >= 0.8f && dist_square <= 9e-4f) { //  【3cm】 原始参数：0.8 0.003 * 0.003
										appendState.Update(confidence, z_dist);
									}
								}
							}

							if (CheckInterpolation) {
								// 不能一直插点，存在相似的就融合
								if (dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize) {
									dot_value = dotxyz(model_camera_n3, interNormalRadius);
									diff_z = fabsf(model_camera_v3.z - interVertexConfidence.z);
									dist_square = squared_distance(model_camera_v3, interVertexConfidence);
									if (dot_value >= 0.8f && diff_z <= 3e-3f) { // 【1cm】 原始参数：0.8 0.003
										fusionInterState.Update(confidence, z_dist, map_x, map_y);
									}
								}
								// 插值Surfel添加
								{
									if (dot_value >= 0.8f && diff_z <= 3e-3f) { //  【3cm】 原始参数：0.8 0.003 * 0.003
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
					// 下面数据是世界坐标系
					geometry_arrays.vertexConfidence[index] = make_float4(model_vertex_confid_0.x, model_vertex_confid_0.y, model_vertex_confid_0.z, model_vertex_confid.w);
					geometry_arrays.normalRadius[index] = make_float4(model_normal_radius_0.x, model_normal_radius_0.y, model_normal_radius_0.z, model_normal_radius.w);
					geometry_arrays.colorTime[index] = model_color_time;
					geometry_arrays.fusedIndicator[index] = 1;
					//debug
					const unsigned int offset = atomicAdd(fusionDepthLiveSurfelNum, 1);
					fusionLiveSurfel[offset] = geometry_arrays.vertexConfidence[index];
				}

				// 处理插值Surfel的融合
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
					// 下面数据是世界坐标系
					geometry_arrays.vertexConfidence[index] = make_float4(model_vertex_confid_0.x, model_vertex_confid_0.y, model_vertex_confid_0.z, model_vertex_confid.w);
					geometry_arrays.normalRadius[index] = make_float4(model_normal_radius_0.x, model_normal_radius_0.y, model_normal_radius_0.z, model_normal_radius.w);
					geometry_arrays.colorTime[index] = model_color_time;
					geometry_arrays.fusedIndicator[index] = 1;
					//debug
					const unsigned int offset = atomicAdd(fusionDepthLiveSurfelNum, 1);
					fusionLiveSurfel[offset] = geometry_arrays.vertexConfidence[index];
				}

				//Check the view direction, and using atomic operation for appending
				// 说明以这个Live域点为中心 4 × 4 的区域满足：
				// 1、窗口内所有的点不满足(夹角小于37° && 空间距离 <= 3cm)   
				// 2、IndexMap这个窗口一个有效的点都没有   
				// 这个观察到的深度稠密点满足：3、这个稠密深度面元满足:设相机坐标系原点O，稠密深度顶点P，法线为n，满足夹角<PO, n> < 66°左右
				if (CheckObservation && appendState.best_confid < -0.01 && model_count == 0 && checkViewDirection(observedVertexConfidence, observedNormalRadius)) {
					const unsigned char mask_value = tex2D<unsigned char>(observation_maps.foregroundMask[CameraID], x, y);
					bool checkMask = (mask_value == (unsigned char)1);
#ifndef REBUILD_WITHOUT_BACKGROUND
					checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

					if (checkMask) {
						// 把这个appendingOffset里的值返回出来，然后再加1
						const unsigned int offset = atomicAdd(appendingOffset, 1);
						appendedPixels[offset] = make_ushort4(x, y, CameraID, 0);
					}
				}
				//if (model_inter_count != 0) printf("model_inter_count = %d\n", model_inter_count);

				// 这个点没有被融合，窗口中只要找到近的点几乎都会被融合，不需要保证窗口没有Geometry的Surfel
				if (CheckInterpolation/* && appendInterState.best_confid < -0.01&& checkViewDirection(interVertexConfidence, interNormalRadius) */) {
					const unsigned char mask_value = tex2D<unsigned char>(observation_maps.foregroundMask[CameraID], x, y);
					bool checkMask = (mask_value == (unsigned char)1);
#ifndef REBUILD_WITHOUT_BACKGROUND
					checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

					if (checkMask) {
						// 插值数据也加入到待添加数据中
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

				// 因为相机读出来的数据是在自己相机坐标系下的, FusionMap也是在对应视角下的数据，所以这里不需要SE3
				const float4 observedVertexConfidence = tex2D<float4>(observation_maps.vertexTimeMap[CameraID], x, y);
				const float4 observedNormalRadius = tex2D<float4>(observation_maps.normalRadiusMap[CameraID], x, y);
				const float4 observedColorTime = tex2D<float4>(observation_maps.colorTimeMap[CameraID], x, y);

				// 加载当前视角下插值数据
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
						
							// 变换到当前相机坐标系，与观察的内容进行对齐
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
								// 插值Surfels
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
				//这里的意思是，没有融合的面元，都要直接添加进来。
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
				if (CheckInterpolation && fusionInterState.best_confid < -1e-2f) {	// 如果是插值的像素，并且这个插值像素周围是存在Geometry的Surfels的
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
				// 存到同一个数组
				appendingIndicator[(CameraNum + CameraID) * clipedImageSize + offset] = interPixelIndicator;


				// 融合
				if (CheckObservation && fusionState.best_confid > 0) {
					atomicAdd(FusedSurfelNum, 1);
					// Live域对应视角坐标系
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
					//这里绑定的就是那个remainingbuffer。这个buffer的大小是和上一帧的live中参考面元的个数
					geometry_arrays.fusedIndicator[index] = 1;
				}

				// 融合插值Surfels
				if (CheckInterpolation && fusionInterState.best_confid > 0) {
					atomicAdd(FusedSurfelNum, 1);
					// Live域对应视角坐标系
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
					//这里绑定的就是那个remainingbuffer。这个buffer的大小是和上一帧的live中参考面元的个数
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
				//说明没融合，需要直接保留
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
		// 当前帧观测数据
		fuser.observation_maps.vertexTimeMap[i] = m_observation.vertexConfidenceMap[i];
		fuser.observation_maps.normalRadiusMap[i] = m_observation.normalRadiusMap[i];
		fuser.observation_maps.colorTimeMap[i] = m_observation.colorTimeMap[i];
		fuser.observation_maps.foregroundMask[i] = m_observation.foregroundMask[i];
		// 当前帧观测插值数据
		fuser.observation_maps.interMarkValidMap[i] = m_observation.interpolatedValidValue[i];
		fuser.observation_maps.interVertexMap[i] = m_observation.interpolatedVertexMap[i];
		fuser.observation_maps.interNormalMap[i] = m_observation.interpolatedNormalMap[i];
		fuser.observation_maps.interColorMap[i] = m_observation.interpolatedColorMap[i];

		fuser.basicInfo.initialCameraSE3[i] = InitialCameraSE3[i];
		fuser.basicInfo.world2camera[i] = m_world2camera[i];

		// 4 × 4的FusionMap
		fuser.render_maps.vertexMap[i] = m_fusion_maps[i].warp_vertex_map;
		fuser.render_maps.normalMap[i] = m_fusion_maps[i].warp_normal_map;
		fuser.render_maps.indexMap[i] = m_fusion_maps[i].index_map;
		fuser.render_maps.colorTimeMap[i] = m_fusion_maps[i].color_time_map;
	}

	// 需要往FusionMap的存储Array中写入的数据
	fuser.geometry_arrays.vertexConfidence = m_fusion_geometry.liveVertexConfidence.RawPtr();
	fuser.geometry_arrays.normalRadius = m_fusion_geometry.liveNormalRadius.RawPtr();
	fuser.geometry_arrays.colorTime = m_fusion_geometry.colorTime.RawPtr();
	fuser.geometry_arrays.fusedIndicator = remainingSurfelIndicator.Ptr();	// 参与融合的一定是需要保留的
	
	// 其他属性
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
	// Live域的稠密点Array的Size
	const size_t num_surfels = m_fusion_geometry.liveVertexConfidence.Size();
	remainingSurfelIndicator.ResizeArrayOrException(num_surfels);

	// 用来记录融合了深度面元的动作面源个数
	CHECKCUDA(cudaMemsetAsync(FusedDepthSurfelNum, 0, sizeof(unsigned int), stream));
	// 用来记录保留的动作面元个数
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
	printf("第%d帧，刷新直接添加的面元 = %d  保留的融合面元 = %d\n", currentTimeInt, append, fusion);
#endif
}

void SparseSurfelFusion::SurfelFusionHandler::processFusionAppendAtomic(cudaStream_t stream)
{
	//Clear the attributes
	CHECKCUDA(cudaMemsetAsync(atomicAppendedPixelIndex, 0, sizeof(unsigned int), stream));
	//debug 用来记录融合了深度面元的动作面源个数
	CHECKCUDA(cudaMemsetAsync(FusedDepthSurfelNum, 0, sizeof(unsigned int), stream));
	//debug用来记录保留的动作面元个数
	CHECKCUDA(cudaMemsetAsync(RemainingLiveSurfelNum, 0, sizeof(unsigned int), stream));

	const size_t num_surfels = m_fusion_geometry.liveVertexConfidence.Size();
	remainingSurfelIndicator.ResizeArrayOrException(num_surfels);

	device::FusionAndMarkAppendedObservationSurfelDevice fuser;// 构造融合器

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
	//debug 获得直接保留的动作面元。
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
	//// fusion是红，remain是绿
	//Visualizer::DrawFusedProcessInCanonicalField(fusion.ArrayView(), remain.ArrayView());
}