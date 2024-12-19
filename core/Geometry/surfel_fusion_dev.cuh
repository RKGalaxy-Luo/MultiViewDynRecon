#pragma once
#include <vector_types.h>
#include <base/EncodeUtils.h>
#include <base/ColorTypeTransfer.h>
#include <math/VectorUtils.h>
#include <math/MatUtils.h>


//The macro constants for maximum weight, only accessed on this file
#define d_max_surfel_weight 2400.0f


namespace SparseSurfelFusion { 
	namespace device {

		//用于融合深度面元
		struct DepthSurfelFusionWindowState {
				int best_map_x;
				int best_map_y;
				float nearest_dist;
				float nearest_color_dist;
				float nearest_nor;
				__host__ __device__ DepthSurfelFusionWindowState() : best_map_x(0), best_map_y(0), nearest_dist(1), nearest_color_dist(255*3), nearest_nor(0){}

				__host__ __device__ __forceinline__ void Update(
					const float dist,
					const float color_dist,
					const float nor,
					int map_x, int map_y
				) {
					bool update = false;
					if (dist < nearest_dist) {
						//if (color_dist < nearest_color_dist) {
							update = true;
						//}
					}
					//if (nor>nearest_nor)
					//{
					//	update = true;
					//}

					if (update) {
						nearest_dist = dist;
						nearest_color_dist = color_dist;
						nearest_nor = nor;
						best_map_x = map_x;
						best_map_y = map_y;
					}
				}


			};

		struct SurfelFusionWindowState {
			int best_map_x;
			int best_map_y;
			float best_confid;
			float nearest_z_dist;
			__host__ __device__ SurfelFusionWindowState() : best_map_x(0), best_map_y(0), best_confid(-1), nearest_z_dist(0){}

			__host__ __device__ __forceinline__ void Update(
				const float confid,
				const float z_dist,
				int map_x, int map_y
			) {
				bool update = false;
				if (confid < best_confid - 1e-3) {
					//Do nothing
				}
				else if (confid > best_confid + 1e-3) {
					//Update best
					update = true;
				}
				else {
					if (nearest_z_dist > z_dist) update = true;
				}

				if (update) {
					best_confid = confid;
					nearest_z_dist = z_dist;
					best_map_x = map_x;
					best_map_y = map_y;
				}
			}

		
		};

		struct SurfelAppendingWindowState {
			float best_confid;
			float nearest_z_dist;
			__host__ __device__ SurfelAppendingWindowState() : best_confid(-1.0f), nearest_z_dist(0) {}

			//The udpate method
			__host__ __device__ __forceinline__ void Update(
				const float confid,
				const float z_dist
			) {
				bool update = false;
				if (confid < best_confid - 1e-3) {
					//Do nothing
				}
				else if (confid > best_confid + 1e-3) {
					//Update best
					update = true;
				}
				else {
					if (nearest_z_dist > z_dist) update = true;
				}

				if (update) {
					best_confid = confid;
					nearest_z_dist = z_dist;
				}
			}
		};

		//用来融合深度面元的
		__device__ __forceinline__ void fuse_depth_surfel(
			const float4& depth_vertex_confid,
			const float4& depth_normal_radius,
			const float4& image_color_time,
			//这三个是写入的地方。所以要先把1份数据放到buffer里，再利用这个函数。
			float4& model_vertex_confid,
			float4& model_normal_radius,
			float4& model_color_time
		) {
			const float fused_weight = min(depth_vertex_confid.w + model_vertex_confid.w, d_max_surfel_weight);

			//Fuse of vertex, note that the last element (confidence) must be fused at last
			//const float3 depth_world_v3 = world2camera.apply_inversed_se3(depth_vertex_confid);
			const float3 depth_world_v3 = make_float3(depth_vertex_confid.x, depth_vertex_confid.y, depth_vertex_confid.z);
			model_vertex_confid.x = (model_vertex_confid.x * model_vertex_confid.w + depth_world_v3.x * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.y = (model_vertex_confid.y * model_vertex_confid.w + depth_world_v3.y * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.z = (model_vertex_confid.z * model_vertex_confid.w + depth_world_v3.z * depth_vertex_confid.w) / fused_weight;

			//Fuse of normal
			//const float3 depth_world_n3 = world2camera.rot.transpose_dot(depth_normal_radius);
			const float3 depth_world_n3 = make_float3(depth_normal_radius.x, depth_normal_radius.y, depth_normal_radius.z);
			model_normal_radius.x = (model_normal_radius.x * model_vertex_confid.w + depth_world_n3.x * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.y = (model_normal_radius.y * model_vertex_confid.w + depth_world_n3.y * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.z = (model_normal_radius.z * model_vertex_confid.w + depth_world_n3.z * depth_vertex_confid.w) / fused_weight;
			//Fuse of radius
			model_normal_radius.w = (model_normal_radius.w * model_vertex_confid.w + depth_normal_radius.w * depth_vertex_confid.w) / fused_weight;

			//The rgb of observation and model
			uchar3 rgb_observation, rgb_model;
			float_decode_rgb(image_color_time.x, rgb_observation);
			float_decode_rgb(model_color_time.x, rgb_model);
			rgb_model.x = __float2uint_rz((float(rgb_model.x) * model_vertex_confid.w + float(rgb_observation.x) * depth_vertex_confid.w) / fused_weight);
			rgb_model.y = __float2uint_rz((float(rgb_model.y) * model_vertex_confid.w + float(rgb_observation.y) * depth_vertex_confid.w) / fused_weight);
			rgb_model.z = __float2uint_rz((float(rgb_model.z) * model_vertex_confid.w + float(rgb_observation.z) * depth_vertex_confid.w) / fused_weight);

			const float encoded_fused = float_encode_rgb(rgb_model);
			model_color_time.x = encoded_fused;
			//
			model_color_time.z = image_color_time.z;

			//Update the weight at last
			model_vertex_confid.w = fused_weight;
		}






		__device__ __forceinline__ void fuse_surfel(
			const float4& depth_vertex_confid,
			const float4& depth_normal_radius,
			const float4& image_color_time,
			const mat34& world2camera,
			const float current_time,
			float4& model_vertex_confid,
			float4& model_normal_radius,
			float4& model_color_time
		) {
			const float fused_weight = min(depth_vertex_confid.w + model_vertex_confid.w, d_max_surfel_weight);

			//Fuse of vertex, note that the last element (confidence) must be fused at last
			const float3 depth_world_v3 = world2camera.apply_inversed_se3(depth_vertex_confid);
			model_vertex_confid.x = (model_vertex_confid.x * model_vertex_confid.w + depth_world_v3.x * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.y = (model_vertex_confid.y * model_vertex_confid.w + depth_world_v3.y * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.z = (model_vertex_confid.z * model_vertex_confid.w + depth_world_v3.z * depth_vertex_confid.w) / fused_weight;

			//Fuse of normal
			const float3 depth_world_n3 = world2camera.rot.transpose_dot(depth_normal_radius);
			model_normal_radius.x = (model_normal_radius.x * model_vertex_confid.w + depth_world_n3.x * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.y = (model_normal_radius.y * model_vertex_confid.w + depth_world_n3.y * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.z = (model_normal_radius.z * model_vertex_confid.w + depth_world_n3.z * depth_vertex_confid.w) / fused_weight;
			//Fuse of radius
			model_normal_radius.w = (model_normal_radius.w * model_vertex_confid.w + depth_normal_radius.w * depth_vertex_confid.w) / fused_weight;

			//The rgb of observation and model
			uchar3 rgb_observation, rgb_model;
			float_decode_rgb(image_color_time.x, rgb_observation);
			float_decode_rgb(model_color_time.x, rgb_model);
			rgb_model.x = __float2uint_rz((float(rgb_model.x) * model_vertex_confid.w + float(rgb_observation.x) * depth_vertex_confid.w) / fused_weight);
			rgb_model.y = __float2uint_rz((float(rgb_model.y) * model_vertex_confid.w + float(rgb_observation.y) * depth_vertex_confid.w) / fused_weight);
			rgb_model.z = __float2uint_rz((float(rgb_model.z) * model_vertex_confid.w + float(rgb_observation.z) * depth_vertex_confid.w) / fused_weight);

			const float encoded_fused = float_encode_rgb(rgb_model);
			model_color_time.x = encoded_fused;
			model_color_time.z = current_time;

			//Update the weight at last
			model_vertex_confid.w = fused_weight;
		}

		__device__ __forceinline__ void fuse_surfel_clip_color(
			const float4& depth_vertex_confid,
			const float4& depth_normal_radius,
			const float4& image_color_time,
			const mat34& world2camera,
			const float current_time,
			float4& model_vertex_confid,
			float4& model_normal_radius,
			float4& model_color_time
		) {
			const float fused_weight = min(depth_vertex_confid.w + model_vertex_confid.w, d_max_surfel_weight);

			//Fuse of vertex, note that the last element (confidence) must be fused at last
			const float3 depth_world_v3 = world2camera.apply_inversed_se3(depth_vertex_confid);
			model_vertex_confid.x = (model_vertex_confid.x * model_vertex_confid.w + depth_world_v3.x * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.y = (model_vertex_confid.y * model_vertex_confid.w + depth_world_v3.y * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.z = (model_vertex_confid.z * model_vertex_confid.w + depth_world_v3.z * depth_vertex_confid.w) / fused_weight;

			//Fuse of normal
			const float3 depth_world_n3 = world2camera.rot.transpose_dot(depth_normal_radius);
			model_normal_radius.x = (model_normal_radius.x * model_vertex_confid.w + depth_world_n3.x * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.y = (model_normal_radius.y * model_vertex_confid.w + depth_world_n3.y * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.z = (model_normal_radius.z * model_vertex_confid.w + depth_world_n3.z * depth_vertex_confid.w) / fused_weight;
			//Fuse of radius
			model_normal_radius.w = (model_normal_radius.w * model_vertex_confid.w + depth_normal_radius.w * depth_vertex_confid.w) / fused_weight;

			//The rgb of observation and model
			uchar3 rgb_observation, rgb_model;
			float_decode_rgb(image_color_time.x, rgb_observation);
			float_decode_rgb(model_color_time.x, rgb_model);

			//Determined by the diff between model and observation
			const auto diff = rgb_diff_abs(rgb_observation, rgb_model);
			if (diff > 40) {
				rgb_model = rgb_observation;
			} else {
				rgb_model.x = __float2uint_rz((float(rgb_model.x) * model_vertex_confid.w + float(rgb_observation.x) * depth_vertex_confid.w) / fused_weight);
				rgb_model.y = __float2uint_rz((float(rgb_model.y) * model_vertex_confid.w + float(rgb_observation.y) * depth_vertex_confid.w) / fused_weight);
				rgb_model.z = __float2uint_rz((float(rgb_model.z) * model_vertex_confid.w + float(rgb_observation.z) * depth_vertex_confid.w) / fused_weight);
			}

			const float encoded_fused = float_encode_rgb(rgb_model);
			model_color_time.x = encoded_fused;
			model_color_time.z = current_time;

			//Update the weight at last
			model_vertex_confid.w = fused_weight;
		}


		__device__ __forceinline__ void fuse_surfel_replace_color(
			const float4& depth_vertex_confid,
			const float4& depth_normal_radius,
			const float4& image_color_time,
			const mat34& world2camera,
			const float current_time,
			float4& model_vertex_confid,
			float4& model_normal_radius,
			float4& model_color_time
		) {
		
			const float fused_weight = min(depth_vertex_confid.w + model_vertex_confid.w, d_max_surfel_weight);

			//Fuse of vertex, note that the last element (confidence) must be fused at last
			const float3 depth_world_v3 = world2camera.apply_inversed_se3(depth_vertex_confid);
			model_vertex_confid.x = (model_vertex_confid.x * model_vertex_confid.w + depth_world_v3.x * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.y = (model_vertex_confid.y * model_vertex_confid.w + depth_world_v3.y * depth_vertex_confid.w) / fused_weight;
			model_vertex_confid.z = (model_vertex_confid.z * model_vertex_confid.w + depth_world_v3.z * depth_vertex_confid.w) / fused_weight;

			//Fuse of normal
			const float3 depth_world_n3 = world2camera.rot.transpose_dot(depth_normal_radius);
			model_normal_radius.x = (model_normal_radius.x * model_vertex_confid.w + depth_world_n3.x * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.y = (model_normal_radius.y * model_vertex_confid.w + depth_world_n3.y * depth_vertex_confid.w) / fused_weight;
			model_normal_radius.z = (model_normal_radius.z * model_vertex_confid.w + depth_world_n3.z * depth_vertex_confid.w) / fused_weight;
			//Fuse of radius
			model_normal_radius.w = (model_normal_radius.w * model_vertex_confid.w + depth_normal_radius.w * depth_vertex_confid.w) / fused_weight;

			//Directly replace rgb with observation rgb value
			//uchar3 rgb_observation;
			//float_decode_rgb(image_color_time.x, rgb_observation);
			//model_color_time.x = float_encode_rgb(rgb_observation);
			model_color_time.x = image_color_time.x;
			model_color_time.z = current_time;

			//Update the weight at last
			model_vertex_confid.w = fused_weight;
		}
	////只有在刷新帧更新world2camera时，采用这个函数做融合  只限刷新函数里使用！！！！！其他地方不用
	//__device__ __forceinline__ void fuse_surfel_replace_color_noworld2camera(
	//	const float4& depth_vertex_confid,
	//	const float4& depth_normal_radius,
	//	const float4& image_color_time,
	//	const float current_time,
	//	float4& model_vertex_confid,
	//	float4& model_normal_radius,
	//	float4& model_color_time
	//) {

	//	const float fused_weight = min(depth_vertex_confid.w + model_vertex_confid.w, d_max_surfel_weight);

	//	//Fuse of vertex, note that the last element (confidence) must be fused at last
	//	//const float3 depth_world_v3 = world2camera.apply_inversed_se3(depth_vertex_confid);
	//	model_vertex_confid.x = (model_vertex_confid.x * model_vertex_confid.w + depth_world_v3.x * depth_vertex_confid.w) / fused_weight;
	//	model_vertex_confid.y = (model_vertex_confid.y * model_vertex_confid.w + depth_world_v3.y * depth_vertex_confid.w) / fused_weight;
	//	model_vertex_confid.z = (model_vertex_confid.z * model_vertex_confid.w + depth_world_v3.z * depth_vertex_confid.w) / fused_weight;

	//	//Fuse of normal
	//	const float3 depth_world_n3 = world2camera.rot.transpose_dot(depth_normal_radius);
	//	model_normal_radius.x = (model_normal_radius.x * model_vertex_confid.w + depth_world_n3.x * depth_vertex_confid.w) / fused_weight;
	//	model_normal_radius.y = (model_normal_radius.y * model_vertex_confid.w + depth_world_n3.y * depth_vertex_confid.w) / fused_weight;
	//	model_normal_radius.z = (model_normal_radius.z * model_vertex_confid.w + depth_world_n3.z * depth_vertex_confid.w) / fused_weight;
	//	//Fuse of radius
	//	model_normal_radius.w = (model_normal_radius.w * model_vertex_confid.w + depth_normal_radius.w * depth_vertex_confid.w) / fused_weight;

	//	//Directly replace rgb with observation rgb value
	//	uchar3 rgb_observation;
	//	float_decode_rgb(image_color_time.x, rgb_observation);
	//	model_color_time.x = float_encode_rgb(rgb_observation);
	//	model_color_time.z = current_time;

	//	//Update the weight at last
	//	model_vertex_confid.w = fused_weight;
	//}


	} // device
} // surfelwarp