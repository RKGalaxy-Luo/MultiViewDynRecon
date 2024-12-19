#include "ReinitRemainingSurfelMarker.h"
#include <core/NonRigidSolver/solver_types.h>
#include <visualization/Visualizer.h>
#include <device_launch_parameters.h>

namespace SparseSurfelFusion { 
	namespace device {

		/**
		 * \brief 将Live域与当前观察内容进行对比，保留Live域中合理的部分，修改并保留Live域中不太合理的部分，删除Live域中错误的部分.
		 */
		struct ReinitSelectValidObervationDevice {
			enum {
				window_halfsize = 2,
				devicesCount = MAX_CAMERA_COUNT
			};
			struct {
				unsigned int VertexNum;
				float4* vertexConfidence;
				float4* normalRadius;
				float4* colorTime;
				const ushort4* surfelKnn;
			} liveGeometryInterface;	// 直接暴露Live域中点的接口

			struct {
				cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];
				cudaTextureObject_t foregroundMask[MAX_CAMERA_COUNT];

			} cameraObservation;		// 当前帧观察到的内容

			unsigned int CameraID;
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3Inverse[MAX_CAMERA_COUNT];
			Intrinsic intrinsic[MAX_CAMERA_COUNT];
			mat34 camera2world[MAX_CAMERA_COUNT];
			mat34 world2camera[MAX_CAMERA_COUNT];

			/**
			 * \brief 将Live域与当前观察内容进行对比，保留Live域中合理的部分，修改并保留Live域中不太合理的部分，删除Live域中错误的部分.
			 * 
			 * \param remainingIndicator 保留区域的Indicator
			 */
			__device__ __forceinline__ void processMarkingAndSelectObsevered(unsigned int* remainingIndicator) {
				const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
				if (idx >= liveGeometryInterface.VertexNum) return;
				// 当前面元是否被融合过，如果融合过则一定保留，如果是新增的面元则fused = 0
				const unsigned int fused = remainingIndicator[idx];

				if (fused > 0) return;

				// 这里都是Live域中的数据
				float4 surfelVertexConfidence = liveGeometryInterface.vertexConfidence[idx];
				float4 surfelNormalRadius = liveGeometryInterface.normalRadius[idx];
				//const float4 surfel_color_time = live_geometry.color_time[idx];

				// 将Live域新增面元扭曲到相机坐标系
				float3 vertex = world2camera[CameraID].rot * surfelVertexConfidence + world2camera[CameraID].trans;
				float3 normal = world2camera[CameraID].rot * surfelNormalRadius;
				vertex = InitialCameraSE3Inverse[CameraID].rot * vertex + InitialCameraSE3Inverse[CameraID].trans;
				normal = InitialCameraSE3Inverse[CameraID].rot * normal;

				// 投影到像素坐标系
				const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic[CameraID].focal_x) + intrinsic[CameraID].principal_x);
				const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic[CameraID].focal_y) + intrinsic[CameraID].principal_y);

				bool hasCorresponded = false;
				bool replaceFlag = false;	// 是否用观察到的面元替换
				float minSquaredDis = 10000.0f;
				int2 CloestMapCoor = make_int2(0xFFFF, 0xFFFF);
				// 遍历像素点附近 4 × 4 窗口的点，判断观测的点是否与观测的深度点近似
				for (int map_y = y - window_halfsize; map_y < y + window_halfsize; map_y++) {
					for (int map_x = x - window_halfsize; map_x < x + window_halfsize; map_x++) {
						// 加载观测到的数据
						const float4 depth_vertex = tex2D<float4>(cameraObservation.vertexMap[CameraID], map_x, map_y);
						const float4 depth_normal = tex2D<float4>(cameraObservation.normalMap[CameraID], map_x, map_y);

						//Compute various values
						const float normalDot = dotxyz(normal, depth_normal);
						float sqauredDis = squared_distance(vertex, depth_vertex);

						// Live域与观察的内容相差不大，将Live域中这个点保留 sqauredDis < 0.03f * 0.03f
						if (sqauredDis < 0.01f * 0.01f && normalDot >= 0.8f) {
							hasCorresponded = true;
						}
						// Live域与观察的内容相差较大，用观察到的面元替换，或者直接舍弃
						else {
							// 相差不算太大
							if (minSquaredDis > sqauredDis && sqauredDis < 0.15f * 0.15f && normalDot >= 0.7f) {

								minSquaredDis = sqauredDis;
								CloestMapCoor = make_int2(map_x, map_y);
								replaceFlag = true;
							}
						}
					}
				} // windows search on depth image

				// 检查点是否在前景上
				unsigned char foregound = tex2D<unsigned char>(cameraObservation.foregroundMask[CameraID], x, y);

				unsigned int remain = 0;
				if ((hasCorresponded == true) && (foregound > 0)) remain = 1;

				else if ((replaceFlag == true) && (foregound > 0)) {
					float4 ObserveVertex = tex2D<float4>(cameraObservation.vertexMap[CameraID], CloestMapCoor.x, CloestMapCoor.y);
					float4 ObserveNormal = tex2D<float4>(cameraObservation.normalMap[CameraID], CloestMapCoor.x, CloestMapCoor.y);
					float3 LiveVertex = InitialCameraSE3[CameraID].rot * ObserveVertex + InitialCameraSE3[CameraID].trans;
					float3 LiveNormal = InitialCameraSE3[CameraID].rot * ObserveNormal;
					LiveVertex = camera2world[CameraID].rot * LiveVertex + camera2world[CameraID].trans;
					LiveNormal = camera2world[CameraID].rot * LiveNormal;
					liveGeometryInterface.vertexConfidence[idx] = make_float4((surfelVertexConfidence.x + LiveVertex.x) / 2.0f, (surfelVertexConfidence.y + LiveVertex.y) / 2.0f, (surfelVertexConfidence.z + LiveVertex.z) / 2.0f, ObserveVertex.w);
					liveGeometryInterface.normalRadius[idx] = make_float4((surfelNormalRadius.x + LiveNormal.x) / 2.0f, (surfelNormalRadius.y + LiveNormal.y) / 2.0f, (surfelNormalRadius.z + LiveNormal.z) / 2.0f, ObserveNormal.w);
					remain = 1;
				}
				// 保留这些点
				remainingIndicator[idx] = remain;
			}
		};

		__global__ void markAndReplaceIrrationalSurfel(
			ReinitSelectValidObervationDevice marker,
			unsigned int* remaining_indicator
		) {
			marker.processMarkingAndSelectObsevered(remaining_indicator);
		}

		//struct ReinitRemainingMarkerDevice {
		//	enum {
		//		window_halfsize = 2,
		//	};

		//	//The geometry model input
		//	struct {
		//		DeviceArrayView<float4> vertex_confid;
		//		const float4* normal_radius;
		//		const float4* color_time;
		//		const ushort4* surfel_knn;
		//	} liveGeometry;

		//	//The observation from camera
		//	struct {
		//		cudaTextureObject_t vertex_map;
		//		cudaTextureObject_t normal_map;
		//		cudaTextureObject_t foreground_mask;
		//	} cameraObservation[MAX_CAMERA_COUNT];

		//	//The information on camera
		//	mat34 InitialCameraSE3Inverse[MAX_CAMERA_COUNT];
		//	mat34 world2camera[MAX_CAMERA_COUNT];
		//	Intrinsic intrinsic[MAX_CAMERA_COUNT];

		//	__device__ __forceinline__ void processMarkingObservedOnly(const unsigned int frameIdx, unsigned* remaining_indicator) const {
		//		const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		//		if (idx >= liveGeometry.vertex_confid.Size()) return;

		//		// 当前面元是否被融合过，如果融合过则一定保留，如果是新增的面元则fused = 0
		//		const unsigned int fused = remaining_indicator[idx];
		//		if (fused > 0) return;
		//		//这里都是0中的数据
		//		const float4 surfel_vertex_confid = liveGeometry.vertex_confid[idx];
		//		const float4 surfel_normal_radius = liveGeometry.normal_radius[idx];
		//	
		//		// 将Live域新增面元扭曲到相机坐标系
		//		float3 vertex = world2camera.rot * surfel_vertex_confid + world2camera.trans;
		//		float3 normal = world2camera.rot * surfel_normal_radius;
		//		vertex = InitialCameraSE3Inverse.rot * vertex + InitialCameraSE3Inverse.trans;
		//		normal = InitialCameraSE3Inverse.rot * normal;


		//		// 投影到像素坐标系
		//		const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
		//		const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);

		//		//Deal with the case where (x, y) is out of the range of the image

		//		//The flag value
		//		bool hasCorresponded = false;

		//		// 遍历像素点附近 4 × 4 窗口的点，判断观测的点是否与观测的深度点近似
		//		for (int map_y = y - window_halfsize; map_y < y + window_halfsize; map_y++) {
		//			for (int map_x = x - window_halfsize; map_x < x + window_halfsize; map_x++) {
		//				//Load the depth image
		//				const float4 depth_vertex = tex2D<float4>(cameraObservation.vertex_map, map_x, map_y);
		//				const float4 depth_normal = tex2D<float4>(cameraObservation.normal_map, map_x, map_y);
		//			
		//				//Compute various values
		//				const float normal_dot = dotxyz(normal, depth_normal);
		//			
		//				// 0.003f * 0.003f     0.8f
		//				if (squared_distance(vertex, depth_vertex) < 9e-6f && normal_dot >= 0.9f)
		//					hasCorresponded = true;
		//			}
		//		} // windows search on depth image

		//		// 检查点是否在前景上
		//		unsigned char foregound = tex2D<unsigned char>(cameraObservation.foreground_mask, x, y);

		//		unsigned int remain = 0;
		//		if ((hasCorresponded == true) && (foregound > 0))
		//			remain = 1;

		//		// 保留这些点
		//		remaining_indicator[idx] = remain;
		//	}


		//	//__device__ __forceinline__ void processMarkingNodeError(
		//	//	const NodeAlignmentError& node_error,
		//	//	float threshold,
		//	//	unsigned* remaining_indicator
		//	//) const {
		//	//	const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		//	//	if (idx >= live_geometry.vertex_confid.Size()) return;

		//	//	//Is this surfel fused? If so, must remain
		//	//	const auto fused = remaining_indicator[idx];
		//	//	if(fused > 0) return;

		//	//	const float4 surfel_vertex_confid = live_geometry.vertex_confid[idx];
		//	//	const float4 surfel_normal_radius = live_geometry.normal_radius[idx];
		//	//	//const float4 surfel_color_time = live_geometry.color_time[idx];
		//	//
		//	//	//Transfer to camera space
		//	//	const float3 vertex = world2camera.rot * surfel_vertex_confid + world2camera.trans;
		//	//	//const float3 normal = world2camera.rot * surfel_normal_radius;

		//	//	//Project to camera image and check foreground
		//	//	const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
		//	//	const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);
		//	//	const auto foregound = tex2D<unsigned char>(camera_observation.foreground_mask, x, y);

		//	//	//Somehow optimistic
		//	//	unsigned remain = foregound > 0 ? 1 : 0;

		//	//	//Check the error
		//	//	const ushort4 knn_nodes = live_geometry.surfel_knn[idx];
		//	//	const unsigned short* knn_nodes_flat = (const unsigned short*)(&knn_nodes);
		//	//	for(int i = 0; i < 4; i++) {
		//	//		const unsigned short node = knn_nodes_flat[i];
		//	//		const float accumlate_error = node_error.nodeAccumlatedError[node];
		//	//		const float accumlate_weight = node_error.nodeAccumlateWeight[node];
		//	//		if(accumlate_weight * threshold > accumlate_error)
		//	//			remain = 0;
		//	//	}

		//	//	//Write to output
		//	//	remaining_indicator[idx] = remain;
		//	//}
		//};


		//__global__ void markReinitRemainingSurfelObservedOnlyKernel(
		//	const ReinitRemainingMarkerDevice marker,
		//	const unsigned int frameIdx,
		//	unsigned* remaining_indicator
		//) {
		//	marker.processMarkingObservedOnly(frameIdx, remaining_indicator);
		//}


		//__global__ void markReinitRemainingSurfelNodeErrorKernel(
		//	const ReinitRemainingMarkerDevice marker,
		//	const NodeAlignmentError node_error,
		//	float threshold,
		//	unsigned* remaining_indicator
		//) {
		//	marker.processMarkingNodeError(node_error, threshold, remaining_indicator);
		//}

	}
}

__device__ __forceinline__ void SparseSurfelFusion::device::ReinitRemainingMarkerDevice::processMarkingObservedOnly(const unsigned int remainingSurfelsCount, const unsigned int mapCols, const unsigned int mapRows, unsigned int* remainingIndicator) const
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int CameraID = threadIdx.y + blockDim.y * blockIdx.y;
	if (idx >= remainingSurfelsCount) return;

	// 当前面元是否被融合过，如果融合过则一定保留，如果是新增的面元则fused = 0
	const unsigned int fused = remainingIndicator[idx];
	if (fused > 0) return;
	//这里都是0中的数据
	const float4 surfel_vertex_confid = liveGeometry.vertexConfidence[idx];
	const float4 surfel_normal_radius = liveGeometry.normalRadius[idx];
	
	// 将Live域新增面元扭曲到相机坐标系
	float3 vertex = world2camera[CameraID].rot * surfel_vertex_confid + world2camera[CameraID].trans;
	float3 normal = world2camera[CameraID].rot * surfel_normal_radius;
	vertex = InitialCameraSE3Inverse[CameraID].rot * vertex + InitialCameraSE3Inverse[CameraID].trans;
	normal = InitialCameraSE3Inverse[CameraID].rot * normal;


	// 投影到像素坐标系
	const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic[CameraID].focal_x) + intrinsic[CameraID].principal_x);
	const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic[CameraID].focal_y) + intrinsic[CameraID].principal_y);

	//Deal with the case where (x, y) is out of the range of the image
	if (x < window_halfsize || x > mapCols - window_halfsize || y < window_halfsize || y > mapRows - window_halfsize) return;
	//The flag value
	bool hasCorresponded = false;

	// 遍历像素点附近 4 × 4 窗口的点，判断观测的点是否与观测的深度点近似
	for (int map_y = y - window_halfsize; map_y < y + window_halfsize; map_y++) {
		for (int map_x = x - window_halfsize; map_x < x + window_halfsize; map_x++) {
			//Load the depth image
			const float4 depth_vertex = tex2D<float4>(cameraObservation[CameraID].vertexMap, map_x, map_y);
			const float4 depth_normal = tex2D<float4>(cameraObservation[CameraID].normalMap, map_x, map_y);
			
			//Compute various values
			const float normal_dot = dotxyz(normal, depth_normal);
			
			// 0.003f * 0.003f     0.8f
			if (squared_distance(vertex, depth_vertex) < 9e-6f && normal_dot >= 0.9f)
				hasCorresponded = true;
		}
	} // windows search on depth image

	// 检查点是否在前景上
	unsigned char foregound = tex2D<unsigned char>(cameraObservation[CameraID].foregroundMask, x, y);

	unsigned int remain = 0;
	if ((hasCorresponded == true) && (foregound > 0))
		remain = 1;

	// 保留这些点
	//remainingIndicator[idx] = remain;
	atomicOr(&remainingIndicator[idx], remain);	// 只要存在1，就为1
}
__global__ void SparseSurfelFusion::device::markReinitRemainingSurfelObservedOnlyKernel(const ReinitRemainingMarkerDevice marker, const unsigned int remainingSurfelsCount, const unsigned int mapCols, const unsigned int mapRows, unsigned int* remainingIndicator)
{
	marker.processMarkingObservedOnly(remainingSurfelsCount, mapCols, mapRows, remainingIndicator);
}
void SparseSurfelFusion::ReinitRemainingSurfelMarker::prepareMarkerArguments(device::ReinitRemainingMarkerDevice& marker) {
	marker.liveGeometry.vertexConfidence = m_surfel_geometry.liveVertexConfidence.ArrayView();
	marker.liveGeometry.normalRadius = m_surfel_geometry.liveNormalRadius.RawPtr();
	marker.liveGeometry.colorTime = m_surfel_geometry.colorTime.RawPtr();
	marker.liveGeometry.surfelKnn = m_surfel_geometry.surfelKnn.RawPtr();
	for (int i = 0; i < devicesCount; i++) {
		marker.cameraObservation[i].vertexMap = m_observation.vertexConfidenceMap[i];
		marker.cameraObservation[i].normalMap = m_observation.normalRadiusMap[i];
		marker.cameraObservation[i].foregroundMask = m_observation.foregroundMask[i];

		marker.world2camera[i] = m_world2camera[i];
		marker.intrinsic[i] = m_intrinsic[i];
		marker.InitialCameraSE3Inverse[i] = InitialCameraSE3[i].inverse();
	}

}

void SparseSurfelFusion::ReinitRemainingSurfelMarker::prepareMarkerAndReplaceArguments(void* raw_marker, unsigned int cameraID)
{
	device::ReinitSelectValidObervationDevice& marker = *((device::ReinitSelectValidObervationDevice*)raw_marker);
	marker.liveGeometryInterface.VertexNum = m_surfel_geometry.liveVertexConfidence.ArrayView().Size();
	marker.liveGeometryInterface.vertexConfidence = m_surfel_geometry.liveVertexConfidence.RawPtr();
	marker.liveGeometryInterface.normalRadius = m_surfel_geometry.liveNormalRadius.RawPtr();
	marker.liveGeometryInterface.colorTime = m_surfel_geometry.colorTime.RawPtr();
	marker.liveGeometryInterface.surfelKnn = m_surfel_geometry.surfelKnn.RawPtr();
	marker.CameraID = cameraID;

	for (int i = 0; i < devicesCount; i++) {
		marker.cameraObservation.vertexMap[i] = m_observation.vertexConfidenceMap[i];
		marker.cameraObservation.normalMap[i] = m_observation.normalRadiusMap[i];
		marker.cameraObservation.foregroundMask[i] = m_observation.foregroundMask[i];
		marker.world2camera[i] = m_world2camera[i];
		marker.camera2world[i] = m_world2camera[i].inverse();
		marker.InitialCameraSE3[i] = InitialCameraSE3[i];
		marker.InitialCameraSE3Inverse[i] = InitialCameraSE3[i].inverse();
		marker.intrinsic[i] = m_intrinsic[i];
	}

}


void SparseSurfelFusion::ReinitRemainingSurfelMarker::MarkRemainingSurfelObservedOnly(cudaStream_t stream) {
	device::ReinitRemainingMarkerDevice marker;
	prepareMarkerArguments(marker);
	
	const unsigned int remainingSurfelsCount = m_remaining_surfel_indicator.Size();
	dim3 block(256, 1);
	dim3 grid(divUp(remainingSurfelsCount, block.x), devicesCount);
	device::markReinitRemainingSurfelObservedOnlyKernel << <grid, block, 0, stream >> > (marker, remainingSurfelsCount, mapCols, mapRows, m_remaining_surfel_indicator.RawPtr());

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));

#endif
}


void SparseSurfelFusion::ReinitRemainingSurfelMarker::MarkRemainingSurfelNodeError(
	const NodeAlignmentError & node_error, 
	float threshold, 
	cudaStream_t stream
) {
//	//Construct the argument
//	device::ReinitRemainingMarkerDevice marker;
//	prepareMarkerArguments((void*)&marker);
//
//	//Invoke the kernel
//	dim3 blk(256);
//	dim3 grid(divUp(m_remaining_surfel_indicator.Size(), blk.x));
//	device::markReinitRemainingSurfelNodeErrorKernel<<<grid, blk, 0, stream>>>(
//		marker,
//		node_error, 
//		threshold,
//		m_remaining_surfel_indicator.RawPtr()
//	);
//
//
//	//Sync and check error
//#if defined(CUDA_DEBUG_SYNC_CHECK)
//	cudaSafeCall(cudaStreamSynchronize(stream));
//	cudaSafeCall(cudaGetLastError());
//#endif
}