#include "Renderer.h"
#include <core/NonRigidSolver/solver_constants.h>
#include <base/data_transfer.h>

namespace SparseSurfelFusion {
	namespace device {

		__global__ void FilterSolverMapsIndexMapKernel(SolverMapFilteringInterface solverMapInterface, const unsigned int mapCols, const unsigned int mapRows, const unsigned int devicesCount, unsigned int* singleViewTotalSurfels, unsigned int* singleViewFilteredSurfels,
			unsigned int* totalValidSurfels, unsigned int* outlierSurfels) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= mapCols || y >= mapRows || CameraID >= devicesCount) return;
			// Live域的vertex转到CameraID号相机的像素空间
			const unsigned int surfelIndex = tex2D<unsigned int>(solverMapInterface.warpIndexMap[CameraID], x, y);
			bool checkMask = false;
			bool checkSquaredDis = false;
			bool checkNormal = false;

			unsigned int thisSolverMapSurfelFrom = 0xFFFFFFFF;
			if (surfelIndex != d_invalid_index) {
				float4 warpColorViewTime = tex2D<float4>(solverMapInterface.warpColorViewTimeMap[CameraID], x, y);
				atomicAdd(&singleViewTotalSurfels[CameraID], 1);
				thisSolverMapSurfelFrom = decodeSolverMapCameraView(warpColorViewTime);
				
				// 第一个条件：验证这个像素位置（x，y）在不在mask的有效区域里
				const unsigned char maskValue = tex2D<unsigned char>(solverMapInterface.observedForeground[CameraID], x, y);
				if (maskValue == (unsigned char)1) checkMask = true;

				// 第二个条件：验证模型面元与对应深度面元的距离是否在条件内，liveVertexCameraLiveField是将0号相机空间Live域的点通过位姿反变换到CameraID号相机空间的Live域上,这里必须加上world2camera，因为map的输出都是没用对齐当前帧的数据，如果要与当前帧的深度面元数据匹配，必须world2camera
				const float4 liveVertexCameraLiveField = tex2D<float4>(solverMapInterface.warpVertexMap[CameraID], x, y);
				float3 liveVertexCameraView = solverMapInterface.Live2Camera[CameraID].rot * liveVertexCameraLiveField + solverMapInterface.Live2Camera[CameraID].trans;
				const float4 observedVertex = tex2D<float4>(solverMapInterface.observedVertexMap[CameraID], x, y);
				const float dis = squared_distance(liveVertexCameraView, observedVertex);
				if (dis < d_distance_max) checkSquaredDis = true;
					
				// 第三个条件：模型面元的法向量与深度面元的法向量间的夹角要够小 要用world2camera
				const float4 liveNormalCameraLiveField = tex2D<float4>(solverMapInterface.warpNormalMap[CameraID], x, y);
				const float4 observedNormal = tex2D<float4>(solverMapInterface.observedNormalMap[CameraID], x, y);
				float3 liveNormalCameraView = solverMapInterface.Live2Camera[CameraID].rot * liveNormalCameraLiveField;
				if (dotxyz(liveNormalCameraView, observedNormal) > d_live_observed_normal_cos_threshold) checkNormal = true;
			
				// 统计误差点
				atomicAdd(&totalValidSurfels[CameraID], 1);
				if (dis > 0.025f * 0.025f || dotxyz(liveNormalCameraView, observedNormal) <= 0.8f || !checkMask) {
					atomicAdd(&outlierSurfels[CameraID], 1);
				}
			}
#ifndef REBUILD_WITHOUT_BACKGROUND
			checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND

			if (CameraID == thisSolverMapSurfelFrom && checkMask && checkSquaredDis && checkNormal) {
				surf2Dwrite(surfelIndex, solverMapInterface.filteredIndexMap[CameraID], x * sizeof(unsigned int), y);
			}
			else {
				if (surfelIndex != d_invalid_index) atomicAdd(&singleViewFilteredSurfels[CameraID], 1);
				surf2Dwrite(d_invalid_index, solverMapInterface.filteredIndexMap[CameraID], x * sizeof(unsigned int), y);
			}
		}

		__global__ void FilterFusionMapsIndexMapKernel(FusionMapFilteringInterface fusionMapInterface, const unsigned int mapCols, const unsigned int mapRows, const unsigned int devicesCount, const unsigned int fusionMapScale) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= mapCols || y >= mapRows || CameraID >= devicesCount) return;

			const unsigned int fusionMapCenterX = fusionMapScale * x;
			const unsigned int fusionMapCenterY = fusionMapScale * y;

			const float4 observedVertex = tex2D<float4>(fusionMapInterface.observedVertexMap[CameraID], x, y);
			const float4 observedNormal = tex2D<float4>(fusionMapInterface.observedNormalMap[CameraID], x, y);
			const unsigned char maskValue = tex2D<unsigned char>(fusionMapInterface.observedForeground[CameraID], x, y);
			bool checkMask = (maskValue == (unsigned char)1);
#ifndef REBUILD_WITHOUT_BACKGROUND
			checkMask = true;
#endif // !REBUILD_WITHOUT_BACKGROUND
			if (checkMask) {
				//上采样遍历
				for (unsigned int dx = 0; dx < fusionMapScale; dx++) {
					for (unsigned int dy = 0; dy < fusionMapScale; dy++) {
						unsigned int upsamplingMapX = fusionMapCenterX + dx;
						unsigned int upsamplingMapY = fusionMapCenterY + dy;
						bool checkIndex = false;
						bool checkDistance = false;
						bool checkNormal = false;
						float4 warpColorViewTime = tex2D<float4>(fusionMapInterface.warpColorViewTimeMap[CameraID], upsamplingMapX, upsamplingMapY);

						unsigned int thisFusionMapSurfelFrom = decodeFusionMapCameraView(warpColorViewTime);
						//printf("thisFusionMapSurfelFrom = %d\n", thisFusionMapSurfelFrom);
						//printf("ColorViewTimeMap[%d, %d] = (%.5f,%.5f,%.5f,%.5f)\n", upsamplingMapX, upsamplingMapY, A.x, A.y, A.z, A.w);
						//条件1: fusionMap上的有效点
						const unsigned int surfelIndex = tex2D<unsigned int>(fusionMapInterface.warpIndexMap[CameraID], upsamplingMapX, upsamplingMapY);

						if (surfelIndex != d_invalid_index) checkIndex = true;
						
						//条件2: 在CameraID视角下，FusionMap中的Live域的点应该与当前帧观察的深度vertexMap的面元距离相近
						const float4 liveVertexCameraLiveField = tex2D<float4>(fusionMapInterface.warpVertexMap[CameraID], upsamplingMapX, upsamplingMapY);
						float3 liveVertexCameraView = fusionMapInterface.World2Camera[CameraID].rot * liveVertexCameraLiveField + fusionMapInterface.World2Camera[CameraID].trans;
						const float squaredDis = squared_distance(liveVertexCameraView, observedVertex);
						if (squaredDis < d_distance_max) checkDistance = true;

						//条件3 在CameraID视角下，FusionMap中的Live域的点应该与当前帧观察的深度vertexMap的面元法线相近
						const float4 liveNormalCameraLiveField = tex2D<float4>(fusionMapInterface.warpNormalMap[CameraID], upsamplingMapX, upsamplingMapY);
						float3 liveNormalCameraView = fusionMapInterface.World2Camera[CameraID].rot * liveNormalCameraLiveField;
						if (dotxyz(liveNormalCameraView, observedNormal) > d_live_observed_normal_cos_threshold) checkNormal = true;

						////判断条件，写入结果
						//if (checkIndex && checkDistance && checkNormal) {
						//	surf2Dwrite(surfelIndex, fusionMapInterface.filteredIndexMap[CameraID], upsamplingMapX * sizeof(unsigned int), upsamplingMapY);
						//}
						//else {
						//	surf2Dwrite(d_invalid_index, fusionMapInterface.filteredIndexMap[CameraID], upsamplingMapX * sizeof(unsigned int), upsamplingMapY);
						//}

						if (thisFusionMapSurfelFrom == CameraID && checkIndex && checkDistance && checkNormal) {
							surf2Dwrite(surfelIndex, fusionMapInterface.filteredIndexMap[CameraID], upsamplingMapX * sizeof(unsigned int), upsamplingMapY);
						}
						else {
							surf2Dwrite(d_invalid_index, fusionMapInterface.filteredIndexMap[CameraID], upsamplingMapX * sizeof(unsigned int), upsamplingMapY);
						}
					}
				}
			}
			else {
				for (unsigned int dx = 0; dx < fusionMapScale; dx++) {
					for (unsigned int dy = 0; dy < fusionMapScale; dy++) {
						unsigned int upsamplingMapX = fusionMapCenterX + dx;
						unsigned int upsamplingMapY = fusionMapCenterY + dy;

						surf2Dwrite(d_invalid_index, fusionMapInterface.filteredIndexMap[CameraID], upsamplingMapX * sizeof(unsigned int), upsamplingMapY);
					}
				}
			}
		}
	}
}


void SparseSurfelFusion::Renderer::FilterSolverMapsIndexMap(SolverMaps* maps, CameraObservation& observation, mat34* Live2Camera, float filterRatio, cudaStream_t stream)
{
	// 清空上一帧数据
	CHECKCUDA(cudaMemsetAsync(singleViewTotalSurfels.ptr(), 0, sizeof(unsigned int) * deviceCount, stream));
	CHECKCUDA(cudaMemsetAsync(singleViewFilteredSurfels.ptr(), 0, sizeof(unsigned int) * deviceCount, stream));

	unsigned int* totalValidData_Dev = NULL;
	unsigned int* outlierData_Dev = NULL;

	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&totalValidData_Dev), sizeof(unsigned int) * MAX_CAMERA_COUNT, stream));
	CHECKCUDA(cudaMemsetAsync(totalValidData_Dev, 0, sizeof(unsigned int) * MAX_CAMERA_COUNT, stream));
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&outlierData_Dev), sizeof(unsigned int) * MAX_CAMERA_COUNT, stream));
	CHECKCUDA(cudaMemsetAsync(outlierData_Dev, 0, sizeof(unsigned int) * MAX_CAMERA_COUNT, stream));

	for (int i = 0; i < deviceCount; i++) {
		solverMapFilteringInterface.warpVertexMap[i] = maps[i].warp_vertex_map;
		solverMapFilteringInterface.warpNormalMap[i] = maps[i].warp_normal_map;
		solverMapFilteringInterface.warpIndexMap[i] = maps[i].index_map;
		solverMapFilteringInterface.warpColorViewTimeMap[i] = maps[i].normalized_rgb_map;

		solverMapFilteringInterface.observedVertexMap[i] = observation.PreviousVertexConfidenceMap[i];
		solverMapFilteringInterface.observedNormalMap[i] = observation.PreviousNormalRadiusMap[i];
		solverMapFilteringInterface.observedForeground[i] = observation.foregroundMaskPrevious[i];

		solverMapFilteringInterface.Live2Camera[i] = Live2Camera[i];

		solverMapFilteringInterface.filteredIndexMap[i] = solverMapFilteredIndexMap[i].surface;
	}
	dim3 block(16, 16, 1);
	dim3 grid(divUp(imageWidth, block.x), divUp(imageHeight, block.y), divUp(deviceCount, block.z));
	device::FilterSolverMapsIndexMapKernel << <grid, block, 0, stream >> > (solverMapFilteringInterface, imageWidth, imageHeight, deviceCount, singleViewTotalSurfels.ptr(), singleViewFilteredSurfels.ptr(), totalValidData_Dev, outlierData_Dev);
	CHECKCUDA(cudaMemcpyAsync(singleViewTotalSurfelsHost, singleViewTotalSurfels.ptr(), sizeof(unsigned int) * deviceCount, cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaMemcpyAsync(singleViewFilteredSurfelsHost, singleViewFilteredSurfels.ptr(), sizeof(unsigned int) * deviceCount, cudaMemcpyDeviceToHost, stream));

	CHECKCUDA(cudaStreamSynchronize(stream));
#ifdef USE_DYNAMICAL_REFRESH
	for (int i = 0; i < deviceCount; i++) {
		maps[i].index_map = solverMapFilteredIndexMap[i].texture;
	}
#endif // USE_DYNAMICAL_REFRESH



	CHECKCUDA(cudaMemcpyAsync(totalValidData, totalValidData_Dev, sizeof(unsigned int) * MAX_CAMERA_COUNT, cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaMemcpyAsync(outlierData, outlierData_Dev, sizeof(unsigned int) * MAX_CAMERA_COUNT, cudaMemcpyDeviceToHost, stream));


	shouldRefresh = CheckFilteredSolverMapsSurfelRatio(filterRatio);

	CHECKCUDA(cudaFreeAsync(totalValidData_Dev, stream));
	CHECKCUDA(cudaFreeAsync(outlierData_Dev, stream));

}
bool SparseSurfelFusion::Renderer::CheckFilteredSolverMapsSurfelRatio(float ratio)
{
	bool refresh = false;
	if (ratio < 0.0f) ratio = 0.0f;
	else if (ratio > 1.0f) ratio = 1.0f;
	for (int i = 0; i < deviceCount; i++) {
		float filteredRatio = singleViewFilteredSurfelsHost[i] * 1.0f / singleViewTotalSurfelsHost[i];
		if (filteredRatio > ratio) {
			refresh = true;
		}
#ifdef DEBUG_RUNNING_INFO
		printf("CameraID = %d   Ratio = %.3f\n", i, filteredRatio);
#endif // DEBUG_RUNNING_INFO
	}

	//float totalValidSurfels = 0;
	//float totalOutlierSurfels = 0;
	//for (int i = 0; i < deviceCount; i++) {
	//	totalValidSurfels += totalValidData[i] * 1.0f;
	//	totalOutlierSurfels += outlierData[i] * 1.0f;
	//}
	//printf("%.5f, ", totalOutlierSurfels / totalValidSurfels);

	return refresh;
}
void SparseSurfelFusion::Renderer::FilterFusionMapsIndexMap(FusionMaps* maps, CameraObservation& observation, mat34* World2Camera, cudaStream_t stream)
{

	for (int i = 0; i < deviceCount; i++) {
		fusionMapFilteringInterface.warpVertexMap[i] = maps[i].warp_vertex_map;
		fusionMapFilteringInterface.warpNormalMap[i] = maps[i].warp_normal_map;
		fusionMapFilteringInterface.warpIndexMap[i] = maps[i].index_map;
		fusionMapFilteringInterface.warpColorViewTimeMap[i] = maps[i].color_time_map;

		fusionMapFilteringInterface.observedVertexMap[i] = observation.vertexConfidenceMap[i];
		fusionMapFilteringInterface.observedNormalMap[i] = observation.normalRadiusMap[i];
		fusionMapFilteringInterface.observedForeground[i] = observation.foregroundMask[i];

		fusionMapFilteringInterface.World2Camera[i] = World2Camera[i];
		fusionMapFilteringInterface.initialCameraSE3Inverse[i] = InitialCameraSE3[i].inverse();

		fusionMapFilteringInterface.filteredIndexMap[i] = fusionMapFilteredIndexMap[i].surface;
	}

	dim3 block(16, 16, 1);
	dim3 grid(divUp(imageWidth, block.x), divUp(imageHeight, block.y), divUp(deviceCount, block.z));
	device::FilterFusionMapsIndexMapKernel << <grid, block, 0, stream >> > (fusionMapFilteringInterface, imageWidth, imageHeight, deviceCount, FusionMapScale);
	CHECKCUDA(cudaStreamSynchronize(stream));
#ifdef USE_DYNAMICAL_REFRESH
	for (int i = 0; i < deviceCount; i++) {
		maps[i].index_map = fusionMapFilteredIndexMap[i].texture;
	}
#endif // USE_DYNAMICAL_REFRESH



}