#include <math/MatUtils.h>
#include "DoubleBufferCompactor.h"
#include <device_launch_parameters.h>
#include <visualization/Visualizer.h>

namespace SparseSurfelFusion { 
	namespace device {

		__device__ unsigned int devicesCount = MAX_CAMERA_COUNT;

		struct LiveSurfelKNNCompactionOutput {
			float4* live_vertex_confid;
			float4* live_normal_radius;
			float4* color_time;
			ushort4* surfel_knn;
			float4* surfel_knnweight;
		};
	
		struct LiveSurfelCompactionOutputInterface {
			LiveSurfelKNNCompactionOutput compactOutputInterface[MAX_CAMERA_COUNT];
		};

		__global__ void compactRemainingAndAppendedSurfelKNNKernel(
			const AppendedObservationSurfelKNN appended_observation,
			const RemainingLiveSurfel remaining_surfel,
			const RemainingSurfelKNN remaining_knn,
			const unsigned int* num_remaining_surfels,
			//The output
			device::LiveSurfelCompactionOutputInterface compaction_output
		) {
			//Query the size at first
			const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
			const unsigned int unfiltered_remaining_size = remaining_surfel.remainingIndicator.Size();
			const unsigned int unfiltered_appended_size = appended_observation.validityIndicator.Size();

			//There is only two types, thus just let it go...
			if (idx < unfiltered_remaining_size) {
				if (remaining_surfel.remainingIndicator[idx] > 0) {
					const unsigned int offset = remaining_surfel.remainingIndicatorPrefixsum[idx] - 1;
					for (int i = 0; i < device::devicesCount; i++) {
						compaction_output.compactOutputInterface[i].live_vertex_confid[offset] = remaining_surfel.liveVertexConfidence[idx];
						compaction_output.compactOutputInterface[i].live_normal_radius[offset] = remaining_surfel.liveNormalRadius[idx];
						compaction_output.compactOutputInterface[i].color_time[offset] = remaining_surfel.colorTime[idx];
						compaction_output.compactOutputInterface[i].surfel_knn[offset] = remaining_knn.surfelKnn[idx];
						compaction_output.compactOutputInterface[i].surfel_knnweight[offset] = remaining_knn.surfelKnnWeight[idx];
					}
				}
			}
			else if (idx >= unfiltered_remaining_size && idx < (unfiltered_remaining_size + unfiltered_appended_size)) {
				const unsigned int append_idx = idx - unfiltered_remaining_size;
				if (appended_observation.validityIndicator[append_idx] > 0) {
					const unsigned int offset = appended_observation.validityIndicatorPrefixsum[append_idx] + (*num_remaining_surfels) - 1;
					for (int i = 0; i < device::devicesCount; i++) {
						compaction_output.compactOutputInterface[i].live_vertex_confid[offset] = appended_observation.surfelVertexConfidence[append_idx];
						compaction_output.compactOutputInterface[i].live_normal_radius[offset] = appended_observation.surfelNormalRadius[append_idx];
						compaction_output.compactOutputInterface[i].color_time[offset] = appended_observation.surfelColorTime[append_idx];
						compaction_output.compactOutputInterface[i].surfel_knn[offset] = appended_observation.surfelKnn[append_idx];
						compaction_output.compactOutputInterface[i].surfel_knnweight[offset] = appended_observation.surfelKnnWeight[append_idx];
					}
				}
			}
		}

		//The method and kernel for compaction of the geometry
		struct ReinitCompactionOutput {
			float4* reference_vertex_confid;
			float4* reference_normal_radius;
			float4* live_vertex_confid;
			float4* live_normal_radius;
			float4* color_time;
		};

		struct ReinitCompactionOutputInterface {
			ReinitCompactionOutput compactOutputInterface[MAX_CAMERA_COUNT];
			mat34 World2Camera[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		};

		__global__ void compactReinitSurfelKernel(
			const ReinitAppendedObservationSurfel appended_observation,
			const RemainingLiveSurfel remaining_surfel,
			const unsigned int* num_remaining_surfels,
			const unsigned int image_cols,
			const unsigned int image_rows,
			const unsigned int cameraNum,
			device::ReinitCompactionOutputInterface compaction_output
		) {
			//Query the size at first
			const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
			const unsigned int unfiltered_remaining_size = remaining_surfel.remainingIndicator.Size();
			const unsigned int unfiltered_appended_size = appended_observation.validityIndicator.Size();
			const unsigned int unfiltered_observed_appened_size = unfiltered_appended_size / 2;
			const unsigned int clipedImageSize = image_cols * image_rows;

			//There is only two types, thus just let it go...
			//待保留的（融合的+需保留的）
			if (idx < unfiltered_remaining_size) {
				if (remaining_surfel.remainingIndicator[idx] == 1) {
					const unsigned int offset = remaining_surfel.remainingIndicatorPrefixsum[idx] - 1;
					const float4 vertex_confid = remaining_surfel.liveVertexConfidence[idx];
					const float4 normal_radius = remaining_surfel.liveNormalRadius[idx];
					for (int i = 0; i < device::devicesCount; i++) {
						float3 vertex_confid_world2camera = compaction_output.World2Camera[i].rot * vertex_confid + compaction_output.World2Camera[i].trans;
						float3 normal_radius_world2camera = compaction_output.World2Camera[i].rot * normal_radius;
						compaction_output.compactOutputInterface[i].reference_vertex_confid[offset] = make_float4(vertex_confid_world2camera.x, vertex_confid_world2camera.y, vertex_confid_world2camera.z, vertex_confid.w);
						compaction_output.compactOutputInterface[i].live_vertex_confid[offset] = make_float4(vertex_confid_world2camera.x, vertex_confid_world2camera.y, vertex_confid_world2camera.z, vertex_confid.w);
						compaction_output.compactOutputInterface[i].reference_normal_radius[offset] = make_float4(normal_radius_world2camera.x, normal_radius_world2camera.y, normal_radius_world2camera.z, normal_radius.w);
						compaction_output.compactOutputInterface[i].live_normal_radius[offset] = make_float4(normal_radius_world2camera.x, normal_radius_world2camera.y, normal_radius_world2camera.z, normal_radius.w);
						compaction_output.compactOutputInterface[i].color_time[offset] = remaining_surfel.colorTime[idx];
					}
				}
			}
			else if (idx >= unfiltered_remaining_size && idx < unfiltered_remaining_size + unfiltered_observed_appened_size){
				const unsigned int appendArrayOffset = idx - unfiltered_remaining_size;
				const unsigned int CameraID = appendArrayOffset / clipedImageSize;
				const unsigned int appendIdx = appendArrayOffset - CameraID * clipedImageSize;
				if (appended_observation.validityIndicator[appendArrayOffset] == 1) {
					const unsigned int offset = appended_observation.validityIndicatorPrefixsum[appendArrayOffset] + (*num_remaining_surfels) - 1;

					// 获得这些新增面元像素坐标系中的位置
					const unsigned int x = appendIdx % image_cols;
					const unsigned int y = appendIdx / image_cols;

					// 获取CameraID号相机的vertex_confid
					const float4 vertex_confid = tex2D<float4>(appended_observation.observedVertexMap[CameraID], x, y);
					const float4 normal_radius = tex2D<float4>(appended_observation.observedNormalMap[CameraID], x, y);

					// 将CameraID号相机的vertex_confid转换到0号坐标系(Canonical域)中
					const float3 CanonicalVertex = compaction_output.InitialCameraSE3[CameraID].rot * vertex_confid + compaction_output.InitialCameraSE3[CameraID].trans;
					const float3 CanonicalNormal = compaction_output.InitialCameraSE3[CameraID].rot * normal_radius;
					const float4 color_time = tex2D<float4>(appended_observation.observedColorMap[CameraID], x, y);
					for (int i = 0; i < device::devicesCount; i++) {
						compaction_output.compactOutputInterface[i].reference_vertex_confid[offset] = make_float4(CanonicalVertex.x, CanonicalVertex.y, CanonicalVertex.z, vertex_confid.w);
						compaction_output.compactOutputInterface[i].live_vertex_confid[offset] = make_float4(CanonicalVertex.x, CanonicalVertex.y, CanonicalVertex.z, vertex_confid.w);
						compaction_output.compactOutputInterface[i].reference_normal_radius[offset] = make_float4(CanonicalNormal.x, CanonicalNormal.y, CanonicalNormal.z, normal_radius.w);
						compaction_output.compactOutputInterface[i].live_normal_radius[offset] = make_float4(CanonicalNormal.x, CanonicalNormal.y, CanonicalNormal.z, normal_radius.w);
						compaction_output.compactOutputInterface[i].color_time[offset] = color_time;
					}
				}
			}
			else if (idx >= unfiltered_remaining_size + unfiltered_observed_appened_size && idx < unfiltered_remaining_size + unfiltered_appended_size) {
				const unsigned int appendArrayOffset = idx - unfiltered_remaining_size;
				const unsigned int interAppendArrayOffset = appendArrayOffset - unfiltered_observed_appened_size;
				const unsigned int CameraID = interAppendArrayOffset / clipedImageSize;
				const unsigned int appendIdx = appendArrayOffset - (CameraID + cameraNum) * clipedImageSize;
				if (appended_observation.validityIndicator[appendArrayOffset] == 1) {	// 因为是同一个数组，所以是同一个appendArrayOffset
					const unsigned int offset = appended_observation.validityIndicatorPrefixsum[appendArrayOffset] + (*num_remaining_surfels) - 1;

					// 获得这些新增面元像素坐标系中的位置
					const unsigned int x = appendIdx % image_cols;
					const unsigned int y = appendIdx / image_cols;
					// 获取CameraID号相机的vertex_confid
					const float4 interVertexConfidence = appended_observation.interVertexMap[CameraID](y, x);
					const float4 interNormalRadius = appended_observation.interNormalMap[CameraID](y, x);

					// 将CameraID号相机的vertex_confid转换到0号坐标系(Canonical域)中
					const float3 CanonicalVertex = compaction_output.InitialCameraSE3[CameraID].rot * interVertexConfidence + compaction_output.InitialCameraSE3[CameraID].trans;
					const float3 CanonicalNormal = compaction_output.InitialCameraSE3[CameraID].rot * interNormalRadius;
					const float4 colorTime = appended_observation.interColorMap[CameraID](y, x);

					//if(is_zero_vertex(interVertexConfidence))printf("Indicator = %d   interVertexConfidence(%.3f, %.3f, %.3f, %.3f)\n", appended_observation.validityIndicator[appendArrayOffset], interVertexConfidence.x, interVertexConfidence.y, interVertexConfidence.z, interVertexConfidence.w);
					for (int i = 0; i < device::devicesCount; i++) {
						compaction_output.compactOutputInterface[i].reference_vertex_confid[offset] = make_float4(CanonicalVertex.x, CanonicalVertex.y, CanonicalVertex.z, interVertexConfidence.w);
						compaction_output.compactOutputInterface[i].live_vertex_confid[offset] = make_float4(CanonicalVertex.x, CanonicalVertex.y, CanonicalVertex.z, interVertexConfidence.w);
						compaction_output.compactOutputInterface[i].reference_normal_radius[offset] = make_float4(CanonicalNormal.x, CanonicalNormal.y, CanonicalNormal.z, interNormalRadius.w);
						compaction_output.compactOutputInterface[i].live_normal_radius[offset] = make_float4(CanonicalNormal.x, CanonicalNormal.y, CanonicalNormal.z, interNormalRadius.w);
						compaction_output.compactOutputInterface[i].color_time[offset] = colorTime;
					}
				}
			}
		}
	}
}

void SparseSurfelFusion::DoubleBufferCompactor::PerformCompactionGeometryKNNSync(unsigned int& num_valid_remaining_surfels, unsigned int& num_valid_append_surfels, cudaStream_t stream)
{
	const size_t unfiltered_remaining_surfel_size = m_remaining_surfel.remainingIndicator.Size();
	const size_t unfiltered_appended_surfel_size = m_appended_surfel_knn.validityIndicator.Size();

	//The number of remaining surfel is the last element of inclusive sum
	const unsigned int* num_remaining_surfel_dev = m_remaining_surfel.remainingIndicatorPrefixsum + unfiltered_remaining_surfel_size - 1;
	const unsigned int* num_appended_surfel_dev = m_appended_surfel_knn.validityIndicatorPrefixsum + unfiltered_appended_surfel_size - 1;
	//Sync and query the size
	CHECKCUDA(cudaMemcpyAsync(&num_valid_remaining_surfels, num_remaining_surfel_dev, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	if (unfiltered_appended_surfel_size != 0) {
		CHECKCUDA(cudaMemcpyAsync(&num_valid_append_surfels, num_appended_surfel_dev, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	}
	else {
		num_valid_append_surfels = 0;
	}
	CHECKCUDA(cudaStreamSynchronize(stream));

	for (int i = 0; i < devicesCount; i++) {
		m_compact_to_geometry[i][updatedGeometryIndex]->ResizeValidSurfelArrays(num_valid_append_surfels + num_valid_remaining_surfels);
	}
	//Construct the output
	//这是结果, 里边都是指针, 就是另一个surfelgeometry的初始化！！
	device::LiveSurfelCompactionOutputInterface compaction_output;
	for (int i = 0; i < devicesCount; i++) {
		compaction_output.compactOutputInterface[i].live_vertex_confid = m_compact_to_geometry[i][updatedGeometryIndex]->LiveVertexConfidence.Ptr();
		compaction_output.compactOutputInterface[i].live_normal_radius = m_compact_to_geometry[i][updatedGeometryIndex]->LiveNormalRadius.Ptr();
		compaction_output.compactOutputInterface[i].color_time = m_compact_to_geometry[i][updatedGeometryIndex]->ColorTime.Ptr();
		compaction_output.compactOutputInterface[i].surfel_knn = m_compact_to_geometry[i][updatedGeometryIndex]->surfelKNN.Ptr();
		compaction_output.compactOutputInterface[i].surfel_knnweight = m_compact_to_geometry[i][updatedGeometryIndex]->surfelKNNWeight.Ptr();
	}

#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif

	dim3 block(256);
	dim3 grid(divUp(unfiltered_appended_surfel_size + unfiltered_remaining_surfel_size, block.x));
	device::compactRemainingAndAppendedSurfelKNNKernel << <grid, block, 0, stream >> > (
		m_appended_surfel_knn,
		m_remaining_surfel,
		m_remaining_knn,
		num_remaining_surfel_dev,
		compaction_output
	);
	CHECKCUDA(cudaStreamSynchronize(stream));

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif


#ifdef DEBUG_RUNNING_INFO
	printf("Compact Geometry数量 = %u\n", m_compact_to_geometry[0][updatedGeometryIndex]->getCanonicalVertexConfidence().Size(), num_valid_append_surfels + num_valid_remaining_surfels);
#endif // DEBUG_RUNNING_INFO
}

void SparseSurfelFusion::DoubleBufferCompactor::PerformComapctionGeometryOnlySync(unsigned int& num_valid_remaining_surfels, unsigned int& num_valid_append_surfels, unsigned int* number, mat34* world2camera, cudaStream_t stream)
{
	//The number of remaining surfel is the last element of inclusive sum
	const unsigned int* num_remaining_surfel_dev = m_remaining_surfel.remainingIndicatorPrefixsum + m_remaining_surfel.remainingIndicator.Size() - 1;
	const unsigned int* numberDevice[MAX_CAMERA_COUNT];
	device::ReinitCompactionOutputInterface compaction_output;	// 经压缩后的输出
	//printf("Buffer = %d\n", m_compact_to_geometry[0][updatedGeometryIndex]->CanonicalVertexConfidence.BufferSize());
	for (int i = 0; i < devicesCount; i++) {
		numberDevice[i] = m_reinit_append_surfel.validityIndicatorPrefixsum + (i + 1) * m_image_cols * m_image_rows - 1;
		FUNCTION_CHECK(!(m_compact_to_geometry[i][updatedGeometryIndex] == nullptr));
		compaction_output.compactOutputInterface[i].reference_vertex_confid = m_compact_to_geometry[i][updatedGeometryIndex]->CanonicalVertexConfidence.Ptr();
		compaction_output.compactOutputInterface[i].reference_normal_radius = m_compact_to_geometry[i][updatedGeometryIndex]->CanonicalNormalRadius.Ptr();
		compaction_output.compactOutputInterface[i].live_vertex_confid = m_compact_to_geometry[i][updatedGeometryIndex]->LiveVertexConfidence.Ptr();
		compaction_output.compactOutputInterface[i].live_normal_radius = m_compact_to_geometry[i][updatedGeometryIndex]->LiveNormalRadius.Ptr();
		compaction_output.compactOutputInterface[i].color_time = m_compact_to_geometry[i][updatedGeometryIndex]->ColorTime.Ptr();
		compaction_output.World2Camera[i] = world2camera[i];
		compaction_output.InitialCameraSE3[i] = InitialCameraSE3[i];
	}

	//Seems ready for compaction
	const unsigned int unfiltered_remaining_surfel_size = m_remaining_surfel.remainingIndicator.Size();
	const unsigned int unfiltered_appended_surfel_size = m_reinit_append_surfel.validityIndicator.Size();//这个大小应该是600*360*2
	dim3 block(256);
	dim3 grid(divUp(unfiltered_appended_surfel_size + unfiltered_remaining_surfel_size, block.x));
	device::compactReinitSurfelKernel << <grid, block, 0, stream >> > (
		m_reinit_append_surfel,
		m_remaining_surfel,
		num_remaining_surfel_dev,
		m_image_cols,
		m_image_rows,
		devicesCount,
		compaction_output
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif

	//Sync and query the size
	CHECKCUDA(cudaMemcpyAsync(&num_valid_remaining_surfels, num_remaining_surfel_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaMemcpyAsync(
		&num_valid_append_surfels,
		m_reinit_append_surfel.validityIndicatorPrefixsum + m_reinit_append_surfel.validityIndicator.Size() - 1,
		sizeof(unsigned int),
		cudaMemcpyDeviceToHost,
		stream
	));
	unsigned int numberHost[MAX_CAMERA_COUNT];
	for (int i = 0; i < devicesCount; i++) {
		CHECKCUDA(cudaMemcpyAsync(&numberHost[i], numberDevice[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
		m_compact_to_geometry[i][updatedGeometryIndex]->ResizeValidSurfelArrays(num_valid_append_surfels + num_valid_remaining_surfels);
	}
	CHECKCUDA(cudaStreamSynchronize(stream));

	for (int i = 0; i < devicesCount; i++) {
		if (i == 0) number[i] = numberHost[i];
		else number[i] = numberHost[i] - numberHost[i - 1];
	}

#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif

}
