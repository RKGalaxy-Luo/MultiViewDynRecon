#include "sanity_check.h"
#include <base/Logging.h>
#include "solver_constants.h"
#include "DenseDepthHandler.h"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		enum {
			window_halfsize = 1,
			alignment_error_block_size = 256,
			alignment_error_warps_num = alignment_error_block_size / 32,
		};

		__device__ __forceinline__ float computeAlignmentErrorWindowSearch(
			cudaTextureObject_t depth_vertex_confid_map,
			cudaTextureObject_t depth_normal_radius_map,
			cudaTextureObject_t filter_foreground_mask,
			cudaTextureObject_t reference_vertex_map,
			cudaTextureObject_t reference_normal_map,
			cudaTextureObject_t index_map,
			const DeviceArrayView2D<KNNAndWeight> knn_map,
			const DualQuaternion* device_warp_field,
			const Intrinsic& intrinsic, 
			const mat34& initialCameraPose,
			const mat34& initialCameraPoseInv,
			const mat34& world2camera
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= knn_map.Cols() || y >= knn_map.Rows()) return 0.0f;

			//The residual value
			float alignment_error = 0.0f;
			const auto surfel_index = tex2D<unsigned>(index_map, x, y);
			if (surfel_index != d_invalid_index) {
				//Get the vertex
				const float4 can_vertex4 = tex2D<float4>(reference_vertex_map, x, y);
				const float4 can_normal4 = tex2D<float4>(reference_normal_map, x, y);
				const float3 can_vertex_world = initialCameraPose.rot * can_vertex4 + initialCameraPose.trans;
				const float3 can_normal_world = initialCameraPose.rot * can_normal4;
				const KNNAndWeight knn = knn_map(y, x);
				DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn.knn, knn.weight);
				const mat34 se3 = dq_average.se3_matrix();

				//And warp it
				const float3 warped_vertex = se3.rot * can_vertex_world + se3.trans;
				const float3 warped_normal = se3.rot * can_normal_world;

				//Transfer to the camera frame
				float3 warped_vertex_camera = world2camera.rot * warped_vertex + world2camera.trans;
				float3 warped_normal_camera = world2camera.rot * warped_normal;

				warped_vertex_camera = initialCameraPoseInv.rot * warped_vertex_camera + initialCameraPoseInv.trans;
				warped_normal_camera = initialCameraPoseInv.rot * warped_normal_camera;

				const int2 img_coord = {
					__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
					__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
				};

				//Use window search
				alignment_error = d_maximum_alignment_error;
				bool depth_vertex_found = false;
				for (int depth_y = img_coord.y - window_halfsize; depth_y <= img_coord.y + window_halfsize; depth_y++) {
					for (int depth_x = img_coord.x - window_halfsize; depth_x <= img_coord.x + window_halfsize; depth_x++) {
						const float4 depth_vertex = tex2D<float4>(depth_vertex_confid_map, depth_x, depth_y);
						const float4 depth_normal = tex2D<float4>(depth_normal_radius_map, depth_x, depth_y);
						if (!is_zero_vertex(depth_vertex) && dotxyz(warped_normal_camera, depth_normal) > 0.7f)
							depth_vertex_found = true;
						const float error = fabsf_diff_xyz(warped_vertex_camera, depth_vertex);
						if (error < alignment_error) {
							alignment_error = error;
						}
					}
				}

				//If there is no depth pixel, check the foreground mask
				if (!depth_vertex_found) {
					const float filter_foreground_value = tex2D<float>(filter_foreground_mask, img_coord.x, img_coord.y);
					if (filter_foreground_value < 0.9f) { // This is on boundary or foreground 这是在边界或前景上
						// 0.05[m] (5 cm) is the approximate maximum value (corresponded to 1.0 foreground value)
						// if the surfel is on the boundary of the image.
						// alignment_error = 0.03f * filter_foreground_value;
						alignment_error = 0.03f * filter_foreground_value;
					}
					else {
						alignment_error = d_maximum_alignment_error;
					}
				}
			}
			//Return the value for further processing
			return alignment_error;
		}

		__global__ void computeAlignmentErrorMapKernel(
			ObservationDenseDepthHandlerInterface observation,
			GeometryMapDenseDepthHandlerInterface geometryMap,
			const unsigned int knnMapCols,
			const unsigned int knnMapRows,
			const unsigned int devicesCount,
			const DualQuaternion* deviceWarpField,//节点的SE3
			//the output
			NodeAccumulatedErrorAndWeight errorAndWeight
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= knnMapCols || y >= knnMapRows || CameraID >= devicesCount) return;

			// "刚性对齐 + 非刚性对齐" 误差，无效点和对齐点 alignment_error = 0
			const float alignmentError = computeAlignmentErrorWindowSearch(
				observation.vertexMap[CameraID],
				observation.normalMap[CameraID],
				observation.filteredForegroundMap[CameraID],
				geometryMap.referenceVertexMap[CameraID],
				geometryMap.referenceNormalMap[CameraID],
				geometryMap.indexMap[CameraID],
				geometryMap.knnMap[CameraID],
				deviceWarpField,
				geometryMap.intrinsic[CameraID],
				geometryMap.InitialCameraSE3[CameraID],
				geometryMap.InitialCameraSE3Inverse[CameraID],
				geometryMap.world2Camera[CameraID]
			);

			//Write the value to surface
			surf2Dwrite(alignmentError, errorAndWeight.alignmentErrorMap[CameraID].surface, x * sizeof(float), y);
		}


		__global__ void computeNodeAlignmentErrorFromMapKernel(
			ObservationDenseDepthHandlerInterface observation,
			GeometryMapDenseDepthHandlerInterface geometryMap,
			const unsigned int knnMapCols,
			const unsigned int knnMapRows,
			const unsigned int devicesCount, 
			const DualQuaternion* deviceWarpField,//节点的SE3
			//the output
			float* nodeAccumulatedError,
			float* nodeAccumulatedWeight
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= knnMapCols || y >= knnMapRows || CameraID >= devicesCount) return;

			// "刚性对齐 + 非刚性对齐" 误差，无效点和对齐点 alignment_error = 0
			const float alignmentError = computeAlignmentErrorWindowSearch(
				observation.vertexMap[CameraID],
				observation.normalMap[CameraID],
				observation.filteredForegroundMap[CameraID],
				geometryMap.referenceVertexMap[CameraID],
				geometryMap.referenceNormalMap[CameraID],
				geometryMap.indexMap[CameraID],
				geometryMap.knnMap[CameraID],
				deviceWarpField,
				geometryMap.intrinsic[CameraID],
				geometryMap.InitialCameraSE3[CameraID],
				geometryMap.InitialCameraSE3Inverse[CameraID],
				geometryMap.world2Camera[CameraID]
			);

			// knn和knnWeight被用作蒙皮插值，里面存着的都是稠密点最邻近的节点node
			const KNNAndWeight knn = geometryMap.knnMap[CameraID](y, x);
			const unsigned short* nodeArray = (const unsigned short*)(&knn.knn);
			const float* nodeWeightArray = (const float*)(&knn.weight);
			if (alignmentError > 1e-6f) {	// "无效点"和"对齐点"的 alignmentError == 0, 因此不存在无效点被累加误差
				// 下述会存在同一节点在不同视角下误差的都出现时，误差会被累加多次
				for (int i = 0; i < 4; i++) {
					const unsigned short node = nodeArray[i];
					const float nodeWeight = nodeWeightArray[i];
					atomicAdd(&nodeAccumulatedError[node], nodeWeight * alignmentError);
					atomicAdd(&nodeAccumulatedWeight[node], nodeWeight);
				}
			}
		}

		__global__ void computeLargeErrorNodeNum(
			const float* nodeAccumulatedError,
			const float* nodeAccumulatedWeight,
			const unsigned int nodesNum,
			const float threshold,
			float* nodeUnitedError,
			unsigned int* markLargeNode
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= nodesNum) return;
			nodeUnitedError[idx] = (nodeAccumulatedError[idx] / (nodeAccumulatedWeight[idx] + 1e-4f)) / d_maximum_alignment_error;
			if (nodeUnitedError[idx] > threshold) markLargeNode[idx] = 1;
			else markLargeNode[idx] = 0;
			if (nodeAccumulatedError[idx] < 1e-10f) {
				markLargeNode[idx] = 1;			// 这个节点没一个符合条件的稠密点
				nodeUnitedError[idx] = 1.0f;	// 归一化节点Error直接设为1
			}
		}

		__global__ void collectAlignmentErrorMapFromNodeKernel(
			const float* nodeAlignmentError,
			const float* nodeAccumlateWeight,
			const unsigned int knnCols,
			const unsigned int knnRows,
			const unsigned int deviceCount,
			GeometryMapDenseDepthHandlerInterface geometryMap,
			//the output
			NodeAccumulatedErrorAndWeight errorMap
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int cameraID = threadIdx.z + blockDim.z * blockIdx.z;

			if (x >= knnCols || y >= knnRows || cameraID >= deviceCount) return;

			float filter_alignment_error = 0.0f;
			const unsigned int surfel_index = tex2D<unsigned int>(geometryMap.indexMap[cameraID], x, y);
			if (surfel_index != d_invalid_index) {
				//The knn and weight is used to interplate
				const KNNAndWeight knn = geometryMap.knnMap[cameraID](y, x);
				const unsigned short* nodeArray = (const unsigned short*)(&knn.knn);
				const float* nodeWeightArray = (const float*)(&knn.weight);

				//Load from node
				float accumlateError = 0.0f;
				float accumlateWeight = 0.0f;
				for (int i = 0; i < 4; i++) {
					const unsigned short node = nodeArray[i];
					const float nodeWeight = nodeWeightArray[i];
					const float nodeError = nodeAlignmentError[node];
					const float nodeTotalWeight = nodeAccumlateWeight[node];
					const float nodeUnitError = nodeError / nodeTotalWeight;
					accumlateError += nodeUnitError * nodeWeight;
					accumlateWeight += nodeWeight;
				}
				filter_alignment_error = accumlateError / (accumlateWeight + 1e-4f);
			}
			surf2Dwrite(filter_alignment_error, errorMap.alignmentErrorMap[cameraID].surface, x * sizeof(float), y);
		}
	}
}


void SparseSurfelFusion::DenseDepthHandler::ComputeAlignmentErrorMapDirect(const DeviceArrayView<DualQuaternion>& node_se3, cudaStream_t stream) {
	//Check the size
	FUNCTION_CHECK(m_node_se3.Size() == node_se3.Size());

	dim3 block(16, 16, 1);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y), devicesCount);
	device::computeAlignmentErrorMapKernel << <grid, block, 0, stream >> > (
		observedDenseDepthHandlerInterface,
		geometryDenseDepthHandlerInterface,
		m_image_width,
		m_image_height,
		devicesCount,
		node_se3.RawPtr(),
		nodeAccumulatedErrorAndWeight
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif
}

void SparseSurfelFusion::DenseDepthHandler::ComputeNodewiseError(const DeviceArrayView<DualQuaternion>& node_se3, bool printNodeError, cudaStream_t stream) {
	const unsigned int nodesNum = m_node_se3.Size();
	FUNCTION_CHECK(node_se3.Size() == nodesNum);

	nodeAccumulatedError.ResizeArrayOrException(nodesNum);
	nodeAccumulatedWeight.ResizeArrayOrException(nodesNum);
	CHECKCUDA(cudaMemsetAsync(nodeAccumulatedError.Ptr(), 0, sizeof(float) * nodesNum, stream));
	CHECKCUDA(cudaMemsetAsync(nodeAccumulatedWeight.Ptr(), 0, sizeof(float) * nodesNum, stream));

	// 将Live域点投影到相机坐标系上，与观察到的节点进行匹配，并结合Knn，计算整个节点累计误差
	// 如果某点被多个视角看到，那么该点的误差是每个视角计算出的误差的累加
	dim3 block_1(16, 16, 1);
	dim3 grid_1(divUp(m_image_width, block_1.x), divUp(m_image_height, block_1.y), divUp(devicesCount, block_1.z));
	device::computeNodeAlignmentErrorFromMapKernel << <grid_1, block_1, 0, stream >> > (
		observedDenseDepthHandlerInterface,
		geometryDenseDepthHandlerInterface,
		m_image_width,
		m_image_height, 
		devicesCount,
		node_se3.RawPtr(),
		nodeAccumulatedError.Ptr(),
		nodeAccumulatedWeight.Ptr()
	);

	nodeUnitedAlignmentError.ResizeArrayOrException(nodesNum);
	nodeLargeNodeErrorNum.ResizeArrayOrException(nodesNum);
	dim3 block_2(128); 
	dim3 grid_2(divUp(nodesNum, block_2.x));
	device::computeLargeErrorNodeNum << <grid_2, block_2, 0, stream >> > (nodeAccumulatedError.Ptr(),nodeAccumulatedWeight.Ptr(), nodesNum, 0.2f, nodeUnitedAlignmentError.Ptr(), nodeLargeNodeErrorNum.Ptr());
	
	if (printNodeError) {
		int sum = 0;
		std::vector<unsigned int> largeErrorNode(nodesNum);
		nodeLargeNodeErrorNum.ArrayView().Download(largeErrorNode);
		for (int i = 0; i < nodesNum; i++) {
			if (largeErrorNode[i] == 1) sum++;
		}
		printf("%.5f, ", sum * 1.0f / nodesNum);
	}

#ifdef DEBUG_RUNNING_INFO
	std::vector<float> nodeErrorHost(nodesNum);
	std::vector<float> nodeErrorWeightHost(nodesNum);
	nodeAccumulatedError.ArrayView().Download(nodeErrorHost);
	nodeAccumulatedWeight.ArrayView().Download(nodeErrorWeightHost);
	float maxUnitError = -1.0f;
	float errorWeight = 0.0f;
	float nodeError = 0.0f;
	for (int i = 0; i < nodesNum; i++) {
		float unitNodeError = nodeErrorHost[i] / (nodeErrorWeightHost[i] + 1e-4f);
		if (unitNodeError > maxUnitError) {
			errorWeight = nodeErrorWeightHost[i];
			nodeError = nodeErrorHost[i];
			maxUnitError = unitNodeError;
		}
	}
	printf("maxUnitError = %.5f   nodeError = %.5f   maxErrorWeight = %.5f\n", maxUnitError, nodeError, errorWeight);// maxErrorThreshold = 2.0f 作为阈值比较合适？ 


	std::vector<unsigned int> largeErrorNodes(nodesNum);
	markLargeErrorNode.ArrayView().Download(largeErrorNodes);
	unsigned int totalLargeErrorNodesNum = 0;
	for (int i = 0; i < nodesNum; i++) {
		if (largeErrorNodes[i] == 1) totalLargeErrorNodesNum++;
	}
	printf("totalLargeErrorNodesNum = %u\n", totalLargeErrorNodesNum);
#endif // DEBUG_RUNNING_INFO
 
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::DenseDepthHandler::CorrectWarpFieldSE3(WarpField& warpField)
{

}

void SparseSurfelFusion::DenseDepthHandler::distributeNodeErrorOnMap(cudaStream_t stream) {
	dim3 block(16, 16, 1);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y), devicesCount);
	device::collectAlignmentErrorMapFromNodeKernel << <grid, block, 0, stream >> > (
		nodeAccumulatedError.Ptr(),
		nodeAccumulatedWeight.Ptr(),
		m_image_width,
		m_image_height,
		devicesCount,
		geometryDenseDepthHandlerInterface,
		nodeAccumulatedErrorAndWeight
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif
}



void SparseSurfelFusion::DenseDepthHandler::ComputeAlignmentErrorMapFromNode(const DeviceArrayView<DualQuaternion>& node_se3, cudaStream_t stream) {
	//ComputeNodewiseError(node_se3, stream);
	distributeNodeErrorOnMap(stream);
}