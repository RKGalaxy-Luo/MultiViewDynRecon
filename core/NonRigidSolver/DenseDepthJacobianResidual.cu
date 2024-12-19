#include "sanity_check.h"
#include <base/Logging.h>
#include "DenseDepthHandler.h"
#include "solver_constants.h"
#include "geometry_icp_jacobian.cuh"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {
		__device__ unsigned int CompactedDifferentDensityMapOffsetDpt[MAX_CAMERA_COUNT];
	}
}

__device__ unsigned int SparseSurfelFusion::device::CalculateDensityMapCameraViewDpt(const unsigned int idx, const unsigned int devicesCount)
{
	for (int i = 0; i < devicesCount; i++) {
		if (i == devicesCount - 1) return i;
		else {
			if (CompactedDifferentDensityMapOffsetDpt[i] <= idx && idx < CompactedDifferentDensityMapOffsetDpt[i + 1]) return i;
		}
	}
}

__global__ void SparseSurfelFusion::device::computeDenseDepthJacobianKernel(ObservationDenseDepthHandlerInterface observedDenseDepthHandlerInterface, GeometryMapDenseDepthHandlerInterface geometryDenseDepthHandlerInterface, const unsigned int imgRows, const unsigned int imgCols, const unsigned int totalPotentialPixels, const unsigned int devicesCount, DeviceArrayView<ushort3> potentialMatchedPixels, const ushort4* potentialMatchedKnn, const float4* potentialMatchedKnnWeight, const DualQuaternion* nodeSE3, TwistGradientOfScalarCost* twistGradient, float* residual)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalPotentialPixels) return;
	const unsigned int CameraID = CalculateDensityMapCameraViewDpt(idx, devicesCount);
	// These value will definited be written to global memory   这些值将被定义并写入全局内存
	float pixelResidual = 0.0f;
	float pixelGradient[6] = { 0.0f };
	TwistGradientOfScalarCost* pixelTwist = (TwistGradientOfScalarCost*)pixelGradient;

	//Now, query the pixel, knn and their weight
	const ushort3 potentialPixel = potentialMatchedPixels[idx];	// 在对应视角以及筛了一遍了，不会透射
	const ushort4 knn = potentialMatchedKnn[idx];
	const float4 knnWeight = potentialMatchedKnnWeight[idx];
		 
	// 各自视角下的Canonical域点
	const float4 canVertexFloat4 = tex2D<float4>(geometryDenseDepthHandlerInterface.referenceVertexMap[CameraID], potentialPixel.x, potentialPixel.y);
	const float4 canNormalFloat4 = tex2D<float4>(geometryDenseDepthHandlerInterface.referenceNormalMap[CameraID], potentialPixel.x, potentialPixel.y);
	
	// 将点转到0号坐标系下的Canonical域
	const float3 canVertexCam_0 = geometryDenseDepthHandlerInterface.InitialCameraSE3[CameraID].rot * canVertexFloat4 + geometryDenseDepthHandlerInterface.InitialCameraSE3[CameraID].trans;
	const float3 canNormalCam_0 = geometryDenseDepthHandlerInterface.InitialCameraSE3[CameraID].rot * canNormalFloat4;

	DualQuaternion dqAverage = averageDualQuaternion(nodeSE3, knn, knnWeight);
	mat34 se3 = dqAverage.se3_matrix();
	// 0号坐标系：将Canonical域转到Live域
	float3 warpedVertexLiveField = se3.rot * canVertexCam_0 + se3.trans;
	float3 warpedNormalLiveField = se3.rot * canNormalCam_0;

	//Transfer to the camera frame
	float3 warpedVertexCamera = geometryDenseDepthHandlerInterface.world2Camera[CameraID].rot * warpedVertexLiveField + geometryDenseDepthHandlerInterface.world2Camera[CameraID].trans;
	float3 warpedNormalCamera = geometryDenseDepthHandlerInterface.world2Camera[CameraID].rot * warpedNormalLiveField;
	// 完成SE3扭曲后再转到CameraID视角下与当前帧观察到的内容匹配
	warpedVertexCamera = geometryDenseDepthHandlerInterface.InitialCameraSE3Inverse[CameraID].rot * warpedVertexCamera + geometryDenseDepthHandlerInterface.InitialCameraSE3Inverse[CameraID].trans;
	warpedNormalCamera = geometryDenseDepthHandlerInterface.InitialCameraSE3Inverse[CameraID].rot * warpedNormalCamera;


	//Project the vertex into image
	const int2 imgCoord = {
		__float2int_rn(((warpedVertexCamera.x / (warpedVertexCamera.z + 1e-10)) * geometryDenseDepthHandlerInterface.intrinsic[CameraID].focal_x) + geometryDenseDepthHandlerInterface.intrinsic[CameraID].principal_x),
		__float2int_rn(((warpedVertexCamera.y / (warpedVertexCamera.z + 1e-10)) * geometryDenseDepthHandlerInterface.intrinsic[CameraID].focal_y) + geometryDenseDepthHandlerInterface.intrinsic[CameraID].principal_y)
	};
	if (imgCoord.x >= 0 && imgCoord.x < imgCols && imgCoord.y >= 0 && imgCoord.y < imgRows) {
		//Query the depth image
		const float4 depthVertexFloat4 = tex2D<float4>(observedDenseDepthHandlerInterface.vertexMap[CameraID], imgCoord.x, imgCoord.y);
		const float4 depthNormalFloat4 = tex2D<float4>(observedDenseDepthHandlerInterface.normalMap[CameraID], imgCoord.x, imgCoord.y);
		const float3 depthVertex = make_float3(depthVertexFloat4.x, depthVertexFloat4.y, depthVertexFloat4.z);
		const float3 depthNormal = make_float3(depthNormalFloat4.x, depthNormalFloat4.y, depthNormalFloat4.z);

		//Check the matched
		bool validPair = true;

		//The depth pixel is not valid
		if (is_zero_vertex(depthVertexFloat4)) {
			validPair = false;
		}

		//The orientation is not matched
		if (dot(depthNormal, warpedNormalCamera) < d_correspondence_normal_dot_threshold) {
			validPair = false;
		}

		//The distance is too far away
		if (squared_norm(depthVertex - warpedVertexCamera) > d_correspondence_distance_threshold_square) {
			validPair = false;
		}


		//This pair is valid, compute the jacobian and residual
		if (validPair) {
			pixelResidual = dot(depthNormal, warpedVertexCamera - depthVertex);
			pixelTwist->translation = geometryDenseDepthHandlerInterface.InitialCameraSE3[CameraID].rot.dot(depthNormal);				// 转到0号坐标系
			pixelTwist->translation = geometryDenseDepthHandlerInterface.world2Camera[CameraID].rot.transpose_dot(pixelTwist->translation);		// 在0号坐标系下从camera转到world
			pixelTwist->rotation = cross(warpedVertexLiveField, pixelTwist->translation); // cross(warp_vertex, depth_world_normal)
		}
	} // This pixel is projected inside

	//Write it to global memory
	residual[idx] = pixelResidual;
	twistGradient[idx] = *pixelTwist;
}

/* The method and buffer for gradient computation
 */
void SparseSurfelFusion::DenseDepthHandler::ComputeJacobianTermsFreeIndex(cudaStream_t stream) {
//	//Correct the size of array
//	m_term_twist_gradient.ResizeArrayOrException(m_valid_pixel_pairs.ArraySize());
//	m_term_residual.ResizeArrayOrException(m_valid_pixel_pairs.ArraySize());
//
//	//Invoke kernel
//	dim3 blk(128);
//	dim3 grid(divUp(m_valid_pixel_pairs.ArraySize(), blk.x));
//	device::computeDenseDepthJacobianKernel << <grid, blk, 0, stream >> > (
//		m_depth_observation.vertex_map,
//		m_depth_observation.normal_map,
//		m_geometry_maps.reference_vertex_map,
//		//Pixels and KNN
//		m_valid_pixel_pairs.ArrayView(),
//		m_dense_depth_knn.Ptr(),
//		m_dense_depth_knn_weight.Ptr(),
//		//The deformation
//		m_node_se3.RawPtr(),
//		m_camera2world,
//		//Output
//		m_term_twist_gradient.Ptr(),
//		m_term_residual.Ptr()
//		);
//
//	//Sync and check error
//#if defined(CUDA_DEBUG_SYNC_CHECK)
//	cudaSafeCall(cudaStreamSynchronize(stream));
//	cudaSafeCall(cudaGetLastError());
//#endif
}


void SparseSurfelFusion::DenseDepthHandler::ComputeJacobianTermsFixedIndex(std::vector<unsigned int> differenceOffsetImageKnnFetcher, cudaStream_t stream) {
	const unsigned int totalPotentialPixels = m_potential_pixels_knn.pixels.Size();
	m_term_residual.ResizeArrayOrException(totalPotentialPixels);//这里用到了indexmap上像素点的坐标了
	m_term_twist_gradient.ResizeArrayOrException(totalPotentialPixels);
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::CompactedDifferentDensityMapOffsetDpt, differenceOffsetImageKnnFetcher.data(), sizeof(unsigned int) * devicesCount, 0, cudaMemcpyHostToDevice, stream));
	dim3 block(128);
	dim3 grid(divUp(totalPotentialPixels, block.x));
	for (int i = 0; i < devicesCount; i++) {
		device::computeDenseDepthJacobianKernel << <grid, block, 0, stream >> > (
			observedDenseDepthHandlerInterface,
			geometryDenseDepthHandlerInterface,
			m_image_height,
			m_image_width,
			totalPotentialPixels,
			devicesCount,
			//The pixel pairs and knn
			m_potential_pixels_knn.pixels,
			m_potential_pixels_knn.node_knn.RawPtr(),
			m_potential_pixels_knn.knn_weight.RawPtr(),
			//The deformation
			m_node_se3.RawPtr(),
			//The output
			m_term_twist_gradient.Ptr(),
			m_term_residual.Ptr()
		);

	}

#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


SparseSurfelFusion::DenseDepthTerm2Jacobian SparseSurfelFusion::DenseDepthHandler::Term2JacobianMap() const {
	DenseDepthTerm2Jacobian term2jacobian;
	term2jacobian.knn_array = m_potential_pixels_knn.node_knn;
	term2jacobian.knn_weight_array = m_potential_pixels_knn.knn_weight;
	term2jacobian.residual_array = m_term_residual.ArrayView();
	term2jacobian.twist_gradient_array = m_term_twist_gradient.ArrayView();
	term2jacobian.check_size();

	//Check correct
	return term2jacobian;
}