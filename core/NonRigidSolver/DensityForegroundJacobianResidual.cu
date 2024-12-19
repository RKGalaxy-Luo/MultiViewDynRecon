#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>

#include "sanity_check.h"
#include <math/MatUtils.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include "solver_constants.h"
#include "DensityForegroundMapHandler.h"
#include "density_map_jacobian.cuh"
#include <device_launch_parameters.h>

/* Compute the gradient of density map and foreground mask
 */
namespace SparseSurfelFusion {
	namespace device {
		__device__ unsigned int CompactedDifferentForegroundMapOffsetFgr[MAX_CAMERA_COUNT];
		__device__ unsigned int CompactedDifferentDensityMapOffsetFgr[MAX_CAMERA_COUNT];
	} 
}

__device__ unsigned int SparseSurfelFusion::device::CalculateForegroundMapCameraViewFgr(const unsigned int idx, const unsigned int devicesCount)
{
	for (int i = 0; i < devicesCount; i++) {
		if (i == devicesCount - 1) return i;
		else {
			if (CompactedDifferentForegroundMapOffsetFgr[i] <= idx && idx < CompactedDifferentForegroundMapOffsetFgr[i + 1]) return i;
		}
	}
}

__device__ unsigned int SparseSurfelFusion::device::CalculateDensityMapCameraViewFgr(const unsigned int idx, const unsigned int devicesCount)
{
	for (int i = 0; i < devicesCount; i++) {
		if (i == devicesCount - 1) return i;
		else {
			if (CompactedDifferentDensityMapOffsetFgr[i] <= idx && idx < CompactedDifferentDensityMapOffsetFgr[i + 1]) return i;
		}
	}
}

__global__ void SparseSurfelFusion::device::computeDensityMapJacobian(DepthObservationForegroundInterface depthObservationForegroundInterface, GeometryMapForegroundInterface geometryMapForegroundInterface, const unsigned int width, const unsigned int height, const unsigned int devicesCount, const unsigned int totalTermsNum, const DeviceArrayView<ushort3> densityTermPixels, const ushort4* densityTermKnn, const float4* densityTermKnnWeight, const DualQuaternion* deviceWarpField, TwistGradientOfScalarCost* gradient, float* residualArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalTermsNum) return; //The function does not involve warp/block sync
	const unsigned int CameraID = CalculateDensityMapCameraViewFgr(idx, devicesCount);
	//Prepare the data
	const ushort3 pixel = densityTermPixels[idx];
	const ushort4 knn = densityTermKnn[idx];
	const float4 knn_weight = densityTermKnnWeight[idx];
	const float4 reference_vertex = tex2D<float4>(geometryMapForegroundInterface.referenceVertexMap[CameraID], pixel.x, pixel.y);
	const float4 rendered_rgb = tex2D<float4>(geometryMapForegroundInterface.normalizedRgbMap[CameraID], pixel.x, pixel.y);
	const float geometry_density = rgb2density(rendered_rgb);
	
	//Compute the jacobian
	TwistGradientOfScalarCost twist_graident;
	float residual;
#if defined(USE_IMAGE_HUBER_WEIGHT)
	computeImageDensityJacobainAndResidualHuberWeight(
		density_map,
		density_gradient_map,
		width, height,
		reference_vertex,
		geometry_density,
		knn, knn_weight,
		device_warp_field,
		intrinsic, world2camera,
		twist_graident, residual,
		d_density_map_cutoff
	);
#else
	computeImageDensityJacobainAndResidual(
		geometryMapForegroundInterface.initialCameraSE3[CameraID],
		CameraID,
		depthObservationForegroundInterface.densityMap[CameraID],
		depthObservationForegroundInterface.densityGradientMap[CameraID],
		width, height,
		reference_vertex,
		geometry_density,
		knn, knn_weight,
		deviceWarpField,
		geometryMapForegroundInterface.intrinsic[CameraID], 
		geometryMapForegroundInterface.world2Camera[CameraID],
		twist_graident,
		residual
	);
#endif
	
	//Output
	gradient[idx] = twist_graident;
	residualArray[idx] = residual;
}

__global__ void SparseSurfelFusion::device::computeForegroundMaskJacobian(DepthObservationForegroundInterface depthObservationForegroundInterface, GeometryMapForegroundInterface geometryMapForegroundInterface, const unsigned int width, const unsigned int height, const unsigned int totalMaskPixels, const unsigned int devicesCount, const DeviceArrayView<ushort3> foreground_term_pixels, const ushort4* foreground_term_knn, const float4* foreground_term_knn_weight, const DualQuaternion* device_warp_field, TwistGradientOfScalarCost* gradient, float* residual_array)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalMaskPixels) return; //The function does not involve warp/block sync
	const unsigned int CameraID = CalculateForegroundMapCameraViewFgr(idx, devicesCount);
	//Prepare the data
	const ushort3 pixel = foreground_term_pixels[idx];
	const ushort4 knn = foreground_term_knn[idx];
	const float4 knn_weight = foreground_term_knn_weight[idx];
	const float4 reference_vertex = tex2D<float4>(geometryMapForegroundInterface.referenceVertexMap[CameraID], pixel.x, pixel.y);
	const float geometry_density = 0.0f; // An occupied pixel should be marked as 0 on filter foreground mask
	
	//Compute the jacobian
	TwistGradientOfScalarCost twist_graident;
	float residual;
#if defined(USE_IMAGE_HUBER_WEIGHT)
	computeImageDensityJacobainAndResidualHuberWeight(
		filter_foreground_mask,
		foreground_gradient_map,
		width, height,
		reference_vertex,
		geometry_density,
		knn, knn_weight,
		device_warp_field,
		intrinsic, world2camera,
		twist_graident, residual,
		d_foreground_cutoff
	);
#else
	computeImageDensityJacobainAndResidual(
		geometryMapForegroundInterface.initialCameraSE3[CameraID],
		CameraID,
		depthObservationForegroundInterface.filteredForegroundMask[CameraID],
		depthObservationForegroundInterface.foregroundMaskGradientMap[CameraID],
		width, height,
		reference_vertex,
		geometry_density,
		knn, knn_weight,
		device_warp_field,
		geometryMapForegroundInterface.intrinsic[CameraID],
		geometryMapForegroundInterface.world2Camera[CameraID],
		twist_graident, 
		residual
	);
#endif
	
	//Output
	gradient[idx] = twist_graident;
	residual_array[idx] = residual;
}

void SparseSurfelFusion::DensityForegroundMapHandler::computeDensityMapTwistGradient(std::vector<unsigned int> differenceOffsetImageKnnFetcher, cudaStream_t stream)
{
	//Correct the size of output
	const size_t num_pixels = m_potential_pixels_knn.pixels.Size();
	m_color_residual.ResizeArrayOrException(num_pixels);
	m_color_twist_gradient.ResizeArrayOrException(num_pixels);
	//If the size is zero, just return
	if (num_pixels == 0) {
		LOGGING(INFO) << "imageKnnFetcher没值";
		return;
	}

	CHECKCUDA(cudaMemcpyToSymbolAsync(device::CompactedDifferentDensityMapOffsetFgr, differenceOffsetImageKnnFetcher.data(), sizeof(unsigned int) * devicesCount, 0, cudaMemcpyHostToDevice, stream));

	dim3 block(128);
	dim3 grid(divUp(num_pixels, block.x));
	device::computeDensityMapJacobian << <grid, block, 0, stream >> > (
		depthObservationForegroundInterface,
		geometryMapForegroundInterface,
		m_image_width, 
		m_image_height,
		devicesCount,
		num_pixels,
		m_potential_pixels_knn.pixels,
		m_potential_pixels_knn.node_knn.RawPtr(),
		m_potential_pixels_knn.knn_weight.RawPtr(),
		m_node_se3,
		m_color_twist_gradient.Ptr(),
		m_color_residual.Ptr()
	);
	
	

#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::DensityForegroundMapHandler::computeForegroundMaskTwistGradient(std::vector<unsigned int> differenceOffsetForegroundHandler, cudaStream_t stream)
{
	//Correct the size of output
	const unsigned int totalValidMaskPixels = m_valid_mask_pixel.ArraySize();
	m_foreground_residual.ResizeArrayOrException(totalValidMaskPixels);
	m_foreground_twist_gradient.ResizeArrayOrException(totalValidMaskPixels);

	//If the size is zero, just return
	if (totalValidMaskPixels == 0) {
		LOGGING(INFO) << "m_density_foreground_handler是空的，没值";
		return;
	}
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::CompactedDifferentForegroundMapOffsetFgr, differenceOffsetForegroundHandler.data(), sizeof(unsigned int) * devicesCount, 0, cudaMemcpyHostToDevice, stream));

	//Invoke the kernel
	dim3 block(128);
	dim3 grid(divUp(totalValidMaskPixels, block.x));
	device::computeForegroundMaskJacobian << <grid, block, 0, stream >> > (
		depthObservationForegroundInterface,
		geometryMapForegroundInterface,
		m_image_width,
		m_image_height, 
		totalValidMaskPixels,
		devicesCount,
		m_valid_mask_pixel.ArrayView(),
		m_valid_mask_pixel_knn.Ptr(),
		m_valid_mask_pixel_knn_weight.Ptr(),
		m_node_se3,
		m_foreground_twist_gradient.Ptr(),
		m_foreground_residual.Ptr()
	);

	
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


void SparseSurfelFusion::DensityForegroundMapHandler::ComputeTwistGradient(std::vector<unsigned int> differenceOffsetImageKnnFetcher, vector<unsigned int> differenceOffsetForegroundHandler, cudaStream_t colorStream, cudaStream_t foregroundStream) {
	computeDensityMapTwistGradient(differenceOffsetImageKnnFetcher, colorStream);
	computeForegroundMaskTwistGradient(differenceOffsetForegroundHandler, foregroundStream);
}


void SparseSurfelFusion::DensityForegroundMapHandler::Term2JacobianMaps(
	DensityMapTerm2Jacobian& density_term2jacobian,
	ForegroundMaskTerm2Jacobian& foreground_term2jacobian
) {
	density_term2jacobian.knn_array = m_potential_pixels_knn.node_knn;
	density_term2jacobian.knn_weight_array = m_potential_pixels_knn.knn_weight;
	density_term2jacobian.residual_array = m_color_residual.ArrayView();
	density_term2jacobian.twist_gradient_array = m_color_twist_gradient.ArrayView();
	density_term2jacobian.check_size();

	foreground_term2jacobian.knn_array = m_valid_mask_pixel_knn.ArrayView();
	foreground_term2jacobian.knn_weight_array = m_valid_mask_pixel_knn_weight.ArrayView();
	foreground_term2jacobian.residual_array = m_foreground_residual.ArrayView();
	foreground_term2jacobian.twist_gradient_array = m_foreground_twist_gradient.ArrayView();
	foreground_term2jacobian.check_size();
}