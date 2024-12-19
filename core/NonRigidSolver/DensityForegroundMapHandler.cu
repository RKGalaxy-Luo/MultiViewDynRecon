#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include "sanity_check.h"
#include <math/MatUtils.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include "solver_constants.h"
#include "DensityForegroundMapHandler.h"
#include "density_map_jacobian.cuh"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {
		enum {
			valid_color_halfsize = 1
		};

		__device__ __forceinline__ bool isValidColorPixel(cudaTextureObject_t rendered_rgb_map, cudaTextureObject_t index_map, int x_center, int y_center) {
#if defined(USE_DENSE_SOLVER_MAPS)
			for (auto y = y_center - valid_color_halfsize; y <= y_center + valid_color_halfsize; y++) {
				for (auto x = x_center - valid_color_halfsize; x <= x_center + valid_color_halfsize; x++) {
					if (tex2D<unsigned>(index_map, x, y) == d_invalid_index) return false;
				}
			}
#endif
			return true;
		}

		__device__ __forceinline__ bool isValidForegroundMaskPixel(cudaTextureObject_t filter_foreground_mask, int x, int y) {
			const float mask_value = tex2D<float>(filter_foreground_mask, x, y);
			if (mask_value > 1e-5f) { return true; }
			else { return false; }
		}

		__global__ void markValidColorForegroundMapsPixelKernel(
			DepthObservationForegroundInterface observeInterface,
			GeometryMapForegroundInterface geometryMapInterface,
			DeviceArrayView<DualQuaternion> deviceWarpField,
			const unsigned int knnCols,
			const unsigned int knnRows,
			const unsigned int devicesCount,
			unsigned int* validRgbIndicatorArray,
			unsigned int* validForegoundMaskIndicatorArray
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= knnCols || y >= knnRows || CameraID >= devicesCount) return;
			//The indicator will must be written to pixel_occupied_array
			unsigned int validRgb = 0;
			unsigned int validForegroundMask = 0;
			const unsigned int offset = x + y * knnCols + CameraID * knnCols * knnRows;

			//Read the value on index map
			const unsigned int surfelIndex = tex2D<unsigned int>(geometryMapInterface.indexMap[CameraID], x, y);
			if (surfelIndex != d_invalid_index) {
				//reference_vertex_map ���CameraID�ӽ��µ�Canonical��ĵ�
				const float4 can_vertex4 = tex2D<float4>(geometryMapInterface.referenceVertexMap[CameraID], x, y);
				const float4 can_normal4 = tex2D<float4>(geometryMapInterface.referenceNormalMap[CameraID], x, y);

				// ��Canonical��ĵ�ת��0������ϵ��
				const float3 canVertexCam_0 = geometryMapInterface.initialCameraSE3[CameraID].rot * can_vertex4 + geometryMapInterface.initialCameraSE3[CameraID].trans;
				const float3 canNormalCam_0 = geometryMapInterface.initialCameraSE3[CameraID].rot * can_normal4;

				//�����knn���ϱ��Ǹ�ref�����knn
				const KNNAndWeight knn = geometryMapInterface.knnMap[CameraID](y, x);
				//�ó���ƽ��xy��Ӧ�����ص��ref�����άӳ��㣬�ҵ�knn����Ȩ�أ������ż��Ԫ��
				//ע�⣬�����õ�knn����Ȩ�ض�����һ֡��
				DualQuaternion dq_average = averageDualQuaternion(deviceWarpField, knn.knn, knn.weight);
				mat34 se3 = dq_average.se3_matrix();	// ��Ԫ���������0������ϵ�µ���Ԫ��

				// 0������ϵ����Canonical��ת��Live��
				float3 warped_vertex = se3.rot * canVertexCam_0 + se3.trans;
				float3 warped_normal = se3.rot * canNormalCam_0;

				float3 warped_vertex_camera = geometryMapInterface.world2Camera[CameraID].rot * warped_vertex + geometryMapInterface.world2Camera[CameraID].trans;
				float3 warped_normal_camera = geometryMapInterface.world2Camera[CameraID].rot * warped_normal;
				// ���SE3Ť������ת��CameraID�ӽ����뵱ǰ֡�۲쵽������ƥ��
				warped_vertex_camera = geometryMapInterface.initialCameraSE3Inverse[CameraID].rot * warped_vertex_camera + geometryMapInterface.initialCameraSE3Inverse[CameraID].trans;
				warped_normal_camera = geometryMapInterface.initialCameraSE3Inverse[CameraID].rot * warped_normal_camera;

				//Check the normal of this pixel
				const float3 view_direction = normalized(warped_vertex_camera);
				const float viewAngleCosine = -dot(view_direction, warped_normal_camera);//����ɸѡ�������
				//Project the vertex into image
				//The image coordinate of this vertex
				//��һ֡�Ķ���ģ�͵�����һ֡�µ�����ӽ������ĸ�������
				const int2 img_coord = {
					__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * geometryMapInterface.intrinsic[CameraID].focal_x) + geometryMapInterface.intrinsic[CameraID].principal_x),
					__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * geometryMapInterface.intrinsic[CameraID].focal_y) + geometryMapInterface.intrinsic[CameraID].principal_y)
				};
				//Check that the pixel projected to and the view angle
				if (img_coord.x >= 0 && img_coord.x < knnCols && img_coord.y >= 0 && img_coord.y < knnRows && viewAngleCosine > d_valid_color_dot_threshold) {
					if (isValidColorPixel(geometryMapInterface.normalizedRgbMap[CameraID], geometryMapInterface.indexMap[CameraID], x, y)) validRgb = 1;	//index_map�еĶ�����Чֵ�����Ķ����Ƿ���Ч
					if (isValidForegroundMaskPixel(observeInterface.filteredForegroundMask[CameraID], img_coord.x, img_coord.y)) validForegroundMask = 1;			//���������ص��Ƿ���ǰ��
				} // The vertex project to a valid image pixel

				//Mark the rgb as always valid if the pixel is valid?
				validRgb = 1;

			} // The reference vertex is valid

			//ֻҪ��ǰ����Ԫ����һ֡����Ч����ô��rgb��Ч
			validRgbIndicatorArray[offset] = validRgb;
			validForegoundMaskIndicatorArray[offset] = validForegroundMask;
		}



		__global__ void compactValidPixelKernel(
			const unsigned int* validIndicatorArray,
			const unsigned int* prefixsumIndicatorArray,
			GeometryMapForegroundInterface geometryMapInterface,
			const unsigned int knnCols,
			const unsigned int knnRows,
			const unsigned int devicesCount,
			ushort3* compactedPixelCoordinate,
			ushort4* pixelKnn,
			float4* pixelKnnWeight,
			unsigned int* differentViewsOffset
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= knnCols || y >= knnRows || CameraID >= devicesCount) return;
			const unsigned int knnSize = knnRows * knnCols;
			const unsigned int flattenIdx = x + y * knnCols + CameraID * knnSize;
			if (validIndicatorArray[flattenIdx] > 0) {
				const unsigned int offset = prefixsumIndicatorArray[flattenIdx] - 1;
				const KNNAndWeight knn = geometryMapInterface.knnMap[CameraID](y, x);
				compactedPixelCoordinate[offset] = make_ushort3(x, y, (unsigned short)CameraID);
				pixelKnn[offset] = knn.knn;
				pixelKnnWeight[offset] = knn.weight;
			}

			if (flattenIdx % knnSize == 0) {
				if (CameraID == 0) {
					differentViewsOffset[CameraID] = prefixsumIndicatorArray[knnSize - 1];
				}
				else {
					differentViewsOffset[CameraID] = prefixsumIndicatorArray[(CameraID + 1) * knnSize - 1] - prefixsumIndicatorArray[CameraID * knnSize - 1];
				}
			}
		}
	}
}



/* The method to mark the valid pixels
 */
void SparseSurfelFusion::DensityForegroundMapHandler::MarkValidColorForegroundMaskPixels(cudaStream_t stream) {

	dim3 block(16, 16, 1);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y), divUp(devicesCount, block.z));
	device::markValidColorForegroundMapsPixelKernel << <grid, block, 0, stream >> > (
		depthObservationForegroundInterface,
		geometryMapForegroundInterface,
		m_node_se3,
		m_image_width,
		m_image_height,
		devicesCount,
		//Output
		m_color_pixel_indicator_map.ptr(),
		m_mask_pixel_indicator_map.ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


/* The method for compaction
 */
void SparseSurfelFusion::DensityForegroundMapHandler::CompactValidColorPixel(cudaStream_t stream) {
	//Do a prefix sum
	m_color_pixel_indicator_prefixsum.InclusiveSum(m_color_pixel_indicator_map, stream);
	//Invoke the kernel
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	for (int i = 0; i < devicesCount; i++) { 
		
		//device::compactValidPixelKernel << <grid, blk, 0, stream >> > (
		//	i,
		//	m_color_pixel_indicator_map,
		//	m_color_pixel_indicator_prefixsum.valid_prefixsum_array,
		//	m_knn_map[i],
		//	//The output
		//	m_valid_color_pixel.Ptr(),
		//	m_valid_color_pixel_knn.Ptr(),
		//	m_valid_color_pixel_knn_weight.Ptr(),
		//	differentViewsForegroundMapOffset.Ptr()
		//	);
	}
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void SparseSurfelFusion::DensityForegroundMapHandler::QueryCompactedColorPixelArraySize(cudaStream_t stream)
{
	//Sync the stream and query the size
	unsigned int num_valid_pairs;
	CHECKCUDA(cudaMemcpyAsync(
		&num_valid_pairs,
		m_color_pixel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_color_pixel_indicator_map.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost, stream
	));
	CHECKCUDA(cudaStreamSynchronize(stream));

	//Correct the size of the array
	m_valid_color_pixel.ResizeArrayOrException(num_valid_pairs);
	m_valid_color_pixel_knn.ResizeArrayOrException(num_valid_pairs);
	m_valid_color_pixel_knn_weight.ResizeArrayOrException(num_valid_pairs);

	//Also write to potential pixels
	m_potential_pixels_knn.pixels = m_valid_color_pixel.ArrayReadOnly();
	m_potential_pixels_knn.node_knn = m_valid_color_pixel_knn.ArrayReadOnly();
	m_potential_pixels_knn.knn_weight = m_valid_color_pixel_knn_weight.ArrayReadOnly();
	
	//Check the output
	//LOG(INFO) << "The number of valid color pixel is " << m_valid_color_pixel.ArraySize();
}


void SparseSurfelFusion::DensityForegroundMapHandler::CompactValidMaskPixel(cudaStream_t stream) {
	//Do a prefix sum
	m_mask_pixel_indicator_prefixsum.InclusiveSum(m_mask_pixel_indicator_map, stream);
	//Invoke the kernel
	dim3 block(16, 16, 1);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y), divUp(devicesCount, block.z));
	device::compactValidPixelKernel << <grid, block, 0, stream >> > (
		m_mask_pixel_indicator_map,
		m_mask_pixel_indicator_prefixsum.valid_prefixsum_array,
		geometryMapForegroundInterface,
		m_image_width,
		m_image_height,
		devicesCount,
		//The output
		m_valid_mask_pixel.Ptr(),
		m_valid_mask_pixel_knn.Ptr(),
		m_valid_mask_pixel_knn_weight.Ptr(),
		differentViewsForegroundMapOffset.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


void SparseSurfelFusion::DensityForegroundMapHandler::QueryCompactedMaskPixelArraySize(cudaStream_t stream)
{
	//Sync the stream and query the size
	CHECKCUDA(cudaMemcpyAsync(
		m_num_mask_pixel,
		m_mask_pixel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_mask_pixel_indicator_map.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost, stream
	));
	//Sync before use
	CHECKCUDA(cudaStreamSynchronize(stream));
	//printf("m_num_mask_pixel = %d\n", *m_num_mask_pixel);
	//Correct the size of the array
	m_valid_mask_pixel.ResizeArrayOrException(*m_num_mask_pixel);
	m_valid_mask_pixel_knn.ResizeArrayOrException(*m_num_mask_pixel);
	m_valid_mask_pixel_knn_weight.ResizeArrayOrException(*m_num_mask_pixel);
	differentViewsForegroundMapOffset.ResizeArrayOrException(devicesCount);
	//printf("m_num_mask_pixel = %d\n", *m_num_mask_pixel);

}








