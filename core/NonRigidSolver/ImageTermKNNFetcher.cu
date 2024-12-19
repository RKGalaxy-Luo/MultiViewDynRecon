#include "ImageTermKNNFetcher.h"
#include "solver_constants.h"

#include <device_launch_parameters.h>


namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief 选择至少在一个视角中可见的点.
		 * 
		 * \param indexMap 某一视角的IndexMap
		 * \param CameraID 视角相机ID
		 * \param imgRows 裁剪图像高
		 * \param imgCols 裁剪图像宽
		 * \param referencePixelIndicator 标记有效面元
		 */
		__global__ void markPotentialValidImageTermPixelKernel(ImageKnnFetcherInterface fetcherInterface, unsigned int imgRows, unsigned int imgCols, unsigned int devicesCount, PtrSize<unsigned> referencePixelIndicator) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= imgCols || y >= imgRows || CameraID >= devicesCount) return;
			// 以视角为单位做偏移
			const unsigned int offset = x + y * imgCols + CameraID * imgCols * imgRows;
			const unsigned int surfelIndex = tex2D<unsigned int>(fetcherInterface.IndexMap[CameraID], x, y);
			unsigned int indicator = 0;
			if (surfelIndex != d_invalid_index) { indicator = 1; }
			else { indicator = 0; }
			referencePixelIndicator[offset] = indicator;
		}

		__global__ void compactPontentialImageTermPixelsKernel(
			ImageKnnFetcherInterface fetcherInterface,
			const unsigned int knnCols,
			const unsigned int knnRows,
			const unsigned int devicesCount,
			const unsigned* potentialPixelIndicator,
			const unsigned* prefixsumPixelIndicator,
			ushort3* potentialPixels,
			ushort4* potentialPixelsKnn,
			float4* potentialPixelsKnnWeight,
			unsigned int* differentViewsOffset
		) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			const unsigned int CameraID = threadIdx.z + blockDim.z * blockIdx.z;
			if (x >= knnCols || y >= knnRows || CameraID >= devicesCount) return;
			const unsigned int knnSize = knnCols * knnRows;
			const unsigned int flattenIdx = x + y * knnCols + CameraID * knnSize;

			if (potentialPixelIndicator[flattenIdx] > 0) {
				const unsigned int offset = prefixsumPixelIndicator[flattenIdx] - 1;
				KNNAndWeight knn;
				knn = fetcherInterface.KnnMap[CameraID](y, x);
				potentialPixels[offset] = make_ushort3(x, y, (unsigned short)CameraID);
				potentialPixelsKnn[offset] = knn.knn;
				potentialPixelsKnnWeight[offset] = knn.weight;
			}
			if (flattenIdx % knnSize == 0) {
				if (CameraID == 0) {
					differentViewsOffset[CameraID] = prefixsumPixelIndicator[knnSize - 1];
				}
				else {
					differentViewsOffset[CameraID] = prefixsumPixelIndicator[(CameraID + 1) * knnSize - 1] - prefixsumPixelIndicator[CameraID * knnSize - 1];
				}
			}
		}
	}
}


void SparseSurfelFusion::ImageTermKNNFetcher::MarkPotentialMatchedPixels(cudaStream_t stream) {

	dim3 block(16, 16, 1);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y), divUp(devicesCount, block.z));

	device::markPotentialValidImageTermPixelKernel << <grid, block, 0, stream >> > (
		imageKnnFetcherInterface,
		m_image_height,
		m_image_width,
		devicesCount,
		m_potential_pixel_indicator
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}



void SparseSurfelFusion::ImageTermKNNFetcher::CompactPotentialValidPixels(cudaStream_t stream) {
	//Do a prefix sum
	m_indicator_prefixsum.InclusiveSum(m_potential_pixel_indicator, stream);
	//Invoke the kernel
	dim3 block(16, 16, 1);
	dim3 grid(divUp(m_image_width, block.x), divUp(m_image_height, block.y), divUp(devicesCount, block.z));
	device::compactPontentialImageTermPixelsKernel << <grid, block, 0, stream >> > (
		imageKnnFetcherInterface,
		m_image_width,
		m_image_height,
		devicesCount,
		m_potential_pixel_indicator,
		m_indicator_prefixsum.valid_prefixsum_array,
		m_potential_pixels.Ptr(),
		m_dense_image_knn.Ptr(),
		m_dense_image_knn_weight.Ptr(),
		differenceViewOffset.Ptr()
	);
	
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif

}


void SparseSurfelFusion::ImageTermKNNFetcher::SyncQueryCompactedPotentialPixelSize(cudaStream_t stream) {

	CHECKCUDA(cudaMemcpyAsync(
		m_num_potential_pixel,
		m_indicator_prefixsum.valid_prefixsum_array.ptr() + m_potential_pixel_indicator.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	//Sync before use
	CHECKCUDA(cudaStreamSynchronize(stream));

	m_potential_pixels.ResizeArrayOrException(*m_num_potential_pixel);
	m_dense_image_knn.ResizeArrayOrException(*m_num_potential_pixel);
	m_dense_image_knn_weight.ResizeArrayOrException(*m_num_potential_pixel);
	differenceViewOffset.ResizeArrayOrException(devicesCount);
	//printf("m_num_potential_pixel = %d\n", *m_num_potential_pixel);
}