/*****************************************************************//**
 * \file   ForegroundSegmenter.cu
 * \brief  前景分割器
 * 
 * \author LUO
 * \date   March 22nd 2024
 *********************************************************************/
#include "ForegroundSegmenter.h"

__global__ void SparseSurfelFusion::device::SwapChannelAndNormalizeValueKernel(PtrSize<uchar3> UCharInputArray, PtrSize<float> NormalizedInputArray, const unsigned int rawSize)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= rawSize)	return;
	const uchar3 rawPixel = UCharInputArray[idx];
	NormalizedInputArray[idx + 0 * rawSize] = (float)rawPixel.z / 255.0f;
	NormalizedInputArray[idx + 1 * rawSize] = (float)rawPixel.y / 255.0f;
	NormalizedInputArray[idx + 2 * rawSize] = (float)rawPixel.x / 255.0f;
}

__global__ void SparseSurfelFusion::device::CollectAndClipMaskKernel(PtrSize<float> MaskArray, float threshold, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t rawForeground)
{
	const unsigned int clip_x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int clip_y = threadIdx.y + blockDim.y * blockIdx.y;
	if (clip_x >= clipedCols || clip_y >= clipedRows) return;
	// 对应cliped_x和clip_y在原始图像山的位置
	const unsigned int raw_x = clip_x + CLIP_BOUNDARY;
	const unsigned int raw_y = clip_y + CLIP_BOUNDARY;
	const unsigned int ColsOffset = clipedCols + 2 * CLIP_BOUNDARY;
	// 将这些像素化成1维
	const unsigned int raw_flatten = raw_x + raw_y * ColsOffset;
	unsigned char maskValue;
	if (MaskArray[raw_flatten] > threshold) {
		maskValue = 1;
	}
	else {
		maskValue = 0;
	}
	surf2Dwrite(maskValue, rawForeground, clip_x * sizeof(unsigned char), clip_y);
}

__global__ void SparseSurfelFusion::device::ErodeRawForeground(cudaTextureObject_t rawForeground, const int erodeRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t MaskSurface)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedCols || y >= clipedRows) return;
	const unsigned int flatten = y * clipedCols + x;
	if (erodeRadius <= x && x < clipedCols - erodeRadius && erodeRadius <= y && y < clipedRows - erodeRadius) {
		unsigned char minPix = 1;
		bool isEdge = false;
		for (int i = -erodeRadius; i <= erodeRadius; i++) {
			for (int j = -erodeRadius; j <= erodeRadius; j++) {
				unsigned char rawValue = tex2D<unsigned char>(rawForeground, x + i, y + j);
				if (rawValue == 0) { isEdge = true; break; }
			}
			if (isEdge) break;
		}
		if (isEdge) minPix = 0;
		else minPix = 1;
		surf2Dwrite(minPix, MaskSurface, x * sizeof(unsigned char), y);
	}
	else {
		unsigned char minPix = 0;
		surf2Dwrite(minPix, MaskSurface, x * sizeof(unsigned char), y);
	}
}

__global__ void SparseSurfelFusion::device::DilateRawForeground(cudaTextureObject_t rawMask, const int dilateRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t dilateMaskSurface)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedCols || y >= clipedRows) return;
	if (dilateRadius <= x && x < clipedCols - dilateRadius && dilateRadius <= y && y < clipedRows - dilateRadius) {
		unsigned char maxPix = 0;
		bool isInside = false;
		for (int i = -dilateRadius; i <= dilateRadius; i++) {
			for (int j = -dilateRadius; j <= dilateRadius; j++) {
				unsigned char rawValue = tex2D<unsigned char>(rawMask, x + i, y + j);
				if (rawValue == 1) { isInside = true; break; }
			}
			if (isInside) break;
		}
		if (isInside) maxPix = 1;
		else maxPix = 0;
		surf2Dwrite(maxPix, dilateMaskSurface, x * sizeof(unsigned char), y);
	}
	else {
		unsigned char maxPix = 0;
		surf2Dwrite(maxPix, dilateMaskSurface, x * sizeof(unsigned char), y);
	}

}

__global__ void SparseSurfelFusion::device::ErodeRawForeground(cudaTextureObject_t rawForeground, const int erodeRadius, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t MaskSurface)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedCols || y >= clipedRows) return;
	const unsigned int flatten = y * clipedCols + x;
	if (erodeRadius <= x && x < clipedCols - erodeRadius && erodeRadius <= y && y < clipedRows - erodeRadius) {
		unsigned char minPix = 1;
		bool isEdge = false;
		for (int i = -erodeRadius; i <= erodeRadius; i++) {
			for (int j = -erodeRadius; j <= erodeRadius; j++) {
				if (i * i + j * j <= squaredRadius) {
					unsigned char rawValue = tex2D<unsigned char>(rawForeground, x + i, y + j);
					if (rawValue == 0) { isEdge = true; break; }
				}
			}
			if (isEdge) break;
		}
		if (isEdge) minPix = 0;
		else minPix = 1;
		surf2Dwrite(minPix, MaskSurface, x * sizeof(unsigned char), y);
	}
	else {
		unsigned char minPix = 0;
		surf2Dwrite(minPix, MaskSurface, x * sizeof(unsigned char), y);
	}

}

__global__ void SparseSurfelFusion::device::DilateRawForeground(cudaTextureObject_t rawMask, const int dilateRadius, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t dilateMaskSurface)
{

	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedCols || y >= clipedRows) return;
	if (dilateRadius <= x && x < clipedCols - dilateRadius && dilateRadius <= y && y < clipedRows - dilateRadius) {
		unsigned char maxPix = 0;
		bool isInside = false;
		for (int i = -dilateRadius; i <= dilateRadius; i++) {
			for (int j = -dilateRadius; j <= dilateRadius; j++) {
				if (i * i + j * j <= squaredRadius) {
					unsigned char rawValue = tex2D<unsigned char>(rawMask, x + i, y + j);
					if (rawValue == 1) { isInside = true; break; }
				}
			}
			if (isInside) break;
		}
		if (isInside) maxPix = 1;
		else maxPix = 0;
		surf2Dwrite(maxPix, dilateMaskSurface, x * sizeof(unsigned char), y);
	}
	else {
		unsigned char maxPix = 0;
		surf2Dwrite(maxPix, dilateMaskSurface, x * sizeof(unsigned char), y);
	}
}

__global__ void SparseSurfelFusion::device::ForegroundEdgeMask(cudaTextureObject_t clipedMask, const int edgeThickness, const float squaredRadius, const unsigned int clipedRows, const unsigned int clipedCols, cudaSurfaceObject_t edgeMask)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedCols || y >= clipedRows) return;
	unsigned char isEdge = 0;
	if (edgeThickness <= x && x < clipedCols - edgeThickness && edgeThickness <= y && y < clipedRows - edgeThickness) {
		bool outMask = false;
		bool inMask = false;
		for (int i = -edgeThickness; i <= edgeThickness; i++) {
			for (int j = -edgeThickness; j <= edgeThickness; j++) {
				if (i * i + j * j <= squaredRadius) {
					unsigned char mask = tex2D<unsigned char>(clipedMask, x + i, y + j);
					if (mask == (unsigned char)0 && outMask == false) {
						outMask = true;
					}
					if (mask == (unsigned char)1 && inMask == false){
						inMask = true;
					}
				}
			}
		}
		if (outMask && inMask) {
			isEdge = 1; 
		}
	}
	surf2Dwrite(isEdge, edgeMask, x * sizeof(unsigned char), y);
}

__global__ void SparseSurfelFusion::device::CollectPreviousForegroundMaskKernel(cudaTextureObject_t mask, cudaSurfaceObject_t previousMask, const unsigned int clipedRows, const unsigned int clipedCols)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedCols || y >= clipedRows) return;
	unsigned char maskValue = tex2D<unsigned char>(mask, x, y);
	surf2Dwrite(maskValue, previousMask, x * sizeof(unsigned char), y);
}

__global__ void SparseSurfelFusion::device::filterForegroundMaskKernel(cudaTextureObject_t foregroundMask, unsigned maskRows, unsigned maskCols, const float sigma, cudaSurfaceObject_t filteredMask)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= maskCols || y >= maskRows) return;

	//A window search
	const int halfsize = __float2uint_ru(sigma) * 2;
	float total_weight = 0.0f;
	float total_value = 0.0f;
	for (int neighbor_y = y - halfsize; neighbor_y <= y + halfsize; neighbor_y++) {
		for (int neighbor_x = x - halfsize; neighbor_x <= x + halfsize; neighbor_x++) {
			//Retrieve the mask value at neigbour
			const unsigned char neighbor_foreground = tex2D<unsigned char>(foregroundMask, neighbor_x, neighbor_y);

			//Compute the gaussian weight
			const float diff_x_square = (neighbor_x - x) * (neighbor_x - x);
			const float diff_y_square = (neighbor_y - y) * (neighbor_y - y);
			const float weight = __expf(0.5f * (diff_x_square + diff_y_square) / (sigma * sigma));

			//Accumlate it
			if (neighbor_x >= 0 && neighbor_x < maskCols && neighbor_y >= 0 && neighbor_y < maskRows)
			{
				total_weight += weight;
				total_value += weight * float(1 - neighbor_foreground);
			}
		}
	}


	//Compute the value locally
	const unsigned char foreground_indicator = tex2D<unsigned char>(foregroundMask, x, y);
	float filter_value = 0.0;
	if (foreground_indicator == 0) {
		filter_value = total_value / (total_weight + 1e-3f);
	}
	else {
		filter_value = 1.0f;
	}

	//Write to the surface
	surf2Dwrite(filter_value, filteredMask, x * sizeof(float), y);

}

void SparseSurfelFusion::ForegroundSegmenter::createClipedMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect)
{
	createClipedMaskTextureSurface(clipedRows, clipedCols, collect.texture, collect.surface, collect.cudaArray);
}

void SparseSurfelFusion::ForegroundSegmenter::createClipedFilteredMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, CudaTextureSurface& collect)
{
	createClipedFilteredMaskTextureSurface(clipedRows, clipedCols, collect.texture, collect.surface, collect.cudaArray);
}

void SparseSurfelFusion::ForegroundSegmenter::createClipedMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
	cudaTextureDesc maskTextureDescription;
	createDefault2DTextureDescriptor(maskTextureDescription);
	cudaChannelFormatDesc maskChannelDescription = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);	// 单通道uint8

	CHECKCUDA(cudaMallocArray(&cudaArray, &maskChannelDescription, clipedCols, clipedRows));

	cudaResourceDesc resourceDescription;
	memset(&resourceDescription, 0, sizeof(cudaResourceDesc));
	resourceDescription.resType = cudaResourceTypeArray;
	resourceDescription.res.array.array = cudaArray;

	CHECKCUDA(cudaCreateTextureObject(&texture, &resourceDescription, &maskTextureDescription, 0));
	CHECKCUDA(cudaCreateSurfaceObject(&surface, &resourceDescription));
}
void SparseSurfelFusion::ForegroundSegmenter::createClipedFilteredMaskTextureSurface(const unsigned clipedRows, const unsigned clipedCols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray)
{
	cudaTextureDesc maskTextureDescription;
	createDefault2DTextureDescriptor(maskTextureDescription);
	cudaChannelFormatDesc maskChannelDescription = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);	// 单通道float32

	CHECKCUDA(cudaMallocArray(&cudaArray, &maskChannelDescription, clipedCols, clipedRows));

	cudaResourceDesc resourceDescription;
	memset(&resourceDescription, 0, sizeof(cudaResourceDesc));
	resourceDescription.resType = cudaResourceTypeArray;
	resourceDescription.res.array.array = cudaArray;

	CHECKCUDA(cudaCreateTextureObject(&texture, &resourceDescription, &maskTextureDescription, 0));
	CHECKCUDA(cudaCreateSurfaceObject(&surface, &resourceDescription));
}
void SparseSurfelFusion::ForegroundSegmenter::SwapChannelAndNormalizeValue(DeviceArray<uchar3>& colorImages, DeviceArray<float>& normalizedSrc, cudaStream_t stream)
{
	dim3 block(256);
	dim3 grid(divUp(ImageSize, block.x));
	device::SwapChannelAndNormalizeValueKernel << <grid, block, 0, stream >> > (colorImages, normalizedSrc, rawImageRows * rawImageCols);
}

void SparseSurfelFusion::ForegroundSegmenter::CollectPreviousForegroundMask(cudaTextureObject_t mask, cudaSurfaceObject_t previousMask, const unsigned int clipedRows, const unsigned int clipedCols, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(clipedCols, block.x), divUp(clipedRows, block.y));
	device::CollectPreviousForegroundMaskKernel << <grid, block, 0, stream >> > (mask, previousMask, clipedRows, clipedCols);
}

void SparseSurfelFusion::ForegroundSegmenter::CollectAndClipMask(const unsigned int clipedRows, const unsigned int clipedCols, CudaTextureSurface rawForeground, CudaTextureSurface maskSurface, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(clipedCols, block.x), divUp(clipedRows, block.y));
	device::CollectAndClipMaskKernel << <grid, block, 0, stream >> > (MaskDeviceArray, 0, clipedRows, clipedCols, rawForeground.surface);
	device::ForegroundEdgeMask << <grid, block, 0, stream >> > (rawForeground.texture, edgeThickness, (edgeThickness + 0.5f) * (edgeThickness + 0.5f), clipedRows, clipedCols, ClipedEdgeMask.surface);
	device::ErodeRawForeground << <grid, block, 0, stream >> > (rawForeground.texture, erodeRadius, (erodeRadius + 0.5f) * (erodeRadius + 0.5f), clipedRows, clipedCols, maskSurface.surface);
	device::DilateRawForeground << <grid, block, 0, stream >> > (maskSurface.texture, dilateRadius, (dilateRadius + 0.5f) * (dilateRadius + 0.5f), clipedRows, clipedCols, maskSurface.surface);
}

void SparseSurfelFusion::ForegroundSegmenter::FilterForefroundMask(cudaTextureObject_t foregroundMask, const unsigned int clipedRows, const unsigned int clipedCols, float sigma, cudaSurfaceObject_t filteredMask, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(clipedCols, block.x), divUp(clipedRows, block.y));
	device::filterForegroundMaskKernel << <grid, block, 0, stream >> > (foregroundMask, clipedRows, clipedCols, sigma, filteredMask);
	CHECKCUDA(cudaStreamSynchronize(stream));		// 阻塞该线程，为了获得Mask的纹理

}
