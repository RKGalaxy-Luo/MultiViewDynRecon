#include "CrossViewEdgeCorrespondence.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif

__host__ __device__ __forceinline__ unsigned int SparseSurfelFusion::device::EncodeCrossViewPixelPair(unsigned int x, unsigned int y, unsigned int view)
{
	unsigned int encode = x + (y << 14) + (view << 28);
	return encode;
}

__device__ __forceinline__ bool SparseSurfelFusion::device::GetClosetCrossViewPairs(CrossViewCorrInput& input, const unsigned int& view, const unsigned int& nextView, const unsigned int& lastView, const unsigned int& x, const unsigned int& y, const unsigned int& clipedCols, const unsigned int& clipedRows, const float& pairsSquDisThreshold, CrossViewCorrPairs& crossPairs)
{
	unsigned char isEdge = tex2D<unsigned char>(input.EdgeMask[view], x, y);
	float4 vertex = tex2D<float4>(input.VertexMap[view], x, y);		// 当前视角上的vertex
	if (isEdge == (unsigned char)1 && !is_zero_vertex(vertex)) {	// 在边缘上
		//if (input.CorrPairsMap[view](y, x).x != 0xFFFF) {
		//	crossPairs.PixelPairs = make_ushort4(x, y, x, y);
		//	crossPairs.PixelViews = make_ushort2(view, view);
		//	return true;
		//}
		//else {
		//	return false;
		//}

		mat34 nextViewTransSe3 = input.InitialCameraSE3Inv[nextView] * input.InitialCameraSE3[view];
		float3 nextViewVertex = nextViewTransSe3.rot * vertex + nextViewTransSe3.trans;
		const ushort2 nextViewImageCoor = {
			__float2uint_rn(((nextViewVertex.x / (nextViewVertex.z + 1e-10)) * input.intrinsic[nextView].focal_x) + input.intrinsic[nextView].principal_x),
			__float2uint_rn(((nextViewVertex.y / (nextViewVertex.z + 1e-10)) * input.intrinsic[nextView].focal_y) + input.intrinsic[nextView].principal_y)
		};
		mat34 lastViewTransSe3 = input.InitialCameraSE3Inv[lastView] * input.InitialCameraSE3[view];
		float3 lastViewVertex = lastViewTransSe3.rot * vertex + lastViewTransSe3.trans;
		const ushort2 lastViewImageCoor = {
			__float2uint_rn(((lastViewVertex.x / (lastViewVertex.z + 1e-10)) * input.intrinsic[lastView].focal_x) + input.intrinsic[lastView].principal_x),
			__float2uint_rn(((lastViewVertex.y / (lastViewVertex.z + 1e-10)) * input.intrinsic[lastView].focal_y) + input.intrinsic[lastView].principal_y)
		};

		bool checkNextView = false, checkLastView = false;
		if (CrossViewSearchSize < nextViewImageCoor.x && nextViewImageCoor.x < clipedCols - CrossViewSearchSize &&
			CrossViewSearchSize < nextViewImageCoor.y && nextViewImageCoor.y < clipedRows - CrossViewSearchSize)
			checkNextView = true;
		if (CrossViewSearchSize < lastViewImageCoor.x && lastViewImageCoor.x < clipedCols - CrossViewSearchSize &&
			CrossViewSearchSize < lastViewImageCoor.y && lastViewImageCoor.y < clipedRows - CrossViewSearchSize)
			checkLastView = true;
		// 当前视角的点投影到下一个视角：1、在上一个或者下一个视角的图片范围。2、在上一个或者下一个视角的前景Mask上
		if (checkNextView || checkLastView) {
			float minSquaredDis = 1e6f;
			ushort2 validCorrPairs = make_ushort2(0xFFFF, 0xFFFF);
			bool nextIsClosest = true;
			for (int i = -CrossViewSearchSize; i <= CrossViewSearchSize; i += CrossViewSearchStep) {
				for (int j = -CrossViewSearchSize; j <= CrossViewSearchSize; j += CrossViewSearchStep) {
					ushort2 nextViewPixel = make_ushort2(nextViewImageCoor.x + i, nextViewImageCoor.y + j);
					float4 thisWindowVertex = tex2D<float4>(input.VertexMap[nextView], nextViewPixel.x, nextViewPixel.y);
					// 下一个视角存在有效的光流
					if (checkNextView && !is_zero_vertex(thisWindowVertex)) {
						// 遍历到的这个nextView像素点对应的vertex
						float squaredDis = squared_distance(nextViewVertex, thisWindowVertex);
						// 小于5cm
						if (squaredDis < pairsSquDisThreshold && squaredDis < minSquaredDis) {
							minSquaredDis = squaredDis;
							validCorrPairs = nextViewPixel;
							nextIsClosest = true;
						}
					}
					ushort2 lastViewPixel = make_ushort2(lastViewImageCoor.x + i, lastViewImageCoor.y + j);
					// 上一个视角存在有效的光流
					thisWindowVertex = tex2D<float4>(input.VertexMap[lastView], lastViewPixel.x, lastViewPixel.y);
					if (checkLastView && !is_zero_vertex(thisWindowVertex)) {
						// 遍历到的这个lastView像素点对应的vertex
						float squaredDis = squared_distance(lastViewVertex, thisWindowVertex);
						// 小于5cm
						if (squaredDis < pairsSquDisThreshold && squaredDis < minSquaredDis) {
							minSquaredDis = squaredDis;
							validCorrPairs = lastViewPixel;
							nextIsClosest = false;
						}
					}
				}
			}
			// 找到了有效的CorrPairs
			if (validCorrPairs.x != 0xFFFF) {
				crossPairs.PixelPairs = make_ushort4(x, y, validCorrPairs.x, validCorrPairs.y);
				if (nextIsClosest) crossPairs.PixelViews = make_ushort2(view, nextView);
				else crossPairs.PixelViews = make_ushort2(view, lastView);
				return true;
			}
		}
	}
	return false;
}

__device__ __forceinline__ bool SparseSurfelFusion::device::GetTrackedCrossViewPairs(CrossViewCorrInput& input, const CrossViewCorrPairs& closestPairs, CrossViewCorrPairs& trackedPairs)
{
	bool isTrackedPixel_1 = false, isTrackedPixel_2 = false;
	ushort2 trackedPixel_1 = make_ushort2(0xFFFF, 0xFFFF);
	ushort2 trackedPixel_2 = make_ushort2(0xFFFF, 0xFFFF);
	ushort view_1 = closestPairs.PixelViews.x;
	ushort view_2 = closestPairs.PixelViews.y;
	ushort2 center_1 = make_ushort2(closestPairs.PixelPairs.x, closestPairs.PixelPairs.y);
	ushort2 center_2 = make_ushort2(closestPairs.PixelPairs.z, closestPairs.PixelPairs.w);
	// OpticalSearchSize < CrossViewSearchSize, 因此无需判断是否出界
	for (int i = -OpticalSearchSize; i <= OpticalSearchSize; i += OpticalSearchStep) {
		for (int j = -OpticalSearchSize; j <= OpticalSearchSize; j += OpticalSearchStep) {
			if (isTrackedPixel_1 && isTrackedPixel_2) break;	//两者都回溯到了

			if (!isTrackedPixel_1) {
				ushort2 pixel_1 = make_ushort2(center_1.x + i, center_1.y + j);
				float squaredDis_1 = squared_distance(tex2D<float4>(input.VertexMap[view_1], center_1.x, center_1.y), tex2D<float4>(input.VertexMap[view_1], pixel_1.x, pixel_1.y));
				if (input.CorrPairsMap[view_1](pixel_1.y, pixel_1.x).x != 0xFFFF && squaredDis_1 < 9e-4f) {	// 有效光流, 并且距离中心vertex距离不能超过1cm
					trackedPixel_1 = pixel_1;
					isTrackedPixel_1 = true;
				}
			}

			if (!isTrackedPixel_2) {
				ushort2 pixel_2 = make_ushort2(center_2.x + i, center_2.y + j);
				float squaredDis_2 = squared_distance(tex2D<float4>(input.VertexMap[view_2], center_2.x, center_2.y), tex2D<float4>(input.VertexMap[view_2], pixel_2.x, pixel_2.y));
				if (input.CorrPairsMap[view_2](pixel_2.y, pixel_2.x).x != 0xFFFF && squaredDis_2 < 9e-4f) {	// 有效光流, 并且距离中心vertex距离不能超过1cm
					trackedPixel_2 = pixel_2;
					isTrackedPixel_2 = true;
				}
			}
		}
	}
	// 中心位置优先
	if (input.CorrPairsMap[view_1](center_1.y, center_1.x).x != 0xFFFF && input.CorrPairsMap[view_2](center_2.y, center_2.x).x != 0xFFFF) {
		trackedPairs.PixelPairs = make_ushort4(center_1.x, center_1.y, center_2.x, center_2.y);
		trackedPairs.PixelViews = make_ushort2(view_1, view_2);
		return true;
	}
	// 中心位置并不满足，且两者都回溯到了
	if (isTrackedPixel_1 && isTrackedPixel_2) {
		trackedPairs.PixelPairs = make_ushort4(trackedPixel_1.x, trackedPixel_1.y, trackedPixel_2.x, trackedPixel_2.y);
		trackedPairs.PixelViews = make_ushort2(view_1, view_2);
		return true;
	}
	return false;
}


__global__ void SparseSurfelFusion::device::MarkValidPairPixels(CrossViewCorrInput input, const unsigned int clipedCols, const unsigned int clipedRows, const unsigned int clipedImageSize, const unsigned int cameraNum, const float pairsSquDisThreshold, CrossViewCorrPairs* corrPairs, unsigned char* markValidEdgePixel)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	const unsigned int view = threadIdx.z + blockDim.z * blockIdx.z;
	if (x >= clipedCols || y >= clipedRows || view >= cameraNum) return;
	const unsigned int flattenIdx = x + y * clipedCols + view * clipedImageSize;
	const unsigned int nextView = (view + cameraNum + 1) % cameraNum;
	const unsigned int lastView = (view + cameraNum - 1) % cameraNum;
	markValidEdgePixel[flattenIdx] = (unsigned char)0;
	CrossViewCorrPairs crossPair;		// 跨视角找Pairs，目前不考虑BackTrack的问题
	bool isPairValid = GetClosetCrossViewPairs(input, view, nextView, lastView, x, y, clipedCols, clipedRows, pairsSquDisThreshold, crossPair);

	if (isPairValid) {
		corrPairs[flattenIdx] = crossPair;
		markValidEdgePixel[flattenIdx] = (unsigned char)1;
	}
}

__global__ void SparseSurfelFusion::device::SelectCrossViewBackTracingPairs(DeviceArrayView<CrossViewCorrPairs> uniquePairs, CrossViewCorrInput input, const unsigned int uniquePairsNum, unsigned char* markBackTracingPairs)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniquePairsNum) return;
	markBackTracingPairs[idx] = (unsigned char)0;
	CrossViewCorrPairs crossPair = uniquePairs[idx];
	CrossViewCorrPairs trackedPairs;	// 能用匹配点追踪的
	bool isTrackedPairs = GetTrackedCrossViewPairs(input, crossPair, trackedPairs);
	if (isTrackedPairs) {
		markBackTracingPairs[idx] = (unsigned char)1;
	}
}

__global__ void SparseSurfelFusion::device::EncodeDiffViewPixelCoor(DeviceArrayView<CrossViewCorrPairs> validCorrPairs, const unsigned int PairsNum, unsigned int* diffViewCoorKey)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= PairsNum) return;
	diffViewCoorKey[idx] = EncodeCrossViewPixelPair(validCorrPairs[idx].PixelPairs.z, validCorrPairs[idx].PixelPairs.w, validCorrPairs[idx].PixelViews.y);
}

__global__ void SparseSurfelFusion::device::EncodeSameViewPixelCoor(DeviceArrayView<CrossViewCorrPairs> validCorrPairs, const unsigned int PairsNum, unsigned int* sameViewCoorKey)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= PairsNum) return;
	sameViewCoorKey[idx] = EncodeCrossViewPixelPair(validCorrPairs[idx].PixelPairs.x, validCorrPairs[idx].PixelPairs.y, validCorrPairs[idx].PixelViews.x);
}

__global__ void SparseSurfelFusion::device::LabelSortedPixelCoor(PtrSize<unsigned int> sortedKey, const unsigned int PairsNum, unsigned int* corrPairsLabel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= PairsNum) return;
	if (idx == 0) corrPairsLabel[idx] = 1;
	else {
		if (sortedKey[idx] != sortedKey[idx - 1]) {
			corrPairsLabel[idx] = 1;
		}
		else {
			corrPairsLabel[idx] = 0;
		}
	}
}

__global__ void SparseSurfelFusion::device::ComputePairsArrayOffset(const unsigned int* pairsLabel, const unsigned int* prefixSumLabel, const unsigned int PairsNum, const unsigned int uniquePairsNum, unsigned int* offset)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= PairsNum) return;
	if (pairsLabel[idx] == 1) {
		offset[prefixSumLabel[idx] - 1] = idx;	//将这个时候的偏移的idx给compactedOffset存储
	}
	if (idx == 0) {
		offset[uniquePairsNum] = PairsNum;		// 最后一个值记录一共有多少个有效的体素点(包括重复的体素)，
	}
}

__global__ void SparseSurfelFusion::device::SelectUniqueCrossViewMatchingPairs(DeviceArrayView<CrossViewCorrPairs> sortedPairs, CrossViewCorrInput input, const unsigned int* offset, const unsigned int uniquePairsNum, CrossViewCorrPairs* uniquePairs)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniquePairsNum) return;	// 遍历安全
	float minSquaredDis = 1e6f;
	int minDisIdx = -1;
	// 匹配到同一个点的所有pair比长度
	for (int i = offset[idx]; i < offset[idx + 1]; i++) {
		float4 vertex_1 = tex2D<float4>(input.VertexMap[sortedPairs[idx].PixelViews.x], sortedPairs[idx].PixelPairs.x, sortedPairs[idx].PixelPairs.y);
		float4 vertex_2 = tex2D<float4>(input.VertexMap[sortedPairs[idx].PixelViews.y], sortedPairs[idx].PixelPairs.z, sortedPairs[idx].PixelPairs.w);
		float3 vertex_1_world = input.InitialCameraSE3[sortedPairs[idx].PixelViews.x].rot * vertex_1 + input.InitialCameraSE3[sortedPairs[idx].PixelViews.x].trans;
		float3 vertex_2_world = input.InitialCameraSE3[sortedPairs[idx].PixelViews.y].rot * vertex_2 + input.InitialCameraSE3[sortedPairs[idx].PixelViews.y].trans;
		float squaredDis = squared_distance(vertex_1_world, vertex_2_world);
		if (squaredDis < minSquaredDis) {
			minSquaredDis = squaredDis;
			minDisIdx = i;
		}
	}
	uniquePairs[idx] = sortedPairs[minDisIdx];
}



void SparseSurfelFusion::CrossViewEdgeCorrespondence::AllocateBuffer()
{
	corrPairs.AllocateBuffer(devicesCount * clipedImageSize);
	markValidPairs.AllocateBuffer(devicesCount * clipedImageSize);
	validCorrPairs.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	viewCoorKey.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	corrPairsSort.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	corrPairsLabel.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	prefixSum.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	uniqueCrossMatchingPairs.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	sortedPairsOffset.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	uniqueCrossViewBackTracingPairs.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	markValidBackTracingPairs.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::SelectValidCrossViewPairs(cudaStream_t stream)
{
	dim3 block(16, 16, 1);
	dim3 grid(divUp(ImageColsCliped, block.x), divUp(ImageRowsCliped, block.y), divUp(devicesCount, block.z));
	device::MarkValidPairPixels << <grid, block, 0, stream >> > (crossCorrInput, ImageColsCliped, ImageRowsCliped, clipedImageSize, devicesCount, Constants::CrossViewPairsSquaredDisThreshold, corrPairs.Ptr(), markValidPairs.Ptr());

	// 筛选出有效跨镜匹配点
	int* validCrossMatchedPairsNumDev = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validCrossMatchedPairsNumDev), sizeof(int), stream));
	void* d_temp_storage = NULL;    // 中间变量，用完即可释放
	size_t temp_storage_bytes = 0;  // 中间变量
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, corrPairs.Ptr(), markValidPairs.Ptr(), validCorrPairs.Ptr(), validCrossMatchedPairsNumDev, devicesCount * clipedImageSize, stream, false));	// 确定临时设备存储需求
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, corrPairs.Ptr(), markValidPairs.Ptr(), validCorrPairs.Ptr(), validCrossMatchedPairsNumDev, devicesCount * clipedImageSize, stream, false));	// 筛选	

	CHECKCUDA(cudaMemcpyAsync(&validCrossViewPairsNum, validCrossMatchedPairsNumDev, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaFreeAsync(validCrossMatchedPairsNumDev, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	validCorrPairs.ResizeArrayOrException(validCrossViewPairsNum);
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::FilterSameViewRepetitiveMatching(cudaStream_t stream)
{
	// 重复利用
	viewCoorKey.ResizeArrayOrException(UniqueMatchingNum);
	dim3 block_1(64);
	dim3 grid_1(divUp(UniqueMatchingNum, block_1.x));
	device::EncodeSameViewPixelCoor << <grid_1, block_1, 0, stream >> > (uniqueCrossMatchingPairs.ArrayView(), UniqueMatchingNum, viewCoorKey.Ptr());
	corrPairsLabel.ResizeArrayOrException(UniqueMatchingNum);
	corrPairsSort.Sort(viewCoorKey.ArrayView(), uniqueCrossMatchingPairs.ArrayView(), stream);
	DeviceArray<unsigned int>& SortedKey = corrPairsSort.valid_sorted_key;
	DeviceArrayView<CrossViewCorrPairs> SortedValue(corrPairsSort.valid_sorted_value.ptr(), UniqueMatchingNum);
	device::LabelSortedPixelCoor << <grid_1, block_1, 0, stream >> > (SortedKey, UniqueMatchingNum, corrPairsLabel.Ptr());
	prefixSum.InclusiveSum(corrPairsLabel.Array(), stream);

	// 前缀和的GPU地址给prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = prefixSum.valid_prefixsum_array;
	// 将前缀和最后一个元素即为总数
	CHECKCUDA(cudaMemcpyAsync(&UniqueMatchingNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	sortedPairsOffset.ResizeArrayOrException(UniqueMatchingNum + 1);
	uniqueCrossMatchingPairs.ResizeArrayOrException(UniqueMatchingNum);
	device::ComputePairsArrayOffset << <grid_1, block_1, 0, stream >> > (corrPairsLabel.Ptr(), prefixsumLabel.ptr(), viewCoorKey.ArraySize(), UniqueMatchingNum, sortedPairsOffset.Ptr());

	dim3 block_2(64);
	dim3 grid_2(divUp(UniqueMatchingNum, block_2.x));
	device::SelectUniqueCrossViewMatchingPairs << <grid_2, block_2, 0, stream >> > (SortedValue, crossCorrInput, sortedPairsOffset.Ptr(), UniqueMatchingNum, uniqueCrossMatchingPairs.Ptr());
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::FilterDiffViewRepetitiveMatching(cudaStream_t stream)
{
	viewCoorKey.ResizeArrayOrException(validCrossViewPairsNum);
	dim3 block_1(64);
	dim3 grid_1(divUp(validCrossViewPairsNum, block_1.x));
	device::EncodeDiffViewPixelCoor << <grid_1, block_1, 0, stream >> > (validCorrPairs.ArrayView(), validCrossViewPairsNum, viewCoorKey.Ptr());
	corrPairsLabel.ResizeArrayOrException(validCrossViewPairsNum);
	corrPairsSort.Sort(viewCoorKey.ArrayView(), validCorrPairs.ArrayView(), stream);
	DeviceArray<unsigned int>& SortedKey = corrPairsSort.valid_sorted_key;
	DeviceArrayView<CrossViewCorrPairs> SortedValue(corrPairsSort.valid_sorted_value.ptr(), validCrossViewPairsNum);
	device::LabelSortedPixelCoor << <grid_1, block_1, 0, stream >> > (SortedKey, validCrossViewPairsNum, corrPairsLabel.Ptr());
	prefixSum.InclusiveSum(corrPairsLabel.Array(), stream);

	// 前缀和的GPU地址给prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = prefixSum.valid_prefixsum_array;
	// 将前缀和最后一个元素即为总数
	CHECKCUDA(cudaMemcpyAsync(&UniqueMatchingNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	sortedPairsOffset.ResizeArrayOrException(UniqueMatchingNum + 1);
	uniqueCrossMatchingPairs.ResizeArrayOrException(UniqueMatchingNum);
	device::ComputePairsArrayOffset << <grid_1, block_1, 0, stream >> > (corrPairsLabel.Ptr(), prefixsumLabel.ptr(), viewCoorKey.ArraySize(), UniqueMatchingNum, sortedPairsOffset.Ptr());
	dim3 block_2(64);
	dim3 grid_2(divUp(UniqueMatchingNum, block_2.x));
	device::SelectUniqueCrossViewMatchingPairs << <grid_2, block_2, 0, stream >> > (SortedValue, crossCorrInput, sortedPairsOffset.Ptr(), UniqueMatchingNum, uniqueCrossMatchingPairs.Ptr());
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::FilterSeveralForOneMatchingPairs(cudaStream_t stream)
{
	FilterDiffViewRepetitiveMatching(stream);
	FilterSameViewRepetitiveMatching(stream);
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::SelectCrossViewBackTracingPairs(cudaStream_t stream)
{
	markValidBackTracingPairs.ResizeArrayOrException(UniqueMatchingNum);
	dim3 block(64);
	dim3 grid(divUp(UniqueMatchingNum, block.x));
	device::SelectCrossViewBackTracingPairs << <grid, block, 0, stream >> > (uniqueCrossMatchingPairs.ArrayView(), crossCorrInput, UniqueMatchingNum, markValidBackTracingPairs.Ptr());
	
	int* validCrossViewTracingPairsNum = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validCrossViewTracingPairsNum), sizeof(int), stream));
	void* d_temp_storage = NULL;    // 中间变量，用完即可释放
	size_t temp_storage_bytes = 0;  // 中间变量
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, uniqueCrossMatchingPairs.Ptr(), markValidBackTracingPairs.Ptr(), uniqueCrossViewBackTracingPairs.Ptr(), validCrossViewTracingPairsNum, UniqueMatchingNum, stream, false));	// 确定临时设备存储需求
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, uniqueCrossMatchingPairs.Ptr(), markValidBackTracingPairs.Ptr(), uniqueCrossViewBackTracingPairs.Ptr(), validCrossViewTracingPairsNum, UniqueMatchingNum, stream, false));	// 筛选	

	CHECKCUDA(cudaMemcpyAsync(&UniqueBackTracingPairsNum, validCrossViewTracingPairsNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaFreeAsync(validCrossViewTracingPairsNum, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	uniqueCrossViewBackTracingPairs.ResizeArrayOrException(UniqueBackTracingPairsNum);
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::FindCrossViewEdgeMatchedPairs(cudaStream_t stream)
{
	if (devicesCount >= 2) {
		SelectValidCrossViewPairs(stream);
		FilterSeveralForOneMatchingPairs(stream);
		SelectCrossViewBackTracingPairs(stream);
	}
	else {	// 小于2个视角根本无需跨镜匹配
		uniqueCrossMatchingPairs.ResizeArrayOrException(0);
	}
}
