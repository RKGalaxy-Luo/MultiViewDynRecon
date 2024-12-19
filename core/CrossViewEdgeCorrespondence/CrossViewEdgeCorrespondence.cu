#include "CrossViewEdgeCorrespondence.h"
#if defined(__CUDACC__)		//�����NVCC����������
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
	float4 vertex = tex2D<float4>(input.VertexMap[view], x, y);		// ��ǰ�ӽ��ϵ�vertex
	if (isEdge == (unsigned char)1 && !is_zero_vertex(vertex)) {	// �ڱ�Ե��
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
		// ��ǰ�ӽǵĵ�ͶӰ����һ���ӽǣ�1������һ��������һ���ӽǵ�ͼƬ��Χ��2������һ��������һ���ӽǵ�ǰ��Mask��
		if (checkNextView || checkLastView) {
			float minSquaredDis = 1e6f;
			ushort2 validCorrPairs = make_ushort2(0xFFFF, 0xFFFF);
			bool nextIsClosest = true;
			for (int i = -CrossViewSearchSize; i <= CrossViewSearchSize; i += CrossViewSearchStep) {
				for (int j = -CrossViewSearchSize; j <= CrossViewSearchSize; j += CrossViewSearchStep) {
					ushort2 nextViewPixel = make_ushort2(nextViewImageCoor.x + i, nextViewImageCoor.y + j);
					float4 thisWindowVertex = tex2D<float4>(input.VertexMap[nextView], nextViewPixel.x, nextViewPixel.y);
					// ��һ���ӽǴ�����Ч�Ĺ���
					if (checkNextView && !is_zero_vertex(thisWindowVertex)) {
						// �����������nextView���ص��Ӧ��vertex
						float squaredDis = squared_distance(nextViewVertex, thisWindowVertex);
						// С��5cm
						if (squaredDis < pairsSquDisThreshold && squaredDis < minSquaredDis) {
							minSquaredDis = squaredDis;
							validCorrPairs = nextViewPixel;
							nextIsClosest = true;
						}
					}
					ushort2 lastViewPixel = make_ushort2(lastViewImageCoor.x + i, lastViewImageCoor.y + j);
					// ��һ���ӽǴ�����Ч�Ĺ���
					thisWindowVertex = tex2D<float4>(input.VertexMap[lastView], lastViewPixel.x, lastViewPixel.y);
					if (checkLastView && !is_zero_vertex(thisWindowVertex)) {
						// �����������lastView���ص��Ӧ��vertex
						float squaredDis = squared_distance(lastViewVertex, thisWindowVertex);
						// С��5cm
						if (squaredDis < pairsSquDisThreshold && squaredDis < minSquaredDis) {
							minSquaredDis = squaredDis;
							validCorrPairs = lastViewPixel;
							nextIsClosest = false;
						}
					}
				}
			}
			// �ҵ�����Ч��CorrPairs
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
	// OpticalSearchSize < CrossViewSearchSize, ��������ж��Ƿ����
	for (int i = -OpticalSearchSize; i <= OpticalSearchSize; i += OpticalSearchStep) {
		for (int j = -OpticalSearchSize; j <= OpticalSearchSize; j += OpticalSearchStep) {
			if (isTrackedPixel_1 && isTrackedPixel_2) break;	//���߶����ݵ���

			if (!isTrackedPixel_1) {
				ushort2 pixel_1 = make_ushort2(center_1.x + i, center_1.y + j);
				float squaredDis_1 = squared_distance(tex2D<float4>(input.VertexMap[view_1], center_1.x, center_1.y), tex2D<float4>(input.VertexMap[view_1], pixel_1.x, pixel_1.y));
				if (input.CorrPairsMap[view_1](pixel_1.y, pixel_1.x).x != 0xFFFF && squaredDis_1 < 9e-4f) {	// ��Ч����, ���Ҿ�������vertex���벻�ܳ���1cm
					trackedPixel_1 = pixel_1;
					isTrackedPixel_1 = true;
				}
			}

			if (!isTrackedPixel_2) {
				ushort2 pixel_2 = make_ushort2(center_2.x + i, center_2.y + j);
				float squaredDis_2 = squared_distance(tex2D<float4>(input.VertexMap[view_2], center_2.x, center_2.y), tex2D<float4>(input.VertexMap[view_2], pixel_2.x, pixel_2.y));
				if (input.CorrPairsMap[view_2](pixel_2.y, pixel_2.x).x != 0xFFFF && squaredDis_2 < 9e-4f) {	// ��Ч����, ���Ҿ�������vertex���벻�ܳ���1cm
					trackedPixel_2 = pixel_2;
					isTrackedPixel_2 = true;
				}
			}
		}
	}
	// ����λ������
	if (input.CorrPairsMap[view_1](center_1.y, center_1.x).x != 0xFFFF && input.CorrPairsMap[view_2](center_2.y, center_2.x).x != 0xFFFF) {
		trackedPairs.PixelPairs = make_ushort4(center_1.x, center_1.y, center_2.x, center_2.y);
		trackedPairs.PixelViews = make_ushort2(view_1, view_2);
		return true;
	}
	// ����λ�ò������㣬�����߶����ݵ���
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
	CrossViewCorrPairs crossPair;		// ���ӽ���Pairs��Ŀǰ������BackTrack������
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
	CrossViewCorrPairs trackedPairs;	// ����ƥ���׷�ٵ�
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
		offset[prefixSumLabel[idx] - 1] = idx;	//�����ʱ���ƫ�Ƶ�idx��compactedOffset�洢
	}
	if (idx == 0) {
		offset[uniquePairsNum] = PairsNum;		// ���һ��ֵ��¼һ���ж��ٸ���Ч�����ص�(�����ظ�������)��
	}
}

__global__ void SparseSurfelFusion::device::SelectUniqueCrossViewMatchingPairs(DeviceArrayView<CrossViewCorrPairs> sortedPairs, CrossViewCorrInput input, const unsigned int* offset, const unsigned int uniquePairsNum, CrossViewCorrPairs* uniquePairs)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= uniquePairsNum) return;	// ������ȫ
	float minSquaredDis = 1e6f;
	int minDisIdx = -1;
	// ƥ�䵽ͬһ���������pair�ȳ���
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

	// ɸѡ����Ч�羵ƥ���
	int* validCrossMatchedPairsNumDev = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&validCrossMatchedPairsNumDev), sizeof(int), stream));
	void* d_temp_storage = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, corrPairs.Ptr(), markValidPairs.Ptr(), validCorrPairs.Ptr(), validCrossMatchedPairsNumDev, devicesCount * clipedImageSize, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, corrPairs.Ptr(), markValidPairs.Ptr(), validCorrPairs.Ptr(), validCrossMatchedPairsNumDev, devicesCount * clipedImageSize, stream, false));	// ɸѡ	

	CHECKCUDA(cudaMemcpyAsync(&validCrossViewPairsNum, validCrossMatchedPairsNumDev, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaFreeAsync(validCrossMatchedPairsNumDev, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	validCorrPairs.ResizeArrayOrException(validCrossViewPairsNum);
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::FilterSameViewRepetitiveMatching(cudaStream_t stream)
{
	// �ظ�����
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

	// ǰ׺�͵�GPU��ַ��prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = prefixSum.valid_prefixsum_array;
	// ��ǰ׺�����һ��Ԫ�ؼ�Ϊ����
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

	// ǰ׺�͵�GPU��ַ��prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = prefixSum.valid_prefixsum_array;
	// ��ǰ׺�����һ��Ԫ�ؼ�Ϊ����
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
	void* d_temp_storage = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, uniqueCrossMatchingPairs.Ptr(), markValidBackTracingPairs.Ptr(), uniqueCrossViewBackTracingPairs.Ptr(), validCrossViewTracingPairsNum, UniqueMatchingNum, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, uniqueCrossMatchingPairs.Ptr(), markValidBackTracingPairs.Ptr(), uniqueCrossViewBackTracingPairs.Ptr(), validCrossViewTracingPairsNum, UniqueMatchingNum, stream, false));	// ɸѡ	

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
	else {	// С��2���ӽǸ�������羵ƥ��
		uniqueCrossMatchingPairs.ResizeArrayOrException(0);
	}
}
