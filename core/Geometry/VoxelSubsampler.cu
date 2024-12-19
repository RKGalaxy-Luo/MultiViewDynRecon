/*****************************************************************//**
 * \file   VoxelSubsampler.h
 * \brief  �����Գ��ܶ�������²���
 *
 * \author LUO
 * \date   February 26th 2024
 *********************************************************************/
#include "VoxelSubsampler.h"

__global__ void SparseSurfelFusion::device::createVoxelKeyKernel(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= points.Size()) return;
	const int voxelX = __float2int_rd(points[idx].x / voxelSize); //��λ������һ��voxel_sizeΪһ����λ
	const int voxelY = __float2int_rd(points[idx].y / voxelSize);
	const int voxelZ = __float2int_rd(points[idx].z / voxelSize);
	const int encoded = encodeVoxel(voxelX, voxelY, voxelZ);
	encodedVoxelKey[idx] = encoded;
}

__global__ void SparseSurfelFusion::device::createVoxelKeyKernelDebug(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize, float4* debugCanVertices, int* debugencodevoxel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= points.Size()) return;
	const int voxelX = __float2int_rd(points[idx].x / voxelSize); //��λ������һ��voxel_sizeΪһ����λ
	const int voxelY = __float2int_rd(points[idx].y / voxelSize);
	const int voxelZ = __float2int_rd(points[idx].z / voxelSize);
	const int encoded = encodeVoxel(voxelX, voxelY, voxelZ);
	encodedVoxelKey[idx] = encoded;
	
	debugCanVertices[idx] = points[idx];
	debugencodevoxel[idx] = encoded;

}

__global__ void SparseSurfelFusion::device::labelSortedVoxelKeyKernel(const PtrSize<const int> sortedVoxelKey, unsigned int* keyLabel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx == 0) keyLabel[0] = 1;
	else {
		if (sortedVoxelKey[idx] != sortedVoxelKey[idx - 1])
			keyLabel[idx] = 1;
		else
			keyLabel[idx] = 0;
	}
}

__global__ void SparseSurfelFusion::device::compactedVoxelKeyKernel(const PtrSize<const int> sortedVoxelKey, const unsigned int* voxelKeyLabel, const unsigned int* prefixsumedLabel, int* compactedKey, DeviceArrayHandle<int> compactedOffset)
{
	const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedVoxelKey.size) return;
	if (voxelKeyLabel[idx] == 1) {	//�������ֵ��ǰһ����ͬʱ
		compactedKey[prefixsumedLabel[idx] - 1] = sortedVoxelKey[idx];	//����ʱ�ļ�ֵ��compactedKey
		compactedOffset[prefixsumedLabel[idx] - 1] = idx;	//�����ʱ���ƫ�Ƶ�idx��compactedOffset�洢
		//������֤��compactedKey��compactedOffset��Ӧ��compactedKey[i]������ǰһ����һ���ı����ֵ��compactedOffset[i]������������ֵ��voxelKeyLabel�еĵڼ���
	}
	if (idx == 0) {
		// compactedVoxelKey �����ǣ�voxelsNum
		// compactedOffset   �����ǣ�voxelsNum + 1
		compactedOffset[compactedOffset.Size() - 1] = sortedVoxelKey.size;	// ���һ��ֵ��¼һ���ж��ٸ���Ч�����ص�(�����ظ�������)��
	}
}

__global__ void SparseSurfelFusion::device::samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float voxelSize, float4* sampledPoints)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= compactedKey.Size()) return;
	// ����λ��
	const int encoded = compactedKey[idx];  //�ھ���ѹ����key��Ѱ��
	int x, y, z;
	decodeVoxel(encoded, x, y, z);//����key
	//voxel_center���������ı�׼����
	const float3 voxel_center = make_float3(float(x + 0.5) * voxelSize, float(y + 0.5) * voxelSize, float(z + 0.5) * voxelSize);

	// �ҵ��������ĵ��Ǹ�
	float min_dist_square = 1e5; //10^5
	int min_dist_idx = compactedOffset[idx];
	//�ڱ����ֵһ����sorted_points��Ѱ�Ҿ������voxel_center���Ǹ���
	for (int i = compactedOffset[idx]; i < compactedOffset[idx + 1]; i++) {
		const float4 point4 = sortedPoints[i];
		const float3 point = make_float3(point4.x, point4.y, point4.z);
		const float new_dist = squared_norm(point - voxel_center);
		if (new_dist < min_dist_square) {
			min_dist_square = new_dist;
			min_dist_idx = i;
		}
	}

	// ������洢��ȫ���ڴ��У��ҵ���ӽ���׼������Ǹ�������Ϊ����ѹ����Ķ���
	sampledPoints[idx] = sortedPoints[min_dist_idx];
}

__global__ void SparseSurfelFusion::device::samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float4* sortedColorViewTime, const float voxelSize, float4* sampledPoints)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= compactedKey.Size()) return;
	// ����λ��
	const int encoded = compactedKey[idx];  //�ھ���ѹ����key��Ѱ��
	int x, y, z;
	decodeVoxel(encoded, x, y, z);//����key
	//voxel_center���������ı�׼����
	const float3 voxel_center = make_float3(float(x + 0.5) * voxelSize, float(y + 0.5) * voxelSize, float(z + 0.5) * voxelSize);

	// �ҵ��������ĵ��Ǹ�
	float min_dist_square = 1e5; //10^5
	int min_dist_idx = compactedOffset[idx];
	//�ڱ����ֵһ����sorted_points��Ѱ�Ҿ������voxel_center���Ǹ���
	for (int i = compactedOffset[idx]; i < compactedOffset[idx + 1]; i++) {
		const float4 point4 = sortedPoints[i];
		const float3 point = make_float3(point4.x, point4.y, point4.z);
		const float new_dist = squared_norm(point - voxel_center);
		if (new_dist < min_dist_square) {
			min_dist_square = new_dist;
			min_dist_idx = i;
		}
	}

	// ������洢��ȫ���ڴ��У��ҵ���ӽ���׼������Ǹ�������Ϊ����ѹ����Ķ���
	sampledPoints[idx] = sortedPoints[min_dist_idx];
	sampledPoints[idx].w = sortedColorViewTime[min_dist_idx].y;	// �ڵ��w�����洢�ڵ���Դ���������Ŷ�

}


void SparseSurfelFusion::VoxelSubsampler::buildVoxelKeyForPoints(const DeviceArrayView<float4>& points, const float voxel_size, cudaStream_t stream)
{
	// ��������Ĵ�С
	pointKey.ResizeArrayOrException(points.Size());
	// ��������
	dim3 block(256); //һ��block����256���߳�
	dim3 grid(divUp(points.Size(), block.x));
	device::createVoxelKeyKernel << <grid, block, 0, stream >> > (points, pointKey, voxel_size);
}

void SparseSurfelFusion::VoxelSubsampler::sortCompactVoxelKeys(const DeviceArrayView<float4>& points, cudaStream_t stream)
{
	//ִ�����򣬸��ݼ�ֵ���������򣩣�ͬʱҲ��points����������m_point_key��points��һһ��Ӧ��
	pointKeySort.Sort(pointKey.ArrayReadOnly(), points, stream);
	//ֻҪ������m_point_key_sort����ôm_point_key_sort.valid_sorted_key��m_point_key_sort.valid_sorted_value�������к�������׵�ַ
	//���������ļ�
	voxelLabel.ResizeArrayOrException(points.Size());
	dim3 block(128);
	dim3 grid(divUp(points.Size(), block.x));
	device::labelSortedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel.ArrayHandle());


	//ǰ��ͣ�����ط�ֱ�ӽ�����ط�ֱ�ӽ����������ǰ׺���׵�ַ��ֵ����m_voxel_label_prefixsum.valid_prefixsum_array
	voxelLabelPrefixsum.InclusiveSum(voxelLabel.ArrayView(), stream); // ������ͬ���ص�ֵ����һֱ�ǵ�ǰ���ص�index

	//��ѯ������(CPU����������)
	unsigned int voxelsNum;
	//ǰ׺�͵�GPU��ַ��prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = voxelLabelPrefixsum.valid_prefixsum_array;

	//��ǰ׺�ʹ�GPU�п�����voxelsNum�У���ǵ����һ���������ص�������
	CHECKCUDA(cudaMemcpyAsync(&voxelsNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));

	//����ѹ������
	compactedVoxelKey.ResizeArrayOrException(voxelsNum);			//��compactedVoxelKey���ٿռ�
	compactedVoxelOffset.ResizeArrayOrException(voxelsNum + 1);		//��compactedVoxelOffset���ٿռ�
	device::compactedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel, prefixsumLabel, compactedVoxelKey, compactedVoxelOffset.ArrayHandle());
}


void SparseSurfelFusion::VoxelSubsampler::sortCompactVoxelKeys(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, cudaStream_t stream)
{
	//ִ�����򣬸��ݼ�ֵ���������򣩣�ͬʱҲ��points����������m_point_key��points��һһ��Ӧ��
	pointKeySort.Sort(pointKey.ArrayReadOnly(), points, stream);
	//ֻҪ������m_point_key_sort����ôm_point_key_sort.valid_sorted_key��m_point_key_sort.valid_sorted_value�������к�������׵�ַ
	//���������ļ�
	voxelLabel.ResizeArrayOrException(points.Size());
	dim3 block(128);
	dim3 grid(divUp(points.Size(), block.x));
	device::labelSortedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel.ArrayHandle());

	// ��Point����һֱ������׷��
	colorViewTimeSort.Sort(pointKey.ArrayReadOnly(), colorViewTime, stream);

	//ǰ��ͣ�����ط�ֱ�ӽ�����ط�ֱ�ӽ����������ǰ׺���׵�ַ��ֵ����m_voxel_label_prefixsum.valid_prefixsum_array
	voxelLabelPrefixsum.InclusiveSum(voxelLabel.ArrayView(), stream); // ������ͬ���ص�ֵ����һֱ�ǵ�ǰ���ص�index

	//��ѯ������(CPU����������)
	unsigned int voxelsNum;
	//ǰ׺�͵�GPU��ַ��prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = voxelLabelPrefixsum.valid_prefixsum_array;

	//��ǰ׺�ʹ�GPU�п�����voxelsNum�У���ǵ����һ���������ص�������
	CHECKCUDA(cudaMemcpyAsync(&voxelsNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	//����ѹ������
	compactedVoxelKey.ResizeArrayOrException(voxelsNum);			//��compactedVoxelKey���ٿռ�
	compactedVoxelOffset.ResizeArrayOrException(voxelsNum + 1);		//��compactedVoxelOffset���ٿռ�
	device::compactedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel, prefixsumLabel, compactedVoxelKey, compactedVoxelOffset.ArrayHandle());

}
void SparseSurfelFusion::VoxelSubsampler::collectSubsampledPoint(DeviceBufferArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream)
{
	// У��subsampled_points��С
	const unsigned int voxelsNum = compactedVoxelKey.ArraySize();
	subsampled_points.ResizeArrayOrException(voxelsNum);

	dim3 block(128);
	dim3 grid(divUp(voxelsNum, block.x));
	device::samplingPointsKernel << <grid, block, 0, stream >> > (compactedVoxelKey.ArrayView(), compactedVoxelOffset, pointKeySort.valid_sorted_value, voxel_size, subsampled_points);

}

void SparseSurfelFusion::VoxelSubsampler::collectSynchronizeSubsampledPoint(SynchronizeArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream)
{
	//���ٿռ�
	const unsigned int num_voxels = compactedVoxelKey.ArraySize();
	subsampled_points.ResizeArrayOrException(num_voxels);

	//��������GPU
	DeviceArrayHandle<float4> subsampled_points_slice = subsampled_points.DeviceArrayReadWrite();
	dim3 block(128);
	dim3 grid(divUp(num_voxels, block.x));
	device::samplingPointsKernel << <grid, block, 0, stream >> > (compactedVoxelKey.ArrayView(), compactedVoxelOffset, pointKeySort.valid_sorted_value, colorViewTimeSort.valid_sorted_value, voxel_size, subsampled_points_slice);

	//������ͬ�����ȴ����е�������֮���ٽ�����һ��
	subsampled_points.SynchronizeToHost(stream);
}
