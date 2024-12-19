/*****************************************************************//**
 * \file   VoxelSubsampler.h
 * \brief  用作对稠密顶点进行下采样
 *
 * \author LUO
 * \date   February 26th 2024
 *********************************************************************/
#include "VoxelSubsampler.h"

__global__ void SparseSurfelFusion::device::createVoxelKeyKernel(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= points.Size()) return;
	const int voxelX = __float2int_rd(points[idx].x / voxelSize); //单位化，以一个voxel_size为一个单位
	const int voxelY = __float2int_rd(points[idx].y / voxelSize);
	const int voxelZ = __float2int_rd(points[idx].z / voxelSize);
	const int encoded = encodeVoxel(voxelX, voxelY, voxelZ);
	encodedVoxelKey[idx] = encoded;
}

__global__ void SparseSurfelFusion::device::createVoxelKeyKernelDebug(DeviceArrayView<float4> points, int* encodedVoxelKey, const float voxelSize, float4* debugCanVertices, int* debugencodevoxel)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= points.Size()) return;
	const int voxelX = __float2int_rd(points[idx].x / voxelSize); //单位化，以一个voxel_size为一个单位
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
	if (voxelKeyLabel[idx] == 1) {	//当编码键值与前一个不同时
		compactedKey[prefixsumedLabel[idx] - 1] = sortedVoxelKey[idx];	//将此时的键值给compactedKey
		compactedOffset[prefixsumedLabel[idx] - 1] = idx;	//将这个时候的偏移的idx给compactedOffset存储
		//上述保证了compactedKey与compactedOffset对应：compactedKey[i]储存与前一个不一样的编码键值，compactedOffset[i]储存这个编码键值在voxelKeyLabel中的第几个
	}
	if (idx == 0) {
		// compactedVoxelKey 数量是：voxelsNum
		// compactedOffset   数量是：voxelsNum + 1
		compactedOffset[compactedOffset.Size() - 1] = sortedVoxelKey.size;	// 最后一个值记录一共有多少个有效的体素点(包括重复的体素)，
	}
}

__global__ void SparseSurfelFusion::device::samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float voxelSize, float4* sampledPoints)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= compactedKey.Size()) return;
	// 体素位置
	const int encoded = compactedKey[idx];  //在经过压缩的key中寻找
	int x, y, z;
	decodeVoxel(encoded, x, y, z);//解码key
	//voxel_center是这个编码的标准体素
	const float3 voxel_center = make_float3(float(x + 0.5) * voxelSize, float(y + 0.5) * voxelSize, float(z + 0.5) * voxelSize);

	// 找到靠近中心的那个
	float min_dist_square = 1e5; //10^5
	int min_dist_idx = compactedOffset[idx];
	//在编码键值一样的sorted_points中寻找距离最近voxel_center的那个点
	for (int i = compactedOffset[idx]; i < compactedOffset[idx + 1]; i++) {
		const float4 point4 = sortedPoints[i];
		const float3 point = make_float3(point4.x, point4.y, point4.z);
		const float new_dist = squared_norm(point - voxel_center);
		if (new_dist < min_dist_square) {
			min_dist_square = new_dist;
			min_dist_idx = i;
		}
	}

	// 将结果存储到全局内存中：找到最接近标准编码的那个顶点作为最终压缩后的顶点
	sampledPoints[idx] = sortedPoints[min_dist_idx];
}

__global__ void SparseSurfelFusion::device::samplingPointsKernel(const DeviceArrayView<int> compactedKey, const int* compactedOffset, const float4* sortedPoints, const float4* sortedColorViewTime, const float voxelSize, float4* sampledPoints)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= compactedKey.Size()) return;
	// 体素位置
	const int encoded = compactedKey[idx];  //在经过压缩的key中寻找
	int x, y, z;
	decodeVoxel(encoded, x, y, z);//解码key
	//voxel_center是这个编码的标准体素
	const float3 voxel_center = make_float3(float(x + 0.5) * voxelSize, float(y + 0.5) * voxelSize, float(z + 0.5) * voxelSize);

	// 找到靠近中心的那个
	float min_dist_square = 1e5; //10^5
	int min_dist_idx = compactedOffset[idx];
	//在编码键值一样的sorted_points中寻找距离最近voxel_center的那个点
	for (int i = compactedOffset[idx]; i < compactedOffset[idx + 1]; i++) {
		const float4 point4 = sortedPoints[i];
		const float3 point = make_float3(point4.x, point4.y, point4.z);
		const float new_dist = squared_norm(point - voxel_center);
		if (new_dist < min_dist_square) {
			min_dist_square = new_dist;
			min_dist_idx = i;
		}
	}

	// 将结果存储到全局内存中：找到最接近标准编码的那个顶点作为最终压缩后的顶点
	sampledPoints[idx] = sortedPoints[min_dist_idx];
	sampledPoints[idx].w = sortedColorViewTime[min_dist_idx].y;	// 节点的w分量存储节点来源，而非置信度

}


void SparseSurfelFusion::VoxelSubsampler::buildVoxelKeyForPoints(const DeviceArrayView<float4>& points, const float voxel_size, cudaStream_t stream)
{
	// 纠正数组的大小
	pointKey.ResizeArrayOrException(points.Size());
	// 构建体素
	dim3 block(256); //一个block中有256个线程
	dim3 grid(divUp(points.Size(), block.x));
	device::createVoxelKeyKernel << <grid, block, 0, stream >> > (points, pointKey, voxel_size);
}

void SparseSurfelFusion::VoxelSubsampler::sortCompactVoxelKeys(const DeviceArrayView<float4>& points, cudaStream_t stream)
{
	//执行排序，根据键值对排序（升序），同时也对points进行了排序，m_point_key和points是一一对应的
	pointKeySort.Sort(pointKey.ArrayReadOnly(), points, stream);
	//只要运行了m_point_key_sort，那么m_point_key_sort.valid_sorted_key和m_point_key_sort.valid_sorted_value便是排列好数组的首地址
	//标记已排序的键
	voxelLabel.ResizeArrayOrException(points.Size());
	dim3 block(128);
	dim3 grid(divUp(points.Size(), block.x));
	device::labelSortedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel.ArrayHandle());


	//前项和，这个地方直接将这个地方直接将计算出来的前缀和首地址赋值给了m_voxel_label_prefixsum.valid_prefixsum_array
	voxelLabelPrefixsum.InclusiveSum(voxelLabel.ArrayView(), stream); // 这样相同体素的值就是一直是当前体素的index

	//查询体素数(CPU中声明的数)
	unsigned int voxelsNum;
	//前缀和的GPU地址给prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = voxelLabelPrefixsum.valid_prefixsum_array;

	//将前缀和从GPU中拷贝到voxelsNum中，标记到最后一个就是体素的总数量
	CHECKCUDA(cudaMemcpyAsync(&voxelsNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));

	//构造压缩数组
	compactedVoxelKey.ResizeArrayOrException(voxelsNum);			//给compactedVoxelKey开辟空间
	compactedVoxelOffset.ResizeArrayOrException(voxelsNum + 1);		//给compactedVoxelOffset开辟空间
	device::compactedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel, prefixsumLabel, compactedVoxelKey, compactedVoxelOffset.ArrayHandle());
}


void SparseSurfelFusion::VoxelSubsampler::sortCompactVoxelKeys(const DeviceArrayView<float4>& points, const DeviceArrayView<float4>& colorViewTime, cudaStream_t stream)
{
	//执行排序，根据键值对排序（升序），同时也对points进行了排序，m_point_key和points是一一对应的
	pointKeySort.Sort(pointKey.ArrayReadOnly(), points, stream);
	//只要运行了m_point_key_sort，那么m_point_key_sort.valid_sorted_key和m_point_key_sort.valid_sorted_value便是排列好数组的首地址
	//标记已排序的键
	voxelLabel.ResizeArrayOrException(points.Size());
	dim3 block(128);
	dim3 grid(divUp(points.Size(), block.x));
	device::labelSortedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel.ArrayHandle());

	// 与Point保持一直，方便追溯
	colorViewTimeSort.Sort(pointKey.ArrayReadOnly(), colorViewTime, stream);

	//前项和，这个地方直接将这个地方直接将计算出来的前缀和首地址赋值给了m_voxel_label_prefixsum.valid_prefixsum_array
	voxelLabelPrefixsum.InclusiveSum(voxelLabel.ArrayView(), stream); // 这样相同体素的值就是一直是当前体素的index

	//查询体素数(CPU中声明的数)
	unsigned int voxelsNum;
	//前缀和的GPU地址给prefixsum_label
	const DeviceArray<unsigned int>& prefixsumLabel = voxelLabelPrefixsum.valid_prefixsum_array;

	//将前缀和从GPU中拷贝到voxelsNum中，标记到最后一个就是体素的总数量
	CHECKCUDA(cudaMemcpyAsync(&voxelsNum, prefixsumLabel.ptr() + prefixsumLabel.size() - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	//构造压缩数组
	compactedVoxelKey.ResizeArrayOrException(voxelsNum);			//给compactedVoxelKey开辟空间
	compactedVoxelOffset.ResizeArrayOrException(voxelsNum + 1);		//给compactedVoxelOffset开辟空间
	device::compactedVoxelKeyKernel << <grid, block, 0, stream >> > (pointKeySort.valid_sorted_key, voxelLabel, prefixsumLabel, compactedVoxelKey, compactedVoxelOffset.ArrayHandle());

}
void SparseSurfelFusion::VoxelSubsampler::collectSubsampledPoint(DeviceBufferArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream)
{
	// 校正subsampled_points大小
	const unsigned int voxelsNum = compactedVoxelKey.ArraySize();
	subsampled_points.ResizeArrayOrException(voxelsNum);

	dim3 block(128);
	dim3 grid(divUp(voxelsNum, block.x));
	device::samplingPointsKernel << <grid, block, 0, stream >> > (compactedVoxelKey.ArrayView(), compactedVoxelOffset, pointKeySort.valid_sorted_value, voxel_size, subsampled_points);

}

void SparseSurfelFusion::VoxelSubsampler::collectSynchronizeSubsampledPoint(SynchronizeArray<float4>& subsampled_points, const float voxel_size, cudaStream_t stream)
{
	//开辟空间
	const unsigned int num_voxels = compactedVoxelKey.ArraySize();
	subsampled_points.ResizeArrayOrException(num_voxels);

	//把它交给GPU
	DeviceArrayHandle<float4> subsampled_points_slice = subsampled_points.DeviceArrayReadWrite();
	dim3 block(128);
	dim3 grid(divUp(num_voxels, block.x));
	device::samplingPointsKernel << <grid, block, 0, stream >> > (compactedVoxelKey.ArrayView(), compactedVoxelOffset, pointKeySort.valid_sorted_value, colorViewTimeSort.valid_sorted_value, voxel_size, subsampled_points_slice);

	//与主机同步：等待所有点采样完成之后再进行下一步
	subsampled_points.SynchronizeToHost(stream);
}
