#include "MergeSurface.h"
#if defined(__CUDACC__)		//如果由NVCC编译器编译
#include <cub/cub.cuh>
#endif


namespace SparseSurfelFusion {
	namespace device {
		__constant__ float TSDFThreshold = TSDF_THRESHOLD;

		__constant__ ushort2 InvalidPixelPos = { 0xFFFF, 0xFFFF };

		//The transfer from into to float			int转float
		union uint2float_union_device {
			unsigned i_value;
			float f_value;
		};

		__device__ __forceinline__ void uint_decode_rgb_device(const unsigned encoded, uchar3& rgb) {
			rgb.x = ((encoded & 0x00ff0000) >> 16);
			rgb.y = ((encoded & 0x0000ff00) >> 8);
			rgb.z = ((encoded & 0x000000ff) /*0*/);
		}
		__device__ __forceinline__ unsigned float_as_uint_device(float f_value) {
			device::uint2float_union_device u;
			u.f_value = f_value;
			return u.i_value;
		}
		__device__ __forceinline__ void float_decode_rgb_device(const float encoded, uchar3& rgb) {
			const unsigned int unsigned_encoded = float_as_uint_device(encoded);
			uint_decode_rgb_device(unsigned_encoded, rgb);
		}
	}
}

__global__ void SparseSurfelFusion::device::CountOverlappingSurfelsKernel(const mat34 Camera_1_SE3, const Intrinsic Camera_1_intrinsic, const mat34 Camera_2_SE3, const Intrinsic Camera_2_intrinsic, const mat34 InterprolateCameraSE3, const Intrinsic InterprolateCameraIntrinsic, const unsigned int clipedCols, const unsigned int clipedRows, const unsigned int Camera_1_SurfelsNums, const unsigned int totalSurfelNum, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, unsigned short* OverlappingOrderMap, unsigned int* overlappingSurfelsCountMap, ushort2* surfelProjectedPixelPos)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalSurfelNum) return;
	if (idx < Camera_1_SurfelsNums) {	// 遍历第一个相机的面元
		// 转换到世界坐标系
		const float3 CanonicalFieldSurfelPos = Camera_1_SE3.rot * Camera_1_DepthSurfels[idx].VertexAndConfidence + Camera_1_SE3.trans;
		// 将从世界坐标系转换到插值相机坐标系
		const float3 InterprolateSurfelPos = InterprolateCameraSE3.apply_inversed_se3(CanonicalFieldSurfelPos);
		// 转换到插值相机的像素坐标系
		const ushort2 InterprolateCoordinate = {
			__float2uint_rn(((InterprolateSurfelPos.x / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_x) + InterprolateCameraIntrinsic.principal_x),
			__float2uint_rn(((InterprolateSurfelPos.y / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_y) + InterprolateCameraIntrinsic.principal_y)
		};
		//像素点在插值相机的像素坐标系中
		if (InterprolateCoordinate.x < clipedCols && InterprolateCoordinate.y < clipedRows) {
			surfelProjectedPixelPos[idx] = make_ushort2(InterprolateCoordinate.x, InterprolateCoordinate.y);
			OverlappingOrderMap[idx] = atomicAdd(&overlappingSurfelsCountMap[InterprolateCoordinate.y * clipedCols + InterprolateCoordinate.x], 1);	// 统计重叠像素的个数
		}
		else {
			OverlappingOrderMap[idx] = 0xFFFF;
			surfelProjectedPixelPos[idx] = device::InvalidPixelPos;
		}
	}
	else {	// 遍历第二个相机的面元
		const unsigned int offset = idx - Camera_1_SurfelsNums;
		// 转换到世界坐标系
		const float3 CanonicalFieldSurfelPos = Camera_2_SE3.rot * Camera_2_DepthSurfels[offset].VertexAndConfidence + Camera_2_SE3.trans;
		// 将从世界坐标系转换到插值相机坐标系
		const float3 InterprolateSurfelPos = InterprolateCameraSE3.apply_inversed_se3(CanonicalFieldSurfelPos);
		// 转换到插值相机的像素坐标系
		const ushort2 InterprolateCoordinate = {
			__float2uint_rn(((InterprolateSurfelPos.x / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_x) + InterprolateCameraIntrinsic.principal_x),
			__float2uint_rn(((InterprolateSurfelPos.y / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_y) + InterprolateCameraIntrinsic.principal_y)
		};
		//像素点在插值相机的像素坐标系中
		if (InterprolateCoordinate.x < clipedCols && InterprolateCoordinate.y < clipedRows) {
			surfelProjectedPixelPos[idx] = make_ushort2(InterprolateCoordinate.x, InterprolateCoordinate.y);
			OverlappingOrderMap[idx] = atomicAdd(&overlappingSurfelsCountMap[InterprolateCoordinate.y * clipedCols + InterprolateCoordinate.x], 1);	// 统计重叠像素的个数
		}
		else {
			OverlappingOrderMap[idx] = 0xFFFF;
			surfelProjectedPixelPos[idx] = device::InvalidPixelPos;
		}
	}
}

__global__ void SparseSurfelFusion::device::reduceMaxOverlappingSurfelCountKernel(const unsigned int* overlappingSurfelsCountMap, const unsigned int clipedImageSize, unsigned int* MaxCountData)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float MaxValue[ReduceThreadsPerBlock];

	// 赋初值
	if (idx < clipedImageSize) {
		MaxValue[threadIdx.x] = overlappingSurfelsCountMap[idx];
	}
	else {
		MaxValue[threadIdx.x] = -1e6;
	}
	__syncthreads();

	// 规约比大小
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			float lhs, rhs;
			/** 最大值 **/
			lhs = MaxValue[threadIdx.x];
			rhs = MaxValue[threadIdx.x + stride];
			MaxValue[threadIdx.x] = lhs < rhs ? rhs : lhs;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		MaxCountData[blockIdx.x] = MaxValue[threadIdx.x];	// 找到每一个block的最大值
	}
}

__host__ void SparseSurfelFusion::device::findMaxValueOfOverlappingCount(const unsigned int* maxCountDataHost, const unsigned int gridNum, unsigned int& MaxValue)
{
	for (int i = 0; i < gridNum; i++) {
		if (MaxValue < maxCountDataHost[i]) {
			MaxValue = maxCountDataHost[i];
		}
	}
}

__global__ void SparseSurfelFusion::device::CollectOverlappingSurfelInPixel(const DepthSurfel* Camera_1_DepthSurfels, const DepthSurfel* Camera_2_DepthSurfels, const unsigned int* overlappingSurfelsCountMap, const unsigned short* OverlappingOrderMap, const ushort2* surfelProjectedPixelPos, const unsigned int Camera_1_SurfelCount, const unsigned int totalSurfels, const unsigned int clipedCols, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, unsigned int* SurfelIndexArray)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalSurfels || OverlappingOrderMap[idx] == 0xFFFF || surfelProjectedPixelPos[idx].x == device::InvalidPixelPos.x) return;	// 为有效点
	const unsigned int flattenIndex = surfelProjectedPixelPos[idx].y * clipedCols + surfelProjectedPixelPos[idx].x;
	const unsigned int offsetFlattenIndex = MaxOverlappingSurfelNum * flattenIndex;
	// 重叠面元数N满足：2 <= N <= MaxOverlappingSurfelNum
	if (2 <= overlappingSurfelsCountMap[flattenIndex] && overlappingSurfelsCountMap[flattenIndex] <= MaxOverlappingSurfelNum) {
		if (idx < Camera_1_SurfelCount) {	// 相机1面元
			MergeArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = Camera_1_DepthSurfels[idx];
			SurfelIndexArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = idx;
		}
		else {	// 相机2面元
			MergeArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = Camera_2_DepthSurfels[idx - Camera_1_SurfelCount];
			SurfelIndexArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = idx;
		}
	}


}

__global__ void SparseSurfelFusion::device::CalculateOverlappingSurfelTSDF(const CameraParameters Camera_1_Para, CameraParameters Camera_2_Para, CameraParameters Camera_inter_Para, const float maxMergeSquaredDistance, const unsigned int* overlappingSurfelsCountMap, const unsigned int* SurfelIndexArray, const unsigned int Camera_1_SurfelNum, const unsigned int clipedImageSize, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, float* MergeSurfelTSDFValue, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, bool* markValidMergedSurfel)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= clipedImageSize) return;	// 少于2个点就没有融合的必要了
	const unsigned int offset = idx * MaxOverlappingSurfelNum;
	const unsigned int overlappingSurfelNum = overlappingSurfelsCountMap[idx];
	bool fromSameSurface = true;		// 是否来自同一平面
	bool validMergeOperation = false;	// 是否是有效的融合操作，即是否进行了有效的TSDF计算
	// 重叠面元数N满足：2 <= N <= MaxOverlappingSurfelNum
	if (2 <= overlappingSurfelNum && overlappingSurfelNum <= MaxOverlappingSurfelNum) {
		for (int i = 0; i < overlappingSurfelNum - 1; i++) {
			if (MergeArray[offset + i].CameraID != MergeArray[offset + i + 1].CameraID) {
				fromSameSurface = false;
				break;
			}
		}
		if (!fromSameSurface) {	// 来自两个不同面，说明需要TSDF融合
			for (int i = 0; i < overlappingSurfelNum; i++) {
				if (MergeArray[offset + i].isMerged == true) continue;	//如果该面元被融合过，则不再融合当前面元
				// 计算当前点
				float3 CanonicalPivotSurfelPos, InterprolatePivotSurfelPos;
				// 将零点以及其他点在插值相机坐标系下的位置算出来，以z替代深度
				if (MergeArray[offset + i].CameraID == Camera_1_Para.ID) {
					CanonicalPivotSurfelPos = Camera_1_Para.SE3.rot * MergeArray[offset + i].VertexAndConfidence + Camera_1_Para.SE3.trans;// 转换到世界坐标系
					InterprolatePivotSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalPivotSurfelPos);// 将从世界坐标系转换到插值相机坐标系
				}
				else if (MergeArray[offset + i].CameraID == Camera_2_Para.ID) {
					CanonicalPivotSurfelPos = Camera_2_Para.SE3.rot * MergeArray[offset + i].VertexAndConfidence + Camera_2_Para.SE3.trans;// 转换到世界坐标系
					InterprolatePivotSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalPivotSurfelPos);// 将从世界坐标系转换到插值相机坐标系
				}
				else { /* 无效点 */ }

				for (int j = 0; j < overlappingSurfelNum; j++) {
					float disSquare = CalculateSquaredDistance(MergeArray[offset + i].VertexAndConfidence, MergeArray[offset + j].VertexAndConfidence);
					if (i == j || MergeArray[offset + j].isMerged == true || disSquare > maxMergeSquaredDistance) continue;	// 当TSDF累加到自身或者是累加到已经融合过的点，不融合此点
					else {
						validMergeOperation = true;	// 进行了有效的融合操作
						float3 CanonicalOtherSurfelPos, InterprolateOtherSurfelPos;
						if (MergeArray[offset + j].CameraID == Camera_1_Para.ID) {
							CanonicalOtherSurfelPos = Camera_1_Para.SE3.rot * MergeArray[offset + j].VertexAndConfidence + Camera_1_Para.SE3.trans;// 转换到世界坐标系
							InterprolateOtherSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalOtherSurfelPos);// 将从世界坐标系转换到插值相机坐标系
							MergeSurfelTSDFValue[offset + i] += CalculateTSDFValue(InterprolatePivotSurfelPos.z, InterprolateOtherSurfelPos.z);
						}
						else if (MergeArray[offset + j].CameraID == Camera_2_Para.ID) {
							CanonicalOtherSurfelPos = Camera_2_Para.SE3.rot * MergeArray[offset + j].VertexAndConfidence + Camera_2_Para.SE3.trans;// 转换到世界坐标系
							InterprolateOtherSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalOtherSurfelPos);// 将从世界坐标系转换到插值相机坐标系
							MergeSurfelTSDFValue[offset + i] += CalculateTSDFValue(InterprolatePivotSurfelPos.z, InterprolateOtherSurfelPos.z);
						}
						else { /* 无效点 */ }
						// 以z坐标替代深度
					}
				}
			}

			if (validMergeOperation) {	// 只要是有效的TSDF融合操作，即可将标志位置为true
				for (int i = 0; i < overlappingSurfelNum; i++) {
					MergeArray[offset + i].isMerged = true;
					if (MergeArray[offset + i].CameraID == Camera_1_Para.ID) { Camera_1_DepthSurfels[SurfelIndexArray[offset + i]].isMerged = true; }
					else if (MergeArray[offset + i].CameraID == Camera_2_Para.ID) { Camera_2_DepthSurfels[SurfelIndexArray[offset + i] - Camera_1_SurfelNum].isMerged = true; }
					else { /* 无效点 */ }
				}

			}
			if (validMergeOperation) {	// 必须是有效的TSDF融合操作
				unsigned int negativeClosestZeroIdx = 0;	// 在曲面后面最接近曲面的点
				float negativeClosestZeroTSDF = 1e6;		// 在曲面后面最接近曲面的点的TSDF
				unsigned int positiveClosestZeroIdx = 0;	// 在曲面前面最接近曲面的点
				float positiveClosestZeroTSDF = 1e6;		// 在曲面前面最接近曲面的点的TSDF

				for (int i = 0; i < overlappingSurfelNum; i++) {
					// 最接近0的负TSDF
					if (MergeSurfelTSDFValue[offset + i] < 0 && -MergeSurfelTSDFValue[offset + i] < negativeClosestZeroTSDF) {
						negativeClosestZeroTSDF = -MergeSurfelTSDFValue[offset + i];
						negativeClosestZeroIdx = i;
					}
					// 最接近0的正TSDF
					else if (MergeSurfelTSDFValue[offset + i] > 0 && MergeSurfelTSDFValue[offset + i] < positiveClosestZeroTSDF) {
						positiveClosestZeroTSDF = MergeSurfelTSDFValue[offset + i];
						positiveClosestZeroIdx = i;
					}
					else {
						continue;
					}
				}

				DepthSurfel MergedSurfel;	// 开始融合，所有参数目前以线性融合
				float t0 = negativeClosestZeroTSDF / (negativeClosestZeroTSDF + positiveClosestZeroTSDF);	// 负近点到零面占负近点到正近点的比例
				float t1 = 1.0f - t0;
				float3 CanonicalNegativeClosestZeroPoint, CanonicalPositiveClosestZeroPoint, CanonicalNegativeClosestZeroNormal, CanonicalPositiveClosestZeroNormal;
				if (MergeArray[offset + negativeClosestZeroIdx].CameraID == Camera_1_Para.ID) {
					CanonicalNegativeClosestZeroPoint = Camera_1_Para.SE3.rot * MergeArray[offset + negativeClosestZeroIdx].VertexAndConfidence + Camera_1_Para.SE3.trans;
					CanonicalNegativeClosestZeroNormal = Camera_1_Para.SE3.rot * MergeArray[offset + negativeClosestZeroIdx].NormalAndRadius;
				}
				else if (MergeArray[offset + negativeClosestZeroIdx].CameraID == Camera_2_Para.ID) {
					CanonicalNegativeClosestZeroPoint = Camera_2_Para.SE3.rot * MergeArray[offset + negativeClosestZeroIdx].VertexAndConfidence + Camera_2_Para.SE3.trans;
					CanonicalNegativeClosestZeroNormal = Camera_2_Para.SE3.rot * MergeArray[offset + negativeClosestZeroIdx].NormalAndRadius;
				}
				else { /* 无效点 */ }
				if (MergeArray[offset + positiveClosestZeroIdx].CameraID == Camera_1_Para.ID) {
					CanonicalPositiveClosestZeroPoint = Camera_1_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence + Camera_1_Para.SE3.trans;
					CanonicalPositiveClosestZeroNormal = Camera_1_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].NormalAndRadius;
				}
				else if (MergeArray[offset + positiveClosestZeroIdx].CameraID == Camera_2_Para.ID) {
					CanonicalPositiveClosestZeroPoint = Camera_2_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence + Camera_2_Para.SE3.trans;
					CanonicalPositiveClosestZeroNormal = Camera_2_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].NormalAndRadius;
				}
				else { /* 无效点 */ }
				float Vx = t1 * CanonicalNegativeClosestZeroPoint.x + t0 * CanonicalPositiveClosestZeroPoint.x;
				float Vy = t1 * CanonicalNegativeClosestZeroPoint.y + t0 * CanonicalPositiveClosestZeroPoint.y;
				float Vz = t1 * CanonicalNegativeClosestZeroPoint.z + t0 * CanonicalPositiveClosestZeroPoint.z;
				float Vconfidence = t1 * MergeArray[offset + negativeClosestZeroIdx].VertexAndConfidence.w + t0 * MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence.w;
				MergedSurfel.VertexAndConfidence = make_float4(Vx, Vy, Vz, Vconfidence);
				float Nx = t0 * CanonicalNegativeClosestZeroNormal.x + t1 * CanonicalPositiveClosestZeroNormal.x;
				float Ny = t0 * CanonicalNegativeClosestZeroNormal.y + t1 * CanonicalPositiveClosestZeroNormal.y;
				float Nz = t0 * CanonicalNegativeClosestZeroNormal.z + t1 * CanonicalPositiveClosestZeroNormal.z;
				float Nradius = t0 * MergeArray[offset + negativeClosestZeroIdx].NormalAndRadius.w + t1 * MergeArray[offset + positiveClosestZeroIdx].NormalAndRadius.w;
				float3 normalizedNormal = NormalizeNormal(Nx, Ny, Nz);
				MergedSurfel.NormalAndRadius = make_float4(normalizedNormal.x, normalizedNormal.y, normalizedNormal.z, Nradius);
				uchar3 rgb_0, rgb_1;
				// 获得两个面原始rgb数值
				float_decode_rgb_device(MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.x, rgb_0);
				float_decode_rgb_device(MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.x, rgb_1);
				//printf("idx = %d   rgb_0(%d, %d, %d)\n", idx, (int)rgb_0.x, (int)rgb_0.y, (int)rgb_0.z);
				int mergedR = (int)(t1 * (int)rgb_0.x * 1.0f + t0 * (int)rgb_1.x * 1.0f);
				int mergedG = (int)(t1 * (int)rgb_0.y * 1.0f + t0 * (int)rgb_1.y * 1.0f);
				int mergedB = (int)(t1 * (int)rgb_0.z * 1.0f + t0 * (int)rgb_1.z * 1.0f);
				const uchar3 mergedRGB = make_uchar3(mergedR, mergedG, mergedB);
				const float encodedMergedRGB = float_encode_rgb(mergedRGB);
				// 观察时间选择最近观察到的时间，相机视角选择置信度最大的那个面元相机视角
				const float observeTime = MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.z > MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.z ? MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.z : MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.z;
				const float initialTime = MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.w > MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.w ? MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.w : MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.w;
				const float cameraView = MergeArray[offset + negativeClosestZeroIdx].VertexAndConfidence.w > MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence.w ? MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.y : MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.y;
				MergedSurfel.ColorAndTime = make_float4(encodedMergedRGB, cameraView, observeTime, initialTime);
				MergedSurfel.pixelCoordinate.col = (unsigned int)(t0 * MergeArray[offset + negativeClosestZeroIdx].pixelCoordinate.col * 1.0f + t1 * MergeArray[offset + positiveClosestZeroIdx].pixelCoordinate.col * 1.0f);
				MergedSurfel.pixelCoordinate.row = (unsigned int)(t0 * MergeArray[offset + negativeClosestZeroIdx].pixelCoordinate.row * 1.0f + t1 * MergeArray[offset + positiveClosestZeroIdx].pixelCoordinate.row * 1.0f);
				MergedSurfel.CameraID = Camera_1_Para.ID;	// 以1号相机为基准
				MergedSurfel.isMerged = false;	// 新生成的面元并未被融合
				markValidMergedSurfel[offset] = true;
				MergeArray[offset] = MergedSurfel;
			}
		}
	}
}

__forceinline__ __device__ float SparseSurfelFusion::device::CalculateSquaredDistance(const float4& v1, const float4& v2)
{
	return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z);
}

__forceinline__ __device__ float SparseSurfelFusion::device::CalculateTSDFValue(float zeroPoint, float otherPoint)
{
	if (zeroPoint < otherPoint) {	// 零点在otherPoint的前面
		float delta = otherPoint - zeroPoint;
		if (delta > TSDFThreshold) return 1.0f;
		else return (1.0f / TSDFThreshold * delta);
	}
	else {
		float delta = zeroPoint - otherPoint;
		if (delta > TSDFThreshold) return -1.0f;
		else return (-1.0f / TSDFThreshold * delta);
	}
}

__forceinline__ __device__ void SparseSurfelFusion::device::DecodeFloatRGB(const float RGBCode, uchar3& color)
{
	unsigned int UnsignedEncode = *reinterpret_cast<const unsigned int*>(&RGBCode);
	color.x = ((UnsignedEncode & 0x00ff0000) >> 16);
	color.y = ((UnsignedEncode & 0x0000ff00) >> 8);
	color.z = ((UnsignedEncode & 0x000000ff) /*0*/);
}

__forceinline__ __device__ float3 SparseSurfelFusion::device::NormalizeNormal(float& nx, float& ny, float& nz)
{
	float3 normalizedNormal;
	float length = sqrtf(nx * nx + ny * ny + nz * nz);
	normalizedNormal.x = nx / length;
	normalizedNormal.y = ny / length;
	normalizedNormal.z = nz / length;
	return normalizedNormal;
}

__global__ void SparseSurfelFusion::device::MarkAndConvertValidNotMergedSurfeltoCanonical(const DepthSurfel* SurfelsArray, const CameraParameters cameraPara, const unsigned int SurfelsNum, DepthSurfel* SurfelsCanonical, bool* markValidNotMergedFlag)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= SurfelsNum) return;
	if (SurfelsArray[idx].isMerged == false) {
		DepthSurfel surfel;
		surfel.CameraID = SurfelsArray[idx].CameraID;
		surfel.isMerged = false;		// true:新生成的面元需要参与后面映射Texture上
		surfel.ColorAndTime = SurfelsArray[idx].ColorAndTime;
		surfel.pixelCoordinate = SurfelsArray[idx].pixelCoordinate;

		float3 surfelCanonicalCoordinate = cameraPara.SE3.rot * SurfelsArray[idx].VertexAndConfidence + cameraPara.SE3.trans;
		float3 surfelCanonicalNormal = cameraPara.SE3.rot * SurfelsArray[idx].NormalAndRadius;
		surfel.VertexAndConfidence = make_float4(surfelCanonicalCoordinate.x, surfelCanonicalCoordinate.y, surfelCanonicalCoordinate.z, SurfelsArray[idx].VertexAndConfidence.w);
		surfel.NormalAndRadius = make_float4(surfelCanonicalNormal.x, surfelCanonicalNormal.y, surfelCanonicalNormal.z, SurfelsArray[idx].NormalAndRadius.w);

		SurfelsCanonical[idx] = surfel;
		markValidNotMergedFlag[idx] = true;
	}
	else {
		markValidNotMergedFlag[idx] = false;
	}
}

void SparseSurfelFusion::MergeSurface::CalculateTSDFMergedSurfels(const CameraParameters Camera_1_Para, const CameraParameters Camera_2_Para, const CameraParameters Camera_Inter_Para, DeviceArray<DepthSurfel> Camera_1_DepthSurfels, DeviceArray<DepthSurfel> Camera_2_DepthSurfels, cudaStream_t stream)
{
	/************************************ Step.1 计算重叠点的个数以及重叠点重叠的顺序 ************************************/
	const unsigned int depthSurfel_1_Count = Camera_1_DepthSurfels.size();
	const unsigned int depthSurfel_2_Count = Camera_2_DepthSurfels.size();
	const unsigned int totalDepthSurfels = depthSurfel_1_Count + depthSurfel_2_Count;
	surfelProjectedPixelPos.ResizeArrayOrException(totalDepthSurfels);
	OverlappingOrderMap.ResizeArrayOrException(totalDepthSurfels);
	dim3 block_1(128);
	dim3 grid_1(divUp(totalDepthSurfels, block_1.x));
	device::CountOverlappingSurfelsKernel << <grid_1, block_1, 0, stream >> > (Camera_1_Para.SE3, Camera_1_Para.intrinsic, Camera_2_Para.SE3, Camera_2_Para.intrinsic, Camera_Inter_Para.SE3, Camera_Inter_Para.intrinsic, clipedCols, clipedRows, depthSurfel_1_Count, totalDepthSurfels, Camera_1_DepthSurfels, Camera_2_DepthSurfels, OverlappingOrderMap.Array().ptr(), OverlappingSurfelsCountMap.Array().ptr(), surfelProjectedPixelPos.Array().ptr());

	//// 检查重叠区域图像
	//std::vector<unsigned int> overlappingCountHost;
	//overlappingCountHost.resize(OverlappingSurfelsCountMap.ArraySize());
	//OverlappingSurfelsCountMap.ArrayView().Download(overlappingCountHost);
	//cv::Mat overlappedMap(CLIP_HEIGHT, CLIP_WIDTH, CV_32SC1, overlappingCountHost.data());
	//overlappedMap.convertTo(overlappedMap, CV_32FC1);
	//cv::Mat normalizedMat, temp;
	////cv::normalize(overlappedMap, normalizedMat, 0, 1, cv::NORM_MINMAX);
	//cv::threshold(overlappedMap, normalizedMat, Constants::MaxOverlappingSurfel, 0, cv::THRESH_TOZERO_INV);
	//cv::threshold(normalizedMat, normalizedMat, 1, 0, cv::THRESH_TOZERO);
	//cv::namedWindow("Check Overlap " + std::to_string(Camera_Inter_Para.ID), cv::WINDOW_NORMAL);
	//cv::imshow("Check Overlap " + std::to_string(Camera_Inter_Para.ID), normalizedMat);

	///************************************ Step.2 寻找最大重叠点的数量 ************************************/
	//const unsigned int gridNum = divUp(ClipedImageSize, device::ReduceThreadsPerBlock);
	//dim3 block_2(device::ReduceThreadsPerBlock);
	//dim3 grid_2(gridNum);
	//unsigned int* maxCountData = NULL;	// 记录每个block的最大值，其大小也应为block的数量
	//CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&maxCountData), sizeof(unsigned int) * gridNum, stream));
	//device::reduceMaxOverlappingSurfelCountKernel << <grid_2, block_2, 0, stream >> > (OverlappingSurfelsCountMap.Array().ptr(), ClipedImageSize, maxCountData);
	//std::vector<unsigned int> maxCountDataHost;
	//maxCountDataHost.resize(gridNum);
	//CHECKCUDA(cudaMemcpyAsync(maxCountDataHost.data(), maxCountData, sizeof(unsigned int) * gridNum, cudaMemcpyDeviceToHost, stream));
	//unsigned int MaxSurfelOverlappingCount = 0;	// 最大重叠点的数量
	//device::findMaxValueOfOverlappingCount(maxCountDataHost.data(), gridNum, MaxSurfelOverlappingCount);
	//printf("第 %d 个插值相机的重叠面元个数 = %d\n", Camera_Inter_Para.ID, MaxSurfelOverlappingCount);
	////printf("MergeArray个数 = %d\n", MaxSurfelOverlappingCount * ClipedImageSize);
	////printf("第 %d 个插值相机surfel_1 = %d  surfel_2 = %d\n", Camera_Inter_Para.ID, depthSurfel_1_Count, depthSurfel_2_Count);
	//if (MaxSurfelOverlappingCount > 50) LOGGING(INFO) << "建议调整插值相机的位姿，当前位置重叠点过多，内存耗费太大！";
	
	/************************************ Step.3 融合表面 ************************************/
	DepthSurfel* MergeArray = NULL;	// 暂时记录两个视角需要融合的面元
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergeArray), sizeof(DepthSurfel) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	CHECKCUDA(cudaMemsetAsync(MergeArray, 0, sizeof(DepthSurfel) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	unsigned int* SurfelIndexArray = NULL;	// 记录MergeArray中的数据对应的Camera_1_DepthSurfels和Camera_2_DepthSurfels中的哪个Surfel
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&SurfelIndexArray), sizeof(unsigned int) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	CHECKCUDA(cudaMemsetAsync(SurfelIndexArray, 0, sizeof(unsigned int) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));

	device::CollectOverlappingSurfelInPixel << <grid_1, block_1, 0, stream >> > (Camera_1_DepthSurfels.ptr(), Camera_2_DepthSurfels.ptr(), OverlappingSurfelsCountMap.Ptr(), OverlappingOrderMap.Ptr(), surfelProjectedPixelPos.Ptr(), depthSurfel_1_Count, totalDepthSurfels, clipedCols, Constants::MaxOverlappingSurfel, MergeArray, SurfelIndexArray);

	float* MergeSurfelTSDFValue = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergeSurfelTSDFValue), sizeof(float) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	CHECKCUDA(cudaMemsetAsync(MergeSurfelTSDFValue, 0.0f, sizeof(float) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));

	bool* ValidMergedSurfelFlag = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ValidMergedSurfelFlag), sizeof(bool) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	CHECKCUDA(cudaMemsetAsync(ValidMergedSurfelFlag, false, sizeof(bool) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));

	dim3 block_3(128);
	dim3 grid_3(divUp(ClipedImageSize, block_3.x));
	device::CalculateOverlappingSurfelTSDF << <grid_3, block_3, 0, stream >> > (Camera_1_Para, Camera_2_Para, Camera_Inter_Para, Constants::MaxTruncatedSquaredDistance, OverlappingSurfelsCountMap.Ptr(), SurfelIndexArray, depthSurfel_1_Count, ClipedImageSize, Constants::MaxOverlappingSurfel, MergeArray, MergeSurfelTSDFValue, Camera_1_DepthSurfels.ptr(), Camera_2_DepthSurfels.ptr(), ValidMergedSurfelFlag);

	DepthSurfel* CompactedMergedSurfelArray = NULL;	
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&CompactedMergedSurfelArray), sizeof(DepthSurfel) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	CHECKCUDA(cudaMemsetAsync(CompactedMergedSurfelArray, 0, sizeof(DepthSurfel) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));

	int* ValidMergedSurfelNums = NULL;		// 融合后的点Device
	int ValidMergedSurfelNumsHost = -1;		// 融合后的点Host
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ValidMergedSurfelNums), sizeof(int), stream));

	void* d_temp_storage = NULL;    // 中间变量，用完即可释放
	size_t temp_storage_bytes = 0;  // 中间变量
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, MergeArray, ValidMergedSurfelFlag, CompactedMergedSurfelArray, ValidMergedSurfelNums, ClipedImageSize * Constants::MaxOverlappingSurfel, stream, false));	// 确定临时设备存储需求
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, MergeArray, ValidMergedSurfelFlag, CompactedMergedSurfelArray, ValidMergedSurfelNums, ClipedImageSize * Constants::MaxOverlappingSurfel, stream, false));	// 筛选	
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaMemcpyAsync(&ValidMergedSurfelNumsHost, ValidMergedSurfelNums, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CHECKCUDA(cudaMemcpyAsync(MergedDepthSurfels.Ptr() + MergedDepthSurfels.ArraySize(), CompactedMergedSurfelArray, sizeof(DepthSurfel) * ValidMergedSurfelNumsHost, cudaMemcpyDeviceToDevice, stream));
	MergedDepthSurfels.ResizeArrayOrException(MergedDepthSurfels.ArraySize() + ValidMergedSurfelNumsHost);

	//CHECKCUDA(cudaFreeAsync(maxCountData, stream));
	CHECKCUDA(cudaFreeAsync(MergeArray, stream));
	CHECKCUDA(cudaFreeAsync(SurfelIndexArray, stream));
	CHECKCUDA(cudaFreeAsync(MergeSurfelTSDFValue, stream));
	CHECKCUDA(cudaFreeAsync(ValidMergedSurfelFlag, stream));
	CHECKCUDA(cudaFreeAsync(CompactedMergedSurfelArray, stream));
	CHECKCUDA(cudaFreeAsync(ValidMergedSurfelNums, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
	
}

void SparseSurfelFusion::MergeSurface::CollectNotMergedSurfels(DeviceBufferArray<DepthSurfel>* depthSurfels, cudaStream_t stream)
{
	for (int i = 0; i < CamerasCount; i++) {
		const unsigned int SurfelsNum = depthSurfels[i].ArraySize();
		DepthSurfel* AllSurfelsCanonical = NULL;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&AllSurfelsCanonical), sizeof(DepthSurfel) * SurfelsNum, stream));
		bool* markValidNotMergedFlag = NULL;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&markValidNotMergedFlag), sizeof(bool) * SurfelsNum, stream));
		dim3 block(128);
		dim3 grid(divUp(SurfelsNum, block.x));
		device::MarkAndConvertValidNotMergedSurfeltoCanonical << < grid, block, 0, stream >> > (depthSurfels[i].Ptr(), CameraParameter[2 * i], SurfelsNum, AllSurfelsCanonical, markValidNotMergedFlag);
	
		DepthSurfel* ValidNotMergedSurfelCanonical = NULL;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ValidNotMergedSurfelCanonical), sizeof(DepthSurfel) * SurfelsNum, stream));
		
		int* ValidNotMergedSurfelNums = NULL;
		int ValidNotMergedSurfelNumsHost = 0;
		CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ValidNotMergedSurfelNums), sizeof(int), stream));

		void* d_temp_storage = NULL;    // 中间变量，用完即可释放
		size_t temp_storage_bytes = 0;  // 中间变量
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, AllSurfelsCanonical, markValidNotMergedFlag, ValidNotMergedSurfelCanonical, ValidNotMergedSurfelNums, SurfelsNum, stream, false));	// 确定临时设备存储需求
		CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, AllSurfelsCanonical, markValidNotMergedFlag, ValidNotMergedSurfelCanonical, ValidNotMergedSurfelNums, SurfelsNum, stream, false));	// 筛选	
		CHECKCUDA(cudaMemcpyAsync(&ValidNotMergedSurfelNumsHost, ValidNotMergedSurfelNums, sizeof(int), cudaMemcpyDeviceToHost, stream));		
	
		CHECKCUDA(cudaStreamSynchronize(stream));
		CHECKCUDA(cudaMemcpyAsync(MergedDepthSurfels.Ptr() + MergedDepthSurfels.ArraySize(), ValidNotMergedSurfelCanonical, sizeof(DepthSurfel) * ValidNotMergedSurfelNumsHost, cudaMemcpyDeviceToDevice, stream));
		MergedDepthSurfels.ResizeArrayOrException(MergedDepthSurfels.ArraySize() + ValidNotMergedSurfelNumsHost);

		CHECKCUDA(cudaFreeAsync(AllSurfelsCanonical, stream));
		CHECKCUDA(cudaFreeAsync(markValidNotMergedFlag, stream));
		CHECKCUDA(cudaFreeAsync(ValidNotMergedSurfelCanonical, stream));
		CHECKCUDA(cudaFreeAsync(ValidNotMergedSurfelNums, stream));
		CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));

	}


}
	