#include "MergeSurface.h"
#if defined(__CUDACC__)		//�����NVCC����������
#include <cub/cub.cuh>
#endif


namespace SparseSurfelFusion {
	namespace device {
		__constant__ float TSDFThreshold = TSDF_THRESHOLD;

		__constant__ ushort2 InvalidPixelPos = { 0xFFFF, 0xFFFF };

		//The transfer from into to float			intתfloat
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
	if (idx < Camera_1_SurfelsNums) {	// ������һ���������Ԫ
		// ת������������ϵ
		const float3 CanonicalFieldSurfelPos = Camera_1_SE3.rot * Camera_1_DepthSurfels[idx].VertexAndConfidence + Camera_1_SE3.trans;
		// ������������ϵת������ֵ�������ϵ
		const float3 InterprolateSurfelPos = InterprolateCameraSE3.apply_inversed_se3(CanonicalFieldSurfelPos);
		// ת������ֵ�������������ϵ
		const ushort2 InterprolateCoordinate = {
			__float2uint_rn(((InterprolateSurfelPos.x / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_x) + InterprolateCameraIntrinsic.principal_x),
			__float2uint_rn(((InterprolateSurfelPos.y / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_y) + InterprolateCameraIntrinsic.principal_y)
		};
		//���ص��ڲ�ֵ�������������ϵ��
		if (InterprolateCoordinate.x < clipedCols && InterprolateCoordinate.y < clipedRows) {
			surfelProjectedPixelPos[idx] = make_ushort2(InterprolateCoordinate.x, InterprolateCoordinate.y);
			OverlappingOrderMap[idx] = atomicAdd(&overlappingSurfelsCountMap[InterprolateCoordinate.y * clipedCols + InterprolateCoordinate.x], 1);	// ͳ���ص����صĸ���
		}
		else {
			OverlappingOrderMap[idx] = 0xFFFF;
			surfelProjectedPixelPos[idx] = device::InvalidPixelPos;
		}
	}
	else {	// �����ڶ����������Ԫ
		const unsigned int offset = idx - Camera_1_SurfelsNums;
		// ת������������ϵ
		const float3 CanonicalFieldSurfelPos = Camera_2_SE3.rot * Camera_2_DepthSurfels[offset].VertexAndConfidence + Camera_2_SE3.trans;
		// ������������ϵת������ֵ�������ϵ
		const float3 InterprolateSurfelPos = InterprolateCameraSE3.apply_inversed_se3(CanonicalFieldSurfelPos);
		// ת������ֵ�������������ϵ
		const ushort2 InterprolateCoordinate = {
			__float2uint_rn(((InterprolateSurfelPos.x / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_x) + InterprolateCameraIntrinsic.principal_x),
			__float2uint_rn(((InterprolateSurfelPos.y / (InterprolateSurfelPos.z + 1e-10)) * InterprolateCameraIntrinsic.focal_y) + InterprolateCameraIntrinsic.principal_y)
		};
		//���ص��ڲ�ֵ�������������ϵ��
		if (InterprolateCoordinate.x < clipedCols && InterprolateCoordinate.y < clipedRows) {
			surfelProjectedPixelPos[idx] = make_ushort2(InterprolateCoordinate.x, InterprolateCoordinate.y);
			OverlappingOrderMap[idx] = atomicAdd(&overlappingSurfelsCountMap[InterprolateCoordinate.y * clipedCols + InterprolateCoordinate.x], 1);	// ͳ���ص����صĸ���
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

	// ����ֵ
	if (idx < clipedImageSize) {
		MaxValue[threadIdx.x] = overlappingSurfelsCountMap[idx];
	}
	else {
		MaxValue[threadIdx.x] = -1e6;
	}
	__syncthreads();

	// ��Լ�ȴ�С
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			float lhs, rhs;
			/** ���ֵ **/
			lhs = MaxValue[threadIdx.x];
			rhs = MaxValue[threadIdx.x + stride];
			MaxValue[threadIdx.x] = lhs < rhs ? rhs : lhs;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		MaxCountData[blockIdx.x] = MaxValue[threadIdx.x];	// �ҵ�ÿһ��block�����ֵ
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
	if (idx >= totalSurfels || OverlappingOrderMap[idx] == 0xFFFF || surfelProjectedPixelPos[idx].x == device::InvalidPixelPos.x) return;	// Ϊ��Ч��
	const unsigned int flattenIndex = surfelProjectedPixelPos[idx].y * clipedCols + surfelProjectedPixelPos[idx].x;
	const unsigned int offsetFlattenIndex = MaxOverlappingSurfelNum * flattenIndex;
	// �ص���Ԫ��N���㣺2 <= N <= MaxOverlappingSurfelNum
	if (2 <= overlappingSurfelsCountMap[flattenIndex] && overlappingSurfelsCountMap[flattenIndex] <= MaxOverlappingSurfelNum) {
		if (idx < Camera_1_SurfelCount) {	// ���1��Ԫ
			MergeArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = Camera_1_DepthSurfels[idx];
			SurfelIndexArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = idx;
		}
		else {	// ���2��Ԫ
			MergeArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = Camera_2_DepthSurfels[idx - Camera_1_SurfelCount];
			SurfelIndexArray[offsetFlattenIndex + OverlappingOrderMap[idx]] = idx;
		}
	}


}

__global__ void SparseSurfelFusion::device::CalculateOverlappingSurfelTSDF(const CameraParameters Camera_1_Para, CameraParameters Camera_2_Para, CameraParameters Camera_inter_Para, const float maxMergeSquaredDistance, const unsigned int* overlappingSurfelsCountMap, const unsigned int* SurfelIndexArray, const unsigned int Camera_1_SurfelNum, const unsigned int clipedImageSize, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, float* MergeSurfelTSDFValue, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, bool* markValidMergedSurfel)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= clipedImageSize) return;	// ����2�����û���ںϵı�Ҫ��
	const unsigned int offset = idx * MaxOverlappingSurfelNum;
	const unsigned int overlappingSurfelNum = overlappingSurfelsCountMap[idx];
	bool fromSameSurface = true;		// �Ƿ�����ͬһƽ��
	bool validMergeOperation = false;	// �Ƿ�����Ч���ںϲ��������Ƿ��������Ч��TSDF����
	// �ص���Ԫ��N���㣺2 <= N <= MaxOverlappingSurfelNum
	if (2 <= overlappingSurfelNum && overlappingSurfelNum <= MaxOverlappingSurfelNum) {
		for (int i = 0; i < overlappingSurfelNum - 1; i++) {
			if (MergeArray[offset + i].CameraID != MergeArray[offset + i + 1].CameraID) {
				fromSameSurface = false;
				break;
			}
		}
		if (!fromSameSurface) {	// ����������ͬ�棬˵����ҪTSDF�ں�
			for (int i = 0; i < overlappingSurfelNum; i++) {
				if (MergeArray[offset + i].isMerged == true) continue;	//�������Ԫ���ںϹ��������ںϵ�ǰ��Ԫ
				// ���㵱ǰ��
				float3 CanonicalPivotSurfelPos, InterprolatePivotSurfelPos;
				// ������Լ��������ڲ�ֵ�������ϵ�µ�λ�����������z������
				if (MergeArray[offset + i].CameraID == Camera_1_Para.ID) {
					CanonicalPivotSurfelPos = Camera_1_Para.SE3.rot * MergeArray[offset + i].VertexAndConfidence + Camera_1_Para.SE3.trans;// ת������������ϵ
					InterprolatePivotSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalPivotSurfelPos);// ������������ϵת������ֵ�������ϵ
				}
				else if (MergeArray[offset + i].CameraID == Camera_2_Para.ID) {
					CanonicalPivotSurfelPos = Camera_2_Para.SE3.rot * MergeArray[offset + i].VertexAndConfidence + Camera_2_Para.SE3.trans;// ת������������ϵ
					InterprolatePivotSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalPivotSurfelPos);// ������������ϵת������ֵ�������ϵ
				}
				else { /* ��Ч�� */ }

				for (int j = 0; j < overlappingSurfelNum; j++) {
					float disSquare = CalculateSquaredDistance(MergeArray[offset + i].VertexAndConfidence, MergeArray[offset + j].VertexAndConfidence);
					if (i == j || MergeArray[offset + j].isMerged == true || disSquare > maxMergeSquaredDistance) continue;	// ��TSDF�ۼӵ�����������ۼӵ��Ѿ��ںϹ��ĵ㣬���ںϴ˵�
					else {
						validMergeOperation = true;	// ��������Ч���ںϲ���
						float3 CanonicalOtherSurfelPos, InterprolateOtherSurfelPos;
						if (MergeArray[offset + j].CameraID == Camera_1_Para.ID) {
							CanonicalOtherSurfelPos = Camera_1_Para.SE3.rot * MergeArray[offset + j].VertexAndConfidence + Camera_1_Para.SE3.trans;// ת������������ϵ
							InterprolateOtherSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalOtherSurfelPos);// ������������ϵת������ֵ�������ϵ
							MergeSurfelTSDFValue[offset + i] += CalculateTSDFValue(InterprolatePivotSurfelPos.z, InterprolateOtherSurfelPos.z);
						}
						else if (MergeArray[offset + j].CameraID == Camera_2_Para.ID) {
							CanonicalOtherSurfelPos = Camera_2_Para.SE3.rot * MergeArray[offset + j].VertexAndConfidence + Camera_2_Para.SE3.trans;// ת������������ϵ
							InterprolateOtherSurfelPos = Camera_inter_Para.SE3.apply_inversed_se3(CanonicalOtherSurfelPos);// ������������ϵת������ֵ�������ϵ
							MergeSurfelTSDFValue[offset + i] += CalculateTSDFValue(InterprolatePivotSurfelPos.z, InterprolateOtherSurfelPos.z);
						}
						else { /* ��Ч�� */ }
						// ��z����������
					}
				}
			}

			if (validMergeOperation) {	// ֻҪ����Ч��TSDF�ںϲ��������ɽ���־λ��Ϊtrue
				for (int i = 0; i < overlappingSurfelNum; i++) {
					MergeArray[offset + i].isMerged = true;
					if (MergeArray[offset + i].CameraID == Camera_1_Para.ID) { Camera_1_DepthSurfels[SurfelIndexArray[offset + i]].isMerged = true; }
					else if (MergeArray[offset + i].CameraID == Camera_2_Para.ID) { Camera_2_DepthSurfels[SurfelIndexArray[offset + i] - Camera_1_SurfelNum].isMerged = true; }
					else { /* ��Ч�� */ }
				}

			}
			if (validMergeOperation) {	// ��������Ч��TSDF�ںϲ���
				unsigned int negativeClosestZeroIdx = 0;	// �����������ӽ�����ĵ�
				float negativeClosestZeroTSDF = 1e6;		// �����������ӽ�����ĵ��TSDF
				unsigned int positiveClosestZeroIdx = 0;	// ������ǰ����ӽ�����ĵ�
				float positiveClosestZeroTSDF = 1e6;		// ������ǰ����ӽ�����ĵ��TSDF

				for (int i = 0; i < overlappingSurfelNum; i++) {
					// ��ӽ�0�ĸ�TSDF
					if (MergeSurfelTSDFValue[offset + i] < 0 && -MergeSurfelTSDFValue[offset + i] < negativeClosestZeroTSDF) {
						negativeClosestZeroTSDF = -MergeSurfelTSDFValue[offset + i];
						negativeClosestZeroIdx = i;
					}
					// ��ӽ�0����TSDF
					else if (MergeSurfelTSDFValue[offset + i] > 0 && MergeSurfelTSDFValue[offset + i] < positiveClosestZeroTSDF) {
						positiveClosestZeroTSDF = MergeSurfelTSDFValue[offset + i];
						positiveClosestZeroIdx = i;
					}
					else {
						continue;
					}
				}

				DepthSurfel MergedSurfel;	// ��ʼ�ںϣ����в���Ŀǰ�������ں�
				float t0 = negativeClosestZeroTSDF / (negativeClosestZeroTSDF + positiveClosestZeroTSDF);	// �����㵽����ռ�����㵽������ı���
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
				else { /* ��Ч�� */ }
				if (MergeArray[offset + positiveClosestZeroIdx].CameraID == Camera_1_Para.ID) {
					CanonicalPositiveClosestZeroPoint = Camera_1_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence + Camera_1_Para.SE3.trans;
					CanonicalPositiveClosestZeroNormal = Camera_1_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].NormalAndRadius;
				}
				else if (MergeArray[offset + positiveClosestZeroIdx].CameraID == Camera_2_Para.ID) {
					CanonicalPositiveClosestZeroPoint = Camera_2_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence + Camera_2_Para.SE3.trans;
					CanonicalPositiveClosestZeroNormal = Camera_2_Para.SE3.rot * MergeArray[offset + positiveClosestZeroIdx].NormalAndRadius;
				}
				else { /* ��Ч�� */ }
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
				// ���������ԭʼrgb��ֵ
				float_decode_rgb_device(MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.x, rgb_0);
				float_decode_rgb_device(MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.x, rgb_1);
				//printf("idx = %d   rgb_0(%d, %d, %d)\n", idx, (int)rgb_0.x, (int)rgb_0.y, (int)rgb_0.z);
				int mergedR = (int)(t1 * (int)rgb_0.x * 1.0f + t0 * (int)rgb_1.x * 1.0f);
				int mergedG = (int)(t1 * (int)rgb_0.y * 1.0f + t0 * (int)rgb_1.y * 1.0f);
				int mergedB = (int)(t1 * (int)rgb_0.z * 1.0f + t0 * (int)rgb_1.z * 1.0f);
				const uchar3 mergedRGB = make_uchar3(mergedR, mergedG, mergedB);
				const float encodedMergedRGB = float_encode_rgb(mergedRGB);
				// �۲�ʱ��ѡ������۲쵽��ʱ�䣬����ӽ�ѡ�����Ŷ������Ǹ���Ԫ����ӽ�
				const float observeTime = MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.z > MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.z ? MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.z : MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.z;
				const float initialTime = MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.w > MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.w ? MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.w : MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.w;
				const float cameraView = MergeArray[offset + negativeClosestZeroIdx].VertexAndConfidence.w > MergeArray[offset + positiveClosestZeroIdx].VertexAndConfidence.w ? MergeArray[offset + negativeClosestZeroIdx].ColorAndTime.y : MergeArray[offset + positiveClosestZeroIdx].ColorAndTime.y;
				MergedSurfel.ColorAndTime = make_float4(encodedMergedRGB, cameraView, observeTime, initialTime);
				MergedSurfel.pixelCoordinate.col = (unsigned int)(t0 * MergeArray[offset + negativeClosestZeroIdx].pixelCoordinate.col * 1.0f + t1 * MergeArray[offset + positiveClosestZeroIdx].pixelCoordinate.col * 1.0f);
				MergedSurfel.pixelCoordinate.row = (unsigned int)(t0 * MergeArray[offset + negativeClosestZeroIdx].pixelCoordinate.row * 1.0f + t1 * MergeArray[offset + positiveClosestZeroIdx].pixelCoordinate.row * 1.0f);
				MergedSurfel.CameraID = Camera_1_Para.ID;	// ��1�����Ϊ��׼
				MergedSurfel.isMerged = false;	// �����ɵ���Ԫ��δ���ں�
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
	if (zeroPoint < otherPoint) {	// �����otherPoint��ǰ��
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
		surfel.isMerged = false;		// true:�����ɵ���Ԫ��Ҫ�������ӳ��Texture��
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
	/************************************ Step.1 �����ص���ĸ����Լ��ص����ص���˳�� ************************************/
	const unsigned int depthSurfel_1_Count = Camera_1_DepthSurfels.size();
	const unsigned int depthSurfel_2_Count = Camera_2_DepthSurfels.size();
	const unsigned int totalDepthSurfels = depthSurfel_1_Count + depthSurfel_2_Count;
	surfelProjectedPixelPos.ResizeArrayOrException(totalDepthSurfels);
	OverlappingOrderMap.ResizeArrayOrException(totalDepthSurfels);
	dim3 block_1(128);
	dim3 grid_1(divUp(totalDepthSurfels, block_1.x));
	device::CountOverlappingSurfelsKernel << <grid_1, block_1, 0, stream >> > (Camera_1_Para.SE3, Camera_1_Para.intrinsic, Camera_2_Para.SE3, Camera_2_Para.intrinsic, Camera_Inter_Para.SE3, Camera_Inter_Para.intrinsic, clipedCols, clipedRows, depthSurfel_1_Count, totalDepthSurfels, Camera_1_DepthSurfels, Camera_2_DepthSurfels, OverlappingOrderMap.Array().ptr(), OverlappingSurfelsCountMap.Array().ptr(), surfelProjectedPixelPos.Array().ptr());

	//// ����ص�����ͼ��
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

	///************************************ Step.2 Ѱ������ص�������� ************************************/
	//const unsigned int gridNum = divUp(ClipedImageSize, device::ReduceThreadsPerBlock);
	//dim3 block_2(device::ReduceThreadsPerBlock);
	//dim3 grid_2(gridNum);
	//unsigned int* maxCountData = NULL;	// ��¼ÿ��block�����ֵ�����СҲӦΪblock������
	//CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&maxCountData), sizeof(unsigned int) * gridNum, stream));
	//device::reduceMaxOverlappingSurfelCountKernel << <grid_2, block_2, 0, stream >> > (OverlappingSurfelsCountMap.Array().ptr(), ClipedImageSize, maxCountData);
	//std::vector<unsigned int> maxCountDataHost;
	//maxCountDataHost.resize(gridNum);
	//CHECKCUDA(cudaMemcpyAsync(maxCountDataHost.data(), maxCountData, sizeof(unsigned int) * gridNum, cudaMemcpyDeviceToHost, stream));
	//unsigned int MaxSurfelOverlappingCount = 0;	// ����ص��������
	//device::findMaxValueOfOverlappingCount(maxCountDataHost.data(), gridNum, MaxSurfelOverlappingCount);
	//printf("�� %d ����ֵ������ص���Ԫ���� = %d\n", Camera_Inter_Para.ID, MaxSurfelOverlappingCount);
	////printf("MergeArray���� = %d\n", MaxSurfelOverlappingCount * ClipedImageSize);
	////printf("�� %d ����ֵ���surfel_1 = %d  surfel_2 = %d\n", Camera_Inter_Para.ID, depthSurfel_1_Count, depthSurfel_2_Count);
	//if (MaxSurfelOverlappingCount > 50) LOGGING(INFO) << "���������ֵ�����λ�ˣ���ǰλ���ص�����࣬�ڴ�ķ�̫��";
	
	/************************************ Step.3 �ںϱ��� ************************************/
	DepthSurfel* MergeArray = NULL;	// ��ʱ��¼�����ӽ���Ҫ�ںϵ���Ԫ
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&MergeArray), sizeof(DepthSurfel) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	CHECKCUDA(cudaMemsetAsync(MergeArray, 0, sizeof(DepthSurfel) * ClipedImageSize * Constants::MaxOverlappingSurfel, stream));
	unsigned int* SurfelIndexArray = NULL;	// ��¼MergeArray�е����ݶ�Ӧ��Camera_1_DepthSurfels��Camera_2_DepthSurfels�е��ĸ�Surfel
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

	int* ValidMergedSurfelNums = NULL;		// �ںϺ�ĵ�Device
	int ValidMergedSurfelNumsHost = -1;		// �ںϺ�ĵ�Host
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&ValidMergedSurfelNums), sizeof(int), stream));

	void* d_temp_storage = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, MergeArray, ValidMergedSurfelFlag, CompactedMergedSurfelArray, ValidMergedSurfelNums, ClipedImageSize * Constants::MaxOverlappingSurfel, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, MergeArray, ValidMergedSurfelFlag, CompactedMergedSurfelArray, ValidMergedSurfelNums, ClipedImageSize * Constants::MaxOverlappingSurfel, stream, false));	// ɸѡ	
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

		void* d_temp_storage = NULL;    // �м���������꼴���ͷ�
		size_t temp_storage_bytes = 0;  // �м����
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, AllSurfelsCanonical, markValidNotMergedFlag, ValidNotMergedSurfelCanonical, ValidNotMergedSurfelNums, SurfelsNum, stream, false));	// ȷ����ʱ�豸�洢����
		CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
		CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, AllSurfelsCanonical, markValidNotMergedFlag, ValidNotMergedSurfelCanonical, ValidNotMergedSurfelNums, SurfelsNum, stream, false));	// ɸѡ	
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
	