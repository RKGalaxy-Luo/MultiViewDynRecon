/*****************************************************************//**
 * \file   RigidSolver.cu
 * \brief  ICP������׼
 * 
 * \author LUO
 * \date   February 3rd 2024
 *********************************************************************/
#include "RigidSolver.h"
#include <base/DeviceAPI/device_intrinsics.h>

__device__ __forceinline__ void SparseSurfelFusion::device::Point2PointICP::solverIterationCamerasPosition(DeviceArrayView<float4> referenceVertices, DeviceArrayView<float4> conversionVertices, DeviceArrayView<ushort4> vertexKNN, const unsigned int pointsNum, PtrStep<float> reduceBuffer) const
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Ax = b
	float3 A_Matrix[6] = { 0.0f };				// ����ϵ������A
	float3 b_Vector = { 0.0f };					// �����Ҳ�����

	if (idx < pointsNum) {
		const float4 point = conversionVertices[idx];						// conversion����еĵ�A
		const float4 nearestPoint = referenceVertices[knnIndex[idx].x];		// reference�о����A����ĵ�

		const float3 conversionPre = currentPositionMat34.rot * point + currentPositionMat34.trans;	// ͨ������任����conversion�䵽��reference�����λ��

		A_Matrix[0] = make_float3(0.0f, -conversionPre.z, conversionPre.y);
		A_Matrix[1] = make_float3(conversionPre.z, 0.0f, -conversionPre.x);
		A_Matrix[2] = make_float3(-conversionPre.y, conversionPre.x, 0.0f);
		A_Matrix[3] = make_float3(1.0f, 0.0f, 0.0f);
		A_Matrix[4] = make_float3(0.0f, 1.0f, 0.0f);
		A_Matrix[5] = make_float3(0.0f, 0.0f, 1.0f);


		b_Vector = make_float3(nearestPoint.x - conversionPre.x, nearestPoint.y - conversionPre.y, nearestPoint.z - conversionPre.z);
		//printf("b_Vector = (%.9f, %.9f, %.9f)\n", b_Vector.x, b_Vector.y, b_Vector.z);

		// ���ڽ��й�Լ����
		__shared__ float reduceMember[totalSharedSize][warpsNum]; // global������һ��__shared__���飬������齫���߳̿��������߳��ﹲ��
		unsigned int shift = 0;
		const unsigned int warpIndex = threadIdx.x >> 5;	// ��ǰ����һ���߳���(threadIdx.x / 2^5 = threadIdx.x / 32)
		const unsigned int laneIndex = threadIdx.x & 31;	// ��ǰ�߳����߳�������һ��λ��(31 = 11111��ֻ�ж��������5λ�������߳����е�λ��)

		for (int i = 0; i < 6; i++) { // ��index
			for (int j = i; j < 6; j++) { // ��index��ATA�����ǶԳƾ������ֻ��Ҫ�õ������Ǿ��󼴿�
				float data = dot(A_Matrix[i], A_Matrix[j]);	// ����AT * A����
				data = warp_scan(data);	// data = ATA�е�Ԫ�أ������ڲ�ͳһ���߳������߳�
				if (laneIndex == 31) { // ֻ����ĳ���߳����ĵ�һ���̲߳�����Ҫ�����data
					reduceMember[shift++][warpIndex] = data; // ��warpIndex���߳�����������ӵĽ������ִ�и�ֵ��������ִ��shift++
				}
				__syncthreads(); // �߳�ͬ���������߳�ִ�е��˴���������ȷ���߳̿��е������߳��ڼ���ִ��֮ǰ���Թ����ڴ��ȫ���ڴ�Ķ�д�������Ѿ����
			}
		}

		// ��Լ�õ�����(�reduceMemberǰ��6��) 
		for (int i = 0; i < 6; i++) {
			float data = dot(A_Matrix[i], b_Vector);	// AT*b
			data = warp_scan(data);
			if (laneIndex == 31) {
				reduceMember[shift++][warpIndex] = data;
			}
			__syncthreads(); // �߳�ͬ������ֹreduceMember����ͬ�̶߳�д��������ͻ
		}

		// ������洢��ȫ���ڴ��У�flattenBlock�ǽ�block��һά����ʽ���У��������ǰblock��Index
		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;
		// ��������������threadIdx.x < totalSharedSize���̣߳�һ��block����blockSize = 256���߳̿�
		for (int i = threadIdx.x; i < totalSharedSize; i += 32) { // ѭ����ִֻ��һ�Σ�д��if���ɣ�Ϊ��дfor?
			if (warpIndex == 0) { // ���߳���warpIndex = 0��
				// ����ǰ�߳�����8���������
				const float warpSum = reduceMember[i][0] + reduceMember[i][1] + reduceMember[i][2] + reduceMember[i][3]
					+ reduceMember[i][4] + reduceMember[i][5] + reduceMember[i][6] + reduceMember[i][7];
				reduceBuffer.ptr(i)[flattenBlock] = warpSum;
			}
		}
	}
}

__device__ __forceinline__ void SparseSurfelFusion::device::Point2PointICP::solverIterationFeatureMatchCamerasPosition(DeviceArrayView<float4> pointsPairs, const unsigned int pairsNum, PtrStep<float> MatrixBuffer) const
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Ax = b
	float3 A_Matrix[6] = { 0.0f };				// ����ϵ������A
	float3 b_Vector = { 0.0f };					// �����Ҳ�����

	if (idx < pairsNum) {			// �����ԣ�Array��previous [0,ValidMatchedPointsNum)   current [ValidMatchedPointsNum, 2 * ValidMatchedPointsNum)
		const float4 targetPoint = pointsPairs[idx];			// ��Ҫȥ��׼�ĵ�A
		const float4 sourcePoint = pointsPairs[2 * idx];		// Ŀ���
		//printf("idx = %hu   previous(%.5f, %.5f, %.5f, %.5f)   <-->   current(%.5f, %.5f, %.5f, %.5f)\n", idx, sourcePoint.x, sourcePoint.y, sourcePoint.z, sourcePoint.w, targetPoint.x, targetPoint.y, targetPoint.z, targetPoint.w);

		const float3 sourcePointPre = currentPositionMat34.rot * sourcePoint + currentPositionMat34.trans;	// ͨ������任����conversion�䵽��reference�����λ��
		A_Matrix[0] = make_float3(0.0f, -sourcePointPre.z, sourcePointPre.y);
		A_Matrix[1] = make_float3(sourcePointPre.z, 0.0f, -sourcePointPre.x);
		A_Matrix[2] = make_float3(-sourcePointPre.y, sourcePointPre.x, 0.0f);
		A_Matrix[3] = make_float3(1.0f, 0.0f, 0.0f);
		A_Matrix[4] = make_float3(0.0f, 1.0f, 0.0f);
		A_Matrix[5] = make_float3(0.0f, 0.0f, 1.0f);


		b_Vector = make_float3(targetPoint.x - sourcePointPre.x, targetPoint.y - sourcePointPre.y, targetPoint.z - sourcePointPre.z);
		//printf("b_Vector = (%.9f, %.9f, %.9f)\n", b_Vector.x, b_Vector.y, b_Vector.z);

		__shared__ float MatrixMember[totalSharedSize];
		unsigned int shift = 0;
		for (int i = 0; i < 6; i++) { // ��index
			for (int j = i; j < 6; j++) { // ��index��ATA�����ǶԳƾ������ֻ��Ҫ�õ������Ǿ��󼴿�
				float data = dot(A_Matrix[i], A_Matrix[j]);	// ����AT * A����
				MatrixMember[shift++] += data;
				__syncthreads(); // �߳�ͬ���������߳�ִ�е��˴���������ȷ���߳̿��е������߳��ڼ���ִ��֮ǰ���Թ����ڴ��ȫ���ڴ�Ķ�д�������Ѿ����
			}
		}

		for (int i = 0; i < 6; i++) {
			float data = dot(A_Matrix[i], b_Vector);	// AT*b
			MatrixMember[shift++] += data;
			__syncthreads(); // �߳�ͬ���������߳�ִ�е��˴���������ȷ���߳̿��е������߳��ڼ���ִ��֮ǰ���Թ����ڴ��ȫ���ڴ�Ķ�д�������Ѿ����
		}

		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;	// ��ǰ�ǵڼ���block

		for (int i = 0; i < totalSharedSize; i++) {
			// ��¼��ǰblock [A|b] �������Ԫ�ص�ֵ
			MatrixBuffer.ptr(i)[flattenBlock] = MatrixMember[i];
		}
	}
}


__device__ __forceinline__ void SparseSurfelFusion::device::RigidSolverDevice::rigidSolveIteration(PtrStep<float> reduceBuffer) const
{
	// Ax = b
	float A_Matrix[6] = { 0 };			// ����ϵ������A
	float b_Vector = 0.0f;				// �����Ҳ�����
	
	const unsigned int flatten_pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int x = flatten_pixel_idx % imageCols;
	const unsigned int y = flatten_pixel_idx / imageCols;

	/*
	* ѭ��չ������һЩ����£������ϣ���Զ������ͬʱ���д�������ʹ���߳�����warp������Ĳ����ԡ�
	* ����������£���ʹĳ���̴߳�������س�����ͼ��Χ��Ҳ����Ҫ���أ���Ϊ�����߳̿��Դ�����Ч��Χ�ڵ����ء�
	*/
	// ��ͼ��Χ֮�ڵ����أ�������Foreground���ͼ���ϵ��������迼��
	if (x < imageCols && y < imageRows) {
		// ��conversion�м���ͼ
		const float4 conversionVertex = tex2D<float4>(conversionMaps.vertexMap, x, y);
		const float4 conversionNormal = tex2D<float4>(conversionMaps.normalMap, x, y);
		// conversion��Ҫ�ȱ任����reference�����λ�ã���ֹ����ֲ�����
		// conversionVertexPre�Ǿ���Ԥ����λ�˱任���vertexλ��
		const float3 conversionVertexPre = currentWorldToCamera.rot * conversionVertex + currentWorldToCamera.trans;
		// conversionNormalPre�Ǿ���Ԥ����λ�˱任���Normal��ֵ
		const float3 conversionNormalPre = currentWorldToCamera.rot * conversionNormal;

		// ���������任�Ѿ����½�conversion�����Ŀռ�任��reference�ռ䣬������Ҫ��conversion��Щ����ͶӰ��reference�����ͼ�ϣ��ҵ��ص�λ�ÿ�ʼ��������
		// imageCoordinate����conversion�������ϵ�µĵ�ͶӰ����reference������ص�����
		const ushort2 imageCoordinate = {
			__float2uint_rn(((conversionVertexPre.x / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
			__float2uint_rn(((conversionVertexPre.y / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
		};

		// ͶӰ���ڷ�Χ�ڣ�������Foreground��
		if (imageCoordinate.x < imageCols && imageCoordinate.y < imageRows) {
			// ����ԭʼ��reference�����ͼ����conversionת��reference����reference����ص�������ȥ����
			// referenceͼ���������ϵ�µ�����
			const float4 referenceVertexMap = tex2D<float4>(referenceMaps.vertexMap, imageCoordinate.x, imageCoordinate.y);
			// referenceͼ���������ϵ�µ�ķ���
			const float4 referenceNormalMap = tex2D<float4>(referenceMaps.normalMap, imageCoordinate.x, imageCoordinate.y);

			// ������жϣ�
			// ��reference������ص�������Ϣ����Ϊ0
			// ��reference�����ͨ�����ͼ��õķ��ߣ���conversion����㷨��ת��reference�������ϵ�µķ�����ȣ��Ƕ�cos(theta)< CAMERA_POSE_RIGID_ANGLE_THRESHOLD  <=>  theta > arcCos(CAMERA_POSE_RIGID_ANGLE_THRESHOLD)
			// ��reference����㣬����conversion�����ת��reference�������ϵ�µĵ�ľ���ƽ��(��λ��m)������CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD
			if (is_zero_vertex(referenceVertexMap) || dotxyz(conversionNormalPre, referenceNormalMap) < CAMERA_POSE_RIGID_ANGLE_THRESHOLD || squared_distance(conversionVertexPre, referenceVertexMap) > CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD) {
				// �����̫Զ�����ܻ�����ֲ����Ž⣬PASS��Щ��Ⱥֵ
			}
			else {
				//printf("coor = (%hu, %hu)\n", imageCoordinate.x, imageCoordinate.y);
				// �Ҳ�����(Point-to-Plane ICP):https://zhuanlan.zhihu.com/p/385414929

				b_Vector = -1.0f * dotxyz(referenceNormalMap, make_float4(conversionVertexPre.x - referenceVertexMap.x, conversionVertexPre.y - referenceVertexMap.y, conversionVertexPre.z - referenceVertexMap.z, 0.0f));
				*(float3*)(A_Matrix + 0) = cross_xyz(conversionVertexPre, referenceNormalMap);
				*(float3*)(A_Matrix + 3) = make_float3(referenceNormalMap.x, referenceNormalMap.y, referenceNormalMap.z);
			}
			
		}


		// ���ڽ��й�Լ����
		__shared__ float reduceMember[totalSharedSize][warpsNum]; // global������һ��__shared__���飬������齫���߳̿��������߳��ﹲ��
		unsigned int shift = 0;
		const unsigned int warpIndex = threadIdx.x >> 5;	// ��ǰ����һ���߳���(threadIdx.x / 2^5 = threadIdx.x / 32)
		const unsigned int laneIndex = threadIdx.x & 31;	// ��ǰ�߳����߳�������һ��λ��(31 = 11111��ֻ�ж��������5λ�������߳����е�λ��)

		// ��Լ�õ�����(�reduceMemberǰ6 + 5 + 4 + 3 + 2 + 1 = 21��)
		for (int i = 0; i < 6; i++) { // ��index
			for (int j = i; j < 6; j++) { // ��index��ATA�����ǶԳƾ������ֻ��Ҫ�õ������Ǿ��󼴿�
				float data = (A_Matrix[i] * A_Matrix[j]);	// ����AT * A����
				data = warp_scan(data);	// data = ATA�е�Ԫ�أ������ڲ�ͳһ���߳������߳�
				if (laneIndex == 31) { // ֻ����ĳ���߳����ĵ�һ���̲߳�����Ҫ�����data
					reduceMember[shift++][warpIndex] = data; // ��warpIndex���߳�����������ӵĽ������ִ�и�ֵ��������ִ��shift++
				}
				__syncthreads(); // �߳�ͬ���������߳�ִ�е��˴���������ȷ���߳̿��е������߳��ڼ���ִ��֮ǰ���Թ����ڴ��ȫ���ڴ�Ķ�д�������Ѿ����
			}
		}

		// ��Լ�õ�����(�reduceMemberǰ��6��) 
		for (int i = 0; i < 6; i++) {
			float data = (A_Matrix[i] * b_Vector);	// AT*b
			data = warp_scan(data);
			if (laneIndex == 31) {
				reduceMember[shift++][warpIndex] = data;
			}
			__syncthreads(); // �߳�ͬ������ֹreduceMember����ͬ�̶߳�д��������ͻ
		}

		// ������洢��ȫ���ڴ��У�flattenBlock�ǽ�block��һά����ʽ���У��������ǰblock��Index
		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;
		// ��������������threadIdx.x < totalSharedSize���̣߳�һ��block����blockSize = 256���߳̿�
		for (int i = threadIdx.x; i < totalSharedSize; i += 32) {
			if (warpIndex == 0) { // ���߳���warpIndex = 0��
				// ����ǰ�߳�����8���������
				const float warpSum = reduceMember[i][0] + reduceMember[i][1] + reduceMember[i][2] + reduceMember[i][3]
					+ reduceMember[i][4] + reduceMember[i][5] + reduceMember[i][6] + reduceMember[i][7];
				reduceBuffer.ptr(i)[flattenBlock] = warpSum;
			}
		}
	}
}


__device__ __forceinline__ void SparseSurfelFusion::device::RigidSolverDevice::rigidSolveIterationUseOpticalFlows(PtrStep<float> reduceBuffer) const
{
	// Ax = b
	float A_Matrix[6] = { 0 };			// ����ϵ������A
	float b_Vector = 0.0f;				// �����Ҳ�����
	const unsigned int flatten_pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int x = flatten_pixel_idx % imageCols;
	const unsigned int y = flatten_pixel_idx / imageCols;
	//��0��ʼ����Ҫ-1.�����ܵ���6�Σ�ǰ5�Σ�0 1 2 3 4��������ͨ�ĵ��� ��6�Σ�5����ƥ������
	if (index<(max-1))
	{
		//ʹ����ͨ�ĵ���
		if (x < imageCols && y < imageRows) {
			// ��conversion�м���ͼ
			const float4 conversionVertex = tex2D<float4>(conversionMaps.vertexMap, x, y);
			const float4 conversionNormal = tex2D<float4>(conversionMaps.normalMap, x, y);
			// conversion��Ҫ�ȱ任����reference�����λ�ã���ֹ����ֲ�����
			// conversionVertexPre�Ǿ���Ԥ����λ�˱任���vertexλ��
			const float3 conversionVertexPre = currentWorldToCamera.rot * conversionVertex + currentWorldToCamera.trans;
			// conversionNormalPre�Ǿ���Ԥ����λ�˱任���Normal��ֵ
			const float3 conversionNormalPre = currentWorldToCamera.rot * conversionNormal;

			// ���������任�Ѿ����½�conversion�����Ŀռ�任��reference�ռ䣬������Ҫ��conversion��Щ����ͶӰ��reference�����ͼ�ϣ��ҵ��ص�λ�ÿ�ʼ��������
			// imageCoordinate����conversion�������ϵ�µĵ�ͶӰ����reference������ص�����
			const ushort2 imageCoordinate = {
				__float2uint_rn(((conversionVertexPre.x / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
				__float2uint_rn(((conversionVertexPre.y / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
			};

			// ͶӰ���ڷ�Χ�ڣ�������Foreground��
			if (imageCoordinate.x < imageCols && imageCoordinate.y < imageRows) {

				// ����ԭʼ��reference�����ͼ����conversionת��reference����reference����ص�������ȥ����
				// referenceͼ���������ϵ�µ�����
				const float4 referenceVertexMap = tex2D<float4>(referenceMaps.vertexMap, imageCoordinate.x, imageCoordinate.y);
				// referenceͼ���������ϵ�µ�ķ���
				const float4 referenceNormalMap = tex2D<float4>(referenceMaps.normalMap, imageCoordinate.x, imageCoordinate.y);

				// ������жϣ�
				// ��reference������ص�������Ϣ����Ϊ0
				// ��reference�����ͨ�����ͼ��õķ��ߣ���conversion����㷨��ת��reference�������ϵ�µķ�����ȣ��Ƕ�cos(theta)< CAMERA_POSE_RIGID_ANGLE_THRESHOLD  <=>  theta > arcCos(CAMERA_POSE_RIGID_ANGLE_THRESHOLD)
				// ��reference����㣬����conversion�����ת��reference�������ϵ�µĵ�ľ���ƽ��(��λ��m)������CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD
				if (is_zero_vertex(referenceVertexMap) || dotxyz(conversionNormalPre, referenceNormalMap) < 0.85/*CAMERA_POSE_RIGID_ANGLE_THRESHOLD*/ || squared_distance(conversionVertexPre, referenceVertexMap) >(0.01*0.01) /*CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD*/) {
					
				}
				else {
					b_Vector = -1.0f * dotxyz(referenceNormalMap, make_float4(conversionVertexPre.x - referenceVertexMap.x, conversionVertexPre.y - referenceVertexMap.y, conversionVertexPre.z - referenceVertexMap.z, 0.0f));
					*(float3*)(A_Matrix + 0) = cross_xyz(conversionVertexPre, referenceNormalMap);
					*(float3*)(A_Matrix + 3) = make_float3(referenceNormalMap.x, referenceNormalMap.y, referenceNormalMap.z);
				}
			}
	}
	else
	{
		//ʹ��ƥ���Խ������һ�ε��� ע��������벻��ȡ����
		if (flatten_pixel_idx<pixelpair.Size())
		{
			const auto pair = pixelpair[flatten_pixel_idx];
			// ��conversion�м���ͼ
			const float4 conversionVertex = tex2D<float4>(conversionMaps.vertexMap, pair.x, pair.y);
			const float4 conversionNormal = tex2D<float4>(conversionMaps.normalMap, pair.x, pair.y);
			// conversion��Ҫ�ȱ任����reference�����λ�ã���ֹ����ֲ�����
			// conversionVertexPre�Ǿ���Ԥ����λ�˱任���vertexλ��
			const float3 conversionVertexPre = currentWorldToCamera.rot * conversionVertex + currentWorldToCamera.trans;
			// conversionNormalPre�Ǿ���Ԥ����λ�˱任���Normal��ֵ
			const float3 conversionNormalPre = currentWorldToCamera.rot * conversionNormal;
			// referenceͼ���������ϵ�µ�����
			const float4 referenceVertexMap = tex2D<float4>(referenceMaps.vertexMap, pair.z, pair.w);
			// referenceͼ���������ϵ�µ�ķ���
			const float4 referenceNormalMap = tex2D<float4>(referenceMaps.normalMap, pair.z, pair.w);
			if (is_zero_vertex(referenceVertexMap) || dotxyz(conversionNormalPre, referenceNormalMap) < CAMERA_POSE_RIGID_ANGLE_THRESHOLD || squared_distance(conversionVertexPre, referenceVertexMap) > CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD) {
				//pass
			}
			else {
				b_Vector = -1.0f * dotxyz(referenceNormalMap, make_float4(conversionVertexPre.x - referenceVertexMap.x, conversionVertexPre.y - referenceVertexMap.y, conversionVertexPre.z - referenceVertexMap.z, 0.0f));
				*(float3*)(A_Matrix + 0) = cross_xyz(conversionVertexPre, referenceNormalMap);
				*(float3*)(A_Matrix + 3) = make_float3(referenceNormalMap.x, referenceNormalMap.y, referenceNormalMap.z);
			}
		}
	}



		// ���ڽ��й�Լ����
		__shared__ float reduceMember[totalSharedSize][warpsNum]; // global������һ��__shared__���飬������齫���߳̿��������߳��ﹲ��
		unsigned int shift = 0;
		const unsigned int warpIndex = threadIdx.x >> 5;	// ��ǰ����һ���߳���(threadIdx.x / 2^5 = threadIdx.x / 32)
		const unsigned int laneIndex = threadIdx.x & 31;	// ��ǰ�߳����߳�������һ��λ��(31 = 11111��ֻ�ж��������5λ�������߳����е�λ��)

		// ��Լ�õ�����(�reduceMemberǰ6 + 5 + 4 + 3 + 2 + 1 = 21��)
		for (int i = 0; i < 6; i++) { // ��index
			for (int j = i; j < 6; j++) { // ��index��ATA�����ǶԳƾ������ֻ��Ҫ�õ������Ǿ��󼴿�
				float data = (A_Matrix[i] * A_Matrix[j]);	// ����AT * A����
				data = warp_scan(data);	// data = ATA�е�Ԫ�أ������ڲ�ͳһ���߳������߳�
				if (laneIndex == 31) { // ֻ����ĳ���߳����ĵ�һ���̲߳�����Ҫ�����data
					reduceMember[shift++][warpIndex] = data; // ��warpIndex���߳�����������ӵĽ������ִ�и�ֵ��������ִ��shift++
					//if (data > 1e2) printf("#############################################Abnormal Value#############################################");
				}
				__syncthreads(); // �߳�ͬ���������߳�ִ�е��˴���������ȷ���߳̿��е������߳��ڼ���ִ��֮ǰ���Թ����ڴ��ȫ���ڴ�Ķ�д�������Ѿ����
			}
		}

		// ��Լ�õ�����(�reduceMemberǰ��6��) 
		for (int i = 0; i < 6; i++) {
			float data = (A_Matrix[i] * b_Vector);	// AT*b
			data = warp_scan(data);
			if (laneIndex == 31) {
				reduceMember[shift++][warpIndex] = data;
				//if (data > 1e2) printf("#############################################Abnormal Value#############################################");
			}
			__syncthreads(); // �߳�ͬ������ֹreduceMember����ͬ�̶߳�д��������ͻ
		}

		// ������洢��ȫ���ڴ��У�flattenBlock�ǽ�block��һά����ʽ���У��������ǰblock��Index
		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;
		// ��������������threadIdx.x < totalSharedSize���̣߳�һ��block����blockSize = 256���߳̿�
		for (int i = threadIdx.x; i < totalSharedSize; i += 32) {
			if (warpIndex == 0) { // ���߳���warpIndex = 0��
				// ����ǰ�߳�����8���������
				const float warpSum = reduceMember[i][0] + reduceMember[i][1] + reduceMember[i][2] + reduceMember[i][3]
					+ reduceMember[i][4] + reduceMember[i][5] + reduceMember[i][6] + reduceMember[i][7];
				reduceBuffer.ptr(i)[flattenBlock] = warpSum;
			}
		}
	}
}


__global__ void SparseSurfelFusion::device::rigidSolveIterationCamerasPositionKernel(const Point2PointICP solver, PtrStep<float> reduceBuffer)
{
	solver.solverIterationCamerasPosition(solver.referenceSparseVertices, solver.conversionSparseVertices, solver.knnIndex, solver.conversionSparseVertices.Size(), reduceBuffer);
}

__global__ void SparseSurfelFusion::device::rigidSolveIterationFeatureMatchCamerasPositionKernel(const Point2PointICP solver, const unsigned int pairsNum, PtrStep<float> MatrixBuffer)
{
	solver.solverIterationFeatureMatchCamerasPosition(solver.matchedPoints, pairsNum, MatrixBuffer);
}

__global__ void SparseSurfelFusion::device::columnReduceKernel(const PtrStepSize<const float> globalBuffer, float* target)
{
	const unsigned int idx = threadIdx.x;								// xά�ȵ�32���߳�
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;		// y���������ڴ�Ĵ�С��27������
	float sum = 0.0f;
	// globalBuffer�д���27��globalBuffer.cols��block�����ݣ�����globalBuffer.cols�ܶ࣬��32��Ϊһ������Լ���
	for (int i = 0; i < globalBuffer.cols; i += 32) {
		sum += globalBuffer.ptr(y)[i];
	}

	// �߳�����Լ
	sum = warp_scan(sum);
	if (idx == 31) { // ֻ�������������ս��
		target[y] = sum;
	}
}

__global__ void SparseSurfelFusion::device::columnReduceFeatureMatchKernel(const PtrStepSize<const float> globalBuffer, const unsigned int blockNums, float* target)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;	// 27�������еĵ�idx��Ԫ��
	float sum = 0.0f;
	for (int i = 0; i < blockNums; i++) {	// �����idx��Ԫ�أ�ÿ��Ԫ�ض�ӦblockNums��������ÿ������block�ĺ���Ϊ��idx��Ԫ�ص�ֵ
		sum += globalBuffer.ptr(idx)[i];
	}
	target[idx] = sum;
}

__global__ void SparseSurfelFusion::device::RigidSolveInterationKernel(const RigidSolverDevice solver, PtrStep<float> reduceBuffer)
{
	solver.rigidSolveIteration(reduceBuffer);

}

__global__ void SparseSurfelFusion::device::RigidSolveInterationKernelUseOpticalFlows(const RigidSolverDevice solver, PtrStep<float> reduceBuffer)
{
	solver.rigidSolveIterationUseOpticalFlows(reduceBuffer);
}

__global__ void SparseSurfelFusion::device::getVertexFromDepthSurfelKernel(DeviceArrayView<DepthSurfel> denseDepthSurfel, unsigned int denseSurfelNum, float4* denseVertex)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < denseSurfelNum) {
		denseVertex[idx] = denseDepthSurfel[idx].VertexAndConfidence;
	}
}

__global__ void SparseSurfelFusion::device::addDensePointsToCanonicalFieldKernel(DeviceArrayHandle<DepthSurfel> preAlignedSurfel, DeviceArrayView<DepthSurfel> depthSurfel, mat34 relativePose, const unsigned int pointsNum, const unsigned int offset)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < pointsNum) {
		// ת��point��preAlignedSurfel��
		const float3 convertPoint = relativePose.rot * depthSurfel[idx].VertexAndConfidence + relativePose.trans;
		preAlignedSurfel[offset + idx].VertexAndConfidence = make_float4(convertPoint.x, convertPoint.y, convertPoint.z, depthSurfel[idx].VertexAndConfidence.w);
		const float3 convertNormal = relativePose.rot * depthSurfel[idx].NormalAndRadius;
		preAlignedSurfel[offset + idx].NormalAndRadius = make_float4(convertNormal.x, convertNormal.y, convertNormal.z, depthSurfel[idx].NormalAndRadius.w);
		preAlignedSurfel[offset + idx].pixelCoordinate = depthSurfel[idx].pixelCoordinate;
		preAlignedSurfel[offset + idx].ColorAndTime = depthSurfel[idx].ColorAndTime;
	}

}

void SparseSurfelFusion::RigidSolver::getVertexFromDepthSurfel(DeviceArrayView<DepthSurfel>& denseDepthSurfel, DeviceBufferArray<float4>& denseVertex, cudaStream_t stream)
{
	// ��¶��дָ�룬��DeviceArrayHandle��Ҫ����������[]��������__global__��ֱ�ӷ���Ԫ��
	DeviceArrayHandle<float4> data = denseVertex.ArrayHandle();
	size_t denseSurfelNum = denseDepthSurfel.Size();
	dim3 block(256);
	dim3 grid(divUp(denseSurfelNum, block.x));
	device::getVertexFromDepthSurfelKernel << <grid, block, 0, stream >> > (denseDepthSurfel, denseSurfelNum, data);

}

void SparseSurfelFusion::RigidSolver::addDensePointsToCanonicalField(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, unsigned int CameraID, cudaStream_t stream)
{
	unsigned int offset = 0;
	size_t cameraPointsNum = PreRigidSolver.denseSurfel[CameraID].Size();
	DeviceArrayHandle<DepthSurfel> mergePoints = preAlignedSurfel.ArrayHandle();		// ��¶ָ��
	DeviceArrayView<DepthSurfel> cameraPoints = PreRigidSolver.denseSurfel[CameraID];	// ������Ԫ
	mat34 convert = PreRigidSolver.CamerasRelativePose[CameraID];

	for (int i = 0; i < CameraID; i++) {
		if (i == 0) continue;
		offset += PreRigidSolver.denseSurfel[i - 1].Size();
	}
	dim3 block(256);
	dim3 grid(divUp(cameraPointsNum, block.x));
	device::addDensePointsToCanonicalFieldKernel << <grid, block, 0, stream >> > (mergePoints, cameraPoints, convert, cameraPointsNum, offset);

}

void SparseSurfelFusion::RigidSolver::mergeDenseSurfelToCanonicalField(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, DeviceArrayView<DepthSurfel>& currentValidDepthSurfel, const unsigned int CameraID, const unsigned int offset, cudaStream_t stream)
{
	DeviceArrayHandle<DepthSurfel> mergePoints = preAlignedSurfel.ArrayHandle();		// ��¶ָ��
	DeviceArrayView<DepthSurfel> cameraPoints = currentValidDepthSurfel;				// ������Ԫ
	size_t cameraPointsNum = currentValidDepthSurfel.Size();
	mat34 convert = PreRigidSolver.CamerasRelativePose[CameraID];

	dim3 block(256);
	dim3 grid(divUp(cameraPointsNum, block.x));
	device::addDensePointsToCanonicalFieldKernel << <grid, block, 0, stream >> > (mergePoints, cameraPoints, convert, cameraPointsNum, offset);
}



void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIteration(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream)
{
	device::Point2PointICP GPUSolver;		// ����GPU�����
	GPUSolver.imageCols = imageCols;		// ͼ�񳤿���GPUSolver
	GPUSolver.imageRows = imageRows;		// ͼ�񳤿���GPUSolver

	// ��òο�Model
	GPUSolver.referenceIntrinsic = PreRigidSolver.clipedIntrinsic[referenceIndex];
	// ��ñ任��Model
	GPUSolver.conversionIntrinsic = PreRigidSolver.clipedIntrinsic[conversionIndex];

	// ��������ϵ��ʵ����reference������������ϵ
	GPUSolver.currentPositionMat34 = PreRigidSolver.CamerasRelativePose[conversionIndex];
	// ��Ҫ��conversion�������ϵ�еĵ�ӳ�䵽reference������ص��У��ٸ���reference���ͼ���ص��Ĳ��֣������ص����֣��õ����Խ�

	// �����Ҫ����ϡ���
	GPUSolver.referenceSparseVertices = PreRigidSolver.sparseVertices[referenceIndex];
	GPUSolver.conversionSparseVertices = PreRigidSolver.sparseVertices[conversionIndex];

	GPUSolver.knnIndex = PreRigidSolver.VertexKNN[conversionIndex].ArrayView();

	dim3 block(device::RigidSolverDevice::blockSize);
	dim3 grid(divUp(imageCols * imageRows, block.x));
	device::rigidSolveIterationCamerasPositionKernel << <grid, block, 0, stream >> > (GPUSolver, reduceBuffer);
	dim3 reduceBufferGrid(1, 1);
	dim3 reduceBufferBlock(32, device::RigidSolverDevice::totalSharedSize);
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceBuffer, reduceMatrixVector.DevicePtr());
	reduceMatrixVector.SynchronizeToHost(stream, false); // �������ͬ��������Ҫʹ�����������Host��ʹ��Eigen����㷨

}

void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIterationFeatureMatch(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream)
{
	device::Point2PointICP GPUSolver;	// ����GPU�����

	GPUSolver.currentPositionMat34 = PreRigidSolver.CamerasRelativePose[conversionIndex];

	GPUSolver.matchedPoints = PreRigidSolver.matchedPoints[conversionIndex];

	const unsigned int pairsNum = PreRigidSolver.matchedPoints[conversionIndex].Size() / 2;

	dim3 block(device::RigidSolverDevice::noReduceBlockSize);	// һ����Ч�����㲻�ᳬ��500��
	dim3 grid(divUp(pairsNum, block.x));
	device::rigidSolveIterationFeatureMatchCamerasPositionKernel << <grid, block, 0, stream >> > (GPUSolver, pairsNum, MatrixBuffer);

	const unsigned int blockNum = divUp(pairsNum, block.x);		// block������

	dim3 SumBlock(device::RigidSolverDevice::totalSharedSize);
	dim3 SumGrid(1);
	device::columnReduceFeatureMatchKernel << <SumGrid, SumBlock, 0, stream >> > (MatrixBuffer, blockNum, finalMatrixVector.DevicePtr());

	finalMatrixVector.SynchronizeToHost(stream, false); // �������ͬ��������Ҫʹ�����������Host��ʹ��Eigen����㷨
}



void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIterationUseOpticalFlows(
	const unsigned int CameraID, 
	unsigned index, 
	unsigned max,
	cudaStream_t stream)
{
	device::RigidSolverDevice solver;
	solver.intrinsic = PreRigidSolver.clipedIntrinsic[CameraID];
	solver.currentWorldToCamera = AccumulatePrevious2Current[CameraID];
	//printf("ICP���� ��\n");
	//printf("        %.9f    %.9f    %.9f    %.9f\n", AccumulatePrevious2Current.rot.m00(), AccumulatePrevious2Current.rot.m01(), AccumulatePrevious2Current.rot.m02(), AccumulatePrevious2Current.trans.x);
	//printf("        %.9f    %.9f    %.9f    %.9f\n", AccumulatePrevious2Current.rot.m10(), AccumulatePrevious2Current.rot.m11(), AccumulatePrevious2Current.rot.m12(), AccumulatePrevious2Current.trans.y);
	//printf("        %.9f    %.9f    %.9f    %.9f\n", AccumulatePrevious2Current.rot.m20(), AccumulatePrevious2Current.rot.m21(), AccumulatePrevious2Current.rot.m22(), AccumulatePrevious2Current.trans.z);

	solver.imageRows = imageRows;
	solver.imageCols = imageCols;
	solver.index = index;
	solver.max = max;

	solver.referenceMaps.vertexMap = referenceMap[CameraID].vertexMap;
	solver.referenceMaps.normalMap = referenceMap[CameraID].normalMap;
	solver.referenceMaps.foregroundMask = referenceMap[CameraID].foreground;

	solver.conversionMaps.vertexMap = conversionMap[CameraID].vertexMap;
	solver.conversionMaps.normalMap = conversionMap[CameraID].normalMap;
	solver.conversionMaps.foregroundMask = conversionMap[CameraID].foreground;
	//solver.pixelpair = opticalpixelpair[CameraID];

	dim3 block(device::RigidSolverDevice::blockSize);
	dim3 grid(divUp(imageCols * imageRows, block.x));
	device::RigidSolveInterationKernelUseOpticalFlows << <grid, block, 0, stream >> > (solver, reduceAugmentedMatrix[CameraID]);

	dim3 reduceBufferGrid(1, 1);
	dim3 reduceBufferBlock(32, device::RigidSolverDevice::totalSharedSize);
	// ����Լ
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// ���ﲻ��ͬ������������Host�߳�
}

void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIteration(const unsigned int CameraID, cudaStream_t stream)
{
	device::RigidSolverDevice solver;
	solver.intrinsic = PreRigidSolver.clipedIntrinsic[CameraID];
	solver.currentWorldToCamera = AccumulatePrevious2Current[CameraID];

	solver.imageRows = imageRows;
	solver.imageCols = imageCols;

	solver.referenceMaps.vertexMap = referenceMap[CameraID].vertexMap;
	solver.referenceMaps.normalMap = referenceMap[CameraID].normalMap;
	solver.referenceMaps.foregroundMask = referenceMap[CameraID].foreground;

	solver.conversionMaps.vertexMap = conversionMap[CameraID].vertexMap;
	solver.conversionMaps.normalMap = conversionMap[CameraID].normalMap;
	solver.conversionMaps.foregroundMask = conversionMap[CameraID].foreground;

	dim3 block(device::RigidSolverDevice::blockSize);
	dim3 grid(divUp(imageCols * imageRows, block.x));
	device::RigidSolveInterationKernel << <grid, block, 0, stream >> > (solver, reduceAugmentedMatrix[CameraID]);

	dim3 reduceBufferGrid(1, 1);
	dim3 reduceBufferBlock(32, device::RigidSolverDevice::totalSharedSize);
	// ����Լ
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// ���ﲻ��ͬ������������Host�߳�
}

void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIteration(const unsigned int CameraID, const mat34* currWorld2Camera, cudaStream_t stream)
{
	device::RigidSolverDevice solver;
	solver.intrinsic = PreRigidSolver.clipedIntrinsic[CameraID];
	solver.currentWorldToCamera = currWorld2Camera[CameraID];

	solver.imageRows = imageRows;
	solver.imageCols = imageCols;

	solver.referenceMaps.vertexMap = referenceMap[CameraID].vertexMap;
	solver.referenceMaps.normalMap = referenceMap[CameraID].normalMap;
	solver.referenceMaps.foregroundMask = referenceMap[CameraID].foreground;

	solver.conversionMaps.vertexMap = conversionMap[CameraID].vertexMap;
	solver.conversionMaps.normalMap = conversionMap[CameraID].normalMap;
	solver.conversionMaps.foregroundMask = conversionMap[CameraID].foreground;

	dim3 block(device::RigidSolverDevice::blockSize);
	dim3 grid(divUp(imageCols * imageRows, block.x));
	device::RigidSolveInterationKernel << <grid, block, 0, stream >> > (solver, reduceAugmentedMatrix[CameraID]);

	dim3 reduceBufferGrid(1, 1);
	dim3 reduceBufferBlock(32, device::RigidSolverDevice::totalSharedSize);
	// ����Լ
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// ���ﲻ��ͬ������������Host�߳�
}

void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIterationLive2Observation(const unsigned int CameraID, cudaStream_t stream)
{
	device::RigidSolverDevice solver;
	solver.intrinsic = PreRigidSolver.clipedIntrinsic[CameraID];
	solver.currentWorldToCamera = World2Camera[CameraID];

	solver.imageRows = imageRows;
	solver.imageCols = imageCols;

	solver.referenceMaps.vertexMap = referenceMap[CameraID].vertexMap;
	solver.referenceMaps.normalMap = referenceMap[CameraID].normalMap;

	solver.conversionMaps.vertexMap = conversionMap[CameraID].vertexMap;
	solver.conversionMaps.normalMap = conversionMap[CameraID].normalMap;

	dim3 block(device::RigidSolverDevice::blockSize);
	dim3 grid(divUp(imageCols * imageRows, block.x));
	device::RigidSolveInterationKernel << <grid, block, 0, stream >> > (solver, reduceAugmentedMatrix[CameraID]);

	dim3 reduceBufferGrid(1, 1);
	dim3 reduceBufferBlock(32, device::RigidSolverDevice::totalSharedSize);
	// ����Լ
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// ���ﲻ��ͬ������������Host�߳�
}

__global__ void SparseSurfelFusion::device::checkPreviousAndCurrentTextureKernel(const unsigned int rows, const unsigned int cols, cudaTextureObject_t previousVertex, cudaTextureObject_t previousNormal, cudaTextureObject_t currentVertex, cudaTextureObject_t currentNormal)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;

	float4 previousVertexCheck = tex2D<float4>(previousVertex, x, y);
	float4 previousNormalCheck = tex2D<float4>(previousNormal, x, y);
	float4 currentVertexCheck = tex2D<float4>(currentVertex, x, y);
	float4 currentNormalCheck = tex2D<float4>(currentNormal, x, y);
	printf("i = (%hu,%hu) pVertex = (%.5f,%.5f,%.5f,%.5f) pNormal = (%.5f,%.5f,%.5f,%.5f) cVertex= (%.5f,%.5f,%.5f,%.5f) cNormal = (%.5f,%.5f,%.5f,%.5f)\n", x, y,
		previousVertexCheck.x, previousVertexCheck.y, previousVertexCheck.z, previousVertexCheck.w,
		previousNormalCheck.x, previousNormalCheck.y, previousNormalCheck.z, previousNormalCheck.w,
		currentVertexCheck.x, currentVertexCheck.y, currentVertexCheck.z, currentVertexCheck.w,
		currentNormalCheck.x, currentNormalCheck.y, currentNormalCheck.z, currentNormalCheck.w);
}

void SparseSurfelFusion::RigidSolver::checkPreviousAndCurrentTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t previousVertex, cudaTextureObject_t previousNormal, cudaTextureObject_t currentVertex, cudaTextureObject_t currentNormal, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	device::checkPreviousAndCurrentTextureKernel << <grid, block, 0, stream >> > (rows, cols, previousVertex, previousNormal, currentVertex, currentNormal);
}
















