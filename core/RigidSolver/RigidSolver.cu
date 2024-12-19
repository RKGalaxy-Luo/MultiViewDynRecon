/*****************************************************************//**
 * \file   RigidSolver.cu
 * \brief  ICP刚性配准
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
	float3 A_Matrix[6] = { 0.0f };				// 方程系数矩阵A
	float3 b_Vector = { 0.0f };					// 方程右侧向量

	if (idx < pointsNum) {
		const float4 point = conversionVertices[idx];						// conversion相机中的点A
		const float4 nearestPoint = referenceVertices[knnIndex[idx].x];		// reference中距离点A最近的点

		const float3 conversionPre = currentPositionMat34.rot * point + currentPositionMat34.trans;	// 通过坐标变换，将conversion变到与reference相近的位置

		A_Matrix[0] = make_float3(0.0f, -conversionPre.z, conversionPre.y);
		A_Matrix[1] = make_float3(conversionPre.z, 0.0f, -conversionPre.x);
		A_Matrix[2] = make_float3(-conversionPre.y, conversionPre.x, 0.0f);
		A_Matrix[3] = make_float3(1.0f, 0.0f, 0.0f);
		A_Matrix[4] = make_float3(0.0f, 1.0f, 0.0f);
		A_Matrix[5] = make_float3(0.0f, 0.0f, 1.0f);


		b_Vector = make_float3(nearestPoint.x - conversionPre.x, nearestPoint.y - conversionPre.y, nearestPoint.z - conversionPre.z);
		//printf("b_Vector = (%.9f, %.9f, %.9f)\n", b_Vector.x, b_Vector.y, b_Vector.z);

		// 现在进行归约操作
		__shared__ float reduceMember[totalSharedSize][warpsNum]; // global中声明一个__shared__数组，这个数组将在线程块中所有线程里共享
		unsigned int shift = 0;
		const unsigned int warpIndex = threadIdx.x >> 5;	// 当前是哪一个线程束(threadIdx.x / 2^5 = threadIdx.x / 32)
		const unsigned int laneIndex = threadIdx.x & 31;	// 当前线程在线程束的哪一个位置(31 = 11111，只有二进制最后5位决定在线程束中的位置)

		for (int i = 0; i < 6; i++) { // 行index
			for (int j = i; j < 6; j++) { // 列index，ATA矩阵是对称矩阵，因此只需要得到上三角矩阵即可
				float data = dot(A_Matrix[i], A_Matrix[j]);	// 进行AT * A操作
				data = warp_scan(data);	// data = ATA中的元素，函数内部统一了线程束中线程
				if (laneIndex == 31) { // 只有是某个线程束的第一个线程才算需要存入的data
					reduceMember[shift++][warpIndex] = data; // 第warpIndex个线程束中数据相加的结果，先执行赋值操作，再执行shift++
				}
				__syncthreads(); // 线程同步，所有线程执行到此处被阻塞，确保线程块中的所有线程在继续执行之前，对共享内存和全局内存的读写操作都已经完成
			}
		}

		// 归约得到向量(填补reduceMember前后6行) 
		for (int i = 0; i < 6; i++) {
			float data = dot(A_Matrix[i], b_Vector);	// AT*b
			data = warp_scan(data);
			if (laneIndex == 31) {
				reduceMember[shift++][warpIndex] = data;
			}
			__syncthreads(); // 线程同步：防止reduceMember被不同线程读写，发生冲突
		}

		// 将结果存储到全局内存中，flattenBlock是将block以一维的形式排列，计算出当前block的Index
		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;
		// 若是碰巧遇到了threadIdx.x < totalSharedSize的线程，一个block中有blockSize = 256个线程块
		for (int i = threadIdx.x; i < totalSharedSize; i += 32) { // 循环体只执行一次：写个if即可，为何写for?
			if (warpIndex == 0) { // 当线程束warpIndex = 0，
				// 将当前线程束中8个数据相加
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
	float3 A_Matrix[6] = { 0.0f };				// 方程系数矩阵A
	float3 b_Vector = { 0.0f };					// 方程右侧向量

	if (idx < pairsNum) {			// 传入点对，Array中previous [0,ValidMatchedPointsNum)   current [ValidMatchedPointsNum, 2 * ValidMatchedPointsNum)
		const float4 targetPoint = pointsPairs[idx];			// 需要去对准的点A
		const float4 sourcePoint = pointsPairs[2 * idx];		// 目标点
		//printf("idx = %hu   previous(%.5f, %.5f, %.5f, %.5f)   <-->   current(%.5f, %.5f, %.5f, %.5f)\n", idx, sourcePoint.x, sourcePoint.y, sourcePoint.z, sourcePoint.w, targetPoint.x, targetPoint.y, targetPoint.z, targetPoint.w);

		const float3 sourcePointPre = currentPositionMat34.rot * sourcePoint + currentPositionMat34.trans;	// 通过坐标变换，将conversion变到与reference相近的位置
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
		for (int i = 0; i < 6; i++) { // 行index
			for (int j = i; j < 6; j++) { // 列index，ATA矩阵是对称矩阵，因此只需要得到上三角矩阵即可
				float data = dot(A_Matrix[i], A_Matrix[j]);	// 进行AT * A操作
				MatrixMember[shift++] += data;
				__syncthreads(); // 线程同步，所有线程执行到此处被阻塞，确保线程块中的所有线程在继续执行之前，对共享内存和全局内存的读写操作都已经完成
			}
		}

		for (int i = 0; i < 6; i++) {
			float data = dot(A_Matrix[i], b_Vector);	// AT*b
			MatrixMember[shift++] += data;
			__syncthreads(); // 线程同步，所有线程执行到此处被阻塞，确保线程块中的所有线程在继续执行之前，对共享内存和全局内存的读写操作都已经完成
		}

		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;	// 当前是第几个block

		for (int i = 0; i < totalSharedSize; i++) {
			// 记录当前block [A|b] 增广矩阵元素的值
			MatrixBuffer.ptr(i)[flattenBlock] = MatrixMember[i];
		}
	}
}


__device__ __forceinline__ void SparseSurfelFusion::device::RigidSolverDevice::rigidSolveIteration(PtrStep<float> reduceBuffer) const
{
	// Ax = b
	float A_Matrix[6] = { 0 };			// 方程系数矩阵A
	float b_Vector = 0.0f;				// 方程右侧向量
	
	const unsigned int flatten_pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int x = flatten_pixel_idx % imageCols;
	const unsigned int y = flatten_pixel_idx / imageCols;

	/*
	* 循环展开：在一些情况下，你可能希望对多个像素同时进行处理，例如使用线程束（warp）级别的并行性。
	* 在这种情况下，即使某个线程处理的像素超出了图像范围，也不需要返回，因为其他线程可以处理有效范围内的像素。
	*/
	// 在图像范围之内的像素，并且在Foreground里，非图像上的像素无需考虑
	if (x < imageCols && y < imageRows) {
		// 从conversion中加载图
		const float4 conversionVertex = tex2D<float4>(conversionMaps.vertexMap, x, y);
		const float4 conversionNormal = tex2D<float4>(conversionMaps.normalMap, x, y);
		// conversion需要先变换到与reference相近的位置，防止陷入局部最优
		// conversionVertexPre是经过预处理位姿变换后的vertex位置
		const float3 conversionVertexPre = currentWorldToCamera.rot * conversionVertex + currentWorldToCamera.trans;
		// conversionNormalPre是经过预处理位姿变换后的Normal的值
		const float3 conversionNormalPre = currentWorldToCamera.rot * conversionNormal;

		// 经过上述变换已经大致将conversion所处的空间变换到reference空间，现在需要将conversion这些个点投影到reference的深度图上，找到重叠位置开始迭代即可
		// imageCoordinate点是conversion相机坐标系下的点投影在在reference相机像素的坐标
		const ushort2 imageCoordinate = {
			__float2uint_rn(((conversionVertexPre.x / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
			__float2uint_rn(((conversionVertexPre.y / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
		};

		// 投影点在范围内，并且在Foreground里
		if (imageCoordinate.x < imageCols && imageCoordinate.y < imageRows) {
			// 加载原始的reference的深度图，用conversion转到reference中与reference深度重叠的像素去迭代
			// reference图的相机坐标系下点坐标
			const float4 referenceVertexMap = tex2D<float4>(referenceMaps.vertexMap, imageCoordinate.x, imageCoordinate.y);
			// reference图的相机坐标系下点的法线
			const float4 referenceNormalMap = tex2D<float4>(referenceMaps.normalMap, imageCoordinate.x, imageCoordinate.y);

			// 相关性判断：
			// ①reference这个像素点的深度信息不能为0
			// ②reference这个点通过深度图获得的法线，与conversion相机点法线转到reference相机坐标系下的法线相比，角度cos(theta)< CAMERA_POSE_RIGID_ANGLE_THRESHOLD  <=>  theta > arcCos(CAMERA_POSE_RIGID_ANGLE_THRESHOLD)
			// ③reference这个点，与与conversion相机点转到reference相机坐标系下的点的距离平方(单位：m)，大于CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD
			if (is_zero_vertex(referenceVertexMap) || dotxyz(conversionNormalPre, referenceNormalMap) < CAMERA_POSE_RIGID_ANGLE_THRESHOLD || squared_distance(conversionVertexPre, referenceVertexMap) > CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD) {
				// 点距离太远，可能会陷入局部最优解，PASS这些离群值
			}
			else {
				//printf("coor = (%hu, %hu)\n", imageCoordinate.x, imageCoordinate.y);
				// 右侧向量(Point-to-Plane ICP):https://zhuanlan.zhihu.com/p/385414929

				b_Vector = -1.0f * dotxyz(referenceNormalMap, make_float4(conversionVertexPre.x - referenceVertexMap.x, conversionVertexPre.y - referenceVertexMap.y, conversionVertexPre.z - referenceVertexMap.z, 0.0f));
				*(float3*)(A_Matrix + 0) = cross_xyz(conversionVertexPre, referenceNormalMap);
				*(float3*)(A_Matrix + 3) = make_float3(referenceNormalMap.x, referenceNormalMap.y, referenceNormalMap.z);
			}
			
		}


		// 现在进行归约操作
		__shared__ float reduceMember[totalSharedSize][warpsNum]; // global中声明一个__shared__数组，这个数组将在线程块中所有线程里共享
		unsigned int shift = 0;
		const unsigned int warpIndex = threadIdx.x >> 5;	// 当前是哪一个线程束(threadIdx.x / 2^5 = threadIdx.x / 32)
		const unsigned int laneIndex = threadIdx.x & 31;	// 当前线程在线程束的哪一个位置(31 = 11111，只有二进制最后5位决定在线程束中的位置)

		// 归约得到矩阵(填补reduceMember前6 + 5 + 4 + 3 + 2 + 1 = 21行)
		for (int i = 0; i < 6; i++) { // 行index
			for (int j = i; j < 6; j++) { // 列index，ATA矩阵是对称矩阵，因此只需要得到上三角矩阵即可
				float data = (A_Matrix[i] * A_Matrix[j]);	// 进行AT * A操作
				data = warp_scan(data);	// data = ATA中的元素，函数内部统一了线程束中线程
				if (laneIndex == 31) { // 只有是某个线程束的第一个线程才算需要存入的data
					reduceMember[shift++][warpIndex] = data; // 第warpIndex个线程束中数据相加的结果，先执行赋值操作，再执行shift++
				}
				__syncthreads(); // 线程同步，所有线程执行到此处被阻塞，确保线程块中的所有线程在继续执行之前，对共享内存和全局内存的读写操作都已经完成
			}
		}

		// 归约得到向量(填补reduceMember前后6行) 
		for (int i = 0; i < 6; i++) {
			float data = (A_Matrix[i] * b_Vector);	// AT*b
			data = warp_scan(data);
			if (laneIndex == 31) {
				reduceMember[shift++][warpIndex] = data;
			}
			__syncthreads(); // 线程同步：防止reduceMember被不同线程读写，发生冲突
		}

		// 将结果存储到全局内存中，flattenBlock是将block以一维的形式排列，计算出当前block的Index
		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;
		// 若是碰巧遇到了threadIdx.x < totalSharedSize的线程，一个block中有blockSize = 256个线程块
		for (int i = threadIdx.x; i < totalSharedSize; i += 32) {
			if (warpIndex == 0) { // 当线程束warpIndex = 0，
				// 将当前线程束中8个数据相加
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
	float A_Matrix[6] = { 0 };			// 方程系数矩阵A
	float b_Vector = 0.0f;				// 方程右侧向量
	const unsigned int flatten_pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int x = flatten_pixel_idx % imageCols;
	const unsigned int y = flatten_pixel_idx / imageCols;
	//从0开始，需要-1.比如总迭代6次，前5次（0 1 2 3 4）都用普通的迭代 第6次（5）用匹配点迭代
	if (index<(max-1))
	{
		//使用普通的迭代
		if (x < imageCols && y < imageRows) {
			// 从conversion中加载图
			const float4 conversionVertex = tex2D<float4>(conversionMaps.vertexMap, x, y);
			const float4 conversionNormal = tex2D<float4>(conversionMaps.normalMap, x, y);
			// conversion需要先变换到与reference相近的位置，防止陷入局部最优
			// conversionVertexPre是经过预处理位姿变换后的vertex位置
			const float3 conversionVertexPre = currentWorldToCamera.rot * conversionVertex + currentWorldToCamera.trans;
			// conversionNormalPre是经过预处理位姿变换后的Normal的值
			const float3 conversionNormalPre = currentWorldToCamera.rot * conversionNormal;

			// 经过上述变换已经大致将conversion所处的空间变换到reference空间，现在需要将conversion这些个点投影到reference的深度图上，找到重叠位置开始迭代即可
			// imageCoordinate点是conversion相机坐标系下的点投影在在reference相机像素的坐标
			const ushort2 imageCoordinate = {
				__float2uint_rn(((conversionVertexPre.x / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
				__float2uint_rn(((conversionVertexPre.y / (conversionVertexPre.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
			};

			// 投影点在范围内，并且在Foreground里
			if (imageCoordinate.x < imageCols && imageCoordinate.y < imageRows) {

				// 加载原始的reference的深度图，用conversion转到reference中与reference深度重叠的像素去迭代
				// reference图的相机坐标系下点坐标
				const float4 referenceVertexMap = tex2D<float4>(referenceMaps.vertexMap, imageCoordinate.x, imageCoordinate.y);
				// reference图的相机坐标系下点的法线
				const float4 referenceNormalMap = tex2D<float4>(referenceMaps.normalMap, imageCoordinate.x, imageCoordinate.y);

				// 相关性判断：
				// ①reference这个像素点的深度信息不能为0
				// ②reference这个点通过深度图获得的法线，与conversion相机点法线转到reference相机坐标系下的法线相比，角度cos(theta)< CAMERA_POSE_RIGID_ANGLE_THRESHOLD  <=>  theta > arcCos(CAMERA_POSE_RIGID_ANGLE_THRESHOLD)
				// ③reference这个点，与与conversion相机点转到reference相机坐标系下的点的距离平方(单位：m)，大于CAMERA_POSE_RIGID_DISTANCE_SQUARED_THRESHOLD
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
		//使用匹配点对进行最后一次迭代 注意这里必须不能取错了
		if (flatten_pixel_idx<pixelpair.Size())
		{
			const auto pair = pixelpair[flatten_pixel_idx];
			// 从conversion中加载图
			const float4 conversionVertex = tex2D<float4>(conversionMaps.vertexMap, pair.x, pair.y);
			const float4 conversionNormal = tex2D<float4>(conversionMaps.normalMap, pair.x, pair.y);
			// conversion需要先变换到与reference相近的位置，防止陷入局部最优
			// conversionVertexPre是经过预处理位姿变换后的vertex位置
			const float3 conversionVertexPre = currentWorldToCamera.rot * conversionVertex + currentWorldToCamera.trans;
			// conversionNormalPre是经过预处理位姿变换后的Normal的值
			const float3 conversionNormalPre = currentWorldToCamera.rot * conversionNormal;
			// reference图的相机坐标系下点坐标
			const float4 referenceVertexMap = tex2D<float4>(referenceMaps.vertexMap, pair.z, pair.w);
			// reference图的相机坐标系下点的法线
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



		// 现在进行归约操作
		__shared__ float reduceMember[totalSharedSize][warpsNum]; // global中声明一个__shared__数组，这个数组将在线程块中所有线程里共享
		unsigned int shift = 0;
		const unsigned int warpIndex = threadIdx.x >> 5;	// 当前是哪一个线程束(threadIdx.x / 2^5 = threadIdx.x / 32)
		const unsigned int laneIndex = threadIdx.x & 31;	// 当前线程在线程束的哪一个位置(31 = 11111，只有二进制最后5位决定在线程束中的位置)

		// 归约得到矩阵(填补reduceMember前6 + 5 + 4 + 3 + 2 + 1 = 21行)
		for (int i = 0; i < 6; i++) { // 行index
			for (int j = i; j < 6; j++) { // 列index，ATA矩阵是对称矩阵，因此只需要得到上三角矩阵即可
				float data = (A_Matrix[i] * A_Matrix[j]);	// 进行AT * A操作
				data = warp_scan(data);	// data = ATA中的元素，函数内部统一了线程束中线程
				if (laneIndex == 31) { // 只有是某个线程束的第一个线程才算需要存入的data
					reduceMember[shift++][warpIndex] = data; // 第warpIndex个线程束中数据相加的结果，先执行赋值操作，再执行shift++
					//if (data > 1e2) printf("#############################################Abnormal Value#############################################");
				}
				__syncthreads(); // 线程同步，所有线程执行到此处被阻塞，确保线程块中的所有线程在继续执行之前，对共享内存和全局内存的读写操作都已经完成
			}
		}

		// 归约得到向量(填补reduceMember前后6行) 
		for (int i = 0; i < 6; i++) {
			float data = (A_Matrix[i] * b_Vector);	// AT*b
			data = warp_scan(data);
			if (laneIndex == 31) {
				reduceMember[shift++][warpIndex] = data;
				//if (data > 1e2) printf("#############################################Abnormal Value#############################################");
			}
			__syncthreads(); // 线程同步：防止reduceMember被不同线程读写，发生冲突
		}

		// 将结果存储到全局内存中，flattenBlock是将block以一维的形式排列，计算出当前block的Index
		const unsigned int flattenBlock = blockIdx.x + gridDim.x * blockIdx.y;
		// 若是碰巧遇到了threadIdx.x < totalSharedSize的线程，一个block中有blockSize = 256个线程块
		for (int i = threadIdx.x; i < totalSharedSize; i += 32) {
			if (warpIndex == 0) { // 当线程束warpIndex = 0，
				// 将当前线程束中8个数据相加
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
	const unsigned int idx = threadIdx.x;								// x维度的32个线程
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;		// y方向是总内存的大小，27个数据
	float sum = 0.0f;
	// globalBuffer中存着27行globalBuffer.cols个block的数据，可能globalBuffer.cols很多，以32个为一束，归约相加
	for (int i = 0; i < globalBuffer.cols; i += 32) {
		sum += globalBuffer.ptr(y)[i];
	}

	// 线程束归约
	sum = warp_scan(sum);
	if (idx == 31) { // 只有首数据是最终结果
		target[y] = sum;
	}
}

__global__ void SparseSurfelFusion::device::columnReduceFeatureMatchKernel(const PtrStepSize<const float> globalBuffer, const unsigned int blockNums, float* target)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;	// 27个数据中的第idx个元素
	float sum = 0.0f;
	for (int i = 0; i < blockNums; i++) {	// 计算第idx个元素，每个元素对应blockNums个数，求每个所有block的和作为第idx个元素的值
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
		// 转换point到preAlignedSurfel中
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
	// 暴露读写指针，用DeviceArrayHandle主要是其重载了[]，可以在__global__中直接访问元素
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
	DeviceArrayHandle<DepthSurfel> mergePoints = preAlignedSurfel.ArrayHandle();		// 暴露指针
	DeviceArrayView<DepthSurfel> cameraPoints = PreRigidSolver.denseSurfel[CameraID];	// 稠密面元
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
	DeviceArrayHandle<DepthSurfel> mergePoints = preAlignedSurfel.ArrayHandle();		// 暴露指针
	DeviceArrayView<DepthSurfel> cameraPoints = currentValidDepthSurfel;				// 稠密面元
	size_t cameraPointsNum = currentValidDepthSurfel.Size();
	mat34 convert = PreRigidSolver.CamerasRelativePose[CameraID];

	dim3 block(256);
	dim3 grid(divUp(cameraPointsNum, block.x));
	device::addDensePointsToCanonicalFieldKernel << <grid, block, 0, stream >> > (mergePoints, cameraPoints, convert, cameraPointsNum, offset);
}



void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIteration(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream)
{
	device::Point2PointICP GPUSolver;		// 构造GPU求解器
	GPUSolver.imageCols = imageCols;		// 图像长宽传入GPUSolver
	GPUSolver.imageRows = imageRows;		// 图像长宽传入GPUSolver

	// 获得参考Model
	GPUSolver.referenceIntrinsic = PreRigidSolver.clipedIntrinsic[referenceIndex];
	// 获得变换的Model
	GPUSolver.conversionIntrinsic = PreRigidSolver.clipedIntrinsic[conversionIndex];

	// 世界坐标系其实就是reference相机的相机坐标系
	GPUSolver.currentPositionMat34 = PreRigidSolver.CamerasRelativePose[conversionIndex];
	// 需要将conversion相机坐标系中的点映射到reference相机像素点中，再根据reference深度图中重叠的部分，迭代重叠部分，得到刚性解

	// 获得需要求解的稀疏点
	GPUSolver.referenceSparseVertices = PreRigidSolver.sparseVertices[referenceIndex];
	GPUSolver.conversionSparseVertices = PreRigidSolver.sparseVertices[conversionIndex];

	GPUSolver.knnIndex = PreRigidSolver.VertexKNN[conversionIndex].ArrayView();

	dim3 block(device::RigidSolverDevice::blockSize);
	dim3 grid(divUp(imageCols * imageRows, block.x));
	device::rigidSolveIterationCamerasPositionKernel << <grid, block, 0, stream >> > (GPUSolver, reduceBuffer);
	dim3 reduceBufferGrid(1, 1);
	dim3 reduceBufferBlock(32, device::RigidSolverDevice::totalSharedSize);
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceBuffer, reduceMatrixVector.DevicePtr());
	reduceMatrixVector.SynchronizeToHost(stream, false); // 这里进行同步，后续要使用这个矩阵在Host中使用Eigen库的算法

}

void SparseSurfelFusion::RigidSolver::rigidSolveDeviceIterationFeatureMatch(unsigned int referenceIndex, unsigned int conversionIndex, cudaStream_t stream)
{
	device::Point2PointICP GPUSolver;	// 构造GPU求解器

	GPUSolver.currentPositionMat34 = PreRigidSolver.CamerasRelativePose[conversionIndex];

	GPUSolver.matchedPoints = PreRigidSolver.matchedPoints[conversionIndex];

	const unsigned int pairsNum = PreRigidSolver.matchedPoints[conversionIndex].Size() / 2;

	dim3 block(device::RigidSolverDevice::noReduceBlockSize);	// 一般有效特征点不会超过500个
	dim3 grid(divUp(pairsNum, block.x));
	device::rigidSolveIterationFeatureMatchCamerasPositionKernel << <grid, block, 0, stream >> > (GPUSolver, pairsNum, MatrixBuffer);

	const unsigned int blockNum = divUp(pairsNum, block.x);		// block的数量

	dim3 SumBlock(device::RigidSolverDevice::totalSharedSize);
	dim3 SumGrid(1);
	device::columnReduceFeatureMatchKernel << <SumGrid, SumBlock, 0, stream >> > (MatrixBuffer, blockNum, finalMatrixVector.DevicePtr());

	finalMatrixVector.SynchronizeToHost(stream, false); // 这里进行同步，后续要使用这个矩阵在Host中使用Eigen库的算法
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
	//printf("ICP矩阵 ：\n");
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
	// 做归约
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// 这里不做同步，否则阻塞Host线程
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
	// 做归约
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// 这里不做同步，否则阻塞Host线程
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
	// 做归约
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// 这里不做同步，否则阻塞Host线程
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
	// 做归约
	device::columnReduceKernel << <reduceBufferGrid, reduceBufferBlock, 0, stream >> > (reduceAugmentedMatrix[CameraID], finalAugmentedMatrixVector[CameraID].DevicePtr());

	finalAugmentedMatrixVector[CameraID].SynchronizeToHost(stream, false);	// 这里不做同步，否则阻塞Host线程
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
















