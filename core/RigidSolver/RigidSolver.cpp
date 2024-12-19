/*****************************************************************//**
 * \file   RigidSolver.cpp
 * \brief  ICP刚性配准
 * 
 * \author LUO
 * \date   February 2nd 2024
 *********************************************************************/
#include "RigidSolver.h"

SparseSurfelFusion::RigidSolver::RigidSolver(int devCount, Intrinsic* clipIntrinsic, unsigned int rows, unsigned int cols) : deviceCount(devCount), imageRows(rows), imageCols(cols)
{
	for (int i = 0; i < deviceCount; i++) {
		PreRigidSolver.clipedIntrinsic[i] = clipIntrinsic[i];							// 获得相机内参
		PreRigidSolver.CamerasRelativePose[i] = mat34::identity();						// 初始化相对位姿矩阵
		World2Camera[i] = mat34::identity();
		AccumulatePrevious2Current[i] = mat34::identity();
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}
	//allocateSubsampleBuffer();
	//allocateKNNBuffer();
	allocateReduceBuffer();
	initSolverStreams();
}

SparseSurfelFusion::RigidSolver::~RigidSolver()
{
	//releaseSubsampleBuffer();
	//releaseKNNBuffer();
	releaseReduceBuffer();
	releaseSolverStreams();
}

void SparseSurfelFusion::RigidSolver::performVertexSubsamplingSync(const unsigned int CameraID, DeviceArrayView<DepthSurfel>& denseDepthSurfel, DeviceBufferArray<float4>& sparseVertex, cudaStream_t stream)
{
	PreRigidSolver.denseSurfel[CameraID] = denseDepthSurfel;

	PreRigidSolver.denseVertices[CameraID].ResizeArrayOrException(denseDepthSurfel.Size());
	getVertexFromDepthSurfel(denseDepthSurfel, PreRigidSolver.denseVertices[CameraID], stream);
	DeviceArrayView<float4> denseVerticeView = PreRigidSolver.denseVertices[CameraID].ArrayView();
	// 下采样获得候选点
	PreRigidSolver.vertexSubsampler->PerformSubsample(denseVerticeView, sparseVertex, Constants::VoxelSize, stream);
	// 拷贝一份，不拥有，只读
	PreRigidSolver.sparseVertices[CameraID] = DeviceArrayView<float4>(sparseVertex.Array().ptr(), sparseVertex.Array().size());
}

void SparseSurfelFusion::RigidSolver::allocateSubsampleBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		PreRigidSolver.vertexSubsampler[i].AllocateBuffer(Constants::maxSurfelsNum);						// 分配下采样需要运行的内存
		PreRigidSolver.denseVertices[i].AllocateBuffer(size_t(imageCols) * imageRows * MAX_CAMERA_COUNT);	// 分配稠密点的
	}
}

void SparseSurfelFusion::RigidSolver::releaseSubsampleBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		PreRigidSolver.vertexSubsampler[i].ReleaseBuffer();
		PreRigidSolver.denseVertices[i].ReleaseBuffer();
	}
}

void SparseSurfelFusion::RigidSolver::allocateKNNBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		// 只需要分配最大采样点的Buffer即可
		PreRigidSolver.VertexKNN[i].AllocateBuffer(Constants::maxSubsampledSparsePointsNum);	// 给KNN点分配内存(Buffer缓冲区分配内存)
	}
}

void SparseSurfelFusion::RigidSolver::releaseKNNBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		PreRigidSolver.VertexKNN[i].ReleaseBuffer();
	}
}

void SparseSurfelFusion::RigidSolver::allocateReduceBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		finalAugmentedMatrixVector[i].AllocateBuffer(device::RigidSolverDevice::totalSharedSize);
		finalAugmentedMatrixVector[i].ResizeArrayOrException(device::RigidSolverDevice::totalSharedSize);
		const unsigned int pixelSize = imageRows * imageCols;
		reduceAugmentedMatrix[i].create(device::RigidSolverDevice::totalSharedSize, divUp(pixelSize, device::RigidSolverDevice::blockSize));
	}
}

void SparseSurfelFusion::RigidSolver::releaseReduceBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		finalAugmentedMatrixVector[i].DeviceArray().release();
		finalAugmentedMatrixVector[i].HostArray().clear();
		reduceAugmentedMatrix[i].release();
	}
	//reduceMatrixVector.DeviceArray().release();
	//reduceMatrixVector.HostArray().clear();
	//reduceBuffer.release();
}

void SparseSurfelFusion::RigidSolver::initSolverStreams()
{
	for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
		CHECKCUDA(cudaStreamCreate(&SolverStreams[i]));
		SolverStreams[i] = 0;
	}
}

void SparseSurfelFusion::RigidSolver::releaseSolverStreams()
{
	for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
		CHECKCUDA(cudaStreamDestroy(SolverStreams[i]));
	}
}

void SparseSurfelFusion::RigidSolver::setPreRigidSolverSE3(mat34* initialPose)
{
	for (int i = 0; i < deviceCount; i++) {
		if (initialPose != nullptr) PreRigidSolver.CamerasRelativePose[i] = initialPose[i];
	}
}

void SparseSurfelFusion::RigidSolver::setInitialCameraPose(const unsigned int CameraID)
{
	if (CameraID >= deviceCount) {
		LOGGING(FATAL) << "新增相机需要增设相机初始化SE3参数";
	}
	else {
		PreRigidSolver.CamerasRelativePose[CameraID] = InitialCameraSE3[CameraID];
	}
}

void SparseSurfelFusion::RigidSolver::setCamerasInitialSE3(const unsigned int CameraID)
{
	setInitialCameraPose(CameraID);	// PreRigidSolver.CamerasRelativePose是指针，不存在访存冲突
}

SparseSurfelFusion::mat34* SparseSurfelFusion::RigidSolver::getCamerasInitialSE3()
{
	if (PreRigidSolver.CamerasRelativePose == nullptr) LOGGING(FATAL) << "相机相对于0号相机矩阵为空";
	return PreRigidSolver.CamerasRelativePose;
}

void SparseSurfelFusion::RigidSolver::setPreRigidSolverInput(const unsigned int CameraID, DeviceArrayView<DepthSurfel>& denseSurfel, DeviceArrayView<float4>& pointsPairs)
{
	PreRigidSolver.denseSurfel[CameraID] = denseSurfel;
	PreRigidSolver.matchedPoints[CameraID] = pointsPairs;
}

//void SparseSurfelFusion::RigidSolver::PreSolvePoint2PointICP(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, int maxIteration, cudaStream_t stream)
//{
//	// 获取程序开始时间点
//	auto start = std::chrono::high_resolution_clock::now();
//	size_t totalSparsePointsNum = 0;
//	for (int i = 0; i < deviceCount; i++) {	
//		totalSparsePointsNum += PreRigidSolver.denseSurfel[i].Size();
//		if (i == 0) continue;	// 0号相机的位姿是单位SE3矩阵
//		for (int j = 0; j < maxIteration; j++) {
//			KNNSearchFunction knnSearch(3, 4, stream); // 寻找最邻近点的方法集，用完即释放内存【局部变量】
//			knnSearch.findKNNVertexInDepthSurfel(PreRigidSolver.sparseVertices[i - 1], PreRigidSolver.sparseVertices[i], PreRigidSolver.VertexKNN[i]);
//			rigidSolveDeviceIteration(i - 1, i, stream);
//			rigidSolveHostIterationSync(i - 1, i, stream);
//		}
//		// 转换到相对于0号相机的位姿
//		PreRigidSolver.CamerasRelativePose[i] = PreRigidSolver.CamerasRelativePose[i - 1] * PreRigidSolver.CamerasRelativePose[i];
//	}
//	preAlignedSurfel.ResizeArrayOrException(totalSparsePointsNum);
//	for (int i = 0; i < deviceCount; i++) {
//		addDensePointsToCanonicalField(preAlignedSurfel, i);
//	}
//	CHECKCUDA(cudaDeviceSynchronize());	// 检查流同步
//	// 获取程序结束时间点
//	auto end = std::chrono::high_resolution_clock::now();
//	// 计算程序运行时间（毫秒）
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//	// 输出程序运行时间
//	std::cout <<"ICP迭代次数： "<< maxIteration << "      ICP运行时间: " << duration / (1000.0f) << " 秒" << std::endl;
//}

//void SparseSurfelFusion::RigidSolver::PreSolveMatchedPairsPoint2PointICP(DeviceBufferArray<DepthSurfel>& preAlignedSurfel, int maxIteration, cudaStream_t stream)
//{
//	// 获取程序开始时间点
//	auto start = std::chrono::high_resolution_clock::now();
//	size_t totalSparsePointsNum = 0;
//	for (int i = 0; i < deviceCount; i++) {
//		totalSparsePointsNum += PreRigidSolver.denseSurfel[i].Size();
//		if (i == 0) continue;	// 0号相机的位姿是单位SE3矩阵
//		for (int j = 0; j < maxIteration; j++) {
//			rigidSolveDeviceIterationFeatureMatch(i - 1, i, stream);
//			rigidSolveHostIterationFeatureMatchSync(i - 1, i, stream);
//		}
//		// 转换到相对于0号相机的位姿
//		PreRigidSolver.CamerasRelativePose[i] = PreRigidSolver.CamerasRelativePose[i - 1] * PreRigidSolver.CamerasRelativePose[i];
//	}
//	preAlignedSurfel.ResizeArrayOrException(totalSparsePointsNum);
//	for (int i = 0; i < deviceCount; i++) {
//		addDensePointsToCanonicalField(preAlignedSurfel, i);
//	}
//	CHECKCUDA(cudaDeviceSynchronize());	// 检查流同步
//	// 获取程序结束时间点
//	auto end = std::chrono::high_resolution_clock::now();
//	// 计算程序运行时间（毫秒）
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//	// 输出程序运行时间
//	std::cout << "ICP迭代次数： " << maxIteration << "      ICP运行时间: " << duration / (1000.0f) << " 秒" << std::endl;
//}

SparseSurfelFusion::mat34 SparseSurfelFusion::RigidSolver::getPreSolverMat34(unsigned int CameraID)
{
	return PreRigidSolver.CamerasRelativePose[CameraID];
}

void SparseSurfelFusion::RigidSolver::SynchronizeAllStreams(cudaStream_t* streams, const unsigned int streamsNum)
{
	for (int i = 0; i < streamsNum; i++) {
		CHECKCUDA(cudaStreamSynchronize(streams[i]));
	}
}

void SparseSurfelFusion::RigidSolver::SetRigidInput(CameraObservation& observation, const mat34 world2camera0, const mat34 world2camera1)
{

}

void SparseSurfelFusion::RigidSolver::SolveRigidAlignment(CameraObservation& observation, bool& isfirstframe, unsigned int MaxInteration)
{
	if (isfirstframe) {
		for (int j = 0; j < deviceCount; j++) {
			World2Camera[j] = mat34::identity();
		}
		isfirstframe = false;//用完恢复
	}

	for (int i = 0; i < deviceCount; i++) {//源代码里用的是live的与深度面元做对齐，你这用的两帧深度图分开做对齐
		referenceMap[i].vertexMap = observation.icpvertexConfidenceMap[i];				// 当前作为参考
		referenceMap[i].normalMap = observation.icpnormalRadiusMap[i];
		referenceMap[i].foreground = observation.foregroundMask[i];
		conversionMap[i].vertexMap = observation.PreviousVertexConfidenceMap[i];	// 上一帧对齐当前帧
		conversionMap[i].normalMap = observation.PreviousNormalRadiusMap[i];
		conversionMap[i].foreground = observation.foregroundMaskPrevious[i];
	}

	for (int i = 0; i < MaxInteration; i++) {
		for (int j = 0; j < deviceCount; j++) {
			rigidSolveDeviceIteration(j, SolverStreams[j]);
		}
		SynchronizeAllStreams(SolverStreams, deviceCount);	// 同步所有流
		rigidSolveHostIterationSync();						// 使用默认流
	}
	for (int j = 0; j < deviceCount; j++) {
		Frame2FrameWorld2Camera[j] = AccumulatePrevious2Current[j];
		World2Camera[j] = AccumulatePrevious2Current[j] * World2Camera[j];
		AccumulatePrevious2Current[j] = mat34::identity();						// 用完置零
	}

	Canonical2Live = AverageCanonicalFieldRigidSE3(World2Camera);
}


void SparseSurfelFusion::RigidSolver::SolveRigidAlignment(Renderer::SolverMaps* solverMaps, CameraObservation& observation, const mat34* LiveField2ObservedField, unsigned int MaxInteration)
{

	for (int i = 0; i < deviceCount; i++) {
		referenceMap[i].vertexMap = observation.vertexConfidenceMap[i];			// 当前作为参考
		referenceMap[i].normalMap = observation.normalRadiusMap[i];
		conversionMap[i].vertexMap = solverMaps[i].warp_vertex_map;	// 上一帧对齐当前帧
		conversionMap[i].normalMap = solverMaps[i].warp_normal_map;
		World2Camera[i] = LiveField2ObservedField[i];
	}

	for (int i = 0; i < MaxInteration; i++) {
		for (int j = 0; j < deviceCount; j++) {
			rigidSolveDeviceIteration(j, World2Camera, SolverStreams[j]);
		}
		SynchronizeAllStreams(SolverStreams, deviceCount);	// 同步所有流
		rigidSolveHostIterationSync(World2Camera);
		//printf("ICP矩阵_%d ：\n", i);
		//printf("        %.9f    %.9f    %.9f    %.9f\n", World2Camera[0].rot.m00(), World2Camera[0].rot.m01(), World2Camera[0].rot.m02(), World2Camera[0].trans.x);
		//printf("        %.9f    %.9f    %.9f    %.9f\n", World2Camera[0].rot.m10(), World2Camera[0].rot.m11(), World2Camera[0].rot.m12(), World2Camera[0].trans.y);
		//printf("        %.9f    %.9f    %.9f    %.9f\n", World2Camera[0].rot.m20(), World2Camera[0].rot.m21(), World2Camera[0].rot.m22(), World2Camera[0].trans.z);
	}
	Canonical2Live = AverageCanonicalFieldRigidSE3(World2Camera);
}


SparseSurfelFusion::mat34 SparseSurfelFusion::RigidSolver::AverageCanonicalFieldRigidSE3(const mat34* world2Camera)
{
	mat34 Canonical2Live;
	mat33 Canonical2LiveRotation;
	float3 Canonical2LiveTranslation;

	for (int i = 0; i < deviceCount; i++) {
		//printf("ICP矩阵 ：\n");
		//printf("        %.9f    %.9f    %.9f    %.9f\n", World2Camera[i].rot.m00(), World2Camera[i].rot.m01(), World2Camera[i].rot.m02(), World2Camera[i].trans.x);
		//printf("        %.9f    %.9f    %.9f    %.9f\n", World2Camera[i].rot.m10(), World2Camera[i].rot.m11(), World2Camera[i].rot.m12(), World2Camera[i].trans.y);
		//printf("        %.9f    %.9f    %.9f    %.9f\n", World2Camera[i].rot.m20(), World2Camera[i].rot.m21(), World2Camera[i].rot.m22(), World2Camera[i].trans.z);
		if (i == 0) {
			Canonical2LiveRotation = world2Camera[i].rot;
			Canonical2LiveTranslation = world2Camera[i].trans;
		}
		else {
			mat34 SE3Rigid0 = PreRigidSolver.CamerasRelativePose[i].inverse();
			SE3Rigid0 = world2Camera[i] * SE3Rigid0;
			SE3Rigid0 = PreRigidSolver.CamerasRelativePose[i] * SE3Rigid0;
			Canonical2LiveRotation = Canonical2LiveRotation + SE3Rigid0.rot;
			Canonical2LiveTranslation = Canonical2LiveTranslation + SE3Rigid0.trans;
		}
	}
	Canonical2LiveRotation = Canonical2LiveRotation / (float)deviceCount;
	Canonical2LiveTranslation = make_float3(Canonical2LiveTranslation.x / (float)deviceCount, Canonical2LiveTranslation.y / (float)deviceCount, Canonical2LiveTranslation.z / (float)deviceCount);
	Canonical2Live.rot = Canonical2LiveRotation;
	Canonical2Live.trans = Canonical2LiveTranslation;
	return Canonical2Live;
}

void SparseSurfelFusion::RigidSolver::CheckNaNValue(float3& rot, float3& trans)
{
	if (isnan(rot.x) || isnan(rot.y) || isnan(rot.z)) rot = make_float3(0.0f, 0.0f, 0.0f);
	if (isnan(trans.x) || isnan(trans.y) || isnan(trans.z)) trans = make_float3(0.0f, 0.0f, 0.0f);
}

void SparseSurfelFusion::RigidSolver::rigidSolveHostIterationSync(cudaStream_t stream)
{
	// 前面已经将所有stream同步了

	std::vector<float> HostArrays[MAX_CAMERA_COUNT];
	for (int i = 0; i < deviceCount; i++) {
		HostArrays[i] = finalAugmentedMatrixVector[i].HostArray();
	}

	unsigned int shift = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			float value[MAX_CAMERA_COUNT];
			for (int k = 0; k < deviceCount; k++) {
				value[k] = 0;
			}

			for (int k = 0; k < deviceCount; k++) {
				value[k] += HostArrays[k][shift];
			}
			for (int k = 0; k < deviceCount; k++) {
				ATA_[k](i, j) = value[k];
				ATA_[k](j, i) = value[k];
			}

			shift++;
		}
	}
	for (int i = 0; i < 6; i++) {
		float value[MAX_CAMERA_COUNT];
		for (int k = 0; k < deviceCount; k++) {
			value[k] = 0;
		}

		for (int k = 0; k < deviceCount; k++) {
			value[k] += HostArrays[k][shift];
		}
		for (int k = 0; k < deviceCount; k++) {
			ATb_[k][i] = value[k];
		}
		shift++;
	}

	// 接下来求解AT * A * x = AT * b 矩阵
	// ①ATA_.llt()：ATA_ 进行了 Cholesky 分解（LLT 分解）
	// ②solve(ATb_)：使用该分解求解了线性方程组 ATA_ * x = ATb_
	// ③cast<float>() 是一个类型转换操作，将求解得到的结果转换为 float 类型。
	Eigen::Matrix<float, 6, 1> x[MAX_CAMERA_COUNT];
	float3 rot[MAX_CAMERA_COUNT];
	float3 trans[MAX_CAMERA_COUNT];
	mat34 UpdateSE3[MAX_CAMERA_COUNT];
	for (int k = 0; k < deviceCount; k++) {
		x[k] = ATA_[k].llt().solve(ATb_[k]).cast<float>();
		rot[k] = make_float3(x[k](0), x[k](1), x[k](2));
		trans[k] = make_float3(x[k](3), x[k](4), x[k](5));
		mat34 temp(rot[k], trans[k]);
		UpdateSE3[k] = temp;
		AccumulatePrevious2Current[k] = UpdateSE3[k] * AccumulatePrevious2Current[k];
	}



	//printf("rotate    = (%15.9f, %15.9f, %15.9f)\n", rot.x, rot.y, rot.z);
	//printf("translate = (%15.9f, %15.9f, %15.9f)\n", trans.x, trans.y, trans.z);



}

void SparseSurfelFusion::RigidSolver::rigidSolveHostIterationSync(mat34* world2Camera, cudaStream_t stream)
{
	// 前面已经将所有stream同步了

	std::vector<float> HostArrays[MAX_CAMERA_COUNT];
	for (int i = 0; i < deviceCount; i++) {
		HostArrays[i] = finalAugmentedMatrixVector[i].HostArray();
	}

	unsigned int shift = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			float value[MAX_CAMERA_COUNT];
			for (int k = 0; k < deviceCount; k++) {
				value[k] = 0;
			}

			for (int k = 0; k < deviceCount; k++) {
				value[k] += HostArrays[k][shift];
			}
			for (int k = 0; k < deviceCount; k++) {
				ATA_[k](i, j) = value[k];
				ATA_[k](j, i) = value[k];
			}

			shift++;
		}
	}
	for (int i = 0; i < 6; i++) {
		float value[MAX_CAMERA_COUNT];
		for (int k = 0; k < deviceCount; k++) {
			value[k] = 0;
		}

		for (int k = 0; k < deviceCount; k++) {
			value[k] += HostArrays[k][shift];
		}
		for (int k = 0; k < deviceCount; k++) {
			ATb_[k][i] = value[k];
		}
		shift++;
	}

	// 接下来求解AT * A * x = AT * b 矩阵
	// ①ATA_.llt()：ATA_ 进行了 Cholesky 分解（LLT 分解）
	// ②solve(ATb_)：使用该分解求解了线性方程组 ATA_ * x = ATb_
	// ③cast<float>() 是一个类型转换操作，将求解得到的结果转换为 float 类型。
	Eigen::Matrix<float, 6, 1> x[MAX_CAMERA_COUNT];
	float3 rot[MAX_CAMERA_COUNT];
	float3 trans[MAX_CAMERA_COUNT];
	mat34 UpdateSE3[MAX_CAMERA_COUNT];
	for (int k = 0; k < deviceCount; k++) {
		x[k] = ATA_[k].llt().solve(ATb_[k]).cast<float>();
		rot[k] = make_float3(x[k](0), x[k](1), x[k](2));
		trans[k] = make_float3(x[k](3), x[k](4), x[k](5));
		mat34 temp(rot[k], trans[k]);
		UpdateSE3[k] = temp;
		world2Camera[k] = UpdateSE3[k] * world2Camera[k];
	}

}


void SparseSurfelFusion::RigidSolver::rigidSolveLive2ObservedHostIterationSync(cudaStream_t stream)
{
	// 前面已经将所有stream同步了

	std::vector<float> HostArrays[MAX_CAMERA_COUNT];
	for (int i = 0; i < deviceCount; i++) {
		HostArrays[i] = finalAugmentedMatrixVector[i].HostArray();
	}

	unsigned int shift = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			float value[MAX_CAMERA_COUNT];
			for (int k = 0; k < deviceCount; k++) {
				value[k] = 0;
			}

			for (int k = 0; k < deviceCount; k++) {
				value[k] += HostArrays[k][shift];
			}
			for (int k = 0; k < deviceCount; k++) {
				ATA_[k](i, j) = value[k];
				ATA_[k](j, i) = value[k];
			}

			shift++;
		}
	}
	for (int i = 0; i < 6; i++) {
		float value[MAX_CAMERA_COUNT];
		for (int k = 0; k < deviceCount; k++) {
			value[k] = 0;
		}

		for (int k = 0; k < deviceCount; k++) {
			value[k] += HostArrays[k][shift];
		}
		for (int k = 0; k < deviceCount; k++) {
			ATb_[k][i] = value[k];
		}
		shift++;
	}

	// 接下来求解AT * A * x = AT * b 矩阵
	// ①ATA_.llt()：ATA_ 进行了 Cholesky 分解（LLT 分解）
	// ②solve(ATb_)：使用该分解求解了线性方程组 ATA_ * x = ATb_
	// ③cast<float>() 是一个类型转换操作，将求解得到的结果转换为 float 类型。
	Eigen::Matrix<float, 6, 1> x[MAX_CAMERA_COUNT];
	float3 rot[MAX_CAMERA_COUNT];
	float3 trans[MAX_CAMERA_COUNT];
	mat34 UpdateSE3[MAX_CAMERA_COUNT];
	for (int k = 0; k < deviceCount; k++) {
		x[k] = ATA_[k].llt().solve(ATb_[k]).cast<float>();
		rot[k] = make_float3(x[k](0), x[k](1), x[k](2));
		trans[k] = make_float3(x[k](3), x[k](4), x[k](5));
		mat34 temp(rot[k], trans[k]);
		UpdateSE3[k] = temp;
		World2Camera[k] = UpdateSE3[k] * World2Camera[k];
	}

}

void SparseSurfelFusion::RigidSolver::rigidSolveMultiViewHostIterationSync(cudaStream_t stream)
{
	Eigen::Matrix<float, 6, 6> JtJ_;
	Eigen::Matrix<float, 6, 1> JtErr_;

	std::vector<float> HostArrays[MAX_CAMERA_COUNT];
	for (int i = 0; i < deviceCount; i++) {
		HostArrays[i] = finalAugmentedMatrixVector[i].HostArray();
	}

	unsigned int shift = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			for (int k = 0; k < deviceCount; k++) {
				const float value = HostArrays[k][shift++];
				JtJ_(i, j) += value;
				JtJ_(j, i) += value;
			}
		}
	}
	for (int i = 0; i < 6; i++) {
		for (int k = 0; k < deviceCount; k++) {
			const float value = HostArrays[k][shift++];
			JtErr_[i] += value;
		}
	}

	//Solve it
	Eigen::Matrix<float, 6, 1> x = JtJ_.llt().solve(JtErr_).cast<float>();

	//Update the se3
	const float3 twist_rot = make_float3(x(0), x(1), x(2));
	const float3 twist_trans = make_float3(x(3), x(4), x(5));
	//printf("rotate    = (%15.9f, %15.9f, %15.9f)\n", twist_rot.x, twist_rot.y, twist_rot.z);
	//printf("translate = (%15.9f, %15.9f, %15.9f)\n", twist_trans.x, twist_trans.y, twist_trans.z);
	const mat34 se3_update(twist_rot, twist_trans);

	for (int i = 0; i < deviceCount; i++) {
		World2Camera[i] = se3_update * World2Camera[i];
	}
}





