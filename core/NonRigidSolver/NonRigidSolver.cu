/*****************************************************************//**
 * \file   NonRigidSolver.cu
 * \brief  �Ǹ��������GPU����ʵ��
 * 
 * \author LUO
 * \date   March 28th 2024
 *********************************************************************/
#include "NonRigidSolver.h"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		__global__ void queryPixelKNNKernel(
			cudaTextureObject_t index_map,
			const ushort4* surfel_knn,
			const float4* surfel_knn_weight,
			//Output
			PtrStepSize<KNNAndWeight> knn_map
		) {
			
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x < knn_map.cols && y < knn_map.rows) {   
				KNNAndWeight knn_weight;
				knn_weight.setInvalid();
				const unsigned int index = tex2D<unsigned int>(index_map, x, y);
				if (index != 0xFFFFFFFF) {
					knn_weight.knn = surfel_knn[index];
					knn_weight.weight = surfel_knn_weight[index];
				}
				knn_map.ptr(y)[x] = knn_weight;
			}
		}

		__device__ __forceinline__ bool checkVertexViewDirection(
			const float4& depthVertex, const float4& depthNormal
		)  {
			const float3 viewDirection = -normalized(make_float3(depthVertex.x, depthVertex.y, depthVertex.z));
			const float3 normal = normalized(make_float3(depthNormal.x, depthNormal.y, depthNormal.z));
			return dot(viewDirection, normal) > 0.5f;
		}


		__device__ mat34 computeSuitableOpticalGuidedSe3(const float4& nodeCoor, GuidedNodesInput guidedNodesInput, const unsigned int CameraID, const unsigned int mapCols, const unsigned int mapRows)
		{
			float minDiffz = 1e6f;
			mat34 suitableOpticalSe3 = mat34::identity();	// �Ѿ�ת��0������ϵ��
			// ���ڵ��0������ϵ��������ͬ���
			mat34& initialSE3 = guidedNodesInput.initialCameraSE3[CameraID];
			mat34& initialSE3Inv = guidedNodesInput.initialCameraSE3Inverse[CameraID];
			mat34& world2camera = guidedNodesInput.world2Camera[CameraID];
			Intrinsic& intrinsic = guidedNodesInput.intrinsic[CameraID];
			float3 convertNodeCoor = world2camera.rot * nodeCoor + world2camera.trans;
			convertNodeCoor = initialSE3Inv.rot * convertNodeCoor + initialSE3Inv.trans;
			cudaTextureObject_t preVertexMap = guidedNodesInput.preVertexMap[CameraID];
			cudaTextureObject_t preNormalMap = guidedNodesInput.preNormalMap[CameraID];
			DeviceArrayView2D<mat34>& opticalMap = guidedNodesInput.guidedSe3Map[CameraID];
			// ͶӰ����ǰ���ƽ��
			const uint2 imageCoordinate = {
				__float2uint_rn(((convertNodeCoor.x / (convertNodeCoor.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
				__float2uint_rn(((convertNodeCoor.y / (convertNodeCoor.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
			};

			if (imageCoordinate.x >= (mapCols - FlowSearchWindowHalfSize) || imageCoordinate.x < FlowSearchWindowHalfSize || imageCoordinate.y >= (mapRows - FlowSearchWindowHalfSize) || imageCoordinate.y < FlowSearchWindowHalfSize) return suitableOpticalSe3;
				
			else {
				// live��Բ����ԭ��Ĭ���ǷǸ���warp�ľ���̫С��
				// ��ʱliveӦ�ú���һ֡�۲��vertex�ܽ����ҵ�����������һ֡vertex�������Ĭ��Ϊ��ǰ��У����λ��
				for (unsigned int i = imageCoordinate.x - FlowSearchWindowHalfSize; i <= imageCoordinate.x + FlowSearchWindowHalfSize; i++) {
					for (unsigned int j = imageCoordinate.y - FlowSearchWindowHalfSize; j <= imageCoordinate.y + FlowSearchWindowHalfSize; j++) {
						float4 preVertex = tex2D<float4>(preVertexMap, i, j);	// ��һ֡����
						float4 preNormal = tex2D<float4>(preNormalMap, i, j);	// ��һ֡����
						if (!is_zero_vertex(preVertex)) {	// ��ֹ͸��
							// �Բ����ԭ��  -->  liveNode���ӽ���һ֡Vertex  -->  ��ǰ֡vertexͨ�������õ���һ֡vertex  
							// -->  ��node���ӽ�����һ֡vertex����Ӧ��ǰvertex��Ӧ������se3��Ϊnode��se3
							mat34 opticalSe3 = opticalMap(j, i);	// ��һ֡����ǰ֡
							float diff_z = fabsf(convertNodeCoor.z - preVertex.z);
							float3 correctedNodeCoor = opticalSe3.rot * convertNodeCoor + opticalSe3.trans;
							// �����仯��node��preObservation������1cm
							if (minDiffz > diff_z && diff_z < 1e-2f && squared_distance(correctedNodeCoor, convertNodeCoor) < 4 * NODE_RADIUS_SQUARE) {
								minDiffz = diff_z;
								suitableOpticalSe3 = world2camera.inverse() * initialSE3 * opticalSe3 * initialSE3Inv * world2camera;
							}
						}
					}
				}
				return suitableOpticalSe3;
			}
		}


		__device__ mat34 computeSuitableCorrespondenceGuidedSe3(const float4& nodeCoor, GuidedNodesInput guidedNodesInput, const unsigned int CameraID, const unsigned int mapCols, const unsigned int mapRows)
		{
			float minDis = 1e6f;
			mat34 suitableSe3 = mat34::identity();	// �Ѿ�ת��0������ϵ��
			// ���ڵ��0������ϵ��������ͬ���
			mat34& initialSE3 = guidedNodesInput.initialCameraSE3[CameraID];
			mat34& initialSE3Inv = guidedNodesInput.initialCameraSE3Inverse[CameraID];
			mat34& world2camera = guidedNodesInput.world2Camera[CameraID];
			Intrinsic& intrinsic = guidedNodesInput.intrinsic[CameraID];
			float3 convertNodeCoor = world2camera.rot * nodeCoor + world2camera.trans;
			convertNodeCoor = initialSE3Inv.rot * convertNodeCoor + initialSE3Inv.trans;
			cudaTextureObject_t preVertexMap = guidedNodesInput.preVertexMap[CameraID];
			cudaTextureObject_t preNormalMap = guidedNodesInput.preNormalMap[CameraID];
			DeviceArrayView2D<mat34>& guidedMap = guidedNodesInput.guidedSe3Map[CameraID];
			DeviceArrayView2D<unsigned char>& validMap = guidedNodesInput.markValidSe3Map[CameraID];
			// ͶӰ����ǰ���ƽ��
			const uint2 imageCoordinate = {
				__float2uint_rn(((convertNodeCoor.x / (convertNodeCoor.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
				__float2uint_rn(((convertNodeCoor.y / (convertNodeCoor.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
			};

			if (imageCoordinate.x >= (mapCols - CorrSearchWindowHalfSize) || imageCoordinate.x < CorrSearchWindowHalfSize || imageCoordinate.y >= (mapRows - CorrSearchWindowHalfSize) || imageCoordinate.y < CorrSearchWindowHalfSize) return suitableSe3;

			else {
				// live��Բ����ԭ��Ĭ���ǷǸ���warp�ľ���̫С��
				// ��ʱliveӦ�ú���һ֡�۲��vertex�ܽ����ҵ�����������һ֡vertex�������Ĭ��Ϊ��ǰ��У����λ��
				for (unsigned int i = imageCoordinate.x - CorrSearchWindowHalfSize; i <= imageCoordinate.x + CorrSearchWindowHalfSize; i++) {
					for (unsigned int j = imageCoordinate.y - CorrSearchWindowHalfSize; j <= imageCoordinate.y + CorrSearchWindowHalfSize; j++) {
						float4 preVertex = tex2D<float4>(preVertexMap, i, j);	// ��һ֡����
						float4 preNormal = tex2D<float4>(preNormalMap, i, j);	// ��һ֡����
						if (validMap(j, i) == (unsigned char)1 && !is_zero_vertex(preVertex)) {	// ������Ч��ϡ���
							// �Բ����ԭ��  -->  liveNode���ӽ���һ֡Vertex  -->  ��ǰ֡vertexͨ�������õ���һ֡vertex  
							// -->  ��node���ӽ�����һ֡vertex����Ӧ��ǰvertex��Ӧ������se3��Ϊnode��se3
							mat34 guidedSe3 = guidedMap(j, i);	// ��һ֡����ǰ֡
							float squaredDis = squared_distance(convertNodeCoor, preVertex);
							float3 correctedNodeCoor = guidedSe3.rot * convertNodeCoor + guidedSe3.trans;
							// �����仯��node��ϡ��ƥ��㲻����5cm��node�任���벻����2���ڵ����
							if (minDis > squaredDis && squaredDis < 1e-4f && squared_distance(correctedNodeCoor, convertNodeCoor) < 4 * NODE_RADIUS_SQUARE) {
								minDis = squaredDis;
								suitableSe3 = guidedSe3;
							}
						}
					}
				}
				suitableSe3 = world2camera.inverse() * initialSE3 * suitableSe3 * initialSE3Inv * world2camera;
				return suitableSe3;
			}
		}

		__global__ void computeOpticalFlowGuidedNodesSE3(GuidedNodesInput guidedNodesInput, const float4* liveNodesCoor, const float* nodeUnitedError, const unsigned int liveNodesNum, const unsigned int mapCols, const unsigned int mapRows, const unsigned int devicesCount, DualQuaternion* dqCorrectNodes)
		{
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= liveNodesNum) return;
			const unsigned int CameraID = decodeNodesCameraView(liveNodesCoor[idx]);
			mat34 suitableSe3 = computeSuitableOpticalGuidedSe3(liveNodesCoor[idx], guidedNodesInput, CameraID, mapCols, mapRows);
			dqCorrectNodes[idx] = DualQuaternion(suitableSe3) * nodeUnitedError[idx];	// ����Ȩ�ر������ڵ�ErrorԽ���ô˹���������λ�˱任Խ��				
			
			dqCorrectNodes[idx].set_identity();
			
		}

		__global__ void computeCorrespondenceGuidedNodesSE3(GuidedNodesInput guidedNodesInput, const float4* liveNodesCoor, const float* nodeUnitedError, const unsigned int liveNodesNum, const unsigned int mapCols, const unsigned int mapRows, const unsigned int devicesCount, DualQuaternion* dqCorrectNodes)
		{
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= liveNodesNum) return;
			const unsigned int CameraID = decodeNodesCameraView(liveNodesCoor[idx]);
			if (nodeUnitedError[idx] > 0.2f) {
				mat34 suitableSe3 = computeSuitableCorrespondenceGuidedSe3(liveNodesCoor[idx], guidedNodesInput, CameraID, mapCols, mapRows);
				dqCorrectNodes[idx] = DualQuaternion(suitableSe3) * nodeUnitedError[idx];	// ����Ȩ�ر������ڵ�ErrorԽ���ô˹���������λ�˱任Խ��				
			}
			else {
				dqCorrectNodes[idx].set_identity();
			}
		}

		__global__ void CorrectWarpVertexAndNodeKernel(CorrectInput input, const unsigned int totalPointsNum, const unsigned int denseLiveSurfelNum, const unsigned int devicesCount, float4* correctedNodes, float4* correctedVertex) 
		{
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= totalPointsNum) return;
			ushort4 knn;
			float4 weight;
			float4 liveVertex = make_float4(0, 0, 0, 0);
			float4 liveNormal = make_float4(0, 0, 0, 0);
			if (idx < denseLiveSurfelNum) {
				knn = input.surfelKnn[0][idx];
				weight = input.surfelKnnWeight[0][idx];
				liveVertex = input.denseLiveSurfelsVertex[0][idx];
				liveNormal = input.denseLiveSurfelsNormal[0][idx];
			}
			else if (idx >= denseLiveSurfelNum && idx < totalPointsNum) {
				const int offset = idx - denseLiveSurfelNum;
				knn = input.nodesKNN[offset];
				weight = input.nodesKNNWeight[offset];
				liveVertex = input.liveNodesCoordinate[offset];
			}
			// У��
			DualQuaternion dqAverage = averageDualQuaternion(input.dqCorrectNodes, knn, weight);
			const mat34 se3 = dqAverage.se3_matrix();
			// ����canonical
			float3 v3_live = make_float3(liveVertex.x, liveVertex.y, liveVertex.z);
			float3 n3_live = make_float3(liveNormal.x, liveNormal.y, liveNormal.z);
			v3_live = se3.rot * v3_live + se3.trans;
			n3_live = se3.rot * n3_live;
			liveVertex = make_float4(v3_live.x, v3_live.y, v3_live.z, liveVertex.w);
			liveNormal = make_float4(n3_live.x, n3_live.y, n3_live.z, liveNormal.w);

			if (idx < denseLiveSurfelNum) {
				for (int i = 0; i < devicesCount; i++) {
					input.denseLiveSurfelsVertex[i][idx] = liveVertex;
					input.denseLiveSurfelsNormal[i][idx] = liveNormal;
				}
				correctedVertex[idx] = liveVertex;
			}
			else if (idx >= denseLiveSurfelNum && idx < totalPointsNum) {
				const int offset = idx - denseLiveSurfelNum;
				input.liveNodesCoordinate[offset] = liveVertex;
				// У������se3 * ԭʼnodeSE3  -->  ��ԭʼ��nodeSE3������У��
				input.nodesSE3[offset] = dqAverage * input.nodesSE3[offset];
			}
		}
	}
}

void SparseSurfelFusion::NonRigidSolver::QueryPixelKNN(cudaStream_t stream) {
	dim3 block(16, 16);
	dim3 grid(divUp(imageWidth, block.x), divUp(imageHeight, block.y));

	for (int i = 0; i < devicesCount; i++) {
		device::queryPixelKNNKernel << <grid, block, 0, stream >> > (
			solverMap[i].index_map,
			denseSurfelsInput.surfelKnn,
			denseSurfelsInput.surfelKnnWeight,
			knnMap[i]//���������
		);
	}
}

/* The method to setup and solve Ax=b using pcg solver
 */
void SparseSurfelFusion::NonRigidSolver::allocatePCGSolverBuffer() {
	const auto max_matrix_size = 6 * Constants::maxNodesNum;
	m_pcg_solver = std::make_shared<BlockPCG<6>>(max_matrix_size);
}

void SparseSurfelFusion::NonRigidSolver::releasePCGSolverBuffer() {
}

void SparseSurfelFusion::NonRigidSolver::UpdatePCGSolverStream(cudaStream_t stream) {
	m_pcg_solver->UpdateCudaStream(stream);
}

void SparseSurfelFusion::NonRigidSolver::SolvePCGMatrixFree() {
	//Prepare the data
	const auto inversed_diagonal_preconditioner = m_preconditioner_rhs_builder->InversedPreconditioner();
	const auto rhs = m_preconditioner_rhs_builder->JtDotResidualValue();
	ApplySpMVBase<6>::Ptr apply_spmv_handler = m_apply_jtj_handler;
	DeviceArrayHandle<float> updated_twist = iterationData.CurrentWarpFieldUpdateBuffer();

	//sanity check
	FUNCTION_CHECK_EQ(rhs.Size(), apply_spmv_handler->MatrixSize());
	FUNCTION_CHECK_EQ(updated_twist.Size(), apply_spmv_handler->MatrixSize());
	FUNCTION_CHECK_EQ(inversed_diagonal_preconditioner.Size(), apply_spmv_handler->MatrixSize() * 6);

	//Hand in to warp solver and solve it
	m_pcg_solver->SetSolverInput(inversed_diagonal_preconditioner, apply_spmv_handler, rhs, updated_twist);
	m_pcg_solver->Solve(10);
}

void SparseSurfelFusion::NonRigidSolver::SolvePCGMaterialized(int pcg_iterations) {
	//Prepare the data
	const DeviceArrayView<float> inversed_diagonal_preconditioner = m_preconditioner_rhs_builder->InversedPreconditioner();
	const DeviceArrayView<float> rhs = m_preconditioner_rhs_builder->JtDotResidualValue();
	ApplySpMVBase<6>::Ptr apply_spmv_handler = m_jtj_materializer->GetSpMVHandler();
	DeviceArrayHandle<float> updated_twist = iterationData.CurrentWarpFieldUpdateBuffer();

	//sanity check
	FUNCTION_CHECK_EQ(rhs.Size(), apply_spmv_handler->MatrixSize());
	FUNCTION_CHECK_EQ(updated_twist.Size(), apply_spmv_handler->MatrixSize());
	FUNCTION_CHECK_EQ(inversed_diagonal_preconditioner.Size(), apply_spmv_handler->MatrixSize() * 6);

	//Hand in to warp solver and solve it
	m_pcg_solver->SetSolverInput(inversed_diagonal_preconditioner, apply_spmv_handler, rhs, updated_twist);
	m_pcg_solver->Solve(pcg_iterations);
}


void SparseSurfelFusion::NonRigidSolver::SetComputeOpticalGuideNodesSe3Input(const DeviceArrayView2D<mat34>* opticalMap, CameraObservation observation, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, const unsigned int devicesCount, device::GuidedNodesInput& input)
{
	for (int i = 0; i < devicesCount; i++) {
		input.preVertexMap[i] = observation.PreviousVertexConfidenceMap[i];
		input.preNormalMap[i] = observation.PreviousNormalRadiusMap[i];
		input.initialCameraSE3[i] = InitialCameraSE3Array[i];
		input.initialCameraSE3Inverse[i] = InitialCameraSE3Array[i].inverse();
		input.world2Camera[i] = world2camera[i];
		input.intrinsic[i] = intrinsicArray[i];
		input.guidedSe3Map[i] = opticalMap[i];
	}
}

void SparseSurfelFusion::NonRigidSolver::SetComputeCorrespondenceGuideNodeSe3Input(const DeviceArrayView2D<mat34>* corrMap, const DeviceArrayView2D<unsigned char>* markValidSe3Map, CameraObservation observation, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, const unsigned int devicesCount, device::GuidedNodesInput& input)
{
	for (int i = 0; i < devicesCount; i++) {
		input.preVertexMap[i] = observation.PreviousVertexConfidenceMap[i];
		input.preNormalMap[i] = observation.PreviousNormalRadiusMap[i];
		input.initialCameraSE3[i] = InitialCameraSE3Array[i];
		input.initialCameraSE3Inverse[i] = InitialCameraSE3Array[i].inverse();
		input.world2Camera[i] = world2camera[i];
		input.intrinsic[i] = intrinsicArray[i];
		input.guidedSe3Map[i] = corrMap[i];
		input.markValidSe3Map[i] = markValidSe3Map[i];
	}
}

void SparseSurfelFusion::NonRigidSolver::SetOpticalGuideInput(WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int devicesCount, const unsigned int updatedGeometryIndex, device::CorrectInput& correctInput)
{
	// WarpField:����һ֡����õ���live��ĳ��ܵ㣬live��ڵ㣬�Լ��ڵ�SE3���й���У������¶ָ��ֱ�Ӵ۸�
	correctInput.nodesSE3 = warpField.BindOpticalFlowGuideInfo().nodesSE3.ptr();
	correctInput.nodesKNN = warpField.BindOpticalFlowGuideInfo().nodesKNN.ptr();
	correctInput.nodesKNNWeight = warpField.BindOpticalFlowGuideInfo().nodesKNNWeight.ptr();
	correctInput.liveNodesCoordinate = warpField.BindOpticalFlowGuideInfo().liveNodesCoordinate.ptr();
	correctInput.dqCorrectNodes = dqCorrectNodes.Ptr();
	correctInput.correctedCanonicalSurfelsSE3 = correctedCanonicalSurfelsSE3.Ptr();

	for (int i = 0; i < devicesCount; i++) {
		// SurfelGeometry
		correctInput.denseLiveSurfelsVertex[i] = geometry[i][updatedGeometryIndex]->BindOpticalFlowGuideInfo().denseLiveSurfelsVertex.ptr();
		correctInput.denseLiveSurfelsNormal[i] = geometry[i][updatedGeometryIndex]->BindOpticalFlowGuideInfo().denseLiveSurfelsNormal.ptr();
		correctInput.surfelKnn[i] = geometry[i][updatedGeometryIndex]->BindOpticalFlowGuideInfo().surfelKnn.ptr();
		correctInput.surfelKnnWeight[i] = geometry[i][updatedGeometryIndex]->BindOpticalFlowGuideInfo().surfelKnnWeight.ptr();
	}
}

void SparseSurfelFusion::NonRigidSolver::CorrectLargeErrorNode(const unsigned int frameIdx, const DeviceArrayView2D<mat34>* opticalMap, CameraObservation observation, DeviceArrayView<float> nodeUnitedError, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int updatedGeometryIndex, cudaStream_t stream)
{
	device::CorrectInput correctInput;
	const unsigned int SparseNodesNum = warpField.BindOpticalFlowGuideInfo().nodesSE3.size();
	const unsigned int DenseSurfelsNum = geometry[0][updatedGeometryIndex]->BindOpticalFlowGuideInfo().denseLiveSurfelsVertex.size();
	const unsigned int mapCols = opticalMap[0].Cols();
	const unsigned int mapRows = opticalMap[0].Rows();
	const float4* liveNodesCoorPtr = warpField.BindOpticalFlowGuideInfo().liveNodesCoordinate.ptr();
	const float* nodeUnitedErrorPtr = nodeUnitedError.RawPtr();
	dqCorrectNodes.ResizeArrayOrException(SparseNodesNum);
	DeviceArray<float4> beforeNodes;
	beforeNodes.create(SparseNodesNum);
	CHECKCUDA(cudaMemcpyAsync(beforeNodes.ptr(), liveNodesCoorPtr, sizeof(float4) * SparseNodesNum, cudaMemcpyDeviceToDevice, stream));
	DeviceArray<float4> correctedVertex;
	correctedVertex.create(DenseSurfelsNum);
	CHECKCUDA(cudaDeviceSynchronize());
	// 1.node���ĸ�ƽ���õ��Ը���node���ԣ��ҵ���ӦCamera�Ĺ���Map
	// 2.������node��SE3
	SetComputeOpticalGuideNodesSe3Input(opticalMap, observation, InitialCameraSE3Array, world2camera, intrinsicArray, devicesCount, guidedNodesInput);
	dim3 block_1(64);
	dim3 grid_1(divUp(SparseNodesNum, block_1.x));
	device::computeOpticalFlowGuidedNodesSE3 << <grid_1, block_1, 0, stream >> > (guidedNodesInput, liveNodesCoorPtr, nodeUnitedErrorPtr, SparseNodesNum, mapCols, mapRows, devicesCount, dqCorrectNodes.Ptr());
	CHECKCUDA(cudaDeviceSynchronize());


	// 3.������node��live���ܵ����У������ԭʼNodeSE3����У����nodeArray��С + DenseSurfel��С�ĺ˺�����
	SetOpticalGuideInput(warpField, geometry, devicesCount, updatedGeometryIndex, correctInput);
	dim3 block_2(128);
	dim3 grid_2(divUp(SparseNodesNum + DenseSurfelsNum, block_2.x));
	device::CorrectWarpVertexAndNodeKernel << <grid_2, block_2, 0, stream >> > (correctInput, SparseNodesNum + DenseSurfelsNum, DenseSurfelsNum, devicesCount, beforeNodes.ptr(), correctedVertex.ptr());

	DeviceArray<DualQuaternion> correctedNodeSe3 = warpField.BindOpticalFlowGuideInfo().nodesSE3;
	DeviceArray<DualQuaternion> currentSolvedSe3 = iterationData.CurrentWarpFieldSe3Interface();
	CHECKCUDA(cudaMemcpyAsync(currentSolvedSe3.ptr(), correctedNodeSe3.ptr(), sizeof(DualQuaternion) * SparseNodesNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaDeviceSynchronize());

	DeviceArrayView<float4> liveNodesView = DeviceArrayView<float4>(warpField.BindOpticalFlowGuideInfo().liveNodesCoordinate);
	DeviceArrayView<float4> beforeNodesView = DeviceArrayView<float4>(beforeNodes);
	DeviceArrayView<float4> correctedVertexView = DeviceArrayView<float4>(correctedVertex);
	//if ((frameIdx <= 10) || (frameIdx >= 15 && frameIdx <= 25) || (frameIdx >= 45 && frameIdx <= 50)) {
	//	// ��ɫ��У��ǰ����ɫ��У����
	//	Visualizer::DrawMatchedReferenceAndObseveredPointsPair(beforeNodesView, liveNodesView);
	//	//// �۲�֡ �� ������liveNodes�Ƚ�
	//	//Visualizer::DrawMatchedCloudPair(observation.vertexConfidenceMap[0], correctedVertexView, toEigen(InitialCameraSE3Array[0] * world2Camera[0]));
	//}

	beforeNodes.release();
	correctedVertex.release();

}


void SparseSurfelFusion::NonRigidSolver::CorrectLargeErrorNode(const unsigned int frameIdx, const DeviceArrayView2D<mat34>* correspondenceSe3Map, const DeviceArrayView2D<unsigned char>* markValidSe3Map, CameraObservation observation, DeviceArrayView<float> nodeUnitedError, const mat34* InitialCameraSE3Array, const mat34* world2camera, const Intrinsic* intrinsicArray, WarpField& warpField, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int updatedGeometryIndex, cudaStream_t stream)
{
	device::CorrectInput correctInput;
	const unsigned int SparseNodesNum = warpField.BindOpticalFlowGuideInfo().nodesSE3.size();
	const unsigned int DenseSurfelsNum = geometry[0][updatedGeometryIndex]->BindOpticalFlowGuideInfo().denseLiveSurfelsVertex.size();
	const unsigned int mapCols = correspondenceSe3Map[0].Cols();
	const unsigned int mapRows = correspondenceSe3Map[0].Rows();
	const float4* liveNodesCoorPtr = warpField.BindOpticalFlowGuideInfo().liveNodesCoordinate.ptr();
	const float* nodeUnitedErrorPtr = nodeUnitedError.RawPtr();
	dqCorrectNodes.ResizeArrayOrException(SparseNodesNum);
	DeviceArray<float4> beforeNodes;
	beforeNodes.create(SparseNodesNum);
	CHECKCUDA(cudaMemcpyAsync(beforeNodes.ptr(), liveNodesCoorPtr, sizeof(float4) * SparseNodesNum, cudaMemcpyDeviceToDevice, stream));
	DeviceArray<float4> correctedVertex;
	correctedVertex.create(DenseSurfelsNum);
	CHECKCUDA(cudaDeviceSynchronize());
	// 1.node���ĸ�ƽ���õ��Ը���node���ԣ��ҵ���ӦCamera�Ĺ���Map
	// 2.������node��SE3
	SetComputeCorrespondenceGuideNodeSe3Input(correspondenceSe3Map, markValidSe3Map, observation, InitialCameraSE3Array, world2camera, intrinsicArray, devicesCount, guidedNodesInput);
	dim3 block_1(64);
	dim3 grid_1(divUp(SparseNodesNum, block_1.x));
	device::computeCorrespondenceGuidedNodesSE3 << <grid_1, block_1, 0, stream >> > (guidedNodesInput, liveNodesCoorPtr, nodeUnitedErrorPtr, SparseNodesNum, mapCols, mapRows, devicesCount, dqCorrectNodes.Ptr());
	CHECKCUDA(cudaDeviceSynchronize());


	// 3.������node��live���ܵ����У������ԭʼNodeSE3����У����nodeArray��С + DenseSurfel��С�ĺ˺�����
	SetOpticalGuideInput(warpField, geometry, devicesCount, updatedGeometryIndex, correctInput);
	dim3 block_2(128);
	dim3 grid_2(divUp(SparseNodesNum + DenseSurfelsNum, block_2.x));
	device::CorrectWarpVertexAndNodeKernel << <grid_2, block_2, 0, stream >> > (correctInput, SparseNodesNum + DenseSurfelsNum, DenseSurfelsNum, devicesCount, beforeNodes.ptr(), correctedVertex.ptr());

	DeviceArray<DualQuaternion> correctedNodeSe3 = warpField.BindOpticalFlowGuideInfo().nodesSE3;
	DeviceArray<DualQuaternion> currentSolvedSe3 = iterationData.CurrentWarpFieldSe3Interface();
	CHECKCUDA(cudaMemcpyAsync(currentSolvedSe3.ptr(), correctedNodeSe3.ptr(), sizeof(DualQuaternion) * SparseNodesNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaDeviceSynchronize());

	DeviceArrayView<float4> liveNodesView = DeviceArrayView<float4>(warpField.BindOpticalFlowGuideInfo().liveNodesCoordinate);
	DeviceArrayView<float4> beforeNodesView = DeviceArrayView<float4>(beforeNodes);
	DeviceArrayView<float4> correctedVertexView = DeviceArrayView<float4>(correctedVertex);
	//if ((frameIdx <= 10) || (frameIdx >= 15 && frameIdx <= 25) || (frameIdx >= 45 && frameIdx <= 50)) {
	//	// ��ɫ��У��ǰ����ɫ��У����
	//	Visualizer::DrawMatchedReferenceAndObseveredPointsPair(beforeNodesView, liveNodesView);
	//	// �۲�֡ �� ������liveNodes�Ƚ�
	//	Visualizer::DrawMatchedCloudPair(observation.vertexConfidenceMap[0], correctedVertexView, toEigen(InitialCameraSE3Array[0]));
	//}

	beforeNodes.release();
	correctedVertex.release();

}