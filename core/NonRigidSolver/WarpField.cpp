/*****************************************************************//**
 * \file   WarpField.cpp
 * \brief  构建、更新扭曲场节点，执行向前或者向后扭曲节点
 *
 * \author LUO
 * \date   March 8th 2024
 *********************************************************************/
#include "WarpField.h"

SparseSurfelFusion::WarpField::WarpField()
{
	voxelSubsampler = std::make_shared<VoxelSubsampler>();
	allocateBuffer(Constants::maxNodesNum);
}

SparseSurfelFusion::WarpField::~WarpField()
{
	releaseBuffer();
}

void SparseSurfelFusion::WarpField::ResizeDeviceArrayToNodeSize(const unsigned int nodesNum)
{
	nodesKNN.ResizeArrayOrException(nodesNum);
	nodesKNNWeight.ResizeArrayOrException(nodesNum);
	liveNodesCoordinate.ResizeArrayOrException(nodesNum);
}

unsigned SparseSurfelFusion::WarpField::CheckAndGetNodeSize() const
{
	const auto num_nodes = canonicalNodesCoordinate.DeviceArraySize();
	FUNCTION_CHECK(num_nodes == canonicalNodesCoordinate.HostArraySize());
	FUNCTION_CHECK(num_nodes == nodesSE3.HostArraySize());
	FUNCTION_CHECK(num_nodes == nodesSE3.DeviceArraySize());
	FUNCTION_CHECK(num_nodes == nodesKNN.ArraySize());
	FUNCTION_CHECK(num_nodes == nodesKNNWeight.ArraySize());
	FUNCTION_CHECK(num_nodes == liveNodesCoordinate.ArraySize());
	return num_nodes;
}


void SparseSurfelFusion::WarpField::allocateBuffer(size_t maxNodesNum)
{
	// CPU和GPU上的内存
	canonicalNodesCoordinate.AllocateBuffer(maxNodesNum);
	nodesSE3.AllocateBuffer(maxNodesNum);
	nodesKNN.AllocateBuffer(maxNodesNum);
	nodesKNNWeight.AllocateBuffer(maxNodesNum);

	// GPU缓冲区
	liveNodesCoordinate.AllocateBuffer(maxNodesNum);
	nodesGraph.AllocateBuffer(maxNodesNum * Constants::nodesGraphNeigboursNum);

	// 给下采样器算法分配GPU运行内存
	voxelSubsampler->AllocateBuffer(Constants::maxSurfelsNum);
	candidateNodes.AllocateBuffer(Constants::maxSubsampledSparsePointsNum);
}

void SparseSurfelFusion::WarpField::releaseBuffer()
{
	liveNodesCoordinate.ReleaseBuffer();
	nodesGraph.ReleaseBuffer();
	voxelSubsampler->ReleaseBuffer();
}

SparseSurfelFusion::WarpField::SkinnerInput SparseSurfelFusion::WarpField::BindWarpFieldSkinnerInfo()
{
	SkinnerInput skinnerInput;
	skinnerInput.canonicalNodesCoordinate = canonicalNodesCoordinate.DeviceArrayReadOnly();
	skinnerInput.sparseNodesKnn = nodesKNN.ArrayHandle();
	skinnerInput.sparseNodesKnnWeight = nodesKNNWeight.ArrayHandle();
	return skinnerInput;
}

SparseSurfelFusion::WarpField::NonRigidSolverInput SparseSurfelFusion::WarpField::BindNonRigidSolverInfo() const
{
	NonRigidSolverInput solverInput;
	solverInput.nodesSE3 = nodesSE3.DeviceArrayReadOnly();
	solverInput.canonicalNodesCoordinate = canonicalNodesCoordinate.DeviceArrayReadOnly();
	solverInput.nodesGraph = nodesGraph.ArrayReadOnly();
	return solverInput;
}

SparseSurfelFusion::WarpField::OpticalFlowGuideInput SparseSurfelFusion::WarpField::BindOpticalFlowGuideInfo()
{
	OpticalFlowGuideInput opeticalGuideInput;
	opeticalGuideInput.nodesSE3 = nodesSE3.DeviceArray();
	opeticalGuideInput.nodesKNN = nodesKNN.Array();
	opeticalGuideInput.nodesKNNWeight = nodesKNNWeight.Array();
	opeticalGuideInput.liveNodesCoordinate = liveNodesCoordinate.Array();
	return opeticalGuideInput;
}

SparseSurfelFusion::WarpField::LiveGeometryUpdaterInput SparseSurfelFusion::WarpField::GeometryUpdaterAccess() const
{
	//这些全是0号空间中的
	LiveGeometryUpdaterInput geometry_input;
	geometry_input.live_node_coords = liveNodesCoordinate.ArrayView();
	geometry_input.reference_node_coords = canonicalNodesCoordinate.DeviceArrayReadOnly();
	geometry_input.node_se3 = nodesSE3.DeviceArrayReadOnly();
	return geometry_input;
}

SparseSurfelFusion::DeviceArrayView<float4> SparseSurfelFusion::WarpField::getCanonicalNodesCoordinate()
{
	return canonicalNodesCoordinate.DeviceArrayReadOnly();
}

SparseSurfelFusion::DeviceArrayView<float4> SparseSurfelFusion::WarpField::getLiveNodesCoordinate()
{
	return liveNodesCoordinate.ArrayView();
}

void SparseSurfelFusion::WarpField::UpdateHostDeviceNodeSE3NoSync(DeviceArrayView<DualQuaternion> node_se3, cudaStream_t stream)
{
	FUNCTION_CHECK(node_se3.Size() == nodesSE3.DeviceArraySize());
	CHECKCUDA(cudaMemcpyAsync(
		nodesSE3.DevicePtr(),
		node_se3.RawPtr(),
		sizeof(DualQuaternion) * node_se3.Size(),
		cudaMemcpyDeviceToDevice,
		stream
	));

	//Sync to host
	nodesSE3.SynchronizeToHost(stream, false);
}

void SparseSurfelFusion::WarpField::InitializeCanonicalNodesAndSE3FromMergedVertices(DeviceArrayView<float4>& canonicalVertices, DeviceArrayView<float4>& colorViewTime, cudaStream_t stream)
{
	SurfelsProcessor processor;
	processor.PerformVerticesSubsamplingSync(voxelSubsampler, canonicalVertices, colorViewTime, candidateNodes, stream);
	const std::vector<float4> HostCandidates = candidateNodes.HostArray();
	InitializeCanonicalNodesAndSE3FromCandidates(HostCandidates, stream);
}

void SparseSurfelFusion::WarpField::InitializeCanonicalNodesAndSE3FromCandidates(const std::vector<float4>& nodesCandidates, cudaStream_t stream)
{
	// 仅仅清空Array，不管Buffer中的其他数据
	canonicalNodesCoordinate.ClearArray();
	nodesSE3.ClearArray();

	const float sampleDistanceSquare = (Constants::NodeSamplingRadius) * (Constants::NodeSamplingRadius); //r = 0.85cm * 0.85cm
	std::vector<float4>& hostFinalNodes = canonicalNodesCoordinate.HostArray();	// 暴露指针，最终的节点数据
	std::vector<DualQuaternion>& hostFinalSE3 = nodesSE3.HostArray();			// 暴露指针，最终节点的初始化的SE3

	for (int i = 0; i < nodesCandidates.size(); i++) {
		float4 point = make_float4(nodesCandidates[i].x, nodesCandidates[i].y, nodesCandidates[i].z, nodesCandidates[i].w);
		
		bool isNode = true;
		for (int j = 0; j < hostFinalNodes.size(); j++) {
			if (squared_norm_xyz(point - hostFinalNodes[j]) < sampleDistanceSquare) {
				isNode = false;
				break;
			}
		}
		// 初始化节点位置和SE3
		if (isNode) {
			hostFinalNodes.emplace_back(make_float4(point.x, point.y, point.z, nodesCandidates[i].w));
			hostFinalSE3.emplace_back(DualQuaternion(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0)));
		}
	}

	// 将节点同步到GPU上
	canonicalNodesCoordinate.SynchronizeToDevice(stream);
	nodesSE3.SynchronizeToDevice(stream);
	const size_t nodesNum = canonicalNodesCoordinate.HostArraySize();	// 节点数量
	//std::cout << "Canonical域节点数量：" << nodesNum << " 个" << std::endl;// 打印运行时间
	ResizeDeviceArrayToNodeSize(nodesNum);
	CHECKCUDA(cudaMemcpyAsync(liveNodesCoordinate.Ptr(), canonicalNodesCoordinate.DevicePtr(), sizeof(float4) * nodesNum, cudaMemcpyDeviceToDevice, stream));
}
