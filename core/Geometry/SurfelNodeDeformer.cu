#include "SurfelNodeDeformer.h"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		__device__ unsigned int devicesCount = MAX_CAMERA_COUNT;

		__global__ void forwardWarpVertexAndNodeKernel(
			const DeviceArrayView<float4> canonical_vertex_confid,
			const DeviceArrayView<float4> canonical_normal_radius,
			const ushort4* vertex_knn_array, 
			const float4* vertex_knn_weight,
			const DeviceArrayView<float4> canonical_nodes_coordinate,
			const ushort4* node_knn_array, 
			const float4* node_knn_weight,
			const DualQuaternion* warp_field,
			//mat34  SE3,
			//Output array, shall be size correct
			device::SurfelGeometryInterface geometry,
			float4* live_node_coordinate
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			ushort4 knn;
			float4 weight;
			float4 vertex = make_float4(0, 0, 0, 0);
			float4 normal = make_float4(0, 0, 0, 0);
			if (idx < canonical_vertex_confid.Size()) {
				knn = vertex_knn_array[idx];
				weight = vertex_knn_weight[idx];
				vertex = canonical_vertex_confid[idx];
				normal = canonical_normal_radius[idx];
			}
			else if (idx >= canonical_vertex_confid.Size() && idx < canonical_vertex_confid.Size() + canonical_nodes_coordinate.Size()) {
				const int offset = idx - canonical_vertex_confid.Size();
				knn = node_knn_array[offset];
				weight = node_knn_weight[offset];
				vertex = canonical_nodes_coordinate[offset];
			}

			//Do warpping
			DualQuaternion dqAverage = averageDualQuaternion(warp_field, knn, weight);
			const mat34 se3 = dqAverage.se3_matrix();
			float3 v3 = make_float3(vertex.x, vertex.y, vertex.z);
			float3 n3 = make_float3(normal.x, normal.y, normal.z);
			v3 = se3.rot * v3 + se3.trans;
			n3 = se3.rot * n3;
			vertex = make_float4(v3.x, v3.y, v3.z, vertex.w);
			normal = make_float4(n3.x, n3.y, n3.z, normal.w);

			if (idx < canonical_vertex_confid.Size()) {
				for (int i = 0; i < device::devicesCount; i++) {
					geometry.liveVertexArray[i][idx] = vertex;
					geometry.liveNormalArray[i][idx] = normal;
				}
			}
			else if (idx >= canonical_vertex_confid.Size() && idx < canonical_vertex_confid.Size() + canonical_nodes_coordinate.Size()) {
				const int offset = idx - canonical_vertex_confid.Size();
				live_node_coordinate[offset] = vertex;
			}
		}


		__global__ void inverseWarpVertexNormalKernel(
			const DeviceArrayView<float4> live_vertex_confid_array,
			const float4* live_normal_radius_array,
			const ushort4* vertex_knn_array,
			const float4* vertex_knn_weight,
			const DualQuaternion* device_warp_field,
			const mat34* correct_se3,
			//output
			device::SurfelGeometryInterface geometry
		) {
			const int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < live_vertex_confid_array.Size()) {
				const float4 live_vertex_confid = live_vertex_confid_array[idx];
				const float4 live_normal_radius = live_normal_radius_array[idx];
				const ushort4 knn = vertex_knn_array[idx];
				const float4 knn_weight = vertex_knn_weight[idx];
				DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
				mat34 se3 = dq_average.se3_matrix();
				float3 vertex = make_float3(live_vertex_confid.x, live_vertex_confid.y, live_vertex_confid.z);
				float3 normal = make_float3(live_normal_radius.x, live_normal_radius.y, live_normal_radius.z);
				//Apply the inversed warping without construction of the matrix
				vertex = se3.apply_inversed_se3(vertex);
				vertex = correct_se3[idx].inverse().rot * vertex + correct_se3[idx].inverse().trans;
				normal = se3.rot.transpose_dot(normal);
				normal = correct_se3[idx].inverse().rot * normal;
				for (int i = 0; i < device::devicesCount; i++) {
					geometry.referenceVertexArray[i][idx] = make_float4(vertex.x, vertex.y, vertex.z, live_vertex_confid.w);
					geometry.referenceNormalArray[i][idx] = make_float4(normal.x, normal.y, normal.z, live_normal_radius.w);
				}
			}
		}

		__global__ void inverseWarpVertexNormalKernel(
			const DeviceArrayView<float4> live_vertex_confid_array,
			const float4* live_normal_radius_array,
			const ushort4* vertex_knn_array,
			const float4* vertex_knn_weight,
			const DualQuaternion* device_warp_field,
			//output
			device::SurfelGeometryInterface geometry
		) {
			const int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < live_vertex_confid_array.Size()) {
				const float4 live_vertex_confid = live_vertex_confid_array[idx];
				const float4 live_normal_radius = live_normal_radius_array[idx];
				const ushort4 knn = vertex_knn_array[idx];
				const float4 knn_weight = vertex_knn_weight[idx];
				DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
				mat34 se3 = dq_average.se3_matrix();
				float3 vertex = make_float3(live_vertex_confid.x, live_vertex_confid.y, live_vertex_confid.z);
				float3 normal = make_float3(live_normal_radius.x, live_normal_radius.y, live_normal_radius.z);
				//Apply the inversed warping without construction of the matrix
				vertex = se3.apply_inversed_se3(vertex);
				normal = se3.rot.transpose_dot(normal);
				for (int i = 0; i < device::devicesCount; i++) {
					geometry.referenceVertexArray[i][idx] = make_float4(vertex.x, vertex.y, vertex.z, live_vertex_confid.w);
					geometry.referenceNormalArray[i][idx] = make_float4(normal.x, normal.y, normal.z, live_normal_radius.w);
				}
			}
		}
	}
}

void SparseSurfelFusion::SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
	bool showRender,
	WarpField& warp_field,
	SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
	const unsigned int devicesCount,
	const unsigned int updatedGeometryIndex,
	const DeviceArrayView<DualQuaternion>& node_se3,
	cudaStream_t stream
) {
	for (int i = 0; i < devicesCount; i++) {
		CheckSurfelGeometySize(*(geometry[i][updatedGeometryIndex]));
	}

	//The node se3 should have the same size
	FUNCTION_CHECK(node_se3.Size() == warp_field.nodesSE3.DeviceArraySize());

	//Update the size of live nodes
	warp_field.liveNodesCoordinate.ResizeArrayOrException(warp_field.canonicalNodesCoordinate.DeviceArraySize());
	FUNCTION_CHECK_EQ(warp_field.nodesKNN.ArraySize(), warp_field.liveNodesCoordinate.ArraySize());
	FUNCTION_CHECK_EQ(warp_field.nodesKNNWeight.ArraySize(), warp_field.liveNodesCoordinate.ArraySize());

	// geometry[0],geometry[1],geometry[2]的canonical域中的数据都是完全一致的
	SurfelGeometry::Ptr& CanonicalSurfelGeometry = geometry[0][updatedGeometryIndex];
	// 加载数据
	const DeviceArrayView<float4> reference_vertex = CanonicalSurfelGeometry->CanonicalVertexConfidence.ArrayView();
	const DeviceArrayView<float4> reference_normal = CanonicalSurfelGeometry->CanonicalNormalRadius.ArrayView();

	const DeviceArrayView<ushort4> knn = CanonicalSurfelGeometry->surfelKNN.ArrayView();
	const DeviceArrayView<float4> knn_weight = CanonicalSurfelGeometry->surfelKNNWeight.ArrayView();

	device::SurfelGeometryInterface geometryInterface;


	// 暂存
	DeviceBufferArray<float4> beforeDeformedVertex;		// 形变前的点
	beforeDeformedVertex.AllocateBuffer(geometry[0][updatedGeometryIndex]->CanonicalVertexConfidence.ArrayView().Size());
	CHECKCUDA(cudaMemcpyAsync(beforeDeformedVertex.Array().ptr(), geometry[0][updatedGeometryIndex]->CanonicalVertexConfidence.ArrayView().RawPtr(), sizeof(float4) * geometry[0][updatedGeometryIndex]->CanonicalVertexConfidence.ArrayView().Size(), cudaMemcpyDeviceToDevice, stream));
	beforeDeformedVertex.ResizeArrayOrException(geometry[0][updatedGeometryIndex]->CanonicalVertexConfidence.ArrayView().Size());

	DeviceBufferArray<float4> beforeDeformNodes;		// 形变前的节点
	beforeDeformNodes.AllocateBuffer(warp_field.canonicalNodesCoordinate.DeviceArrayReadOnly().Size());
	CHECKCUDA(cudaMemcpyAsync(beforeDeformNodes.Array().ptr(), warp_field.canonicalNodesCoordinate.DeviceArrayReadOnly().RawPtr(), sizeof(float4) * warp_field.canonicalNodesCoordinate.DeviceArrayReadOnly().Size(), cudaMemcpyDeviceToDevice, stream));
	beforeDeformNodes.ResizeArrayOrException(warp_field.canonicalNodesCoordinate.DeviceArrayReadOnly().Size());

	//注意，这里是.ArraySlice();
	for (int i = 0; i < devicesCount; i++) {
		geometryInterface.liveVertexArray[i] = geometry[i][updatedGeometryIndex]->LiveVertexConfidence.ArrayHandle().RawPtr();
		geometryInterface.liveNormalArray[i] = geometry[i][updatedGeometryIndex]->LiveNormalRadius.ArrayHandle().RawPtr();
	}

	dim3 block(256);// 这里是canonical稠密顶点的大小加上canonical节点大小
	dim3 grid(divUp(reference_vertex.Size() + warp_field.canonicalNodesCoordinate.DeviceArraySize(), block.x));
	device::forwardWarpVertexAndNodeKernel << <grid, block, 0, stream >> > (
		reference_vertex,
		reference_normal,
		knn,
		knn_weight,
		//For nodes
		warp_field.canonicalNodesCoordinate.DeviceArrayReadOnly(),
		warp_field.nodesKNN.Ptr(),
		warp_field.nodesKNNWeight.Ptr(),
		node_se3.RawPtr(),
		//SE3[1],
		//Output
		geometryInterface,
		warp_field.liveNodesCoordinate.Ptr()
	);

	if (showRender)
	// 红色是before，绿色是deformed后的
	Visualizer::DrawMatchedReferenceAndObseveredPointsPair(beforeDeformNodes.ArrayView(), warp_field.liveNodesCoordinate.ArrayView());
	//Visualizer::DrawMatchedReferenceAndObseveredPointsPair(beforeDeformedVertex.ArrayView(), geometry[0][updatedGeometryIndex]->LiveVertexConfidence.ArrayView());

	beforeDeformedVertex.ReleaseBuffer();
	beforeDeformNodes.ReleaseBuffer();
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}

void SparseSurfelFusion::SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
	bool showRender,
	WarpField& warp_field,
	SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
	const unsigned int devicesCount,
	const unsigned int updatedGeometryIndex,
	cudaStream_t stream
) {
	ForwardWarpSurfelsAndNodes(
		showRender,
		warp_field,
		geometry,
		devicesCount,
		updatedGeometryIndex,
		warp_field.nodesSE3.DeviceArrayReadOnly(),
		stream
	);
}


void SparseSurfelFusion::SurfelNodeDeformer::InverseWarpSurfels(
	SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
	const unsigned int devicesCount,
	const unsigned int updatedGeometryIndex,
	const DeviceArrayView<DualQuaternion>& node_se3,
	const DeviceArrayView<mat34>& correct_se3,
	cudaStream_t stream
) {

	for (int i = 0; i < devicesCount; i++) {
		CheckSurfelGeometySize(*geometry[i][updatedGeometryIndex]);
	}

	SurfelGeometry::Ptr& LiveSurfelGeometry = geometry[0][updatedGeometryIndex];

	//Load the data
	const DeviceArrayView<float4> live_vertex = LiveSurfelGeometry->LiveVertexConfidence.ArrayView();
	const DeviceArrayView<float4> live_normal = LiveSurfelGeometry->LiveNormalRadius.ArrayView();
	const DeviceArrayView<ushort4> knn = LiveSurfelGeometry->surfelKNN.ArrayView();
	const DeviceArrayView<float4> knn_weight = LiveSurfelGeometry->surfelKNNWeight.ArrayView();
	//注意是ArrayHandle

	device::SurfelGeometryInterface geometryInterface;

	const unsigned int referenceVertexArraySize = LiveSurfelGeometry->CanonicalVertexConfidence.ArrayHandle().Size();
	for (int i = 0; i < devicesCount; i++) {
		geometryInterface.referenceVertexArray[i] = geometry[i][updatedGeometryIndex]->CanonicalVertexConfidence.ArrayHandle();
		geometryInterface.referenceNormalArray[i] = geometry[i][updatedGeometryIndex]->CanonicalNormalRadius.ArrayHandle();
	}

	//Do warping
	dim3 block(256);
	dim3 grid(divUp(referenceVertexArraySize, block.x));
	device::inverseWarpVertexNormalKernel << <grid, block, 0, stream >> > (
		live_vertex,
		live_normal,
		knn, knn_weight,
		node_se3.RawPtr(),
		correct_se3.RawPtr(),
		//output
		geometryInterface
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}
void SparseSurfelFusion::SurfelNodeDeformer::InverseWarpSurfels(SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int devicesCount, const unsigned int updatedGeometryIndex, const DeviceArrayView<DualQuaternion>& node_se3, cudaStream_t stream)
{
	for (int i = 0; i < devicesCount; i++) {
		CheckSurfelGeometySize(*geometry[i][updatedGeometryIndex]);
	}

	SurfelGeometry::Ptr& LiveSurfelGeometry = geometry[0][updatedGeometryIndex];

	//Load the data
	const DeviceArrayView<float4> live_vertex = LiveSurfelGeometry->LiveVertexConfidence.ArrayView();
	const DeviceArrayView<float4> live_normal = LiveSurfelGeometry->LiveNormalRadius.ArrayView();
	const DeviceArrayView<ushort4> knn = LiveSurfelGeometry->surfelKNN.ArrayView();
	const DeviceArrayView<float4> knn_weight = LiveSurfelGeometry->surfelKNNWeight.ArrayView();
	//注意是ArrayHandle

	device::SurfelGeometryInterface geometryInterface;

	const unsigned int referenceVertexArraySize = LiveSurfelGeometry->CanonicalVertexConfidence.ArrayHandle().Size();
	for (int i = 0; i < devicesCount; i++) {
		geometryInterface.referenceVertexArray[i] = geometry[i][updatedGeometryIndex]->CanonicalVertexConfidence.ArrayHandle();
		geometryInterface.referenceNormalArray[i] = geometry[i][updatedGeometryIndex]->CanonicalNormalRadius.ArrayHandle();
	}

	//Do warping
	dim3 block(256);
	dim3 grid(divUp(referenceVertexArraySize, block.x));
	device::inverseWarpVertexNormalKernel << <grid, block, 0, stream >> > (
		live_vertex,
		live_normal,
		knn, knn_weight,
		node_se3.RawPtr(),
		//output
		geometryInterface
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}
void SparseSurfelFusion::SurfelNodeDeformer::InverseWarpSurfels(
	const WarpField& warp_field,
	SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
	const unsigned int devicesCount,
	const unsigned int updatedGeometryIndex,
	const DeviceArrayView<DualQuaternion>& node_se3, 
	const DeviceArrayView<mat34>& correct_se3,
	cudaStream_t stream
) {
	//The node se3 should have the same size
	FUNCTION_CHECK(node_se3.Size() == warp_field.nodesSE3.DeviceArraySize());
	InverseWarpSurfels(geometry, devicesCount, updatedGeometryIndex, node_se3, correct_se3, stream);
}

void SparseSurfelFusion::SurfelNodeDeformer::InverseWarpSurfels(const WarpField& warp_field, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], const unsigned int devicesCount, const unsigned int updatedGeometryIndex, const DeviceArrayView<DualQuaternion>& node_se3, cudaStream_t stream)
{
	//The node se3 should have the same size
	FUNCTION_CHECK(node_se3.Size() == warp_field.nodesSE3.DeviceArraySize());
	InverseWarpSurfels(geometry, devicesCount, updatedGeometryIndex, node_se3, stream);
}

void SparseSurfelFusion::SurfelNodeDeformer::InverseWarpSurfels(
	const WarpField& warp_field,
	SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2],
	cudaStream_t stream
) {
	//Check the size
	//InverseWarpSurfels(
	//	warp_field,
	//	geometry,
	//	warp_field.nodesSE3.DeviceArrayReadOnly(),
	//	stream
	//);
}

void SparseSurfelFusion::SurfelNodeDeformer::CheckSurfelGeometySize(const SurfelGeometry& geometry) {
	const auto num_surfels = geometry.ValidSurfelsNum();
	FUNCTION_CHECK(geometry.CanonicalVertexConfidence.ArraySize() == num_surfels);
	FUNCTION_CHECK(geometry.CanonicalNormalRadius.ArraySize() == num_surfels);
	FUNCTION_CHECK(geometry.surfelKNN.ArraySize() == num_surfels);
	FUNCTION_CHECK(geometry.surfelKNNWeight.ArraySize() == num_surfels);
	FUNCTION_CHECK(geometry.LiveVertexConfidence.ArraySize() == num_surfels);
	FUNCTION_CHECK(geometry.LiveNormalRadius.ArraySize() == num_surfels);
}
