#include "CrossViewCorrespondenceHandler.h"

__device__ ushort2 SparseSurfelFusion::device::ValidCrossGeometryPixelInWindow(cudaTextureObject_t indexMap, unsigned short center_x, unsigned short center_y)
{
	ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);

	// 在窗口中搜索有效值
	for (auto y = center_y - CrossGeometrySearchHalfSize; y <= center_y + CrossGeometrySearchHalfSize; y++) {
		for (auto x = center_x - CrossGeometrySearchHalfSize; x <= center_x + CrossGeometrySearchHalfSize; x++) {
			if (tex2D<unsigned>(indexMap, x, y) != 0xFFFFFFFF) {
				valid_pixel.x = x;
				valid_pixel.y = y;
				break;
			}
		}
	}

	// 如果窗口中存在有效值，如果窗口中心无效，那么就选择上面的有效值，如果窗口中心有效就更倾向于窗口中心
	if (tex2D<unsigned>(indexMap, center_x, center_y) != 0xFFFFFFFF) {
		valid_pixel.x = center_x;
		valid_pixel.y = center_y;
	}

	//Return it
	return valid_pixel;
}

__device__ ushort2 SparseSurfelFusion::device::ValidCrossGeometryPixelInWindow(cudaTextureObject_t indexMap, cudaTextureObject_t liveVertexMap, float4 observedVertex, unsigned short center_x, unsigned short center_y)
{
	ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);
	float minSquaredDis = 1e6f;
	// 搜索窗口中与Observed最近的Geomtry点
	for (int y = center_y - CrossGeometrySearchHalfSize; y <= center_y + CrossGeometrySearchHalfSize; y += CrossGeometrySearchStep) {
		for (int x = center_x - CrossGeometrySearchHalfSize; x <= center_x + CrossGeometrySearchHalfSize; x += CrossGeometrySearchStep) {
			if (tex2D<unsigned>(indexMap, x, y) != 0xFFFFFFFF) {
				float4 liveVertex = tex2D<float4>(liveVertexMap, x, y);
				float squaredDis = squared_distance(observedVertex, liveVertex);
				if (squaredDis < minSquaredDis) {
					valid_pixel.x = x;
					valid_pixel.y = y;
					minSquaredDis = squaredDis;
				}
			}
		}
	}

	return valid_pixel;
}

__device__ ushort4 SparseSurfelFusion::device::ValidCrossGeometryPixelInWindow(const GeometryMapCrossViewCorrespondenceInterface& geoemtry, const ObservedCrossViewCorrespondenceInterface& observed, const CrossViewCorrPairs& crossPairsCenter, const float3& observed_1, const float3& observed_2, const float& observed_diff_xyz)
{
	ushort4 valid_pixel = make_ushort4(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
	bool isGeometry_1_valid = false, isGeometry_2_valid = false;
	// 光流上一帧
	ushort view_1 = crossPairsCenter.PixelViews.x;
	ushort view_2 = crossPairsCenter.PixelViews.y;
	ushort2 center_1 = make_ushort2(observed.corrMap[view_1](crossPairsCenter.PixelPairs.y, crossPairsCenter.PixelPairs.x).x, observed.corrMap[view_1](crossPairsCenter.PixelPairs.y, crossPairsCenter.PixelPairs.x).y);
	ushort2 center_2 = make_ushort2(observed.corrMap[view_2](crossPairsCenter.PixelPairs.w, crossPairsCenter.PixelPairs.z).x, observed.corrMap[view_2](crossPairsCenter.PixelPairs.w, crossPairsCenter.PixelPairs.z).y);
	float4 canVertex_1, canVertex_2;

	for (int i = -CrossGeometrySearchHalfSize; i <= CrossGeometrySearchHalfSize; i += CrossGeometrySearchStep) {
		for (int j = -CrossGeometrySearchHalfSize; j <= CrossGeometrySearchHalfSize; j += CrossGeometrySearchStep) {
			if (isGeometry_1_valid && isGeometry_2_valid) break;
			ushort2 pixel_1 = make_ushort2(center_1.x + i, center_1.y + j);
			ushort2 pixel_2 = make_ushort2(center_2.x + i, center_2.y + j);
			if (tex2D<unsigned>(geoemtry.indexMap[view_1], pixel_1.x, pixel_1.y) != 0xFFFFFFFF) {
				canVertex_1 = tex2D<float4>(geoemtry.canonicalVertexMap[view_1], pixel_1.x, pixel_1.y);
				valid_pixel.x = pixel_1.x;
				valid_pixel.y = pixel_1.y;
				isGeometry_1_valid = true;
			}
			if (tex2D<unsigned>(geoemtry.indexMap[view_2], pixel_2.x, pixel_2.y) != 0xFFFFFFFF) {
				canVertex_2 = tex2D<float4>(geoemtry.canonicalVertexMap[view_2], pixel_2.x, pixel_2.y);
				valid_pixel.z = pixel_2.x;
				valid_pixel.w = pixel_2.y;
				isGeometry_2_valid = true;
			}
		}
	}

	if (tex2D<unsigned>(geoemtry.indexMap[view_1], center_1.x, center_1.y) != 0xFFFFFFFF &&
		tex2D<unsigned>(geoemtry.indexMap[view_2], center_2.x, center_2.y) != 0xFFFFFFFF) {
		canVertex_1 = tex2D<float4>(geoemtry.canonicalVertexMap[view_1], center_1.x, center_1.y);
		canVertex_2 = tex2D<float4>(geoemtry.canonicalVertexMap[view_2], center_2.x, center_2.y);
		valid_pixel = make_ushort4(center_1.x, center_1.y, center_2.x, center_2.y);
	}
	if (isGeometry_1_valid && isGeometry_2_valid) {
		// 统一坐标系
		float3 canVertexWorld_1 = geoemtry.initialCameraSE3[view_1].rot * canVertex_1 + geoemtry.initialCameraSE3[view_1].trans;
		float3 canVertexWorld_2 = geoemtry.initialCameraSE3[view_2].rot * canVertex_2 + geoemtry.initialCameraSE3[view_2].trans;

		float diff_xyz = fabsf_diff_xyz(canVertexWorld_1, canVertexWorld_2);
		float GeometryDiff = fabsf(observed_diff_xyz / diff_xyz - 1.0f);		// Canonical域模型的差分不大
		if (GeometryDiff < 0.3f) return valid_pixel;
	}
	return make_ushort4(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);
}

__device__ ushort2 SparseSurfelFusion::device::ValidDepthPixelInWindow(cudaTextureObject_t depthVertexMap, unsigned short center_x, unsigned short center_y)
{
	ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);

	// 搜索有效的vertex
	for (auto y = center_y - CrossGeometrySearchHalfSize; y <= center_y + CrossGeometrySearchHalfSize; y++) {
		for (auto x = center_x - CrossGeometrySearchHalfSize; x <= center_x + CrossGeometrySearchHalfSize; x++) {
			const float4 vertex = tex2D<float4>(depthVertexMap, x, y);
			if (!is_zero_vertex(vertex)) {
				valid_pixel.x = x;
				valid_pixel.y = y;
				break;
			}
		}
	}

	// 总是以中心为准
	const float4 center_vertex = tex2D<float4>(depthVertexMap, center_x, center_y);
	if (!is_zero_vertex(center_vertex)) {
		valid_pixel.x = center_x;
		valid_pixel.y = center_y;
	}

	// 获得像素点
	return valid_pixel;
}

__global__ void SparseSurfelFusion::device::ChooseValidCrossCorrPairsKernel(ObservedCrossViewCorrespondenceInterface observed, GeometryMapCrossViewCorrespondenceInterface geometry, const unsigned int rows, const unsigned int cols, const unsigned int pairsNum, unsigned int* indicator, CrossViewCorrPairs* ObCrossCorrPairs, CrossViewCorrPairs* GeoCrossCorrPairs)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= pairsNum) return;
	const ushort4 CandidatePair = observed.crossCorrPairs[idx].PixelPairs;
	const ushort2 CandidatePairView = observed.crossCorrPairs[idx].PixelViews;
	indicator[idx] = 0;
	ushort4 GeometryCrossCorrPairs = make_ushort4(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF);

	const ushort2 observedPixel_1 = ValidDepthPixelInWindow(observed.depthVertexMap[CandidatePairView.x], CandidatePair.x, CandidatePair.y);	//对应哪个深度像素
	const ushort2 observedPixel_2 = ValidDepthPixelInWindow(observed.depthVertexMap[CandidatePairView.y], CandidatePair.z, CandidatePair.w);	//对应哪个深度像素
	if ((observedPixel_1.x < cols) && (observedPixel_1.y < rows) && (observedPixel_2.x < cols) && (observedPixel_2.y < rows)) {
		float4 observedVertex_1 = tex2D<float4>(observed.depthVertexMap[CandidatePairView.x], observedPixel_1.x, observedPixel_1.y);
		float4 observedVertex_2 = tex2D<float4>(observed.depthVertexMap[CandidatePairView.y], observedPixel_2.x, observedPixel_2.y);
		float3 observedVertexWorld_1 = geometry.initialCameraSE3[CandidatePairView.x].rot * observedVertex_1 + geometry.initialCameraSE3[CandidatePairView.x].trans;
		float3 observedVertexWorld_2 = geometry.initialCameraSE3[CandidatePairView.y].rot * observedVertex_2 + geometry.initialCameraSE3[CandidatePairView.y].trans;

		float diff_xyz = fabsf_diff_xyz(observedVertexWorld_1, observedVertexWorld_2);
		GeometryCrossCorrPairs = ValidCrossGeometryPixelInWindow(geometry, observed, observed.crossCorrPairs[idx], observedVertexWorld_1, observedVertexWorld_2, diff_xyz);
		if (GeometryCrossCorrPairs.x != 0xFFFF) {
			GeoCrossCorrPairs[idx].PixelPairs = GeometryCrossCorrPairs;	// 匹配合适
			GeoCrossCorrPairs[idx].PixelViews = CandidatePairView;		// 视角不变
			ObCrossCorrPairs[idx].PixelPairs = make_ushort4(observedPixel_1.x, observedPixel_1.y, observedPixel_2.x, observedPixel_2.y);
			ObCrossCorrPairs[idx].PixelViews = CandidatePairView;		// 
			indicator[idx] = 1;
		}
	}
}

__global__ void SparseSurfelFusion::device::CompactValidCrossCorrPairsKernel(ObservedCrossViewCorrespondenceInterface observed, GeometryMapCrossViewCorrespondenceInterface geometry, const unsigned int totalCrossCorrPairs, const DeviceArrayView<unsigned int> validIndicator, const unsigned int* prefixsumIndicator, DeviceArrayView<CrossViewCorrPairs> ObCrossCorrPairs, DeviceArrayView<CrossViewCorrPairs> GeoCrossCorrPairs, float4* targetVertexArray, float4* canonicalVertexArray, ushort4* knnArray, float4* knnWeightArray)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= totalCrossCorrPairs || validIndicator[idx] == 0) return;
	const unsigned int offset = prefixsumIndicator[idx] - 1;
	const CrossViewCorrPairs& geoPairs = GeoCrossCorrPairs[idx];		// Canonical域几何体上的Pairs
	const CrossViewCorrPairs& obPais = ObCrossCorrPairs[idx];			// Observed的Pairs

	// Canonical域的匹配点和观察帧【注意：前面的点是观察帧，后面的点是Geometry】
	const float4 observedVertex_1 = tex2D<float4>(observed.depthVertexMap[obPais.PixelViews.x], obPais.PixelPairs.x, obPais.PixelPairs.y);
	const float4 observedVertex_2 = tex2D<float4>(observed.depthVertexMap[obPais.PixelViews.y], obPais.PixelPairs.z, obPais.PixelPairs.w);

	float3 observedVertexWorld_1 = geometry.initialCameraSE3[obPais.PixelViews.x].rot* observedVertex_1 + geometry.initialCameraSE3[obPais.PixelViews.x].trans;
	observedVertexWorld_1= geometry.camera2World[obPais.PixelViews.x].rot * observedVertexWorld_1 + geometry.camera2World[obPais.PixelViews.x].trans;
	float3 observedVertexWorld_2 = geometry.initialCameraSE3[obPais.PixelViews.y].rot * observedVertex_2 + geometry.initialCameraSE3[obPais.PixelViews.y].trans;
	observedVertexWorld_2 = geometry.camera2World[obPais.PixelViews.y].rot * observedVertexWorld_2 + geometry.camera2World[obPais.PixelViews.y].trans;

	const float4 canlVertex_1 = tex2D<float4>(geometry.canonicalVertexMap[geoPairs.PixelViews.x], geoPairs.PixelPairs.x, geoPairs.PixelPairs.y);
	const float4 canlVertex_2 = tex2D<float4>(geometry.canonicalVertexMap[geoPairs.PixelViews.y], geoPairs.PixelPairs.z, geoPairs.PixelPairs.w);

	const float3 canVertexWorld_1 = geometry.initialCameraSE3[geoPairs.PixelViews.x].rot * canlVertex_1 + geometry.initialCameraSE3[geoPairs.PixelViews.x].trans;
	const float3 canVertexWorld_2 = geometry.initialCameraSE3[geoPairs.PixelViews.y].rot * canlVertex_2 + geometry.initialCameraSE3[geoPairs.PixelViews.y].trans;

	// CrossCorrPair的中心
	float3 targetVertex = (observedVertexWorld_1 + observedVertexWorld_2);
	targetVertex = make_float3(targetVertex.x / 2.0f, targetVertex.y / 2.0f, targetVertex.z / 2.0f);

	const KNNAndWeight knn_1 = geometry.knnMap[geoPairs.PixelViews.x](geoPairs.PixelPairs.y, geoPairs.PixelPairs.x);
	const KNNAndWeight knn_2 = geometry.knnMap[geoPairs.PixelViews.y](geoPairs.PixelPairs.w, geoPairs.PixelPairs.z);

	// 输出
	targetVertexArray[2 * offset + 0] = make_float4(targetVertex.x, targetVertex.y, targetVertex.z, 1.0f);
	targetVertexArray[2 * offset + 1] = make_float4(targetVertex.x, targetVertex.y, targetVertex.z, 1.0f);

	canonicalVertexArray[2 * offset + 0] = make_float4(canVertexWorld_1.x, canVertexWorld_1.y, canVertexWorld_1.z, canlVertex_1.w);
	canonicalVertexArray[2 * offset + 1] = make_float4(canVertexWorld_2.x, canVertexWorld_2.y, canVertexWorld_2.z, canlVertex_2.w);

	knnArray[2 * offset + 0] = knn_1.knn;
	knnArray[2 * offset + 1] = knn_2.knn;

	knnWeightArray[2 * offset + 0] = knn_1.weight;
	knnWeightArray[2 * offset + 1] = knn_2.weight;
}

__global__ void SparseSurfelFusion::device::ForwardWarpCrossViewFeatureVertexKernel(DeviceArrayView<float4> canonicalVertexArray, const ushort4* vertexKnnArray, const float4* vertexKnnWeightArray, const DualQuaternion* nodeSe3, const unsigned int canonicalVertexNum, float4* warpedVertexArray)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= canonicalVertexNum) return;
	const float4 canVertex = canonicalVertexArray[idx];
	const ushort4 knn = vertexKnnArray[idx];
	const float4 knnweight = vertexKnnWeightArray[idx];
	// 因为已经都是世界坐标系下的顶点, se3可以直接用
	DualQuaternion dq = averageDualQuaternion(nodeSe3, knn, knnweight);
	const mat34 se3 = dq.se3_matrix();
	const float3 warped_vertex = se3.rot * canVertex + se3.trans;
	warpedVertexArray[idx] = make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f);
}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::ChooseValidCrossCorrPairs(cudaStream_t stream)
{
	crossCorrPairsNum = observedCrossViewCorrInterface.crossCorrPairs.Size();
	validCrossCorrIndicator.ResizeArrayOrException(crossCorrPairsNum);
	GeometryCrossCorrPairs.ResizeArrayOrException(crossCorrPairsNum);
	ObservedCrossPairs.ResizeArrayOrException(crossCorrPairsNum);
	if (crossCorrPairsNum == 0) return;
	
	dim3 block(64);
	dim3 grid(divUp(crossCorrPairsNum, block.x));
	device::ChooseValidCrossCorrPairsKernel << <grid, block, 0, stream >> > (observedCrossViewCorrInterface, geometryCrossViewCorrInterface, knnMapRows, knnMapCols, crossCorrPairsNum, validCrossCorrIndicator.Ptr(), ObservedCrossPairs.Ptr(), GeometryCrossCorrPairs.Ptr());
	CHECKCUDA(cudaStreamSynchronize(stream));

}


void SparseSurfelFusion::CrossViewCorrespondenceHandler::CompactCrossViewCorrPairs(cudaStream_t stream)
{
	if (crossCorrPairsNum == 0) return;
	validCorrPrefixSum.InclusiveSum(validCrossCorrIndicator.ArrayView(), stream);
	
	dim3 block(64);
	dim3 grid(divUp(crossCorrPairsNum, block.x));
	device::CompactValidCrossCorrPairsKernel << <grid, block, 0, stream >> > (observedCrossViewCorrInterface, geometryCrossViewCorrInterface, crossCorrPairsNum, validCrossCorrIndicator.ArrayView(), validCorrPrefixSum.valid_prefixsum_array.ptr(), ObservedCrossPairs.ArrayView(), GeometryCrossCorrPairs.ArrayView(), validTargetVertex.Ptr(), validCanonicalVertex.Ptr(), validVertexKnn.Ptr(), validKnnWeight.Ptr());
	CHECKCUDA(cudaStreamSynchronize(stream));

}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::QueryCompactedCrossViewArraySize(cudaStream_t stream)
{
	// 跨镜匹配为空
	if (crossCorrPairsNum == 0) {
		validTargetVertex.ResizeArrayOrException(0);
		validCanonicalVertex.ResizeArrayOrException(0);
		validVertexKnn.ResizeArrayOrException(0);
		validKnnWeight.ResizeArrayOrException(0);
		return;
	}

	DeviceArray<unsigned int>& prefixSumArray = validCorrPrefixSum.valid_prefixsum_array;
	CHECKCUDA(cudaMemcpyAsync(&validCrossCorrPairsNum, prefixSumArray.ptr() + crossCorrPairsNum - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));

	validTargetVertex.ResizeArrayOrException(validCrossCorrPairsNum * 2);
	validCanonicalVertex.ResizeArrayOrException(validCrossCorrPairsNum * 2);
	validVertexKnn.ResizeArrayOrException(validCrossCorrPairsNum * 2);
	validKnnWeight.ResizeArrayOrException(validCrossCorrPairsNum * 2);

	//printf("validCrossCorrPairsNum = %d\n", validCrossCorrPairsNum * 2);
	// 红色是要对齐的，绿色是对齐的目标
	//if (frameIdx >= 90 && frameIdx <= 120) {
	//	Visualizer::DrawMatchedReferenceAndObseveredPointsPair(validCanonicalVertex.ArrayView(), validTargetVertex.ArrayView());
	//}

}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::forwardWarpFeatureVertex(cudaStream_t stream)
{
	validWarpedVertex.ResizeArrayOrException(validCanonicalVertex.ArraySize());

	//Do a forward warp
	dim3 block(128);
	dim3 grid(divUp(validCanonicalVertex.ArraySize(), block.x));
	device::ForwardWarpCrossViewFeatureVertexKernel << <grid, block, 0, stream >> > (validCanonicalVertex.ArrayView(), validVertexKnn.Ptr(), validKnnWeight.Ptr(), NodeSe3.RawPtr(), validCanonicalVertex.ArraySize(), validWarpedVertex.Ptr());
}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::BuildTerm2Jacobian(cudaStream_t stream)
{
	forwardWarpFeatureVertex(stream);
}

SparseSurfelFusion::Point2PointICPTerm2Jacobian SparseSurfelFusion::CrossViewCorrespondenceHandler::Term2JacobianMap() const
{
	Point2PointICPTerm2Jacobian term2jacobian;
	term2jacobian.target_vertex = validTargetVertex.ArrayView();
	term2jacobian.reference_vertex = validCanonicalVertex.ArrayView();
	term2jacobian.knn = validVertexKnn.ArrayView();
	term2jacobian.knn_weight = validKnnWeight.ArrayView();
	term2jacobian.node_se3 = NodeSe3;
	term2jacobian.warped_vertex = validWarpedVertex.ArrayView();

	//Check the size
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.reference_vertex.Size());
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.knn.Size());
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.knn_weight.Size());
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.warped_vertex.Size());

	//Return it
	return term2jacobian;
}
