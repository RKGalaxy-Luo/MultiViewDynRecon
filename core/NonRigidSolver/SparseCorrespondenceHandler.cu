#include "SparseCorrespondenceHandler.h"
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {

		__device__ unsigned int CorresponedPairArrayOffset[MAX_CAMERA_COUNT + 1];

		enum {
			window_halfsize = 1,
		};

		__device__ unsigned int calculateCameraID(unsigned int idx, unsigned int devCount) {
			for (int i = 0; i < devCount; i++) {
				if (CorresponedPairArrayOffset[i] <= idx && idx < CorresponedPairArrayOffset[i + 1]) return i;
			}
		}

		__device__ ushort2 validGeometryPixelInWindow(
			cudaTextureObject_t index_map,
			unsigned short center_x, unsigned short center_y //������ĵ������
		) {
			ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);

			// �ڴ�����������Чֵ
			for (auto y = center_y - window_halfsize; y <= center_y + window_halfsize; y++) {
				for (auto x = center_x - window_halfsize; x <= center_x + window_halfsize; x++) {
					if (tex2D<unsigned>(index_map, x, y) != 0xFFFFFFFF) {
						valid_pixel.x = x;
						valid_pixel.y = y;
						break;
					}
				}
			}

			// ��������д�����Чֵ���������������Ч����ô��ѡ���������Чֵ���������������Ч�͸������ڴ�������
			if (tex2D<unsigned>(index_map, center_x, center_y) != 0xFFFFFFFF) {
				valid_pixel.x = center_x;
				valid_pixel.y = center_y;
			}

			//Return it
			return valid_pixel;
		}

		__device__ ushort2 validGeometryPixelInWindow(cudaTextureObject_t indexMap, cudaTextureObject_t liveVertexMap, float4 observedVertex, unsigned short center_x, unsigned short center_y) {
			ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);
			float minSquaredDis = 100.0f;
			//Perform a window search
			for (int y = center_y - window_halfsize; y <= center_y + window_halfsize; y++) {
				for (int x = center_x - window_halfsize; x <= center_x + window_halfsize; x++) {
					if (tex2D<unsigned>(indexMap, x, y) != 0xFFFFFFFF) {
						float4 liveVertex = tex2D<float4>(liveVertexMap, x, y);
						float squaredDis = squared_distance(observedVertex, liveVertex);
						if (squaredDis < minSquaredDis) {
							valid_pixel.x = x;
							valid_pixel.y = y;
							minSquaredDis = squaredDis;
						}

						//break;
					}
				}
			}

			//Return it
			return valid_pixel;
		}

		__device__ ushort2 validDepthPixelInWindow(
			cudaTextureObject_t depth_vertex_map,
			unsigned short center_x, 
			unsigned short center_y
		) {
			ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);

			//Perform a window search
			for (auto y = center_y - window_halfsize; y <= center_y + window_halfsize; y++) {
				for (auto x = center_x - window_halfsize; x <= center_x + window_halfsize; x++) {
					const float4 vertex = tex2D<float4>(depth_vertex_map, x, y);
					if (!is_zero_vertex(vertex)) {
						valid_pixel.x = x;
						valid_pixel.y = y;
						break;
					}
				}
			}

			//Always prefer the center one
			const float4 center_vertex = tex2D<float4>(depth_vertex_map, center_x, center_y);
			if (!is_zero_vertex(center_vertex)) {
				valid_pixel.x = center_x;
				valid_pixel.y = center_y;
			}

			//Return it
			return valid_pixel;
		}

		__global__ void chooseValidPixelKernel(
			ObservedSparseCorrespondenceInterface observedSparseCorrInterface,
			GeometryMapSparseCorrespondenceInterface geometrySparseCorrInterface,
			const unsigned int totalCorrPairs,
			const unsigned int rows, 
			const unsigned int cols,
			const unsigned int devicesCount,
			unsigned int* validIndicator,
			ushort4* validPixelPairs
		) {
#ifdef USE_NEAREST_DISTANCE_PAIRS
			//��ʵ��������ɵ�������rgbͼƥ�䵽�����ص�ԣ��ҵ�live����ģ���Ϻ�depthͼ����ƥ������ص�ԡ����˸�ת����
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= totalCorrPairs) return;
			const unsigned int CameraID = calculateCameraID(idx, devicesCount);
			const unsigned int offsetIdx = idx - CorresponedPairArrayOffset[CameraID];
			const ushort4 candidatePair = observedSparseCorrInterface.correspondPixelPairs[CameraID][offsetIdx];
			// ��һ֡�����ص���x��y����ǰ֡������Ե���z��w. IndexMap����һ֡live����ģ�͵�ӳ�䡣���Դ�indexmap����x��y�Ķ�Ӧ��cannical�㣬�����治���ڣ����ڣ��ͱ����������λ��
			const ushort2 depthPixel = validDepthPixelInWindow(observedSparseCorrInterface.depthVertexMap[CameraID], candidatePair.z, candidatePair.w);//��Ӧ���ǵ�һ֡���ͼ���ĸ��������
			unsigned char isEdgePixels = tex2D<unsigned char>(observedSparseCorrInterface.edgeMask[CameraID], depthPixel.x, depthPixel.y);
			if ((depthPixel.x < cols) && (depthPixel.y < rows) && isEdgePixels != (unsigned char)1) {	// ���ڱ�Ե��
				float4 observedVertex = tex2D<float4>(observedSparseCorrInterface.depthVertexMap[CameraID], depthPixel.x, depthPixel.y);
				const ushort2 geometryPixel = validGeometryPixelInWindow(geometrySparseCorrInterface.indexMap[CameraID], geometrySparseCorrInterface.liveVertexMap[CameraID], observedVertex, candidatePair.x, candidatePair.y);
				if ((geometryPixel.x < cols) && (geometryPixel.y < rows)) {
					float4 referenceVertex = tex2D<float4>(geometrySparseCorrInterface.referenceVertexMap[CameraID], geometryPixel.x, geometryPixel.y);
					float squaredDis = squared_distance(referenceVertex, observedVertex);
					if (squaredDis < 2.5e-3f) {
						validIndicator[idx] = 1;
						validPixelPairs[idx] = make_ushort4(geometryPixel.x, geometryPixel.y, depthPixel.x, depthPixel.y);
					}
				}
			}
#else
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= totalCorrPairs) return;
			const unsigned int CameraID = calculateCameraID(idx, devicesCount);
			const unsigned int offsetIdx = idx - CorresponedPairArrayOffset[CameraID];
			const ushort4 candidatePair = observedSparseCorrInterface.correspondPixelPairs[CameraID][offsetIdx];
			// ��һ֡�����ص���x��y����ǰ֡������Ե���z��w. IndexMap����һ֡live����ģ�͵�ӳ�䡣���Դ�indexmap����x��y�Ķ�Ӧ��cannical�㣬�����治���ڣ����ڣ��ͱ����������λ��
			const ushort2 geometryPixel = validGeometryPixelInWindow(geometrySparseCorrInterface.indexMap[CameraID], candidatePair.x, candidatePair.y);
			const ushort2 depthPixel = validDepthPixelInWindow(observedSparseCorrInterface.depthVertexMap[CameraID], candidatePair.z, candidatePair.w);//��Ӧ���ǵ�һ֡���ͼ���ĸ��������
			//unsigned char isEdgePixels = tex2D<unsigned char>(observedSparseCorrInterface.edgeMask[CameraID], depthPixel.x, depthPixel.y);
			if (geometryPixel.x < cols && geometryPixel.y < rows && depthPixel.x < cols && depthPixel.y < rows/* && isEdgePixels != (unsigned char)1*/) {
				validIndicator[idx] = 1;
				validPixelPairs[idx] = make_ushort4(geometryPixel.x, geometryPixel.y, depthPixel.x, depthPixel.y);
			}
			else {
				validIndicator[idx] = 0;
			}
#endif // USE_NEAREST_DISTANCE_PAIRS
		}

		__global__ void compactQueryValidPairsKernel(
			device::ObservedSparseCorrespondenceInterface observedSparseCorrInterface,
			device::GeometryMapSparseCorrespondenceInterface geometrySparseCorrInterface,
			const unsigned int devicesCount,
			const unsigned int totalCorrPairs,
			const DeviceArrayView<unsigned int> validIndicator,
			const unsigned int* prefixsumIndicator,
			unsigned int* differentViewsCorrPairsOffsets,
			const ushort4* validPixelPairs,
			float4* targetVertexArray,
			float4* referenceVertexArray,
			ushort4* knnArray,
			float4* knnWeightArray
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= totalCorrPairs) return;
			//���������һ���ҵ���live���������ͼ���ص�ƥ��ԣ��ҵ�ƥ��ĵ㡣����������ضԵ���ض����ת����
			if (validIndicator[idx] != 0) {
				const unsigned int CameraID = calculateCameraID(idx, devicesCount);
				const unsigned int offset = prefixsumIndicator[idx] - 1;	// ����ѹ�������е�ƫ��
				const ushort4 pixelPair = validPixelPairs[idx];
				// CameraID�������ϵ�µ�Canonical��
				float4 reference_vertex = tex2D<float4>(geometrySparseCorrInterface.referenceVertexMap[CameraID], pixelPair.x, pixelPair.y);
				// �����CameraID�ӽ��µ�Canonical��ĵ�ת��0������ϵ��
				float3 referenceVertexCoor_0 = geometrySparseCorrInterface.initialCameraSE3[CameraID].rot * reference_vertex + geometrySparseCorrInterface.initialCameraSE3[CameraID].trans;
				
				// �۲⵽�����ͼ
				float4 depth_vertex = tex2D<float4>(observedSparseCorrInterface.depthVertexMap[CameraID], pixelPair.z, pixelPair.w);
				// Compute the target vertex
				float3 depth_v3 = make_float3(depth_vertex.x, depth_vertex.y, depth_vertex.z);
				depth_v3 = geometrySparseCorrInterface.initialCameraSE3[CameraID].rot * depth_v3 + geometrySparseCorrInterface.initialCameraSE3[CameraID].trans;
				// ע�⣬�������õ�camera2world����depth��ת����live�У���ΪĿ���
				float3 target_v3 = geometrySparseCorrInterface.camera2World[CameraID].rot * depth_v3 + geometrySparseCorrInterface.camera2World[CameraID].trans;
				//Write to output
				//������ǵõ���ref�еĵ��ھ���warpfeildSE3Ť����ĵ�Ӧ����live�е����target����Ҳ�洢��ref���KNN
				targetVertexArray[offset] = make_float4(target_v3.x, target_v3.y, target_v3.z, 1.0f);
				referenceVertexArray[offset] = make_float4(referenceVertexCoor_0.x, referenceVertexCoor_0.y, referenceVertexCoor_0.z, reference_vertex.w);
				
				KNNAndWeight knn = geometrySparseCorrInterface.knnMap[CameraID](pixelPair.y, pixelPair.x);
				knnArray[offset] = knn.knn;
				knnWeightArray[offset] = knn.weight;
				if (idx == CorresponedPairArrayOffset[CameraID]) {
					// ���㲻ͬ�ӽǵ����pairѹ���������е�ƫ�ƣ�ͨ��ƫ�Ƽ���ȷ����ǰpair�����ĸ��ӽ�
					differentViewsCorrPairsOffsets[CameraID] = offset;
				}
			}
		}


		//Forward warp the vertex for better computation of jacobian
		__global__ void forwardWarpFeatureVertexKernel(
			DeviceArrayView<float4> reference_vertex_array,
			const ushort4* vertex_knn_array,
			const float4* vertex_knnweight_array,
			const DualQuaternion* node_se3,
			float4* warped_vertex_array
		) {
			const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx < reference_vertex_array.Size()) {
				const float4 reference_vertex = reference_vertex_array[idx];
				const ushort4 knn = vertex_knn_array[idx];
				const float4 knnweight = vertex_knnweight_array[idx];
				//��Ϊ�Ѿ�����0������ϵ�µĶ����ˣ�se3����ֱ����
				DualQuaternion dq = averageDualQuaternion(node_se3, knn, knnweight);
				const mat34 se3 = dq.se3_matrix();
				const float3 warped_vertex = se3.rot * reference_vertex + se3.trans;
				warped_vertex_array[idx] = make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f);
			}
		}
	}
}



void SparseSurfelFusion::SparseCorrespondenceHandler::ChooseValidPixelPairs(cudaStream_t stream) {
	size_t totalCorrPair = 0;
	for (int i = 0; i < devicesCount; i++) {
		CorrPairsOffsetArray[i] = totalCorrPair;
		totalCorrPair += observedSparseCorrInterface.correspondPixelPairs[i].Size();
	}
	CorrPairsOffsetArray[devicesCount] = totalCorrPair;
	CHECKCUDA(cudaMemsetAsync(m_valid_pixel_indicator.Ptr(), 0, sizeof(unsigned int) * Constants::kMaxMatchedSparseFeature, stream));
	CHECKCUDA(cudaMemcpyToSymbolAsync(device::CorresponedPairArrayOffset, CorrPairsOffsetArray, sizeof(unsigned int) * (devicesCount + 1), 0, cudaMemcpyHostToDevice, stream));

	m_valid_pixel_indicator.ResizeArrayOrException(totalCorrPair);
	m_corrected_pixel_pairs.ResizeArrayOrException(totalCorrPair);

	//The correspondence array might be empty
	if (m_valid_pixel_indicator.ArraySize() == 0) {
		printf("��ǰ֡����һ֡û���ҵ�������ص�ԣ�������\n");
		return;
	}

	// ��indexMap����currentͼƥ��ĵ�
	dim3 block(64);
	dim3 grid(divUp(totalCorrPair, block.x));
	device::chooseValidPixelKernel << <grid, block, 0, stream >> > (
		observedSparseCorrInterface,
		geometrySparseCorrInterface,
		totalCorrPair,
		knnMapRows,
		knnMapCols,
		devicesCount,
		m_valid_pixel_indicator,
		m_corrected_pixel_pairs
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


void SparseSurfelFusion::SparseCorrespondenceHandler::CompactQueryPixelPairs(cudaStream_t stream) {
	const unsigned int totalCorrPair = CorrPairsOffsetArray[devicesCount];
	
	// The correspondence array might be empty
	if (m_valid_pixel_indicator.ArraySize() == 0) return;
	// Inclusive sum
	m_valid_pixel_prefixsum.InclusiveSum(m_valid_pixel_indicator.ArrayView(), stream);

	dim3 block(64);
	dim3 grid(divUp(totalCorrPair, block.x));
	device::compactQueryValidPairsKernel << <grid, block, 0, stream >> > (
		observedSparseCorrInterface,
		geometrySparseCorrInterface,
		devicesCount,
		totalCorrPair,
		//Prefix-sum information
		m_valid_pixel_indicator.ArrayView(),
		m_valid_pixel_prefixsum.valid_prefixsum_array.ptr(),
		differentViewsCorrPairsOffset.Ptr(),
		m_corrected_pixel_pairs.Ptr(),
		//The output
		m_valid_target_vertex.Ptr(),//��������ϵ�µ������ά����
		m_valid_reference_vertex.Ptr(),
		m_valid_vertex_knn.Ptr(),
		m_valid_knn_weight.Ptr()
	);
	


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


void SparseSurfelFusion::SparseCorrespondenceHandler::QueryCompactedArraySize(cudaStream_t stream) {
	//The correspondence array might be empty
	if (m_valid_pixel_indicator.ArraySize() == 0) {
		m_valid_target_vertex.ResizeArrayOrException(0);
		m_valid_reference_vertex.ResizeArrayOrException(0);
		m_valid_vertex_knn.ResizeArrayOrException(0);
		m_valid_knn_weight.ResizeArrayOrException(0);
		differentViewsCorrPairsOffset.ResizeArrayOrException(0);
		return;
	}

	//Non-empty array
	//unsigned valid_array_size;
	CHECKCUDA(cudaMemcpyAsync(
		m_correspondence_array_size,
		m_valid_pixel_prefixsum.valid_prefixsum_array.ptr() + m_valid_pixel_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));

	//Sync before use
	CHECKCUDA(cudaStreamSynchronize(stream));

#ifdef DEBUG_RUNNING_INFO
	printf("ʹ���ںϺ������Ԫ���ӽ�ӳ���ƥ�����ά��Եĸ���Ϊ %d\n", *m_correspondence_array_size);
#endif // DEBUG_RUNNING_INFO

	//Correct the size
	m_valid_target_vertex.ResizeArrayOrException(*m_correspondence_array_size);
	m_valid_reference_vertex.ResizeArrayOrException(*m_correspondence_array_size);
	m_valid_vertex_knn.ResizeArrayOrException(*m_correspondence_array_size);
	m_valid_knn_weight.ResizeArrayOrException(*m_correspondence_array_size);
	differentViewsCorrPairsOffset.ResizeArrayOrException(devicesCount);
	//printf("m_correspondence_array_size = %d\n", *m_correspondence_array_size);

	//����Ϊʲô����һ֡LiveͶӰ�����ص�IndexMap�뵱ǰ�۲쵽���������ص㣿��
	//if (frameIdx % 80 == 0) {
		//Visualizer::DrawMatchedReferenceAndObseveredPointsPair(m_valid_reference_vertex.ArrayView(), m_valid_target_vertex.ArrayView());
	//}
	//Visualizer::DrawMatchedReferenceAndObseveredPointsPair(geometrySparseCorrInterface.referenceVertexMap[1], observedSparseCorrInterface.depthVertexMap[1], Constants::GetInitialCameraSE3(1), m_valid_reference_vertex.ArrayView(), m_valid_target_vertex.ArrayView());
}


/* The method to build the term 2 jacobian map
 */
void SparseSurfelFusion::SparseCorrespondenceHandler::forwardWarpFeatureVertex(cudaStream_t stream) {
	m_valid_warped_vertex.ResizeArrayOrException(m_valid_reference_vertex.ArraySize());

	//Do a forward warp
	dim3 block(128);
	dim3 grid(divUp(m_valid_reference_vertex.ArraySize(), block.x));
	device::forwardWarpFeatureVertexKernel << <grid, block, 0, stream >> > (
		m_valid_reference_vertex.ArrayView(),
		m_valid_vertex_knn.Ptr(), 
		m_valid_knn_weight.Ptr(),
		m_node_se3.RawPtr(),
		m_valid_warped_vertex.Ptr()
	);

	//std::vector<ushort4> knnHost(m_valid_vertex_knn.ArraySize());
	//m_valid_vertex_knn.ArrayView().Download(knnHost);
	//for (int i = 0; i < m_valid_vertex_knn.ArraySize(); i++) {
	//	printf("idx = %d   knn(%d,%d,%d,%d)\n", i, knnHost[i].x, knnHost[i].y, knnHost[i].z, knnHost[i].w);
	//}

	//Visualizer::DrawMatchedReferenceAndObseveredPointsPair(m_valid_reference_vertex.ArrayView(), m_valid_warped_vertex.ArrayView());


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
#endif
}


void SparseSurfelFusion::SparseCorrespondenceHandler::BuildTerm2Jacobian(cudaStream_t stream) {
	forwardWarpFeatureVertex(stream);
}


SparseSurfelFusion::Point2PointICPTerm2Jacobian SparseSurfelFusion::SparseCorrespondenceHandler::Term2JacobianMap() const {
	Point2PointICPTerm2Jacobian term2jacobian;
	term2jacobian.target_vertex = m_valid_target_vertex.ArrayView();
	term2jacobian.reference_vertex = m_valid_reference_vertex.ArrayView();
	term2jacobian.knn = m_valid_vertex_knn.ArrayView();
	term2jacobian.knn_weight = m_valid_knn_weight.ArrayView();
	term2jacobian.node_se3 = m_node_se3;
	term2jacobian.warped_vertex = m_valid_warped_vertex.ArrayView();

	//Check the size
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.reference_vertex.Size());
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.knn.Size());
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.knn_weight.Size());
	FUNCTION_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.warped_vertex.Size());

	//Return it
	return term2jacobian;
}






