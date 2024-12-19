/*****************************************************************//**
 * \file   OpticalFlow.cu
 * \brief  �������ģ��
 * 
 * \author LUOJIAXUAN
 * \date   July 4th 2024
 *********************************************************************/
#include "OpticalFlow.h"
#if defined(__CUDACC__)		//�����NVCC����������
#include <cub/cub.cuh>
#endif

__forceinline__ __device__ float SparseSurfelFusion::device::windowRange(const unsigned short& x, const unsigned short& y, const short& windowRadius, const unsigned int& cols, const unsigned int& rows, cudaTextureObject_t vertexMap)
{
	const unsigned short x_left = windowRadius;
	const unsigned short x_right = cols - windowRadius;
	const unsigned short y_left = windowRadius;
	const unsigned short y_right = rows - windowRadius;
	ushort2 windowCenter;
	if (x_left <= x && x < x_right && y_left <= y && y < y_right) {
		windowCenter = make_ushort2(x, y);
	}
	else if (x < x_left) {
		if (y < y_left) {
			windowCenter = make_ushort2(windowRadius, windowRadius);
		}
		else if (y >= y_right) {
			windowCenter = make_ushort2(windowRadius, y_right - 1);
		}
		else {
			windowCenter = make_ushort2(windowRadius, y);
		}
	}
	else if (x >= x_right) {
		if (y < y_left) {
			windowCenter = make_ushort2(x_right - 1, windowRadius);
		}
		else if (y >= y_right) {
			windowCenter = make_ushort2(x_right - 1, y_right - 1);
		}
		else {
			windowCenter = make_ushort2(x_right - 1, y);
		}
	}
	else {
		if (y < y_left) {
			windowCenter = make_ushort2(x, windowRadius);
		}
		else {	// y_left <= y && y < y_right�������Ѿ��жϹ���
			windowCenter = make_ushort2(x, y_right - 1);
		}
	}
	float minDepth = 10.0f;
	float maxDepth = -10.0f;
	for (short i = -windowRadius; i <= windowRadius; i++) {
		for (short j = -windowRadius; j <= windowRadius; j++) {
			short currX = windowCenter.x + i;
			short currY = windowCenter.y + j;
			
			float4 vertex = tex2D<float4>(vertexMap, currX, currY);

			if (minDepth > vertex.z)	minDepth = vertex.z;
			if (maxDepth < vertex.z)	maxDepth = vertex.z;
		}
	}
	float windowsRange = maxDepth - minDepth;
	if (windowsRange > 10.0f) windowsRange = 0.0f;
	return windowsRange;
}

__global__ void SparseSurfelFusion::device::ConvertBuffer2InputTensorKernel(const uchar3* previousImage, const unsigned short* previousDepth, const uchar3* currentImage, const unsigned short* currentDepth, const unsigned int rawImageSize, float* inputPreviousImage, float* inputPreviousDepth, float* inputCurrentImage, float* inputCurrentDepth)
{
	const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= rawImageSize) return;

	uchar3 colorImage = previousImage[idx];
	inputPreviousImage[idx + 0 * rawImageSize] = (float)colorImage.z / 255.0f;
	inputPreviousImage[idx + 1 * rawImageSize] = (float)colorImage.y / 255.0f;
	inputPreviousImage[idx + 2 * rawImageSize] = (float)colorImage.x / 255.0f;

	colorImage = currentImage[idx];
	inputCurrentImage[idx + 0 * rawImageSize] = (float)colorImage.z / 255.0f;
	inputCurrentImage[idx + 1 * rawImageSize] = (float)colorImage.y / 255.0f;
	inputCurrentImage[idx + 2 * rawImageSize] = (float)colorImage.x / 255.0f;

	inputPreviousDepth[idx] = previousDepth[idx] / 1000.0f;

	inputCurrentDepth[idx] = currentDepth[idx] / 1000.0f;
}
__global__ void SparseSurfelFusion::device::CalculatePixelPairAnd3DOpticalFlowKernel(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t PreviousVertexMap, cudaTextureObject_t CurrentVertexMap, const float* flow2d, const unsigned int clipedImageRows, const unsigned int clipedImageCols, const unsigned int rawImageSize, ushort4* pixelPair, bool* markValidPair, float4* FlowVector3D, float3* FlowVectorOpenGL, ColorVertex* colorVertex, bool* markValidFlow)
{
	const unsigned int clip_x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int clip_y = threadIdx.y + blockDim.y * blockIdx.y;
	if (clip_x >= clipedImageCols || clip_y >= clipedImageRows) return;
	const short windowRadius = 1;

	// ��Ӧcliped_x��clip_y��ԭʼͼ���ϵ�λ��
	const unsigned int clip_boundary = CLIP_BOUNDARY;
	const unsigned int raw_x = clip_x + clip_boundary;
	const unsigned int raw_y = clip_y + clip_boundary;
	const unsigned int ColsOffset = clipedImageCols + 2 * clip_boundary;
	// ����Щ���ػ���1ά
	const unsigned int raw_flatten = raw_x + raw_y * ColsOffset;
	const unsigned int cliped_flatten = clip_x + clip_y * clipedImageCols;
	float currentWindowRange = device::windowRange(raw_x, raw_y, windowRadius, clipedImageCols, clipedImageRows, CurrentVertexMap);
	const unsigned char currMask = tex2D<unsigned char>(currForeground, clip_x, clip_y);
	float PixelOffset_X = flow2d[raw_flatten];					// ��������xƫ��
	float PixelOffset_Y = flow2d[raw_flatten + rawImageSize];	// ��������yƫ��
	float4 invalidFlow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	// ��Ч�Ĺ����������
	ushort4 invalidPair = make_ushort4(0, 0, 0, 0);
	if (currMask != 0 && currentWindowRange < 0.05f) {	// ��ǰ��ǰ��
		int previousPixel_X = raw_x - PixelOffset_X;	// ��һ֡��x����(û����)
		int previousPixel_Y = raw_y - PixelOffset_Y;	// ��һ֡��y����(û����)
		// ���ܳ���cliped textureͼ��Χ
		if (clip_boundary < previousPixel_X && previousPixel_X < clipedImageCols - clip_boundary && clip_boundary < previousPixel_Y && previousPixel_Y < clipedImageRows - clip_boundary) {
			const unsigned char preMask = tex2D<unsigned char>(preForeground, previousPixel_X, previousPixel_Y);
			float previousWindowRange = device::windowRange(previousPixel_X, previousPixel_Y, windowRadius, clipedImageCols, clipedImageRows, PreviousVertexMap);
			if (preMask != 0 && previousWindowRange < 0.05f) {	// ��һ֡ƥ���Ҳ��ǰ����
				const float4 PreVertex = tex2D<float4>(PreviousVertexMap, clip_x, clip_y);
				const float4 CurVertex = tex2D<float4>(CurrentVertexMap, clip_x, clip_y);
				if (PreVertex.z > 0 && CurVertex.z > 0) {	// ��������ǰ����Ҳ�������Ϊ0�ĵ㣬��Ҫ����Щ���޳�
					float3 previousPos = make_float3(PreVertex.x, PreVertex.y, PreVertex.z);
					float3 currentPos = make_float3(CurVertex.x, CurVertex.y, CurVertex.z);
					float distance = points_distance(previousPos, currentPos);
					if (distance < 0.4f) {	// ����С��10cm������Ч�Ĺ���
						pixelPair[cliped_flatten].x = previousPixel_X - clip_boundary; // ��һ֡���ú�����
						pixelPair[cliped_flatten].y = previousPixel_Y - clip_boundary; // ��һ֡���ú�����
						pixelPair[cliped_flatten].z = clip_x;	// ��ǰ֡���ú�����
						pixelPair[cliped_flatten].w = clip_y;	// ��ǰ֡���ú�����
						markValidPair[cliped_flatten] = true;

						float4 FlowVector3D = make_float4(CurVertex.x - PreVertex.x, CurVertex.y - PreVertex.y, CurVertex.z - PreVertex.z, 1.0f);


						// �����ǿ��ӻ���������飬���Ժ��˼���ɾ��
						FlowVectorOpenGL[2 * cliped_flatten + 0] = previousPos;
						FlowVectorOpenGL[2 * cliped_flatten + 1] = currentPos;
						colorVertex[2 * cliped_flatten + 0].coor = previousPos;
						colorVertex[2 * cliped_flatten + 0].color = make_float3(0.0f, 0.0f, 1.0f);	// ������ɫ�е��鷳������ǰһ֡Ϊ��ɫ
						colorVertex[2 * cliped_flatten + 1].coor = currentPos;
						colorVertex[2 * cliped_flatten + 1].color = make_float3(0.0f, 1.0f, 0.0f);	// ��ǰ֡Ϊ��ɫ
						markValidFlow[2 * cliped_flatten + 0] = true;
						markValidFlow[2 * cliped_flatten + 1] = true;
					}
					else {
						pixelPair[cliped_flatten] = invalidPair;
						markValidPair[cliped_flatten] = false;

						// �����ǿ��ӻ���������飬���Ժ��˼���ɾ��
						FlowVectorOpenGL[2 * cliped_flatten + 0] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
						FlowVectorOpenGL[2 * cliped_flatten + 1] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
						colorVertex[2 * cliped_flatten + 0].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
						colorVertex[2 * cliped_flatten + 0].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
						colorVertex[2 * cliped_flatten + 1].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
						colorVertex[2 * cliped_flatten + 1].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
						markValidFlow[2 * cliped_flatten + 0] = false;
						markValidFlow[2 * cliped_flatten + 1] = false;
					}
				}
				else {
					pixelPair[cliped_flatten] = invalidPair;
					markValidPair[cliped_flatten] = false;

					// �����ǿ��ӻ���������飬���Ժ��˼���ɾ��
					FlowVectorOpenGL[2 * cliped_flatten + 0] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
					FlowVectorOpenGL[2 * cliped_flatten + 1] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
					colorVertex[2 * cliped_flatten + 0].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
					colorVertex[2 * cliped_flatten + 0].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
					colorVertex[2 * cliped_flatten + 1].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
					colorVertex[2 * cliped_flatten + 1].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
					markValidFlow[2 * cliped_flatten + 0] = false;
					markValidFlow[2 * cliped_flatten + 1] = false;
				}
			}
			else {
				pixelPair[cliped_flatten] = invalidPair;
				markValidPair[cliped_flatten] = false;

				// �����ǿ��ӻ���������飬���Ժ��˼���ɾ��
				FlowVectorOpenGL[2 * cliped_flatten + 0] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
				FlowVectorOpenGL[2 * cliped_flatten + 1] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
				colorVertex[2 * cliped_flatten + 0].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
				colorVertex[2 * cliped_flatten + 0].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
				colorVertex[2 * cliped_flatten + 1].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
				colorVertex[2 * cliped_flatten + 1].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
				markValidFlow[2 * cliped_flatten + 0] = false;
				markValidFlow[2 * cliped_flatten + 1] = false;
			}
		}
		else {
			pixelPair[cliped_flatten] = invalidPair;
			markValidPair[cliped_flatten] = false;

			// �����ǿ��ӻ���������飬���Ժ��˼���ɾ��
			FlowVectorOpenGL[2 * cliped_flatten + 0] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
			FlowVectorOpenGL[2 * cliped_flatten + 1] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
			colorVertex[2 * cliped_flatten + 0].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
			colorVertex[2 * cliped_flatten + 0].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
			colorVertex[2 * cliped_flatten + 1].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
			colorVertex[2 * cliped_flatten + 1].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
			markValidFlow[2 * cliped_flatten + 0] = false;
			markValidFlow[2 * cliped_flatten + 1] = false;
		}
	}
	else {
		pixelPair[cliped_flatten] = invalidPair;
		markValidPair[cliped_flatten] = false;

		// �����ǿ��ӻ���������飬���Ժ��˼���ɾ��
		FlowVectorOpenGL[2 * cliped_flatten + 0] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
		FlowVectorOpenGL[2 * cliped_flatten + 1] = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
		colorVertex[2 * cliped_flatten + 0].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
		colorVertex[2 * cliped_flatten + 0].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
		colorVertex[2 * cliped_flatten + 1].coor = make_float3(0.0f, 0.0f, 0.0f);  // ��Чֵ
		colorVertex[2 * cliped_flatten + 1].color = make_float3(0.0f, 0.0f, 0.0f); // ��Чֵ
		markValidFlow[2 * cliped_flatten + 0] = false;
		markValidFlow[2 * cliped_flatten + 1] = false;
	}

}
__global__ void SparseSurfelFusion::device::CalculatePixelPairsKernel(cudaTextureObject_t preForeground, cudaTextureObject_t currForeground, cudaTextureObject_t PreviousVertexMap, cudaTextureObject_t CurrentVertexMap, cudaTextureObject_t preNormalMap, cudaTextureObject_t currNormalMap, const mat34 initialCameraSE3, const float* flow2d, const unsigned int clipedImageRows, const unsigned int clipedImageCols, const unsigned int rawImageSize, ushort4* pixelPair, bool* markValidPair, PtrStepSize<ushort4> validPixelPairsMap)
{
	const unsigned int clip_x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int clip_y = threadIdx.y + blockDim.y * blockIdx.y;
	if (clip_x >= clipedImageCols || clip_y >= clipedImageRows) return;	

	const short windowRadius = 0;

	// ��Ӧcliped_x��clip_y��ԭʼͼ���ϵ�λ��
	const unsigned int clip_boundary = CLIP_BOUNDARY;
	const unsigned int raw_x = clip_x + clip_boundary;
	const unsigned int raw_y = clip_y + clip_boundary;
	const unsigned int ColsOffset = clipedImageCols + 2 * clip_boundary;
	// ����Щ���ػ���1ά
	const unsigned int raw_flatten = raw_x + raw_y * ColsOffset;
	const unsigned int cliped_flatten = clip_x + clip_y * clipedImageCols;
	float currentWindowRange = device::windowRange(raw_x, raw_y, windowRadius, clipedImageCols, clipedImageRows, CurrentVertexMap);
	const unsigned char currMask = tex2D<unsigned char>(currForeground, clip_x, clip_y);
	float PixelOffset_X = flow2d[raw_flatten];					// ��������xƫ��
	float PixelOffset_Y = flow2d[raw_flatten + rawImageSize];	// ��������yƫ��
	float4 invalidFlow = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	// ��Ч�Ĺ����������
	ushort4 invalidPair = make_ushort4(0, 0, 0, 0);
	if (currMask == (unsigned char)1/* && currentWindowRange < 0.05f*/) {	// ��ǰ��ǰ��
		int previousPixel_X = raw_x - PixelOffset_X;	// ��һ֡��x����(û����)
		int previousPixel_Y = raw_y - PixelOffset_Y;	// ��һ֡��y����(û����)
		// ���ܳ���cliped textureͼ��Χ
		if (clip_boundary < previousPixel_X && previousPixel_X < clipedImageCols - clip_boundary && clip_boundary < previousPixel_Y && previousPixel_Y < clipedImageRows - clip_boundary) {
			const unsigned char preMask = tex2D<unsigned char>(preForeground, previousPixel_X, previousPixel_Y);
			float previousWindowRange = device::windowRange(previousPixel_X, previousPixel_Y, windowRadius, clipedImageCols, clipedImageRows, PreviousVertexMap);
			if (true/* preMask == (unsigned char)1&& previousWindowRange < 0.05f*/) {	// ��һ֡ƥ���Ҳ��ǰ����
				const float4 PreVertex = tex2D<float4>(PreviousVertexMap, clip_x, clip_y);
				const float4 CurVertex = tex2D<float4>(CurrentVertexMap, clip_x, clip_y);
				if (PreVertex.z > 0 && CurVertex.z > 0) {	// ��������ǰ����Ҳ�������Ϊ0�ĵ㣬��Ҫ����Щ���޳�
					const float4 PreNormal = tex2D<float4>(preNormalMap, clip_x, clip_y);
					const float4 CurNormal = tex2D<float4>(currNormalMap, clip_x, clip_y);
					float squaredDistance = squared_distance(PreVertex, CurVertex);
					float normalCos = dotxyz(PreNormal, CurNormal);
					if (squaredDistance < 2.5e-3f && normalCos > 0.8f) {	// ����С��10cm������Ч�Ĺ���
						ushort4 pairs = make_ushort4(previousPixel_X - clip_boundary, previousPixel_Y - clip_boundary, clip_x, clip_y);
						if (clip_x % 4 == 0 && clip_y % 4 == 0) {
							pixelPair[cliped_flatten] = pairs;
							markValidPair[cliped_flatten] = true;
						}
						// ����ͼ�����н�������ֻɸѡȥ������Ĺ���
						validPixelPairsMap.ptr(clip_y)[clip_x] = pairs;
					}
				}
			}
		}
	}
}

__global__ void SparseSurfelFusion::device::GetValidEdgePixelPairsKernel(cudaTextureObject_t currEdgeMaskMap, const ushort4* pixelPair, const bool* markValidPixelPairs, const unsigned int clipedImageRows, const unsigned int clipedImageCols, PtrStepSize<ushort4> validEdgePixelPairs)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedImageCols || y >= clipedImageRows) return;
	const unsigned char currEdgeMask = tex2D<unsigned char>(currEdgeMaskMap, x, y);
	const unsigned int flatten_idx = x * clipedImageCols + y;
	// ������ڱ�Ե����Чƥ����
	if (markValidPixelPairs[flatten_idx] == true && currEdgeMask == (unsigned char)1) {	
		validEdgePixelPairs.ptr(y)[x] = pixelPair[flatten_idx];
	}
}

__global__ void SparseSurfelFusion::device::LabelSortedCurrPairEncodedCoor(const PtrSize<const unsigned int> sortedCoor, unsigned int* label)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedCoor.size) return;
	if (idx == 0) label[0] = 1;
	else {
		if (sortedCoor[idx] != sortedCoor[idx - 1]) {
			label[idx] = 1;
		}
		else {
			label[idx] = 0;
		}
			
	}
}

__global__ void SparseSurfelFusion::device::CompactDenseCorrespondence(const PtrSize<ushort4> sortedPairs, const unsigned int* pairsLabel, const unsigned int* prefixSum, ushort4* compactedCorrPairs)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= sortedPairs.size) return;
	if (pairsLabel[idx] == 1) {
		compactedCorrPairs[prefixSum[idx] - 1] = sortedPairs[idx];
	}
}

__global__ void SparseSurfelFusion::device::CorrectOpticalFlowSe3MapKernel(const DeviceArrayView2D<unsigned char> markValidFlowSe3Map, const mat34 correctSe3, const unsigned int clipedImageRows, const unsigned int clipedImageCols, PtrStepSize<mat34> vertexFlowSe3)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedImageCols || y >= clipedImageRows) return;
	if (markValidFlowSe3Map(y, x) == (unsigned char)1) {
		vertexFlowSe3.ptr(y)[x] = correctSe3 * vertexFlowSe3.ptr(y)[x];
	}
}

void SparseSurfelFusion::OpticalFlow::ConvertBuffer2InputTensor(DeviceArray<uchar3>& previousImage, DeviceArray<unsigned short>& previousDepth, DeviceArray<uchar3>& currentImage, DeviceArray<unsigned short>& currentDepth, cudaStream_t stream)
{
	dim3 block(256);
	dim3 grid(divUp(rawImageSize, block.x));
	device::ConvertBuffer2InputTensorKernel << <grid, block, 0, stream >> > (previousImage.ptr(), previousDepth.ptr(), currentImage.ptr(), currentDepth.ptr(), rawImageSize, InputPreviousImage.ptr(), InputPreviousDepth.ptr(), InputCurrentImage.ptr(), InputCurrentDepth.ptr());
}


void SparseSurfelFusion::OpticalFlow::CalculatePixelPairs(cudaStream_t stream)
{
	CHECKCUDA(cudaMemsetAsync(correspondencePixelPair.Ptr(), 0, sizeof(ushort4) * clipedImageSize, stream));
	CHECKCUDA(cudaMemsetAsync(markValidPairs.Ptr(), 0, sizeof(bool) * clipedImageSize, stream));
	CHECKCUDA(cudaMemsetAsync(PixelPairsMap.ptr(), 0xFFFF, sizeof(ushort4) * clipedImageSize, stream));

	dim3 block(16, 16);
	dim3 grid(divUp(ImageColsCliped, block.x), divUp(ImageRowsCliped, block.y));
	device::CalculatePixelPairsKernel << <grid, block, 0, stream >> > (PreviousForeground, CurrentForeground, PreviousVertexMap, CurrentVertexMap, PreviousNormalMap, CurrentNormalMap, InitialCameraSE3, Flow2D.ptr(), ImageRowsCliped, ImageColsCliped, rawImageSize, correspondencePixelPair.Ptr(), markValidPairs.Ptr(), PixelPairsMap);
	int validFlowNum = 0;
	// ɸѡ����Ч��ƥ���
	int* FlowNum = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&FlowNum), sizeof(int), stream));
	void* d_temp_storage = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, correspondencePixelPair.Ptr(), markValidPairs.Ptr(), validPixelPairs.Ptr(), FlowNum, clipedImageSize, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, correspondencePixelPair.Ptr(), markValidPairs.Ptr(), validPixelPairs.Ptr(), FlowNum, clipedImageSize, stream, false));	// ɸѡ	

	CHECKCUDA(cudaMemcpyAsync(&validFlowNum, FlowNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaFreeAsync(FlowNum, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	validPixelPairs.ResizeArrayOrException(validFlowNum);

}


#ifdef DRAW_OPTICALFLOW

void SparseSurfelFusion::OpticalFlow::CalculatePixelPairAnd3DOpticalFlow(cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(ImageColsCliped, block.x), divUp(ImageRowsCliped, block.y));
	device::CalculatePixelPairAnd3DOpticalFlowKernel << <grid, block, 0, stream >> > (PreviousForeground, CurrentForeground, PreviousVertexMap, CurrentVertexMap, Flow2D.ptr(), ImageRowsCliped, ImageColsCliped, rawImageSize, correspondencePixelPair.Ptr(), markValidPairs.Ptr(), FlowVector3D, FlowVectorOpenGL, colorVertexPtr, markValidFlow);

	// ɸѡ����Ч��ƥ���
	int* FlowNum = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&FlowNum), sizeof(int), stream));
	void* d_temp_storage = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, correspondencePixelPair.Ptr(), markValidPairs.Ptr(), validPixelPairs.Ptr(), FlowNum, clipedImageSize, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, correspondencePixelPair.Ptr(), markValidPairs.Ptr(), validPixelPairs.Ptr(), FlowNum, clipedImageSize, stream, false));	// ɸѡ	
	CHECKCUDA(cudaMemcpyAsync(&validFlowNum, FlowNum, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECKCUDA(cudaFreeAsync(FlowNum, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
	//printf("��Чƥ�������� = %d\n", validFlowNum);
	validPixelPairs.ResizeArrayOrException(validFlowNum);

	/***************** �����ӻ���ͳ����Ч�Ĺ������·����к�������ɾ�� *****************/
	int* FlowNumDraw = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&FlowNumDraw), sizeof(int), stream));
	int* VertexNum = NULL;
	CHECKCUDA(cudaMallocAsync(reinterpret_cast<void**>(&VertexNum), sizeof(int), stream));
	void* d_temp_storage_1 = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes_1 = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, FlowVectorOpenGL, markValidFlow, validFlowVector, FlowNumDraw, 2 * clipedImageSize, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage_1, temp_storage_bytes_1, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_1, temp_storage_bytes_1, FlowVectorOpenGL, markValidFlow, validFlowVector, FlowNumDraw, 2 * clipedImageSize, stream, false));	// ɸѡ	
	CHECKCUDA(cudaMemcpyAsync(&validFlowNum, FlowNumDraw, sizeof(int), cudaMemcpyDeviceToHost, stream));

	void* d_temp_storage_2 = NULL;    // �м���������꼴���ͷ�
	size_t temp_storage_bytes_2 = 0;  // �м����
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, colorVertexPtr, markValidFlow, validColorVertex, VertexNum, 2 * clipedImageSize, stream, false));	// ȷ����ʱ�豸�洢����
	CHECKCUDA(cudaMallocAsync(&d_temp_storage_2, temp_storage_bytes_2, stream));
	CHECKCUDA(cub::DeviceSelect::Flagged(d_temp_storage_2, temp_storage_bytes_2, colorVertexPtr, markValidFlow, validColorVertex, VertexNum, 2 * clipedImageSize, stream, false));	// ɸѡ	
	CHECKCUDA(cudaMemcpyAsync(&validVertexNum, VertexNum, sizeof(int), cudaMemcpyDeviceToHost, stream));

	assert(validFlowNum == validVertexNum);

	draw->imshow(validFlowVector, validColorVertex, validFlowNum, stream);

	CHECKCUDA(cudaFreeAsync(FlowNumDraw, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage_1, stream));
	CHECKCUDA(cudaFreeAsync(VertexNum, stream));
	CHECKCUDA(cudaFreeAsync(d_temp_storage_2, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ����1���ͷ���ʱ�ڴ棻2����ȷ������ɴ����validFlow��ָ��
}
#endif // DRAW_OPTICALFLOW


