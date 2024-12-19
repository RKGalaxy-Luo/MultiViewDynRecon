#include "SurfelGeometry.h"

__global__ void SparseSurfelFusion::device::initializerCollectDepthSurfelKernel(DeviceArrayView<DepthSurfel> surfelArray, float4* canonicalVertexConfidence, float4* canonicalNormalRadius, float4* liveVertexConfidence, float4* liveNormalRadius, float4* colorTime/*,int* flag*/)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < surfelArray.Size()) {
		const DepthSurfel& surfel = surfelArray[idx];
		canonicalVertexConfidence[idx] = surfel.VertexAndConfidence;	// �ʼ��ʼ��canonical����live����Ԫ��������ȫһ�µ�
		canonicalNormalRadius[idx] = surfel.NormalAndRadius;
		liveVertexConfidence[idx] = surfel.VertexAndConfidence;
		liveNormalRadius[idx] = surfel.NormalAndRadius;
		colorTime[idx] = surfel.ColorAndTime;
	}
}

__global__ void SparseSurfelFusion::device::collectDepthSurfel(DeviceArrayHandle<DepthSurfel> surfelArray, DeviceArrayHandle<DepthSurfel> surfelArrayRef, float4* canonicalVertexConfidence, float4* canonicalNormalRadius, float4* liveVertexConfidence, float4* liveNormalRadius, float4* colorTime) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//��������߾����úô�С
	if (idx < surfelArray.Size()) {
		surfelArray[idx].VertexAndConfidence = liveVertexConfidence[idx];
		surfelArray[idx].NormalAndRadius = liveNormalRadius[idx];
		surfelArray[idx].ColorAndTime = colorTime[idx];

		PixelCoordinate pix;
		surfelArray[idx].pixelCoordinate = pix;
		surfelArrayRef[idx].VertexAndConfidence = canonicalVertexConfidence[idx];
		surfelArrayRef[idx].NormalAndRadius = canonicalNormalRadius[idx];
		surfelArrayRef[idx].ColorAndTime = colorTime[idx];
		surfelArrayRef[idx].pixelCoordinate = pix;
	}
}

void SparseSurfelFusion::SurfelGeometry::initSurfelGeometry(GeometryAttributes geometry, const DeviceArrayView<DepthSurfel>& surfelArray, cudaStream_t stream)
{
	dim3 block(256);
	dim3 grid(divUp(surfelArray.Size(), block.x));
	device::initializerCollectDepthSurfelKernel << <grid, block, 0, stream >> > (surfelArray, geometry.canonicalVertexConfidence, geometry.canonicalNormalRadius, geometry.liveVertexConfidence, geometry.liveNormalRadius, geometry.colorTime);
}

unsigned int SparseSurfelFusion::SurfelGeometry::collectLiveandCanDepthSurfel(cudaStream_t stream)
{
	liveDepthSurfelBuffer.ResizeArrayOrException(CanonicalVertexConfidence.ArraySize());
	canonicalDepthSurfelBuffer.ResizeArrayOrException(LiveVertexConfidence.ArraySize());
	FUNCTION_CHECK_EQ(liveDepthSurfelBuffer.ArraySize(), canonicalDepthSurfelBuffer.ArraySize());
#ifdef DEBUG_RUNNING_INFO
	printf("��ǰ֡��Ԫ���� = %u\n",liveDepthSurfelBuffer.ArraySize());
#endif // DEBUG_RUNNING_INFO

	dim3 block(256);
	dim3 grid(divUp(CanonicalVertexConfidence.ArraySize(), block.x));
	device::collectDepthSurfel << <grid, block, 0, stream >> > (
		liveDepthSurfelBuffer.ArrayHandle(),
		canonicalDepthSurfelBuffer.ArrayHandle(),
		CanonicalVertexConfidence.ArrayHandle(),
		CanonicalNormalRadius.ArrayHandle(),
		LiveVertexConfidence.ArrayHandle(),
		LiveNormalRadius.ArrayHandle(),
		ColorTime.ArrayHandle()
	);
	CHECKCUDA(cudaStreamSynchronize(stream));
	return liveDepthSurfelBuffer.ArraySize();
}