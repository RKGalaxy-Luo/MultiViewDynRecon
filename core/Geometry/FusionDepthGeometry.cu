#include "FusionDepthGeometry.h"
#include <base/EncodeUtils.h>

namespace SparseSurfelFusion {
	namespace device {
		__global__ void initializerFusionDepthGeometry(
			DeviceArrayView<DepthSurfel> surfelArray, 
			float4* canonicalVertexConfidence, 
			float4* canonicalNormalRadius, 
			float4* colorTime
		)
		{
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < surfelArray.Size()) {
				const DepthSurfel& surfel = surfelArray[idx];
				canonicalVertexConfidence[idx] = surfel.VertexAndConfidence;
				canonicalNormalRadius[idx] = surfel.NormalAndRadius;
				colorTime[idx] = surfel.ColorAndTime;
				//if (idx % 5000 == 0) {
				//	//printf("ver %f %f %f %f,  nor %f %f %f %f, color %f %f %f %f\n",
				//	//	canonicalVertexConfidence[idx].x, canonicalVertexConfidence[idx].y, canonicalVertexConfidence[idx].z, canonicalVertexConfidence[idx].w,
				//	//	canonicalNormalRadius[idx].x, canonicalNormalRadius[idx].y, canonicalNormalRadius[idx].z, canonicalNormalRadius[idx].w,
				//	//	colorTime[idx].x, colorTime[idx].y, colorTime[idx].z, colorTime[idx].w);
				//	//printf("surfel.ColorAndTime %f %f %f %f\n", surfel.ColorAndTime.x, surfel.ColorAndTime.y, surfel.ColorAndTime.z, surfel.ColorAndTime.w);
				//	uchar3 rgb;
				//	float_decode_rgb(colorTime[idx].x,rgb);
				//	printf("rgb %u %u %u\n", rgb.x, rgb.y, rgb.z);
				//}
			}
		}
	}

	void FusionDepthGeometry::initSurfelGeometry(SparseSurfelFusion::FusionDepthGeometry::FusionDepthSurfelGeometryAttributes geometry, const DeviceArrayView<DepthSurfel>& surfelArray, cudaStream_t stream)
	{
		dim3 block(256);
		dim3 grid(divUp(surfelArray.Size(), block.x));
		device::initializerFusionDepthGeometry << <grid, block, 0, stream >> > (surfelArray, geometry.canonicalVertexConfidence, geometry.canonicalNormalRadius, geometry.colorTime);
	}

}
