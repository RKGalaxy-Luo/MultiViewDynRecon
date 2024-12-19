#include "FusionDepthGeometry.h"

SparseSurfelFusion::FusionDepthGeometry::FusionDepthGeometry() : validSurfelNum(0)
{
	CanonicalVertexConfidence.ResizeArrayOrException(0);
	CanonicalNormalRadius.ResizeArrayOrException(0);
	ColorTime.ResizeArrayOrException(0);

}

SparseSurfelFusion::FusionDepthGeometry::~FusionDepthGeometry()
{
}

void SparseSurfelFusion::FusionDepthGeometry::ResizeValidSurfelArrays(size_t size)
{
	// 调整Array大小
	CanonicalVertexConfidence.ResizeArrayOrException(size);
	CanonicalNormalRadius.ResizeArrayOrException(size);
	ColorTime.ResizeArrayOrException(size);

	validSurfelNum = size;
}

void SparseSurfelFusion::FusionDepthGeometry::initGeometryFromMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream)
{
	//printf("size %d\n", validSurfelArray.Size());
	ResizeValidSurfelArrays(validSurfelArray.Size());					// 调整各个所需容器的大小
	FusionDepthSurfelGeometryAttributes geometryAttributes = Geometry();					// 获取SurfelGeometry中面元属性的读写权限
	initSurfelGeometry(geometryAttributes, validSurfelArray, stream);	// 将从相机中获取的面元属性赋值给canonical域和live域的属性
}

SparseSurfelFusion::FusionDepthGeometry::FusionDepthSurfelGeometryAttributes SparseSurfelFusion::FusionDepthGeometry::Geometry()
{
	FusionDepthSurfelGeometryAttributes geometryAttributes;
	geometryAttributes.canonicalVertexConfidence = CanonicalVertexConfidence.ArrayHandle();
	geometryAttributes.canonicalNormalRadius = CanonicalNormalRadius.ArrayHandle();
	geometryAttributes.colorTime = ColorTime.ArrayHandle();

	return geometryAttributes;
}

