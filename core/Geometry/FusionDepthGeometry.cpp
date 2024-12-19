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
	// ����Array��С
	CanonicalVertexConfidence.ResizeArrayOrException(size);
	CanonicalNormalRadius.ResizeArrayOrException(size);
	ColorTime.ResizeArrayOrException(size);

	validSurfelNum = size;
}

void SparseSurfelFusion::FusionDepthGeometry::initGeometryFromMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream)
{
	//printf("size %d\n", validSurfelArray.Size());
	ResizeValidSurfelArrays(validSurfelArray.Size());					// �����������������Ĵ�С
	FusionDepthSurfelGeometryAttributes geometryAttributes = Geometry();					// ��ȡSurfelGeometry����Ԫ���ԵĶ�дȨ��
	initSurfelGeometry(geometryAttributes, validSurfelArray, stream);	// ��������л�ȡ����Ԫ���Ը�ֵ��canonical���live�������
}

SparseSurfelFusion::FusionDepthGeometry::FusionDepthSurfelGeometryAttributes SparseSurfelFusion::FusionDepthGeometry::Geometry()
{
	FusionDepthSurfelGeometryAttributes geometryAttributes;
	geometryAttributes.canonicalVertexConfidence = CanonicalVertexConfidence.ArrayHandle();
	geometryAttributes.canonicalNormalRadius = CanonicalNormalRadius.ArrayHandle();
	geometryAttributes.colorTime = ColorTime.ArrayHandle();

	return geometryAttributes;
}

