/*****************************************************************//**
 * \file   SurfelGeometry.cpp
 * \brief  ��Ԫ�ļ��ζ���
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#include "SurfelGeometry.h"

SparseSurfelFusion::SurfelGeometry::SurfelGeometry() : validSurfelNum(0)
{
	// �����ӵ�еĻ���(owned buffer)
	surfelKNN.AllocateBuffer(Constants::maxSurfelsNum);			// buffer�����ģ�Array�ǿ�����������С��
	surfelKNNWeight.AllocateBuffer(Constants::maxSurfelsNum);	// buffer�����ģ�Array�ǿ�����������С��
	liveDepthSurfelBuffer.AllocateBuffer(Constants::maxSurfelsNum);
	canonicalDepthSurfelBuffer.AllocateBuffer(Constants::maxSurfelsNum);

	// ����������Ĵ�С����Ϊ��
	CanonicalVertexConfidence.ResizeArrayOrException(0);
	CanonicalNormalRadius.ResizeArrayOrException(0);
	LiveVertexConfidence.ResizeArrayOrException(0);
	LiveNormalRadius.ResizeArrayOrException(0);
	ColorTime.ResizeArrayOrException(0);

	surfelKNN.ResizeArrayOrException(0);
	surfelKNNWeight.ResizeArrayOrException(0);
}

SparseSurfelFusion::SurfelGeometry::~SurfelGeometry()
{
	surfelKNN.ReleaseBuffer();
	surfelKNNWeight.ReleaseBuffer();
	liveDepthSurfelBuffer.ReleaseBuffer();
	canonicalDepthSurfelBuffer.ReleaseBuffer();
}

void SparseSurfelFusion::SurfelGeometry::ResizeValidSurfelArrays(size_t size)
{
	// ����Array��С
	CanonicalVertexConfidence.ResizeArrayOrException(size);
	CanonicalNormalRadius.ResizeArrayOrException(size);
	LiveVertexConfidence.ResizeArrayOrException(size);
	LiveNormalRadius.ResizeArrayOrException(size);
	ColorTime.ResizeArrayOrException(size);

	surfelKNN.ResizeArrayOrException(size);
	surfelKNNWeight.ResizeArrayOrException(size);

	validSurfelNum = size;
}

void SparseSurfelFusion::SurfelGeometry::initGeometryFromCamera(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream) 
{
	ResizeValidSurfelArrays(validSurfelArray.Size());					// �����������������Ĵ�С
	GeometryAttributes geometryAttributes = Geometry();					// ��ȡSurfelGeometry����Ԫ���ԵĶ�дȨ��
	initSurfelGeometry(geometryAttributes, validSurfelArray, stream);	// ��������л�ȡ����Ԫ���Ը�ֵ��canonical���live�������
}

SparseSurfelFusion::SurfelGeometry::GeometryAttributes SparseSurfelFusion::SurfelGeometry::Geometry()
{
	GeometryAttributes geometryAttributes;
	geometryAttributes.canonicalVertexConfidence = CanonicalVertexConfidence.ArrayHandle();
	geometryAttributes.canonicalNormalRadius = CanonicalNormalRadius.ArrayHandle();
	geometryAttributes.liveVertexConfidence = LiveVertexConfidence.ArrayHandle();
	geometryAttributes.liveNormalRadius = LiveNormalRadius.ArrayHandle();
	geometryAttributes.colorTime = ColorTime.ArrayHandle();
	return geometryAttributes;
}

SparseSurfelFusion::SurfelGeometry::SkinnerInput SparseSurfelFusion::SurfelGeometry::BindSurfelGeometrySkinnerInfo()
{
	SkinnerInput skinnerInput;
	skinnerInput.canonicalVerticesConfidence = CanonicalVertexConfidence.ArrayView();
	skinnerInput.denseSurfelsKnn = surfelKNN.ArrayHandle();
	skinnerInput.denseSurfelsKnnWeight = surfelKNNWeight.ArrayHandle();
	return skinnerInput;
}

SparseSurfelFusion::SurfelGeometry::NonRigidSolverInput SparseSurfelFusion::SurfelGeometry::BindNonRigidSolverInfo() const
{
	NonRigidSolverInput solverInput;
	solverInput.surfelKnn = surfelKNN.ArrayReadOnly();
	solverInput.surfelKnnWeight = surfelKNNWeight.ArrayReadOnly();
	return solverInput;
}

SparseSurfelFusion::SurfelGeometry::OpticalFlowGuideInput SparseSurfelFusion::SurfelGeometry::BindOpticalFlowGuideInfo()
{
	OpticalFlowGuideInput opeticalGuideInput;
	opeticalGuideInput.surfelKnn = surfelKNN.Array();
	opeticalGuideInput.surfelKnnWeight = surfelKNNWeight.Array();
	opeticalGuideInput.denseLiveSurfelsVertex = DeviceArray<float4>(LiveVertexConfidence.Ptr(), LiveVertexConfidence.ArraySize());
	opeticalGuideInput.denseLiveSurfelsNormal = DeviceArray<float4>(LiveNormalRadius.Ptr(), LiveNormalRadius.ArraySize());
	return opeticalGuideInput;
}

SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::SurfelGeometry::getLiveDepthSurfels()
{
	return liveDepthSurfelBuffer.ArrayView();
}

SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::SurfelGeometry::getCanonicalDepthSurfels()
{
	return canonicalDepthSurfelBuffer.ArrayView();
}

SparseSurfelFusion::DeviceArray<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::SurfelGeometry::getLiveDepthSurfelArrayPtr()
{
	return liveDepthSurfelBuffer.Array();
}

SparseSurfelFusion::DeviceArray<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::SurfelGeometry::getCanonicalDepthSurfelArrayPtr()
{
	return canonicalDepthSurfelBuffer.Array();
}

SparseSurfelFusion::SurfelGeometry::SurfelFusionInput SparseSurfelFusion::SurfelGeometry::SurfelFusionAccess()
{
	SurfelFusionInput fusionInput;
	fusionInput.liveVertexConfidence = LiveVertexConfidence.ArrayHandle();
	fusionInput.liveNormalRadius = LiveNormalRadius.ArrayHandle();
	fusionInput.colorTime = ColorTime.ArrayHandle();
	fusionInput.surfelKnn = surfelKNN.ArrayView();
	fusionInput.surfelKnnWeight = surfelKNNWeight.ArrayView();
	return fusionInput;
}
