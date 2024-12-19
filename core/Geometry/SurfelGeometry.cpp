/*****************************************************************//**
 * \file   SurfelGeometry.cpp
 * \brief  面元的几何对象
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#include "SurfelGeometry.h"

SparseSurfelFusion::SurfelGeometry::SurfelGeometry() : validSurfelNum(0)
{
	// 分配可拥有的缓存(owned buffer)
	surfelKNN.AllocateBuffer(Constants::maxSurfelsNum);			// buffer是死的，Array是可以灵活调整大小的
	surfelKNNWeight.AllocateBuffer(Constants::maxSurfelsNum);	// buffer是死的，Array是可以灵活调整大小的
	liveDepthSurfelBuffer.AllocateBuffer(Constants::maxSurfelsNum);
	canonicalDepthSurfelBuffer.AllocateBuffer(Constants::maxSurfelsNum);

	// 将所有数组的大小调整为零
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
	// 调整Array大小
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
	ResizeValidSurfelArrays(validSurfelArray.Size());					// 调整各个所需容器的大小
	GeometryAttributes geometryAttributes = Geometry();					// 获取SurfelGeometry中面元属性的读写权限
	initSurfelGeometry(geometryAttributes, validSurfelArray, stream);	// 将从相机中获取的面元属性赋值给canonical域和live域的属性
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
