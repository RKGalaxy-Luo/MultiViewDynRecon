#include "CrossViewMatchingInterpolation.h"

SparseSurfelFusion::CrossViewMatchingInterpolation::CrossViewMatchingInterpolation(Intrinsic* intrinsicArray, mat34* initialPoseArray)
{
	for (int i = 0; i < devicesCount; i++) {
		// 赋初值
		interpolationInput.intrinsic[i] = intrinsicArray[i];
		interpolationInput.InitialCameraSE3[i] = initialPoseArray[i];
		interpolationInput.InitialCameraSE3Inv[i] = initialPoseArray[i].inverse();

		// 分配内存
		interVertexMap[i].create(ImageRowsCliped, ImageColsCliped);
		interNormalMap[i].create(ImageRowsCliped, ImageColsCliped);
		interColorMap[i].create(ImageRowsCliped, ImageColsCliped);
		mutexFlag[i].create(ImageRowsCliped, ImageColsCliped);
		markInterValue[i].create(ImageRowsCliped, ImageColsCliped);

		// 转换成可以在GPU上访问的类型
		interpolationOutput.interVertexMap[i] = interVertexMap[i];
		interpolationOutput.interNormalMap[i] = interNormalMap[i];
		interpolationOutput.interColorMap[i] = interColorMap[i];
		interpolationOutput.mutexFlag[i] = mutexFlag[i];
		interpolationOutput.markInterValue[i] = markInterValue[i];

		// 共享显存
		correctedIO.markInterValue[i] = DeviceArrayView2D<unsigned char>(markInterValue[i]);
		correctedIO.interVertexMap[i] = DeviceArrayView2D<float4>(interVertexMap[i]);
		correctedIO.interNormalMap[i] = DeviceArrayView2D<float4>(interNormalMap[i]);
		correctedIO.interColorMap[i] = DeviceArrayView2D<float4>(interColorMap[i]);
	}
	SetInitialValue();
}

SparseSurfelFusion::CrossViewMatchingInterpolation::~CrossViewMatchingInterpolation()
{
	for (int i = 0; i < devicesCount; i++) {
		interVertexMap[i].release();
		interNormalMap[i].release();
		interColorMap[i].release();
		mutexFlag[i].release();
		markInterValue[i].release();
	}
}

void SparseSurfelFusion::CrossViewMatchingInterpolation::SetCrossViewInterpolationInput(const CrossViewInterpolationInput& input, DeviceArrayView<CrossViewCorrPairs> crossPairs)
{
	if (devicesCount <= 1) return;
	crossCorrPairs = crossPairs;
	for (int i = 0; i < devicesCount; i++) {
		interpolationInput.VertexMap[i] = input.vertexMap[i];
		interpolationInput.NormalMap[i] = input.normalMap[i];
		interpolationInput.ColorMap[i] = input.colorMap[i];
	}
}
