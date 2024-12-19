#include "CrossViewMatchingInterpolation.h"

__device__ SparseSurfelFusion::mat34 SparseSurfelFusion::device::ScrewInterpolationMat(DualQuaternion dq, float ratio)
{
	float t = ratio;
	if (t > 1.0f) t = 1.0f;
	else if (t < 1e-10f) t = 1e-10f;

	Quaternion q0Scaled = dq.q0.pow(t);								// 缩放旋转部分
	Quaternion q1Scaled = ((dq.q1 * dq.q0.conjugate()) * t) * q0Scaled;	// 缩放位移部分

	DualQuaternion InterpolationDq = DualQuaternion(q0Scaled, q1Scaled);

	return InterpolationDq;
}

__global__ void SparseSurfelFusion::device::SurfelsInterpolationKernel(CrossViewInterpolateInput input, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, const unsigned int Cols, const unsigned int Rows, const unsigned int pairsNum, const float disThreshold, CrossViewInterpolateOutput output)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= pairsNum) return;

	// 投影平面以View.x为准
	ushort2 crossView = crossCorrPairs[idx].PixelViews;
	ushort4 crossPair = crossCorrPairs[idx].PixelPairs;

	float4 vertex_1 = tex2D<float4>(input.VertexMap[crossView.x], crossPair.x, crossPair.y);
	float4 normal_1 = tex2D<float4>(input.NormalMap[crossView.x], crossPair.x, crossPair.y);

	float4 vertex_2 = tex2D<float4>(input.VertexMap[crossView.y], crossPair.z, crossPair.w);
	float4 normal_2 = tex2D<float4>(input.NormalMap[crossView.y], crossPair.z, crossPair.w);

	float3 vertex_world_1 = input.InitialCameraSE3[crossView.x].rot * vertex_1 + input.InitialCameraSE3[crossView.x].trans;
	float3 normal_world_1 = input.InitialCameraSE3[crossView.x].rot * normal_1;

	float3 vertex_world_2 = input.InitialCameraSE3[crossView.y].rot * vertex_2 + input.InitialCameraSE3[crossView.y].trans;
	float3 normal_world_2 = input.InitialCameraSE3[crossView.y].rot * normal_2;


	unsigned short interpolatedNum = 0;		// interpolatedNum = 0表示不进行任何插值

	float3 interplatedVertex[3] = { (0.25f * vertex_world_2 + 0.75f * vertex_world_1), (0.5f * vertex_world_2 + 0.5f * vertex_world_1), (0.75 * vertex_world_2 + 0.25 * vertex_world_1) };
	float3 interplatedNormal[3] = { (0.25f * normal_world_2 + 0.75f * normal_world_1), (0.5f * normal_world_2 + 0.5f * normal_world_1), (0.75 * normal_world_2 + 0.25 * normal_world_1) };

	float corrPairSquDis = squared_distance(vertex_world_1, vertex_world_2);
	if (1e-4f < corrPairSquDis && corrPairSquDis <= 9e-4f) {		// 距离:(1cm, 3cm]   插1个点
		interpolatedNum = 1;
	}
	else if (9e-4f < corrPairSquDis && corrPairSquDis <= 5e-3f) {	// 距离:(3cm, 7cm]   插3个点
		interpolatedNum = 3;
		// 优先插值中心点
	}
	else {
		// 不插入点
		interpolatedNum = 0;
	}

	//// Check插值是否是正常插值
	//for (int i = 0; i < interpolatedNum; i++) {
	//	// 与两个匹配点各自距离，取最小作为偏移值
	//	float interVertexDeviation = squared_distance(interplatedVertex[i], vertex_world_1);
	//	interVertexDeviation = min(interVertexDeviation, squared_distance(interplatedVertex[i], vertex_world_2));
	//	if (interVertexDeviation > corrPairSquDis || interVertexDeviation > disThreshold) {
	//		interpolatedNum = 0;
	//		break;	// 只要存在一个异常插值，不进行任何插值
	//	}
	//}

	// 正常插值, 将插值点转到
	for (int i = 0; i < interpolatedNum; i++) {
/************************************************ 投影到第1个视角存储 ************************************************/
		float3 interVertexCamera = input.InitialCameraSE3Inv[crossView.x].rot * interplatedVertex[i] + input.InitialCameraSE3Inv[crossView.x].trans;
		// 投影到像素平面
		ushort2 ProjCoor_1 = {
			__float2uint_rn(((interVertexCamera.x / (interVertexCamera.z + 1e-10)) * input.intrinsic[crossView.x].focal_x) + input.intrinsic[crossView.x].principal_x),
			__float2uint_rn(((interVertexCamera.y / (interVertexCamera.z + 1e-10)) * input.intrinsic[crossView.x].focal_y) + input.intrinsic[crossView.x].principal_y)
		};
		if (ProjCoor_1.x < Cols && ProjCoor_1.y < Rows) {
			float3 interNormalCamera = input.InitialCameraSE3Inv[crossView.x].rot * interplatedNormal[i];
			interNormalCamera = normalized(interNormalCamera);
			// 检查原子性，没有被赋值才能在此处赋值，先到先得
			if (atomicCAS(&(output.mutexFlag[crossView.x].ptr(ProjCoor_1.y)[ProjCoor_1.x]), 0, 1) == 0) {

				// 此时似乎是线程安全的
				output.markInterValue[crossView.x].ptr(ProjCoor_1.y)[ProjCoor_1.x] = (unsigned char)1;
				output.interVertexMap[crossView.x].ptr(ProjCoor_1.y)[ProjCoor_1.x] = make_float4(interVertexCamera.x, interVertexCamera.y, interVertexCamera.z, vertex_1.w);
				output.interNormalMap[crossView.x].ptr(ProjCoor_1.y)[ProjCoor_1.x] = make_float4(interNormalCamera.x, interNormalCamera.y, interNormalCamera.z, normal_1.w);
				output.interColorMap[crossView.x].ptr(ProjCoor_1.y)[ProjCoor_1.x] = tex2D<float4>(input.ColorMap[crossView.x], crossPair.x, crossPair.y);
			}
		}

/************************************************ 投影到第2个视角存储 ************************************************/
		interVertexCamera = input.InitialCameraSE3Inv[crossView.y].rot * interplatedVertex[i] + input.InitialCameraSE3Inv[crossView.y].trans;
		// 投影到像素平面
		ushort2 ProjCoor_2 = {
			__float2uint_rn(((interVertexCamera.x / (interVertexCamera.z + 1e-10)) * input.intrinsic[crossView.y].focal_x) + input.intrinsic[crossView.y].principal_x),
			__float2uint_rn(((interVertexCamera.y / (interVertexCamera.z + 1e-10)) * input.intrinsic[crossView.y].focal_y) + input.intrinsic[crossView.y].principal_y)
		};
		if (ProjCoor_2.x < Cols && ProjCoor_2.y < Rows) {
			float3 interNormalCamera = input.InitialCameraSE3Inv[crossView.y].rot * interplatedNormal[i];
			interNormalCamera = normalized(interNormalCamera);
			// 检查原子性，没有被赋值才能在此处赋值，先到先得
			if (atomicCAS(&(output.mutexFlag[crossView.y].ptr(ProjCoor_2.y)[ProjCoor_2.x]), 0, 1) == 0) {
				// 此时似乎是线程安全的
				output.markInterValue[crossView.y].ptr(ProjCoor_2.y)[ProjCoor_2.x] = (unsigned char)1;
				output.interVertexMap[crossView.y].ptr(ProjCoor_2.y)[ProjCoor_2.x] = make_float4(interVertexCamera.x, interVertexCamera.y, interVertexCamera.z, vertex_2.w);
				output.interNormalMap[crossView.y].ptr(ProjCoor_2.y)[ProjCoor_2.x] = make_float4(interNormalCamera.x, interNormalCamera.y, interNormalCamera.z, normal_2.w);
				output.interColorMap[crossView.y].ptr(ProjCoor_2.y)[ProjCoor_2.x] = tex2D<float4>(input.ColorMap[crossView.y], crossPair.z, crossPair.w);
			}
		}
	}
}

__global__ void SparseSurfelFusion::device::CorrectObservedTextureKernel(CorrectTextureIO io, const unsigned int Cols, const unsigned int Rows, const unsigned int CameraNum)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	const unsigned int view = threadIdx.z + blockDim.z * blockIdx.z;
	// 没有有效插值的地方直接略过
	if (x >= Cols || y >= Rows || view >= CameraNum || io.markInterValue[view](y, x) == (unsigned char)0) return;
	
	float4 observedVertex = tex2D<float4>(io.VertexTextureMap[view], x, y);
	if (is_zero_vertex(observedVertex)) {	// 观测值这里是空的
		// 直接用插值的值替代原始观测值
		float4 interVertex, interNormal, interColor;
		interVertex = io.interVertexMap[view](y, x);
		interNormal = io.interNormalMap[view](y, x);
		interColor = io.interColorMap[view](y, x);
		surf2Dwrite(interVertex, io.VertexSurfaceMap[view], x * sizeof(float4), y);
		surf2Dwrite(interNormal, io.NormalSurfaceMap[view], x * sizeof(float4), y);
		surf2Dwrite(interColor, io.ColorSurfaceMap[view], x * sizeof(float4), y);
	}
}

void SparseSurfelFusion::CrossViewMatchingInterpolation::CrossViewInterpolateSurfels(cudaStream_t stream)
{
	if (devicesCount <= 1) return;

	// 初始化数据
	SetInitialValue(stream);
	dim3 block(64);
	dim3 grid(divUp(crossCorrPairs.Size(), block.x));
	device::SurfelsInterpolationKernel << <grid, block, 0, stream >> > (interpolationInput, crossCorrPairs, ImageColsCliped, ImageRowsCliped, crossCorrPairs.Size(), Constants::CrossViewPairsSquaredDisThreshold, interpolationOutput);
}

void SparseSurfelFusion::CrossViewMatchingInterpolation::CorrectVertexNormalColorTexture(CudaTextureSurface* VertexMap, CudaTextureSurface* NormalMap, CudaTextureSurface* ColorMap, cudaStream_t stream)
{
	if (devicesCount <= 1) return;

	for (int i = 0; i < devicesCount; i++) {
		// 观测数据
		correctedIO.VertexTextureMap[i] = VertexMap[i].texture;
		correctedIO.NormalTextureMap[i] = NormalMap[i].texture;
		correctedIO.ColorTextureMap[i] = ColorMap[i].texture;

		correctedIO.VertexSurfaceMap[i] = VertexMap[i].surface;
		correctedIO.NormalSurfaceMap[i] = NormalMap[i].surface;
		correctedIO.ColorSurfaceMap[i] = ColorMap[i].surface;
	}

	dim3 block(16, 16, 1);
	dim3 grid(divUp(ImageColsCliped, block.x), divUp(ImageRowsCliped, block.y), divUp(devicesCount, block.z));
	device::CorrectObservedTextureKernel << <grid, block, 0, stream >> > (correctedIO, ImageColsCliped, ImageRowsCliped, devicesCount);
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::CrossViewMatchingInterpolation::SetInitialValue(cudaStream_t stream)
{
	for (int i = 0; i < devicesCount; i++) {
		CHECKCUDA(cudaMemsetAsync(mutexFlag[i].ptr(), 0, sizeof(unsigned int) * clipedImageSize, stream));
		CHECKCUDA(cudaMemsetAsync(markInterValue[i].ptr(), 0, sizeof(unsigned char) * clipedImageSize, stream));

		CHECKCUDA(cudaMemsetAsync(interVertexMap[i].ptr(), 0, sizeof(float4) * clipedImageSize, stream));
		CHECKCUDA(cudaMemsetAsync(interNormalMap[i].ptr(), 0, sizeof(float4) * clipedImageSize, stream));
		CHECKCUDA(cudaMemsetAsync(interColorMap[i].ptr(), 0, sizeof(float4) * clipedImageSize, stream));
	}

}
