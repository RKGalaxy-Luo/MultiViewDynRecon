#include "DynamicallyDrawPoints.h"

namespace SparseSurfelFusion {
	namespace device {

		//The transfer from into to float			int转float
		union uint2float_union_device_draw {
			unsigned i_value;
			float f_value;
		};

		__device__ __forceinline__ void uint_decode_rgb_device_draw(const unsigned encoded, uchar3& rgb) {
			rgb.x = ((encoded & 0x00ff0000) >> 16);
			rgb.y = ((encoded & 0x0000ff00) >> 8);
			rgb.z = ((encoded & 0x000000ff) /*0*/);
		}
		__device__ __forceinline__ unsigned float_as_uint_device_draw(float f_value) {
			device::uint2float_union_device_draw u;
			u.f_value = f_value;
			return u.i_value;
		}
		__device__ __forceinline__ void float_decode_rgb_device_draw(const float encoded, uchar3& rgb) {
			const unsigned int unsigned_encoded = float_as_uint_device_draw(encoded);
			uint_decode_rgb_device_draw(unsigned_encoded, rgb);
		}
	}
}

__global__ void SparseSurfelFusion::device::adjustModelPositionKernel(DeviceArrayView<float4> SolvedPointsCoor, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedPoints* point)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= PointNum) return;
	point[idx].coordinate = AdjustMatrix.rot * SolvedPointsCoor[idx] + AdjustMatrix.trans;
}

__global__ void SparseSurfelFusion::device::adjustModelPositionKernel(DeviceArrayView<DepthSurfel> SolvedPoints, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedPoints* point)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= PointNum) return;
	point[idx].coordinate = AdjustMatrix.rot * SolvedPoints[idx].VertexAndConfidence + AdjustMatrix.trans;
}

__global__ void SparseSurfelFusion::device::reduceMaxMinKernel(float3* maxBlockData, float3* minBlockData, DeviceArrayView<Renderer::RenderedPoints> points, const unsigned int pointsCount)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float MaxPointX[CudaThreadsPerBlock];
	__shared__ float MaxPointY[CudaThreadsPerBlock];
	__shared__ float MaxPointZ[CudaThreadsPerBlock];
	__shared__ float MinPointX[CudaThreadsPerBlock];
	__shared__ float MinPointY[CudaThreadsPerBlock];
	__shared__ float MinPointZ[CudaThreadsPerBlock];
	if (idx < pointsCount) {	// 下方reduce的时候可能会访存出界(threadIdx.x + stride >= pointsCount)，如果上面直接返回的话，__shared__数组在某个线程上必然有没赋值的情况
		MaxPointX[threadIdx.x] = points[idx].coordinate.x;
		MaxPointY[threadIdx.x] = points[idx].coordinate.y;
		MaxPointZ[threadIdx.x] = points[idx].coordinate.z;
		MinPointX[threadIdx.x] = points[idx].coordinate.x;
		MinPointY[threadIdx.x] = points[idx].coordinate.y;
		MinPointZ[threadIdx.x] = points[idx].coordinate.z;
	}
	else {
		MaxPointX[threadIdx.x] = -1e6;
		MaxPointY[threadIdx.x] = -1e6;
		MaxPointZ[threadIdx.x] = -1e6;
		MinPointX[threadIdx.x] = 1e6;
		MinPointY[threadIdx.x] = 1e6;
		MinPointZ[threadIdx.x] = 1e6;
	}

	__syncthreads();
	// 顺序寻址
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (threadIdx.x < stride) {
			float lhs, rhs;
			/** 最大值 **/
			lhs = MaxPointX[threadIdx.x];
			rhs = MaxPointX[threadIdx.x + stride];
			MaxPointX[threadIdx.x] = lhs < rhs ? rhs : lhs;

			lhs = MaxPointY[threadIdx.x];
			rhs = MaxPointY[threadIdx.x + stride];
			MaxPointY[threadIdx.x] = lhs < rhs ? rhs : lhs;

			lhs = MaxPointZ[threadIdx.x];
			rhs = MaxPointZ[threadIdx.x + stride];
			MaxPointZ[threadIdx.x] = lhs < rhs ? rhs : lhs;

			/** 最小值 **/
			lhs = MinPointX[threadIdx.x];
			rhs = MinPointX[threadIdx.x + stride];
			MinPointX[threadIdx.x] = lhs > rhs ? rhs : lhs;

			lhs = MinPointY[threadIdx.x];
			rhs = MinPointY[threadIdx.x + stride];
			MinPointY[threadIdx.x] = lhs > rhs ? rhs : lhs;

			lhs = MinPointZ[threadIdx.x];
			rhs = MinPointZ[threadIdx.x + stride];
			MinPointZ[threadIdx.x] = lhs > rhs ? rhs : lhs;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {	// 该block的第一个线程
		maxBlockData[blockIdx.x].x = MaxPointX[threadIdx.x];
		maxBlockData[blockIdx.x].y = MaxPointY[threadIdx.x];
		maxBlockData[blockIdx.x].z = MaxPointZ[threadIdx.x];

		minBlockData[blockIdx.x].x = MinPointX[threadIdx.x];
		minBlockData[blockIdx.x].y = MinPointY[threadIdx.x];
		minBlockData[blockIdx.x].z = MinPointZ[threadIdx.x];
	}
}

void SparseSurfelFusion::device::findMaxMinPoint(float3& MaxPoint, float3& MinPoint, float3* maxArray, float3* minArray, const unsigned int GridNum)
{
	// 通常不超过1000
	for (unsigned int i = 0; i < GridNum; i++) {
		//printf("minArray[%d] = (%.5f, %.5f, %.5f)   maxArray[%d] = (%.5f, %.5f, %.5f)\n", i, minArray[i].x, minArray[i].y, minArray[i].z, i, maxArray[i].x, maxArray[i].y, maxArray[i].z);

		MaxPoint.x = MaxPoint.x < maxArray[i].x ? maxArray[i].x : MaxPoint.x;
		MaxPoint.y = MaxPoint.y < maxArray[i].y ? maxArray[i].y : MaxPoint.y;
		MaxPoint.z = MaxPoint.z < maxArray[i].z ? maxArray[i].z : MaxPoint.z;

		MinPoint.x = MinPoint.x > minArray[i].x ? minArray[i].x : MinPoint.x;
		MinPoint.y = MinPoint.y > minArray[i].y ? minArray[i].y : MinPoint.y;
		MinPoint.z = MinPoint.z > minArray[i].z ? minArray[i].z : MinPoint.z;
		//printf("minArray[%d] = (%.5f, %.5f, %.5f)   maxArray[%d] = (%.5f, %.5f, %.5f)\n", i, MinPoint.x, MinPoint.y, MinPoint.z, i, MaxPoint.x, MaxPoint.y, MaxPoint.z);

	}
}

__global__ void SparseSurfelFusion::device::adjustPointsCoordinateAndColorKernel(DeviceArrayView<float4> SolvedPointsColor, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedPoints* points)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsCount) return;
	points[idx].coordinate.x = (points[idx].coordinate.x - center.x) * 2.0f / maxEdge - 1.0f;
	points[idx].coordinate.y = (points[idx].coordinate.y - center.y) * 2.0f / maxEdge - 1.0f;
	points[idx].coordinate.z = (points[idx].coordinate.z - center.z) * 2.0f / maxEdge - 1.0f;
	uchar3 colorUchar;
	float_decode_rgb_device_draw(SolvedPointsColor[idx].x, colorUchar);
	points[idx].rgb.x = colorUchar.x / 255.0f;
	points[idx].rgb.y = colorUchar.y / 255.0f;
	points[idx].rgb.z = colorUchar.z / 255.0f;
}

__global__ void SparseSurfelFusion::device::adjustPointsCoordinateAndColorKernel(DeviceArrayView<DepthSurfel> SolvedPoints, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedPoints* points)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pointsCount) return;
	//points[idx].coordinate.x = (points[idx].coordinate.x - center.x) * 2.0f / maxEdge - 1.0f;
	//points[idx].coordinate.y = (points[idx].coordinate.y - center.y) * 2.0f / maxEdge - 1.0f;
	//points[idx].coordinate.z = (points[idx].coordinate.z - center.z) * 2.0f / maxEdge - 1.0f;
	points[idx].coordinate.x = (points[idx].coordinate.x + 1.59550f) * 2.0f / 1.70285f - 1.0f;
	points[idx].coordinate.y = (points[idx].coordinate.y + 0.74243f) * 2.0f / 1.70285f - 1.0f;
	points[idx].coordinate.z = (points[idx].coordinate.z + 2.22310f) * 2.0f / 1.70285f - 1.0f;
	uchar3 colorUchar;
	float_decode_rgb_device_draw(SolvedPoints[idx].ColorAndTime.x, colorUchar);
	points[idx].rgb.x = colorUchar.x / 255.0f;
	points[idx].rgb.y = colorUchar.y / 255.0f;
	points[idx].rgb.z = colorUchar.z / 255.0f;
	//printf("maxEdge = %.5f    center = (%.5f, %.5f, %.5f)\n", maxEdge, center.x, center.y, center.z);
}

void SparseSurfelFusion::DynamicallyDrawPoints::getBoundingBox(DeviceArrayView<Renderer::RenderedPoints> points, float3& MaxPoint, float3& MinPoint, cudaStream_t stream)
{
	/********************* 归约求解每个Block最大最小Point3D *********************/
	const unsigned int num = points.Size();

	const unsigned int gridNum = divUp(num, device::CudaThreadsPerBlock);
	dim3 blockReduce(device::CudaThreadsPerBlock);
	dim3 gridReduce(gridNum);
	perBlockMaxPoint.ResizeArrayOrException(gridNum);
	perBlockMinPoint.ResizeArrayOrException(gridNum);
	float3* maxArray = perBlockMaxPoint.DeviceArray().ptr();
	float3* minArray = perBlockMinPoint.DeviceArray().ptr();
	device::reduceMaxMinKernel << <gridReduce, blockReduce, 0, stream >> > (maxArray, minArray, points, num);
	/********************* 将每个Block值下载到Host端求解所有点的最值 *********************/
	perBlockMaxPoint.SynchronizeToHost(stream, true);
	perBlockMinPoint.SynchronizeToHost(stream, true);
	float3* maxArrayHost = perBlockMaxPoint.HostArray().data();
	float3* minArrayHost = perBlockMinPoint.HostArray().data();
	device::findMaxMinPoint(MaxPoint, MinPoint, maxArrayHost, minArrayHost, gridNum);
}

void SparseSurfelFusion::DynamicallyDrawPoints::adjustPointsCoordinateAndColor(DeviceArrayView<float4> SolvedPointsColor, const float3 MxPoint, const float3 MnPoint, cudaStream_t stream)
{
	const unsigned int totalPointsNum = SolvedPointsColor.Size();
	RenderedFusedPoints.ResizeArrayOrException(totalPointsNum);
	float3 Center = make_float3(float(MxPoint.x + MnPoint.x) / 2.0f, float(MxPoint.y + MnPoint.y) / 2.0f, float(MxPoint.z + MnPoint.z) / 2.0f);
	float MaxEdge = float(MxPoint.x - MnPoint.x);
	if (MaxEdge < MxPoint.y - MnPoint.y) { MaxEdge = float(MxPoint.y - MnPoint.y); }
	if (MaxEdge < MxPoint.z - MnPoint.z) { MaxEdge = float(MxPoint.z - MnPoint.z); }

	Center = make_float3(Center.x - (MaxEdge / 2.0f), Center.y - (MaxEdge / 2.0f), Center.z - (MaxEdge / 2.0f));
	dim3 block(128);
	dim3 grid(divUp(totalPointsNum, block.x));
	device::adjustPointsCoordinateAndColorKernel << <grid, block, 0, stream >> > (SolvedPointsColor, Center, MaxEdge, totalPointsNum, RenderedFusedPoints.Ptr());
}

void SparseSurfelFusion::DynamicallyDrawPoints::adjustPointsCoordinateAndColor(DeviceArrayView<DepthSurfel> SolvedPoints, const float3 MxPoint, const float3 MnPoint, cudaStream_t stream)
{
	const unsigned int totalPointsNum = SolvedPoints.Size();
	RenderedFusedPoints.ResizeArrayOrException(totalPointsNum);
	float3 Center = make_float3(float(MxPoint.x + MnPoint.x) / 2.0f, float(MxPoint.y + MnPoint.y) / 2.0f, float(MxPoint.z + MnPoint.z) / 2.0f);
	float MaxEdge = float(MxPoint.x - MnPoint.x);
	if (MaxEdge < MxPoint.y - MnPoint.y) { MaxEdge = float(MxPoint.y - MnPoint.y); }
	if (MaxEdge < MxPoint.z - MnPoint.z) { MaxEdge = float(MxPoint.z - MnPoint.z); }

	Center = make_float3(Center.x - (MaxEdge / 2.0f), Center.y - (MaxEdge / 2.0f), Center.z - (MaxEdge / 2.0f));
	dim3 block(128);
	dim3 grid(divUp(totalPointsNum, block.x));
	device::adjustPointsCoordinateAndColorKernel << <grid, block, 0, stream >> > (SolvedPoints, Center, MaxEdge, totalPointsNum, RenderedFusedPoints.Ptr());
}

void SparseSurfelFusion::DynamicallyDrawPoints::adjustModelPosition(DeviceArrayView<float4> SolvedPointsCoor, cudaStream_t stream)
{
	const unsigned int PointsNum = SolvedPointsCoor.Size();
	RenderedFusedPoints.ResizeArrayOrException(PointsNum);
	dim3 block(128);
	dim3 grid(divUp(PointsNum, block.x));
	device::adjustModelPositionKernel << <grid, block, 0, stream >> > (SolvedPointsCoor, AdjustModelSE3, PointsNum, RenderedFusedPoints.Ptr());
}

void SparseSurfelFusion::DynamicallyDrawPoints::adjustModelPosition(DeviceArrayView<DepthSurfel> SolvedPoints, cudaStream_t stream)
{
	const unsigned int PointsNum = SolvedPoints.Size();
	RenderedFusedPoints.ResizeArrayOrException(PointsNum);
	dim3 block(128);
	dim3 grid(divUp(PointsNum, block.x));
	device::adjustModelPositionKernel << <grid, block, 0, stream >> > (SolvedPoints, AdjustModelSE3, PointsNum, RenderedFusedPoints.Ptr());
}