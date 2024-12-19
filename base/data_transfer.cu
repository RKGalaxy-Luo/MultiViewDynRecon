#include "data_transfer.h"
#include <base/CommonUtils.h>
#include <base/EncodeUtils.h>
#include "Logging.h"
#include <base/CommonUtils.h>
#include "common_point_cloud_utils.h"
#include <math/VectorUtils.h>
#include <assert.h>
#include <Eigen/Eigen>
#include <device_launch_parameters.h>

template<typename T>
__global__ void SparseSurfelFusion::device::textureToMap2DKernel(
	cudaTextureObject_t texture,
	PtrStepSize<T> map
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	T element = tex2D<T>(texture, x, y);
	map.ptr(y)[x] = element;
}

__global__ void SparseSurfelFusion::device::textureComponentToMap2DKernel(cudaTextureObject_t texture, bool xComponent, PtrStepSize<float> map)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	float2 element = tex2D<float2>(texture, x, y);
	if (xComponent)	map.ptr(y)[x] = element.x;
	else map.ptr(y)[x] = element.y;

}

__global__ void SparseSurfelFusion::device::textureToMap2DKernelfloat4(
	cudaTextureObject_t texture,
	PtrStepSize<float4> map
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	float4 element = tex2D<float4>(texture, x, y);
	if ((element.x == 1000.0) && (element.y == 1000.0) && (element.z == 1000.0) && (element.z == 1000.0)) {
		map.ptr(y)[x] = make_float4(0, 0, 0, 0);
	}
	else
	{
		
		map.ptr(y)[x] = element;
	}
}



//hsg
__global__ void SparseSurfelFusion::device::indexmapToMap2DKernel(
	cudaTextureObject_t indexmap,
	PtrStepSize<unsigned short> map
	//unsigned max
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	unsigned element = tex2D<unsigned>(indexmap, x, y);
	if (element != 0xFFFFFFFF)
	{
		map.ptr(y)[x] = (unsigned short)0xFFFF;
	}
	else
	{
		map.ptr(y)[x] = (unsigned short)0;
	}
	//unsigned char element = tex2D<unsigned char>(indexmap, x, y);
	//if (element != (unsigned char)0) {
	//	map.ptr(y)[x] = (unsigned short)0xFFFF;
	//}
	//else {
	//	map.ptr(y)[x] = (unsigned short)0;
	//}
}

__global__ void SparseSurfelFusion::device::textureToMap2DKernel(
	cudaTextureObject_t texture,
	PtrStepSize<float4> map,
	mat34 SE3
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	float4 element = tex2D<float4>(texture, x, y);
	if ((element.x == 1000.0) && (element.y == 1000.0) && (element.z == 1000.0) && (element.z == 1000.0)) {
		map.ptr(y)[x] = make_float4(0, 0, 0, 0);
	}
	else
	{
		float3 temp = SE3.rot * element + SE3.trans;
		map.ptr(y)[x] = make_float4(temp.x, temp.y, temp.z, element.w);
	}
}

__global__ void SparseSurfelFusion::device::textureToMap2DKernel(
	cudaTextureObject_t texture,
	PtrStepSize<float4> map,
	const DeviceArrayView2D<mat34> Se3Map
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	float4 element = tex2D<float4>(texture, x, y);
	if ((element.x == 1000.0) && (element.y == 1000.0) && (element.z == 1000.0) && (element.z == 1000.0)) {
		map.ptr(y)[x] = make_float4(0, 0, 0, 0);
	}
	else {
		mat34 SE3 = Se3Map(y, x);
		float3 temp = SE3.rot * element + SE3.trans;
		map.ptr(y)[x] = make_float4(temp.x, temp.y, temp.z, element.w);
	}
}

__global__ void SparseSurfelFusion::device::textureToMap2DUseSE3andCamera2WorldKernel(
	cudaTextureObject_t texture,
	PtrStepSize<float4> map,
	mat34 SE3,
	mat34 camera2world
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	float4 element = tex2D<float4>(texture, x, y);
	if ((element.x == 1000.0) && (element.y == 1000.0) && (element.z == 1000.0) && (element.z == 1000.0)) {
		map.ptr(y)[x] = make_float4(0, 0, 0, 0);
	}
	else
	{
		float3 ele = SE3.rot * element + SE3.trans;
		float3 temp = camera2world.rot * ele + camera2world.trans;
		//float3 ele = camera2world.rot * element + camera2world.trans;
		//float3 temp = SE3.rot * ele + SE3.trans;
		map.ptr(y)[x] = make_float4(temp.x, temp.y, temp.z, element.w);
	}
}

__global__ void SparseSurfelFusion::device::textureToMap2DUseSE3andCamera2WorldAndBackKernel(
	cudaTextureObject_t texture,
	PtrStepSize<float4> map,
	mat34 SE3,
	mat34 camera2world,
	mat34 SE3back
) {
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= map.cols || y >= map.rows) return;
	float4 element = tex2D<float4>(texture, x, y);
	if ((element.x == 1000.0) && (element.y == 1000.0) && (element.z == 1000.0) && (element.z == 1000.0)) {
		map.ptr(y)[x] = make_float4(0, 0, 0, 0);
	}
	else
	{
		float3 ele = SE3.rot * element + SE3.trans;
		float3 temp = camera2world.rot * ele + camera2world.trans;
		float3 result = SE3back.rot * temp + SE3back.trans;
		//float3 ele = camera2world.rot * element + camera2world.trans;
		//float3 temp = SE3.rot * ele + SE3.trans;
		map.ptr(y)[x] = make_float4(result.x, result.y, result.z, element.w);
	}
}


template<typename T>
void SparseSurfelFusion::textureToMap2D(
	cudaTextureObject_t texture,
	DeviceArray2D<T>& map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DKernel<T> << <grid, blk, 0, stream >> > (texture, map);
}

void SparseSurfelFusion::textureComponentToMap2D(cudaTextureObject_t texture, DeviceArray2D<float>& map, std::string orientation, cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	if (orientation == "x" || orientation == "X") {
		device::textureComponentToMap2DKernel << <grid, blk, 0, stream >> > (texture, true, map);
	}
	else {
		device::textureComponentToMap2DKernel << <grid, blk, 0, stream >> > (texture, false, map);
	}
}

void SparseSurfelFusion::textureToMap2Dfloat4(
	cudaTextureObject_t texture,
	DeviceArray2D<float4>& map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DKernelfloat4 << <grid, blk, 0, stream >> > (texture, map);
}


void SparseSurfelFusion::textureToMap2D(
	cudaTextureObject_t texture,
	DeviceArray2D<float4>& map,
	mat34 SE3,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DKernel<< <grid, blk, 0, stream >> > (texture, map,SE3);
}
void SparseSurfelFusion::textureToMap2D(cudaTextureObject_t texture, DeviceArray2D<float4>& map, const DeviceArrayView2D<mat34>& Se3Map, cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DKernel << <grid, blk, 0, stream >> > (texture, map, Se3Map);
}
void SparseSurfelFusion::textureToMap2DUseSE3andCamera2World(
	cudaTextureObject_t texture,
	DeviceArray2D<float4>& map,
	mat34 SE3,
	mat34 camera2world,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DUseSE3andCamera2WorldKernel << <grid, blk, 0, stream >> > (texture, map, SE3,camera2world);
}

void SparseSurfelFusion::textureToMap2DUseSE3andCamera2WorldandBack(
	cudaTextureObject_t texture,
	DeviceArray2D<float4>& map,
	mat34 SE3,
	mat34 camera2world,
	mat34 SE3back,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DUseSE3andCamera2WorldAndBackKernel << <grid, blk, 0, stream >> > (texture, map, SE3, camera2world,SE3back);
}

void SparseSurfelFusion::indexmapToMap2D(
	cudaTextureObject_t indexmap,
	DeviceArray2D<unsigned short>& map,
	unsigned max,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::indexmapToMap2DKernel << <grid, blk, 0, stream >> > (indexmap, map);
}



void SparseSurfelFusion::debugIndexMap(cudaTextureObject_t indexmap0, cudaTextureObject_t indexmap1, cudaTextureObject_t indexmap2, cv::String name)
{   
	CHECKCUDA(cudaDeviceSynchronize());
	//首先把cudaTextureObject读到DeviceArray2D里
	//然后DeviceArray2D变成mat(归一化)
	//输出mat
	//Query the size of texture
	unsigned width = 0, height = 0, max = 0;
	query2DTextureExtent(indexmap0, width, height);
	DeviceArray2D<unsigned short> map0;
	DeviceArray2D<unsigned short> map1;
	DeviceArray2D<unsigned short> map2;
	map0.create(height, width);
	map1.create(height, width);
	map2.create(height, width);

	indexmapToMap2D(indexmap0, map0, max);
	indexmapToMap2D(indexmap1, map1, max);
	indexmapToMap2D(indexmap2, map2, max);

	cv::namedWindow(name + "_0", cv::WINDOW_NORMAL);
	cv::imshow(name + "_0", downloadDepthImage(map0));
	cv::namedWindow(name + "_1", cv::WINDOW_NORMAL);
	cv::imshow(name + "_1", downloadDepthImage(map1));
	cv::namedWindow(name + "_2", cv::WINDOW_NORMAL);
	cv::imshow(name + "_2", downloadDepthImage(map2));

}


void SparseSurfelFusion::debugmask(cudaTextureObject_t mask0, cv::String name)
{
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(mask0, width, height);

	//First download to device array
	DeviceArray2D<unsigned char> map0, map1, map2;
	map0.create(height, width);

	textureToMap2D<unsigned char>(mask0, map0);

	//Donwload to host
	std::vector<unsigned char> normal_map_host0, normal_map_host1, normal_map_host2;
	int cols = width;
	map0.download(normal_map_host0, cols);

	cv::Mat rgb_cpu0(height, width, CV_8UC1);

	for (auto i = 0; i < width; i++) {
		for (auto j = 0; j < height; j++) {
			const auto flatten_idx = i + j * width;
			const unsigned char normal_value0 = normal_map_host0[flatten_idx];
			uchar1 rgb_value0;
			if (normal_value0 == (unsigned char)1) {
				rgb_value0.x = (unsigned char)255;
				rgb_cpu0.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value0.x;
			}
			else {
				rgb_value0.x = (unsigned char)0;
				rgb_cpu0.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value0.x;
			}
		}
	}
	cv::imshow(name, rgb_cpu0);

}

void SparseSurfelFusion::debugmask(cudaTextureObject_t mask0, cudaTextureObject_t mask1, cudaTextureObject_t mask2, cv::String name)
{
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(mask0, width, height);

	//First download to device array
	DeviceArray2D<unsigned char> map0, map1, map2;
	map0.create(height, width);
	map1.create(height, width);
	map2.create(height, width);
	textureToMap2D<unsigned char>(mask0, map0);
	textureToMap2D<unsigned char>(mask1, map1);
	textureToMap2D<unsigned char>(mask2, map2);

	//Donwload to host
	std::vector<unsigned char> normal_map_host0, normal_map_host1, normal_map_host2;
	int cols = width;
	map0.download(normal_map_host0, cols);
	map1.download(normal_map_host1, cols);
	map2.download(normal_map_host2, cols);

	cv::Mat rgb_cpu0(height, width, CV_8UC1);
	cv::Mat rgb_cpu1(height, width, CV_8UC1);
	cv::Mat rgb_cpu2(height, width, CV_8UC1);

	for (auto i = 0; i < width; i++) {
		for (auto j = 0; j < height; j++) {
			const auto flatten_idx = i + j * width;
			const unsigned char normal_value0 = normal_map_host0[flatten_idx];
			const unsigned char normal_value1 = normal_map_host1[flatten_idx];
			const unsigned char normal_value2 = normal_map_host2[flatten_idx];
			uchar1 rgb_value0;
			uchar1 rgb_value1;
			uchar1 rgb_value2;
			if (normal_value0 == (unsigned char)1) {
				rgb_value0.x = (unsigned char)255;
				rgb_cpu0.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value0.x;
			}
			else {
				rgb_value0.x = (unsigned char)0;
				rgb_cpu0.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value0.x;
			}
			if (normal_value1 == (unsigned char)1) {
				rgb_value1.x = (unsigned char)255;
				rgb_cpu1.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value1.x;
			}
			else {
				rgb_value1.x = (unsigned char)0;
				rgb_cpu1.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value1.x;
			}
			if (normal_value2 == (unsigned char)1) {
				rgb_value2.x = (unsigned char)255;
				rgb_cpu2.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value2.x;
			}
			else {
				rgb_value2.x = (unsigned char)0;
				rgb_cpu2.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value2.x;
			}
		}
	}
	cv::imshow(name + "_0", rgb_cpu0);
	cv::imshow(name + "_1", rgb_cpu1);
	cv::imshow(name + "_2", rgb_cpu2);
}

void SparseSurfelFusion::debugwarpedvectexmap(DeviceArray2D<unsigned short> &image_gpu)
{
	//std::vector<unsigned short> host1;
	//image_gpu.download(host1,image_gpu.cols());

	//cv::Mat rgb_cpu1(image_gpu.rows(), image_gpu.cols(), CV_8UC1);

	//for (auto i = 0; i < image_gpu.cols(); i++) {
	//	for (auto j = 0; j < image_gpu.rows(); j++) {
	//		const auto flatten_idx = i + j * image_gpu.cols();
	//		const unsigned char normal_value1 = host1[flatten_idx];
	//		uchar1 rgb_value1;
	//			rgb_value1.x = normal_value1;
	//			rgb_cpu1.at<unsigned char>(j, sizeof(uchar1) * i) = rgb_value1.x;
	//	}
	//}
	//cv::imshow("1mask", rgb_cpu1);
}

__global__ void SparseSurfelFusion::device::debugfusiondepthsurfelmap(
	cudaTextureObject_t vertex,
	cudaTextureObject_t normal,
	cudaTextureObject_t colortime
){
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= 600 || y >= 360) return;
	const float4 vertexconfidence = tex2D<float4>(vertex, x, y);
	const float4 normalradius = tex2D<float4>(normal, x, y);
	const float4 color = tex2D<float4>(colortime, x, y);
	if ((x%300==0)||(y%300==0))
	{
		printf("ver %f %f %f %f,  nor %f %f %f %f, color  %f %f\n",
			vertexconfidence.x, vertexconfidence.y, vertexconfidence.z, vertexconfidence.w,
			normalradius.x, normalradius.y, normalradius.z, normalradius.w,
			color.z, color.w);
		//uchar3 rgb;
		//float_decode_rgb(color.x, rgb);
		//printf("rgb %u %u %u\n", rgb.x, rgb.y, rgb.z);
	}

}



void SparseSurfelFusion::debugfusionDepthSurfelMap(cudaTextureObject_t vertex, cudaTextureObject_t normal, cudaTextureObject_t colortime, cudaStream_t stream)
{
	dim3 blk(16, 16);
	dim3 grid(divUp(600, blk.x), divUp(360, blk.y));
	device::debugfusiondepthsurfelmap << <grid, blk, 0, stream >> > (
		vertex,
		normal,
		colortime
		);

}

__global__ void SparseSurfelFusion::device::debugrefence(
	cudaTextureObject_t indexmap0,
	cudaTextureObject_t indexmap1,
	cudaTextureObject_t referencevertexmap0,
	cudaTextureObject_t referencevertexmap1,
	mat34 SE3, 
	DeviceArrayView<float4> CanonicalVertexConfidence)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= 600 || y >= 360) return;
	const auto surfel_index = tex2D<unsigned>(indexmap1, x, y);
	const auto surfel_index0 = tex2D<unsigned>(indexmap0, x, y);
	if (surfel_index != 0xFFFFFFFF) {
		const float4 can_vertex4 = tex2D<float4>(referencevertexmap1, x, y);
		float3 warp;
		warp = SE3.rot * can_vertex4 + SE3.trans;
		float4 vertex0;
		vertex0 = CanonicalVertexConfidence[surfel_index0];
		//printf("index %u\n", surfel_index);
		//printf("1中换到0中后的位置：%f  %f  %f \n", warp.x, warp.y, warp.z);
		//printf("(%d,%d) %u 0中的位置：%f  %f  %f \n", x,y,surfel_index0,vertex0.x, vertex0.y, vertex0.z);
		//printf("变换误差是： %f  %f  %f \n", (warp.x- vertex0.x), (warp.y- vertex0.y), (warp.z- vertex0.z));
		//printf("0:%f %f %f 1-0:%f %f %f \n", vertex0.x, vertex0.y, vertex0.z, warp.x, warp.y, warp.z);
	}
}

cv::Mat SparseSurfelFusion::downloadmaskImage(const DeviceArray2D<unsigned short>& image_gpu) {
	const auto num_rows = image_gpu.rows();
	const auto num_cols = image_gpu.cols();
	cv::Mat depth_cpu(num_rows, num_cols, CV_16UC1);
	image_gpu.download(depth_cpu.data, sizeof(unsigned short) * num_cols);
	return depth_cpu;
}

cv::Mat SparseSurfelFusion::downloadDepthImage(const DeviceArray2D<unsigned short>& image_gpu)
{
	const auto num_rows = image_gpu.rows();
	const auto num_cols = image_gpu.cols();
	cv::Mat depth_cpu(num_rows, num_cols, CV_16UC1);
	image_gpu.download(depth_cpu.data, sizeof(unsigned short) * num_cols);
	return depth_cpu;
}

cv::Mat SparseSurfelFusion::downloadDepthImage(cudaTextureObject_t image_gpu) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(image_gpu, width, height);
	DeviceArray2D<unsigned short> map;
	map.create(height, width);

	//Transfer and download
	textureToMap2D<unsigned short>(image_gpu, map);
	return downloadDepthImage(map);
}

cv::Mat SparseSurfelFusion::downloadRGBImage(
	const DeviceArray<uchar3>& image_gpu,
	const unsigned rows, const unsigned cols
) {
	assert(rows * cols == image_gpu.size());
	cv::Mat rgb_cpu(rows, cols, CV_8UC3);
	image_gpu.download((uchar3*)(rgb_cpu.data));
	return rgb_cpu;
}

cv::Mat SparseSurfelFusion::downloadNormalizeRGBImage(const DeviceArray2D<float4>& rgb_img) {
	cv::Mat rgb_cpu(rgb_img.rows(), rgb_img.cols(), CV_32FC4);
	rgb_img.download(rgb_cpu.data, sizeof(float4) * rgb_img.cols());
	return rgb_cpu;
}

cv::Mat SparseSurfelFusion::downloadNormalizeRGBImage(cudaTextureObject_t rgb_img) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(rgb_img, width, height);
	DeviceArray2D<float4> map;
	map.create(height, width);

	//Transfer and download
	textureToMap2D<float4>(rgb_img, map);
	return downloadNormalizeRGBImage(map);
}

cv::Mat SparseSurfelFusion::rgbImageFromColorTimeMap(cudaTextureObject_t color_time_map) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(color_time_map, width, height);

	//First download to device array
	DeviceArray2D<float4> map;
	map.create(height, width);
	textureToMap2D<float4>(color_time_map, map);

	//Donwload to host
	std::vector<float4> color_time_host;
	int cols = width;
	map.download(color_time_host, cols);

	cv::Mat rgb_cpu(height, width, CV_8UC3);
	for (auto i = 0; i < width; i++) {
		for (auto j = 0; j < height; j++) {
			const unsigned int flatten_idx = i + j * width;
			const float4 color_time_value = color_time_host[flatten_idx];

			//if (color_time_value.x != 0 && (color_time_value.y < 1e-3f || color_time_value.y > 2.6f)) std::cout << color_time_value.x << "    " << color_time_value.y << std::endl;

			uchar3 rgb_value;
			float_decode_rgb(color_time_value.x, rgb_value);
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 0) = rgb_value.x;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 1) = rgb_value.y;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 2) = rgb_value.z;
		}
	}
	return rgb_cpu;
}

cv::Mat SparseSurfelFusion::normalMapForVisualize(cudaTextureObject_t normal_map) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(normal_map, width, height);

	//First download to device array
	DeviceArray2D<float4> map;
	map.create(height, width);
	textureToMap2D<float4>(normal_map, map);

	//Donwload to host
	std::vector<float4> normal_map_host;
	int cols = width;
	map.download(normal_map_host, cols);

	cv::Mat rgb_cpu(height, width, CV_8UC3);
	for (auto i = 0; i < width; i++) {
		for (auto j = 0; j < height; j++) {
			const auto flatten_idx = i + j * width;
			const float4 normal_value = normal_map_host[flatten_idx];
			uchar3 rgb_value;
			rgb_value.x = (unsigned char)((normal_value.x + 1) * 120.0f);
			rgb_value.y = (unsigned char)((normal_value.y + 1) * 120.0f);
			rgb_value.z = (unsigned char)((normal_value.z + 1) * 120.0f);
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 0) = rgb_value.x;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 1) = rgb_value.y;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 2) = rgb_value.z;
		}
	}
	return rgb_cpu;
}

void SparseSurfelFusion::downloadSegmentationMask(cudaTextureObject_t mask, std::vector<unsigned char>& h_mask) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(mask, width, height);

	//Download it to device
	DeviceArray2D<unsigned char> d_mask;
	d_mask.create(height, width);
	textureToMap2D<unsigned char>(mask, d_mask);

	//Download it to host
	int h_cols;
	d_mask.download(h_mask, h_cols);
}

cv::Mat SparseSurfelFusion::downloadRawSegmentationMask(cudaTextureObject_t mask) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(mask, width, height);

	//Download it to device
	DeviceArray2D<unsigned char> d_mask;
	d_mask.create(height, width);
	textureToMap2D<unsigned char>(mask, d_mask);

	//Download it to host
	std::vector<unsigned char> h_mask_vec;
	int h_cols;
	d_mask.download(h_mask_vec, h_cols);

	cv::Mat raw_mask(height, width, CV_8UC1);
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			const auto offset = col + row * width;
			raw_mask.at<unsigned char>(row, col) = h_mask_vec[offset];
		}
	}

	return raw_mask;
}
cv::Mat SparseSurfelFusion::downloadFilteredSegmentationMask(cudaTextureObject_t filteredMask)
{
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(filteredMask, width, height);

	//Download it to device
	DeviceArray2D<float> d_mask;
	d_mask.create(height, width);
	textureToMap2D<float>(filteredMask, d_mask);

	//Download it to host
	std::vector<float> h_mask_vec;
	int h_cols;
	d_mask.download(h_mask_vec, h_cols);

	cv::Mat filteredMaskMat(height, width, CV_32FC1);
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			const auto offset = col + row * width;
			filteredMaskMat.at<float>(row, col) = h_mask_vec[offset];
		}
	}
	// 归一化到0-1范围
	cv::Mat normalizedMat;
	cv::normalize(filteredMaskMat, normalizedMat, 0, 1, cv::NORM_MINMAX);
	return normalizedMat;
}
void SparseSurfelFusion::downloadGrayScaleImage(cudaTextureObject_t image, cv::Mat& h_image, float scale) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(image, width, height);

	//Download it to device
	DeviceArray2D<float> d_meanfield;
	d_meanfield.create(height, width);
	textureToMap2D<float>(image, d_meanfield);

	//To host
	cv::Mat h_meanfield_prob = cv::Mat(height, width, CV_32FC1);
	d_meanfield.download(h_meanfield_prob.data, sizeof(float) * width);

	//Transfer it
	h_meanfield_prob.convertTo(h_image, CV_8UC1, scale * 255.f);
}

cv::Mat SparseSurfelFusion::downloadGradientComponentImage(cudaTextureObject_t image, std::string orientation)
{

	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(image, width, height);

	//Download it to device
	DeviceArray2D<float> componentMat;
	componentMat.create(height, width);

	textureComponentToMap2D(image, componentMat, orientation);

	//To host
	cv::Mat float1HostMat = cv::Mat(height, width, CV_32FC1);
	componentMat.download(float1HostMat.data, sizeof(float) * width);

	//Transfer it
	// 归一化到0-1范围
	cv::Mat normalizedComponentMat;
	cv::normalize(float1HostMat, normalizedComponentMat, 0, 1, cv::NORM_MINMAX);
	return normalizedComponentMat;
	//normalizedComponentMat.convertTo(h_image, CV_8UC1);
}

void SparseSurfelFusion::downloadTransferBinaryMeanfield(cudaTextureObject_t meanfield_q, cv::Mat& h_meanfield_uchar) {
	downloadGrayScaleImage(meanfield_q, h_meanfield_uchar);
}

/* The point cloud downloading method
 */
PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloud(const SparseSurfelFusion::DeviceArray<float4>& vertex) {
	//printf("数量时  %d\n", vertex.size());
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	std::vector<float4> h_vertex;
	vertex.download(h_vertex);
	setPointCloudSize(point_cloud, vertex.size());
	for (auto idx = 0; idx < vertex.size(); idx++) {
		setPoint(h_vertex[idx].x, h_vertex[idx].y, h_vertex[idx].z, point_cloud, idx);
	}
	return point_cloud;
}

PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloud(const DeviceArray2D<float4>& vertex_map) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));
	size_t valid_count = 0;
	setPointCloudSize(point_cloud, total_size);
	for (int idx = 0; idx < total_size; idx += 1) {
		float x = host_ptr[idx].x;
		float y = host_ptr[idx].y;
		float z = host_ptr[idx].z;
		if (std::abs(x > 1e-3) || std::abs(y > 1e-3) || std::abs(z > 1e-3)) {
			valid_count++;
		}
		setPoint(x, y, z, point_cloud, idx);
	}
	//LOG(INFO) << "The number of valid point cloud is " << valid_count << std::endl;
	delete[] host_ptr;
	return point_cloud;
}

PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloud(
	const DeviceArray2D<float4>& vertex_map,
	DeviceArrayView<unsigned int> indicator) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));

	std::vector<unsigned> h_indicator;
	indicator.Download(h_indicator);
//#ifdef WITH_CILANTRO
//	int valid_point_count = 0;
//	for (int idx = 0; idx < total_size; idx += 1) {
//		if (h_indicator[idx]) valid_point_count++;
//	}
//	setPointCloudSize(point_cloud, valid_point_count);
//#endif

	for (int idx = 0; idx < total_size; idx += 1) {
		if (h_indicator[idx]) {
			setPoint(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z, point_cloud, idx);
		}
	}
	//LOG(INFO) << "The number of valid point cloud is " << valid_count << std::endl;
	delete[] host_ptr;
	return point_cloud;
}

PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloud(
	const DeviceArray2D<float4>& vertex_map0,
	const DeviceArray2D<float4>& vertex_map1,
	DeviceArrayView<ushort3> pixel
) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	const auto num_rows = vertex_map0.rows();
	const auto num_cols = vertex_map0.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr0 = new float4[total_size];
	float4* host_ptr1 = new float4[total_size];
	vertex_map0.download(host_ptr0, num_cols * sizeof(float4));
	vertex_map1.download(host_ptr1, num_cols * sizeof(float4));
	std::vector<ushort3> h_pixels;
	pixel.Download(h_pixels);
	setPointCloudSize(point_cloud, h_pixels.size());
	//这里是两个坐标系下的 ，要改。/
	for (auto i = 0; i < h_pixels.size(); i++) {
		if (h_pixels[i].z==0)
		{
			const auto idx = h_pixels[i].x + h_pixels[i].y * vertex_map0.cols();
			setPoint(host_ptr0[idx].x, host_ptr0[idx].y, host_ptr0[idx].z, point_cloud, i);
		}
		else
		{
			const auto idx = h_pixels[i].x + h_pixels[i].y * vertex_map0.cols();
			setPoint(host_ptr1[idx].x, host_ptr1[idx].y, host_ptr1[idx].z, point_cloud, i);
		}
	}
	delete[] host_ptr0;
	delete[] host_ptr1;
	return point_cloud;
}

void SparseSurfelFusion::downloadPointCloud(const DeviceArray2D<float4>& vertex_map, std::vector<float4>& point_cloud) {
	point_cloud.clear();
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));
	for (int idx = 0; idx < total_size; idx += 1) {
		float4 point;
		point.x = host_ptr[idx].x;
		point.y = host_ptr[idx].y;
		point.z = host_ptr[idx].z;
		if (std::abs(point.x > 1e-3) || std::abs(point.y > 1e-3) || std::abs(point.z > 1e-3))
			point_cloud.push_back(point);
	}
	delete[] host_ptr;
}

PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloud(cudaTextureObject_t vertex_map) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2Dfloat4(vertex_map, vertex_map_array);
	return downloadPointCloud(vertex_map_array);
}

PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloudUseSE3(cudaTextureObject_t vertex_map, mat34 SE3)
{
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D(vertex_map, vertex_map_array, SE3);
	return downloadPointCloud(vertex_map_array);
}
PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloudUseSE3Map(cudaTextureObject_t vertex_map, const DeviceArrayView2D<mat34>& Se3Map)
{
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D(vertex_map, vertex_map_array, Se3Map);
	return downloadPointCloud(vertex_map_array);
}
PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloudUseSE3AndWorld2Camera(cudaTextureObject_t vertex_map, mat34 SE3, mat34 camera2world) 
{
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2DUseSE3andCamera2World(vertex_map, vertex_map_array, SE3,camera2world);
	return downloadPointCloud(vertex_map_array);
}
PointCloud3f_Pointer SparseSurfelFusion::downloadPointCloudUseSE3AndWorld2CameraandBack(
	cudaTextureObject_t vertex_map,
	mat34 SE3,
	mat34 camera2world,
	mat34 SEback) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2DUseSE3andCamera2WorldandBack(vertex_map, vertex_map_array, SE3, camera2world,SEback);
	return downloadPointCloud(vertex_map_array);
}



PointCloud3f_Pointer
SparseSurfelFusion::downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<unsigned int> indicator) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	return downloadPointCloud(vertex_map_array, indicator);
}

PointCloud3f_Pointer
SparseSurfelFusion::downloadPointCloud(cudaTextureObject_t vertex_map0, cudaTextureObject_t vertex_map1, DeviceArrayView<ushort3> pixel) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map0, cols, rows);
	DeviceArray2D<float4> vertex_map_array0, vertex_map_array1;
	vertex_map_array0.create(rows, cols);
	vertex_map_array1.create(rows, cols);
	textureToMap2D<float4>(vertex_map0, vertex_map_array0);
	textureToMap2D<float4>(vertex_map1, vertex_map_array1);
	return downloadPointCloud(vertex_map_array0, vertex_map_array1, pixel);
}

void SparseSurfelFusion::downloadPointCloud(cudaTextureObject_t vertex_map, std::vector<float4>& point_cloud) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	downloadPointCloud(vertex_map_array, point_cloud);
}

//#ifdef WITH_PCL
PointCloudNormal_Pointer SparseSurfelFusion::downloadNormalCloud(const DeviceArray<float4>& d_normal) {
	std::vector<float4> h_normal;
	d_normal.download(h_normal);
	PointCloudNormal_Pointer normal_cloud(new PointCloudNormal);
	for (auto idx = 0; idx < d_normal.size(); idx++) {
		setNormal(h_normal[idx].x, h_normal[idx].y, h_normal[idx].z, normal_cloud, idx);
	}
	return normal_cloud;
}
//#elif defined(WITH_CILANTRO)
//
//void surfelwarp::downloadNormalCloud(const DeviceArray<float4>& d_normal, PointCloudNormal_Pointer& point_cloud) {
//	std::vector<float4> h_normal;
//	d_normal.download(h_normal);
//	setNormalCloudSize(point_cloud, d_normal.size());
//	for (auto idx = 0; idx < d_normal.size(); idx++) {
//		setNormal(h_normal[idx].x, h_normal[idx].y, h_normal[idx].z, point_cloud, idx);
//	}
//}
//
//#endif

//#ifdef WITH_PCL
PointCloudNormal_Pointer SparseSurfelFusion::downloadNormalCloud(const DeviceArray2D<float4>& normal_map) {
	PointCloudNormal_Pointer normal_cloud(new PointCloudNormal);
	const auto num_rows = normal_map.rows();
	const auto num_cols = normal_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	normal_map.download(host_ptr, num_cols * sizeof(float4));
	int valid_count = 0;
	for (int idx = 0; idx < total_size; idx += 1) {
		float4 normal_dev = host_ptr[idx];
		FUNCTION_CHECK(!isnan(normal_dev.x));
		FUNCTION_CHECK(!isnan(normal_dev.y));
		FUNCTION_CHECK(!isnan(normal_dev.z));
		if (norm(make_float3(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z)) > 1e-4) {
			valid_count++;
		}
		setNormal(normal_dev.x, normal_dev.y, normal_dev.z, normal_cloud, idx);
	}
	//LOG(INFO) << "The number of valid normals is " << valid_count;
	delete[] host_ptr;
	return normal_cloud;
}
//#elif defined(WITH_CILANTRO)
//
//void surfelwarp::downloadNormalCloud(const DeviceArray2D<float4>& normal_map, PointCloudNormal_Pointer& point_cloud) {
//	const auto num_rows = normal_map.rows();
//	const auto num_cols = normal_map.cols();
//	const auto total_size = num_cols * num_rows;
//	float4* host_ptr = new float4[total_size];
//	normal_map.download(host_ptr, num_cols * sizeof(float4));
//	int valid_count = 0;
//	setNormalCloudSize(point_cloud, total_size);
//	for (int idx = 0; idx < total_size; idx += 1) {
//		float4 normal_dev = host_ptr[idx];
//		SURFELWARP_CHECK(!isnan(normal_dev.x));
//		SURFELWARP_CHECK(!isnan(normal_dev.y));
//		SURFELWARP_CHECK(!isnan(normal_dev.z));
//		if (norm(make_float3(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z)) > 1e-4) {
//			valid_count++;
//		}
//		setNormal(normal_dev.x, normal_dev.y, normal_dev.z, point_cloud, idx);
//	}
//	//LOG(INFO) << "The number of valid normals is " << valid_count;
//	delete[] host_ptr;
//}
//
//#endif

//#ifdef WITH_PCL
pcl::PointCloud<pcl::Normal>::Ptr SparseSurfelFusion::downloadNormalCloud(cudaTextureObject_t normal_map) {
	unsigned rows, cols;
	query2DTextureExtent(normal_map, cols, rows);
	DeviceArray2D<float4> normal_map_array;
	normal_map_array.create(rows, cols);
	textureToMap2D<float4>(normal_map, normal_map_array);
	return downloadNormalCloud(normal_map_array);
}
//#elif defined(WITH_CILANTRO)
//
//void surfelwarp::downloadNormalCloud(cudaTextureObject_t normal_map, PointCloudNormal_Pointer& point_cloud) {
//	unsigned rows, cols;
//	query2DTextureExtent(normal_map, cols, rows);
//	DeviceArray2D<float4> normal_map_array;
//	normal_map_array.create(rows, cols);
//	textureToMap2D<float4>(normal_map, normal_map_array);
//	downloadNormalCloud(normal_map_array, point_cloud);
//}
//
//#endif

void SparseSurfelFusion::downloadPointNormalCloud(
	const SparseSurfelFusion::DeviceArray<DepthSurfel>& surfel_array,
	PointCloud3f_Pointer& point_cloud,
	PointCloudNormal_Pointer& normal_cloud,
	const float point_scale
){
	//Prepare the data
	point_cloud = make_shared<pcl::PointCloud<pcl::PointXYZ>>();

	normal_cloud = PointCloudNormal_Pointer(new PointCloudNormal);

	std::vector<DepthSurfel> surfel_array_host;
	surfel_array.download(surfel_array_host);

	setPointCloudSize(point_cloud, surfel_array_host.size());
	setNormalCloudSize(normal_cloud, surfel_array_host.size());

	//Construct the output
	for (auto i = 0; i < surfel_array_host.size(); i++) {
		DepthSurfel surfel = surfel_array_host[i];
		setPoint(surfel.VertexAndConfidence.x, surfel.VertexAndConfidence.y, surfel.VertexAndConfidence.z, point_cloud, i, point_scale);
		setNormal(surfel.NormalAndRadius.x, surfel.NormalAndRadius.y, surfel.NormalAndRadius.z, normal_cloud, i);
	}
}

void SparseSurfelFusion::downloadPointNormalCloud(const DeviceArrayView<DepthSurfel> surfel_array, PointCloud3f_Pointer& point_cloud, PointCloudNormal_Pointer& normal_cloud, const float point_scale)
{
	//Prepare the data
	point_cloud = make_shared<pcl::PointCloud<pcl::PointXYZ>>();

	normal_cloud = PointCloudNormal_Pointer(new PointCloudNormal);

	std::vector<DepthSurfel> surfel_array_host;
	surfel_array.Download(surfel_array_host);

	setPointCloudSize(point_cloud, surfel_array_host.size());
	setNormalCloudSize(normal_cloud, surfel_array_host.size());

	//Construct the output
	for (auto i = 0; i < surfel_array_host.size(); i++) {
		DepthSurfel surfel = surfel_array_host[i];
		setPoint(surfel.VertexAndConfidence.x, surfel.VertexAndConfidence.y, surfel.VertexAndConfidence.z, point_cloud, i, point_scale);
		setNormal(surfel.NormalAndRadius.x, surfel.NormalAndRadius.y, surfel.NormalAndRadius.z, normal_cloud, i);
	}
}

void SparseSurfelFusion::downloadPointNormalCloud(
	const SparseSurfelFusion::DeviceArray<DepthSurfel>& surfel_array,
	PointCloud3fRGB_Pointer& point_cloud,
//#ifdef WITH_PCL
	PointCloudNormal_Pointer& normal_cloud,
//#endif
	const float point_scale
) {
	//Prepare the data
	point_cloud = make_shared< pcl::PointCloud<pcl::PointXYZRGB>>();
	normal_cloud = PointCloudNormal_Pointer(new PointCloudNormal);


	//Download it
	std::vector<DepthSurfel> surfel_array_host;
	surfel_array.download(surfel_array_host);

	setPointCloudRGBSize(point_cloud, surfel_array_host.size());
	setNormalCloudSize(normal_cloud, surfel_array_host.size());

	//Construct the output
	for (auto i = 0; i < surfel_array_host.size(); i++) {
		DepthSurfel surfel = surfel_array_host[i];
		uchar3 rgb;
		float_decode_rgb(surfel.ColorAndTime.x, rgb);
		setPointRGB(surfel.VertexAndConfidence.x, surfel.VertexAndConfidence.y, surfel.VertexAndConfidence.z, rgb.z, rgb.y, rgb.x, point_cloud, i, point_scale);
		setNormal(surfel.NormalAndRadius.x, surfel.NormalAndRadius.y, surfel.NormalAndRadius.z, normal_cloud, i);
	}
}




void SparseSurfelFusion::hsgseparateDownloadPointCloud(
	const SparseSurfelFusion::DeviceArrayView<float4>& point_cloud,
	const unsigned remainnumber,
	const unsigned number0view,
	PointCloud3f_Pointer& fused_cloud,
	PointCloud3f_Pointer& view0,
	PointCloud3f_Pointer& view1
)
{
	std::vector<float4> h_surfels;
	point_cloud.Download(h_surfels);

	int i_fused = 0;
	int i_view0 = 0;
	int i_view1 = 0;
	for (auto i = 0; i < h_surfels.size(); i++) {
		const auto flat_point = h_surfels[i];
		if (i< remainnumber) {
			setPoint(flat_point.x, flat_point.y, flat_point.z, fused_cloud, i_fused);
			i_fused++;
		}
		else if(i>= remainnumber&&i< remainnumber+ number0view)
		{
			setPoint(flat_point.x, flat_point.y, flat_point.z, view0, i_view0);
			i_view0++;
		}
		else
		{
			setPoint(flat_point.x, flat_point.y, flat_point.z, view1, i_view1);
			i_view1++;
		}
	}
}

void SparseSurfelFusion::separateDownloadPointCloud(
	const SparseSurfelFusion::DeviceArrayView<float4>& point_cloud,
	const SparseSurfelFusion::DeviceArrayView<unsigned int>& indicator,
	PointCloud3f_Pointer& fused_cloud,
	PointCloud3f_Pointer& unfused_cloud) 
{
	std::vector<float4> h_surfels;
	std::vector<unsigned> h_indicator;
	point_cloud.Download(h_surfels);
	indicator.Download(h_indicator);
	FUNCTION_CHECK(h_indicator.size() == h_surfels.size());

	int i_fused = 0;
	int i_unfused = 0;
	for (auto i = 0; i < h_surfels.size(); i++) {
		const auto indicator = h_indicator[i];
		const auto flat_point = h_surfels[i];
		if (indicator > 0) {
			setPoint(flat_point.x, flat_point.y, flat_point.z, fused_cloud, i_fused);
			i_fused++;
		}
		else {
			setPoint(flat_point.x, flat_point.y, flat_point.z, unfused_cloud, i_unfused);
			i_unfused++;
		}
	}
}

void SparseSurfelFusion::separateDownloadPointCloudWithDiffColor(
	const SparseSurfelFusion::DeviceArrayView<float4>& point_cloud,
	const SparseSurfelFusion::DeviceArrayView<unsigned int>& indicator,
	PointCloud3fRGB_Pointer& fused_cloud,
	PointCloud3fRGB_Pointer& unfused_cloud
) {
	std::vector<float4> h_surfels;
	std::vector<unsigned> h_indicator;
	point_cloud.Download(h_surfels);
	indicator.Download(h_indicator);
	FUNCTION_CHECK(h_indicator.size() == h_surfels.size());

	int i_fused = 0;
	int i_unfused = 0;
	for (auto i = 0; i < h_surfels.size(); i++) {
		const auto indicator = h_indicator[i];
		const auto flat_point = h_surfels[i];
		if (indicator > 0) {
			setPointRGB(flat_point.x, flat_point.y, flat_point.z, 255,0,0, fused_cloud, i_fused);
			i_fused++;
		}
		else {
			setPointRGB(flat_point.x, flat_point.y, flat_point.z,255,255,255, unfused_cloud, i_unfused);
			i_unfused++;
		}
	}
}



void SparseSurfelFusion::separateDownloadPointCloud(
	const SparseSurfelFusion::DeviceArrayView<float4>& point_cloud,
	unsigned num_remaining_surfels,
	PointCloud3f_Pointer& remaining_cloud,
	PointCloud3f_Pointer& appended_cloud
) {
	//Clear the existing point cloud
#ifdef WITH_PCL
	remaining_cloud->points.clear();
	appended_cloud->points.clear();
#endif
	setPointCloudSize(remaining_cloud, num_remaining_surfels);
	setPointCloudSize(appended_cloud, point_cloud.Size() - num_remaining_surfels);

	std::vector<float4> h_surfels;
	point_cloud.Download(h_surfels);
	int i_appended = 0;
	for (auto i = 0; i < point_cloud.Size(); i++) {
		const auto flat_point = h_surfels[i];
		if (i < num_remaining_surfels) {
			setPoint(flat_point.x, flat_point.y, flat_point.z, remaining_cloud, i);
		}
		else {
			setPoint(flat_point.x, flat_point.y, flat_point.z, appended_cloud, i_appended);
			i_appended++;
		}
	}
}

PointCloud3fRGB_Pointer SparseSurfelFusion::download2PointCloudWithDiffColor(
	const SparseSurfelFusion::DeviceArray<float4>& pointcloud0,
	const SparseSurfelFusion::DeviceArray<float4>& pointcloud1
) {
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());

}


/* The download function for colored point cloud
 */
PointCloud3fRGB_Pointer
SparseSurfelFusion::downloadColoredPointCloud(
	const SparseSurfelFusion::DeviceArray<float4>& vertex_confid,
	const SparseSurfelFusion::DeviceArray<float4>& color_time
) {
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());
	std::vector<float4> h_vertex, h_color_time;
	vertex_confid.download(h_vertex);
	color_time.download(h_color_time);
	FUNCTION_CHECK_EQ(h_vertex.size(), h_color_time.size());
	setPointCloudRGBSize(point_cloud, h_vertex.size());
	for (auto idx = 0; idx < h_vertex.size(); idx++) {
		float encoded_rgb = h_color_time[idx].x;
		uchar3 rgb;
		float_decode_rgb(encoded_rgb, rgb);
		setPointRGB(h_vertex[idx].x, h_vertex[idx].y, h_vertex[idx].z,
			rgb.z, rgb.y, rgb.x,
			point_cloud, idx);
	}
	return point_cloud;
}


PointCloud3fRGB_Pointer
SparseSurfelFusion::downloadColoredPointCloud(
	cudaTextureObject_t vertex_map,
	cudaTextureObject_t color_time_map,
	bool flip_color
) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array, color_map_array;
	vertex_map_array.create(rows, cols);
	color_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	textureToMap2D<float4>(color_time_map, color_map_array);

	//Download it
	float4* h_vertex = new float4[rows * cols];
	float4* h_color_time = new float4[rows * cols];
	vertex_map_array.download(h_vertex, cols * sizeof(float4));
	color_map_array.download(h_color_time, cols * sizeof(float4));

	//Construct the point cloud
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());
	setPointCloudRGBSize(point_cloud, rows * cols);
	for (auto i = 0; i < rows * cols; i++) {
		float encoded_rgb = h_color_time[i].x;
		uchar3 rgb;
		float_decode_rgb(encoded_rgb, rgb);

		if (flip_color) {
			setPointRGB(h_vertex[i].x, h_vertex[i].y, h_vertex[i].z,
				rgb.z, rgb.y, rgb.x,
				point_cloud, i);
		}
		else {
			setPointRGB(h_vertex[i].x, h_vertex[i].y, h_vertex[i].z,
				rgb.x, rgb.y, rgb.z,
				point_cloud, i);
		}
	}

	delete[] h_vertex;
	delete[] h_color_time;
	return point_cloud;
}

PointCloud3fRGB_Pointer SparseSurfelFusion::downloadColoredPointCloud(const DeviceArrayView<DepthSurfel>& vertex, const mat34& convert, const std::string type)
{
	const unsigned int vertexCount = vertex.Size();
	std::vector<DepthSurfel> vertexHost(vertexCount);
	vertex.Download(vertexHost);
	PointCloud3fRGB_Pointer pointCloud(new PointCloud3fRGB());
	for (int i = 0; i < vertexCount; i++) {
		float encodedRgb = vertexHost[i].ColorAndTime.x;
		uchar3 rgb;
		float_decode_rgb(encodedRgb, rgb);

		float4 orignalVertex = vertexHost[i].VertexAndConfidence;
		float3 convertedVertex = convert.rot * orignalVertex + convert.trans;
		if (type == "canonical") {
			setPointRGB(convertedVertex.x, convertedVertex.y, convertedVertex.z,
				255, 0, 0,
				pointCloud, i);
		}
		else {
			setPointRGB(convertedVertex.x, convertedVertex.y, convertedVertex.z,
				rgb.z, rgb.y, rgb.x,
				pointCloud, i);
		}

	}
	return pointCloud;
}


//The method to add color to point cloud
PointCloud3fRGB_Pointer SparseSurfelFusion::addColorToPointCloud(
	const PointCloud3f_Pointer& point_cloud,
	uchar4 rgba
) {
	PointCloud3fRGB_Pointer color_cloud(new PointCloud3fRGB());
	setPointCloudRGBSize(color_cloud, point_cloud->size());
	for (auto i = 0; i < point_cloud->size(); i++) {
//#ifdef WITH_PCL
		const auto& point_xyz = point_cloud->points[i];
		float x = point_xyz.x; float y = point_xyz.y; float z = point_xyz.z;
//#elif defined(WITH_CILANTRO)
//		const auto& point_xyz = point_cloud->points.col(i);
//		float x = point_xyz.x(); float y = point_xyz.y(); float z = point_xyz.z();
//#endif
		setPointRGB(x, y, z, rgba.x, rgba.y, rgba.z, color_cloud, i, 1.0f);

	}
	return color_cloud;
}

/* The index map query methods
 */
namespace SparseSurfelFusion {
	namespace device {

		__global__ void queryIndexMapFromPixelKernel(
			cudaTextureObject_t  index_map,
			const DeviceArrayView<ushort4> pixel_array,
			unsigned* index_array
		) {
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < pixel_array.Size()) {

				const auto x = pixel_array[idx].x;
				const auto y = pixel_array[idx].y;
				const auto index = tex2D<unsigned>(index_map, x, y);
				index_array[idx] = index;
			}
		}


	} // namespace device
} // namespace SparseSurfelFusion


void SparseSurfelFusion::queryIndexMapFromPixels(
	cudaTextureObject_t  index_map,
	const DeviceArrayView<ushort4>& pixel_array,
	DeviceArray<unsigned>& index_array
) {
	//Simple sanity check
	FUNCTION_CHECK_EQ(pixel_array.Size(), index_array.size());

	//Invoke the kernel
	dim3 blk(256);
	dim3 grid(pixel_array.Size(), blk.x);
	device::queryIndexMapFromPixelKernel << < grid, blk >> > (index_map, pixel_array, index_array);
}
