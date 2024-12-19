/*****************************************************************//**
 * \file   ImageProcessByGPU.cu
 * \brief  主要涉及一些用GPU处理图像的操作：剪裁，滤波等
 * 
 * \author LUO
 * \date   January 29th 2024
 *********************************************************************/
#include "ImageProcessByGPU.h"

namespace SparseSurfelFusion {
	namespace device {
		__device__ unsigned int devicesCount = MAX_CAMERA_COUNT;
	}
}

__global__ void SparseSurfelFusion::device::clearMapSurfelKernel(MapMergedSurfelInterface mergedSurfel, const unsigned int clipedWidth, const unsigned int clipedHeight) {
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedWidth || y >= clipedHeight) return;
	
	for (int i = 0; i < device::devicesCount; i++) {
		surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 0.0f), mergedSurfel.vertex[i], x * sizeof(float4), y);
		surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 0.0f), mergedSurfel.normal[i], x * sizeof(float4), y);
		surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 0.0f), mergedSurfel.color[i], x * sizeof(float4), y);
	}
}



__global__ void SparseSurfelFusion::device::mapMergedDepthSurfelKernel(const DeviceArrayView<DepthSurfel> validSurfelArray, MapMergedSurfelInterface mergedSurfelInterface, const unsigned int validSurfelNum, const unsigned int clipedWidth, const unsigned int clipedHeight)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= validSurfelNum) return;
	const DepthSurfel mergedSurfels = validSurfelArray[idx];
	for (int i = 0; i < device::devicesCount; i++) {
		if (mergedSurfels.CameraID == i) {
			float3 vertex, normal;
			vertex = mergedSurfelInterface.InitialCameraSE3Inverse[i].rot * mergedSurfels.VertexAndConfidence + mergedSurfelInterface.InitialCameraSE3Inverse[i].trans;
			normal = mergedSurfelInterface.InitialCameraSE3Inverse[i].rot * mergedSurfels.NormalAndRadius;
			float4 vertexConfidence = make_float4(vertex.x, vertex.y, vertex.z, mergedSurfels.VertexAndConfidence.w);
			float4 normalRadius = make_float4(normal.x, normal.y, normal.z, mergedSurfels.NormalAndRadius.w);
			//映射
			const int2 img_coord = {
				__float2int_rn(((vertex.x / (vertex.z + 1e-10)) * mergedSurfelInterface.ClipedIntrinsic[i].focal_x) + mergedSurfelInterface.ClipedIntrinsic[i].principal_x),
				__float2int_rn(((vertex.y / (vertex.z + 1e-10)) * mergedSurfelInterface.ClipedIntrinsic[i].focal_y) + mergedSurfelInterface.ClipedIntrinsic[i].principal_y)
			};
			if (img_coord.x >= 0 && img_coord.x < clipedWidth && img_coord.y >= 0 && img_coord.y < clipedHeight) {
				surf2Dwrite(vertexConfidence, mergedSurfelInterface.vertex[i], img_coord.x * sizeof(float4), img_coord.y);
				surf2Dwrite(normalRadius, mergedSurfelInterface.normal[i], img_coord.x * sizeof(float4), img_coord.y);
				surf2Dwrite(mergedSurfels.ColorAndTime, mergedSurfelInterface.color[i], img_coord.x * sizeof(float4), img_coord.y);
			}
		}
	}
}

__global__ void SparseSurfelFusion::device::constrictMultiViewForegroundKernel(cudaSurfaceObject_t depthMap, cudaSurfaceObject_t vertexMap, cudaSurfaceObject_t normalMap, cudaSurfaceObject_t colorMap, MultiViewMaskInterface MultiViewInterface, const unsigned int CameraID, const unsigned int clipedWidth, const unsigned int clipedHeight)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= clipedWidth || y >= clipedHeight) return;
	unsigned short depth = tex2D<unsigned short>(MultiViewInterface.depthMap[CameraID], x, y);
	const float4 vertex = tex2D<float4>(MultiViewInterface.vertexMap[CameraID], x, y);
	const float4 normal = tex2D<float4>(MultiViewInterface.normalMap[CameraID], x, y);
	const float4 color = tex2D<float4>(MultiViewInterface.colorMap[CameraID], x, y);
	const float4 zeroFloat4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int i = 0; i < device::devicesCount; i++) {
		if (depth != 0) {
			float3 coor = make_float3(0.0f, 0.0f, 0.0f);
			if (CameraID != i) {
				// 先转Canonical域，再转到对应的i的相机坐标系
				coor = MultiViewInterface.InitialCameraSE3[CameraID].rot * vertex + MultiViewInterface.InitialCameraSE3[CameraID].trans;
				coor = MultiViewInterface.InitialCameraSE3Inverse[i].rot * coor + MultiViewInterface.InitialCameraSE3Inverse[i].trans;
			}
			else {
				coor = make_float3(vertex.x, vertex.y, vertex.z);
			}
			const ushort2 imageCoordinate = {
				__float2uint_rn(((coor.x / (coor.z + 1e-10)) * MultiViewInterface.ClipedIntrinsic[i].focal_x) + MultiViewInterface.ClipedIntrinsic[i].principal_x),
				__float2uint_rn(((coor.y / (coor.z + 1e-10)) * MultiViewInterface.ClipedIntrinsic[i].focal_y) + MultiViewInterface.ClipedIntrinsic[i].principal_y)
			};

			if (imageCoordinate.x < clipedWidth && imageCoordinate.y < clipedHeight) {
				unsigned char mask = tex2D<unsigned char>(MultiViewInterface.foreground[i], imageCoordinate.x, imageCoordinate.y);
				if (mask != (unsigned char)1) depth = 0;
			}
			else {
				depth = 0;
			}
		}
		else break;
	}
	surf2Dwrite(depth, depthMap, x * sizeof(unsigned short), y);
	if (depth > 0) {
		surf2Dwrite(vertex, vertexMap, x * sizeof(float4), y);
		surf2Dwrite(normal, normalMap, x * sizeof(float4), y);
		surf2Dwrite(color, colorMap, x * sizeof(float4), y);
	}
	else {
		surf2Dwrite(zeroFloat4, vertexMap, x * sizeof(float4), y);
		surf2Dwrite(zeroFloat4, normalMap, x * sizeof(float4), y);
		surf2Dwrite(zeroFloat4, colorMap, x * sizeof(float4), y);
	}
}


__global__ void SparseSurfelFusion::device::clipFilterDepthKernel(
	cudaTextureObject_t rawDepth, 
	const unsigned int clipImageRows, 
	const unsigned int clipImageCols, 
	const unsigned int clipNear, 
	const unsigned int clipFar, 
	const float sigmaSInverseSquare, 
	const float sigmaRInverseSquare, 
	cudaSurfaceObject_t filterDepth)
{
	//平行于被剪切的图像
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (y >= clipImageRows || x >= clipImageCols) return;
/*********************************** 对深度图进行双边滤波 ***********************************/
	//计算原始深度的中心
	const int half_width = 5;
	const unsigned int raw_x = x + CLIP_BOUNDARY;
	const unsigned int raw_y = y + CLIP_BOUNDARY;
	const unsigned short center_depth = tex2D<unsigned short>(rawDepth, raw_x, raw_y);

	//遍历窗口
	float sum_all = 0.0f; 
	float sum_weight = 0.0f;
	for (auto y_idx = raw_y - half_width; y_idx <= raw_y + half_width; y_idx++) {
		for (auto x_idx = raw_x - half_width; x_idx <= raw_x + half_width; x_idx++) {
			const unsigned short depth = tex2D<unsigned short>(rawDepth, x_idx, y_idx);
			const float depth_diff2 = (depth - center_depth) * (depth - center_depth);
			const float pixel_diff2 = (x_idx - raw_x) * (x_idx - raw_x) + (y_idx - raw_y) * (y_idx - raw_y);
			const float this_weight = (depth > 0) * expf(-sigmaSInverseSquare * pixel_diff2) * expf(-sigmaRInverseSquare * depth_diff2);
			sum_weight += this_weight;
			sum_all += this_weight * depth;
		}
	}
	//滤波后的深度放到filtered_depth_value
	unsigned short filtered_depth_value = __float2uint_rn(sum_all / sum_weight);

/*********************************** 双边滤波完成后的像素传入cuda纹理内存 ***********************************/
	// 如果深度小于最小距离，或者大于最大距离，将其设置为0
	if (filtered_depth_value < clipNear || filtered_depth_value > clipFar) filtered_depth_value = 0;
	// 将filtered_depth_value的值写入filterDepth
	surf2Dwrite(filtered_depth_value, filterDepth, x * sizeof(unsigned short), y);
}

__global__ void SparseSurfelFusion::device::clipNormalizeColorKernel(
	const PtrSize<const uchar3> rawColorImage, 
	const unsigned int clipRows, 
	const unsigned int clipCols, 
	cudaSurfaceObject_t clipColor
)
{
	// 检查这个内核的位置
	const unsigned int clip_x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int clip_y = threadIdx.y + blockDim.y * blockIdx.y;
	if (clip_x >= clipCols || clip_y >= clipRows) return;

	// 从这里开始，对rawColorImage的访问应该在范围内
	const unsigned int clip_boundary = CLIP_BOUNDARY;
	const unsigned int raw_x = clip_x + clip_boundary;
	const unsigned int raw_y = clip_y + clip_boundary;
	const unsigned int raw_flatten = raw_x + raw_y * (clipCols + 2 * clip_boundary); // 计算一维数组中的位置
	// 传入的DeviceArray<uchar3>类型是二维压缩成了一维，现在需要通过二维坐标定位到一维数组中的位置
	const uchar3 raw_pixel = rawColorImage[raw_flatten]; 

	// 归一化并写入输出
	float4 noramlized_rgb;	
	noramlized_rgb.x = float(raw_pixel.x) / 255.0f;
	noramlized_rgb.y = float(raw_pixel.y) / 255.0f;
	noramlized_rgb.z = float(raw_pixel.z) / 255.0f;
	noramlized_rgb.w = 1.0f;
	surf2Dwrite(noramlized_rgb, clipColor, clip_x * sizeof(float4), clip_y);
}

__global__ void SparseSurfelFusion::device::clipNormalizeColorKernel(
	const PtrSize<const uchar3> rawColor, 
	const unsigned int clipRows, 
	const unsigned int clipCols, 
	cudaSurfaceObject_t clipColor, 
	cudaSurfaceObject_t grayScale
)
{
	//检查这个内核的位置
	const auto clip_x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto clip_y = threadIdx.y + blockDim.y * blockIdx.y;
	if (clip_x >= clipCols || clip_y >= clipRows) return;

	//从这里开始，对raw_rgb_img的访问应该在范围内
	const unsigned int boundary_clip = CLIP_BOUNDARY;
	const auto raw_x = clip_x + boundary_clip;
	const auto raw_y = clip_y + boundary_clip;
	const auto raw_flatten = raw_x + raw_y * (clipCols + 2 * boundary_clip);
	const uchar3 raw_pixel = rawColor[raw_flatten];

	//归一化并写入输出
	float4 noramlized_rgb;
	noramlized_rgb.x = float(raw_pixel.x) / 255.0f;
	noramlized_rgb.y = float(raw_pixel.y) / 255.0f;
	noramlized_rgb.z = float(raw_pixel.z) / 255.0f;
	noramlized_rgb.w = 1.0f;
	const float grayScaleImage = rgba2density(noramlized_rgb);//获得密度(灰度)图像

	surf2Dwrite(noramlized_rgb, clipColor, clip_x * sizeof(float4), clip_y);
	surf2Dwrite(grayScaleImage, grayScale, clip_x * sizeof(float), clip_y);
}

__global__ void SparseSurfelFusion::device::filterCrayScaleImageKernel(cudaTextureObject_t grayScale, unsigned int rows, unsigned int cols, cudaSurfaceObject_t filteredGrayScale)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= cols || y >= rows) return;

	const int half_width = 5;
	const float center_density = tex2D<float>(grayScale, x, y);

	// 滑动窗口
	float sum_all = 0.0f;
	float sum_weight = 0.0f;
	for (unsigned int y_idx = y - half_width; y_idx <= y + half_width; y_idx++) {
		for (unsigned int x_idx = x - half_width; x_idx <= x + half_width; x_idx++) {
			const float density = tex2D<float>(grayScale, x_idx, y_idx);
			const float value_diff2 = (center_density - density) * (center_density - density);
			const float pixel_diff2 = (x_idx - x) * (x_idx - x) + (y_idx - y) * (y_idx - y);
			const float this_weight = (density > 0.0f) * expf(-(1.0f / 25) * pixel_diff2) * expf(-(1.0f / 0.01) * value_diff2);
			sum_weight += this_weight;
			sum_all += this_weight * density;
		}
	}

	// 滤波后的值
	float filter_density_value = sum_all / (sum_weight);

	// 将值限定到合适的范围，大于1则为1，小于0则为0
	if (filter_density_value >= 1.0f) {
		filter_density_value = 1.0f;
	}
	else if (filter_density_value >= 0.0f) {

	}
	else {
		filter_density_value = 0.0f;
	}
	surf2Dwrite(filter_density_value, filteredGrayScale, x * sizeof(float), y);
}

__global__ void SparseSurfelFusion::device::copyPreviousVertexAndNormalKernel(cudaSurfaceObject_t collectPreviousVertex, cudaSurfaceObject_t collectPreviousNormal, cudaTextureObject_t previousVertexTexture, cudaTextureObject_t previousNormalTexture, const unsigned int rows, const unsigned int cols)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;
	float4 previousVertex = tex2D<float4>(previousVertexTexture, x, y);
	surf2Dwrite(previousVertex, collectPreviousVertex, x * sizeof(float4), y);
	float4 previousNormal = tex2D<float4>(previousNormalTexture, x, y);
	surf2Dwrite(previousNormal, collectPreviousNormal, x * sizeof(float4), y);
}

__global__ void SparseSurfelFusion::device::createVertexConfidenceMapKernel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, const IntrinsicInverse intrinsicInverse, cudaSurfaceObject_t vertexConfidenceMap)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;

	//获取值并进行反向投影
	const unsigned short raw_depth = tex2D<unsigned short>(depthImage, x, y);
	float4 vertex_confid;

	//将深度缩放到以m(米)为单位，深度图像始终以mm(毫米)为单位
	vertex_confid.z = float(raw_depth) / (1000.f);
	vertex_confid.x = (x - intrinsicInverse.principal_x) * intrinsicInverse.inv_focal_x * vertex_confid.z;
	vertex_confid.y = (y - intrinsicInverse.principal_y) * intrinsicInverse.inv_focal_y * vertex_confid.z;
	vertex_confid.w = 1.0f; // 深度像素的置信度值,用1.0f初始化
	// 上述将顶点映射到了相机坐标系中
	surf2Dwrite(vertex_confid, vertexConfidenceMap, x * sizeof(float4), y);
}

__global__ void SparseSurfelFusion::device::createNormalRadiusMapKernel(cudaTextureObject_t vertexMap, const unsigned int rows, const unsigned int cols, float cameraFocal, cudaSurfaceObject_t normalRadiusMap)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;

	//该值必须在结束时写入surface
	float4 normal_radius_value = make_float4(0, 0, 0, 0);

	//中心的顶点
	const float4 vertex_center = tex2D<float4>(vertexMap, x, y);
	if (!is_zero_vertex(vertex_center)) {//如果顶点不为零
		float4 centeroid = make_float4(0, 0, 0, 0);	// 窗口的中心点坐标
		int counter = 0;
		//首先搜索窗口以确定中心
		for (int cy = y - windowLength; cy <= y + windowLength; cy += 1) {
			for (int cx = x - windowLength; cx <= x + windowLength; cx += 1) {
				const float4 p = tex2D<float4>(vertexMap, cx, cy);
				if (!is_zero_vertex(p)) {
					centeroid.x += p.x;
					centeroid.y += p.y;
					centeroid.z += p.z;
					counter++;
				}
			}
		}

		//至少一半的窗口是有效的
		if (counter > (windowSize / 2)) {
			centeroid *= (1.0f / counter); // 取窗口中有效位置像素的坐标平均值
			float covariance[6] = { 0 };

			//第二个窗口搜索计算法线
			for (int cy = y - windowLength; cy < y + windowLength; cy += 1) {
				for (int cx = x - windowLength; cx < x + windowLength; cx += 1) {
					const float4 p = tex2D<float4>(vertexMap, cx, cy);
					if (!is_zero_vertex(p)) {
						const float4 diff = p - centeroid;
						//计算协方差
						covariance[0] += diff.x * diff.x; //(0, 0)
						covariance[1] += diff.x * diff.y; //(0, 1)
						covariance[2] += diff.x * diff.z; //(0, 2)
						covariance[3] += diff.y * diff.y; //(1, 1)
						covariance[4] += diff.y * diff.z; //(1, 2)
						covariance[5] += diff.z * diff.z; //(2, 2)
					}
				}
			}//第二窗口搜索结束

			//normal的本征值，利用PCL库中拷贝的方法
			eigen33 eigen(covariance);
			float3 normal;
			eigen.compute(normal);
			if (dotxyz(normal, vertex_center) >= 0.0f) normal *= -1;


			//printf("idx = (%hu, %hu) normal = (%10.5f, %10.5f, %10.5f)\n", x, y, normal.x, normal.y, normal.z);
			//if (referenceNormalMap.x > 1e6 || referenceNormalMap.y > 1e6 || referenceNormalMap.z > 1e6) printf("########################################There is NaN Value########################################\n");
			//if (conversionNormalPre.x > 1e6 || conversionNormalPre.y > 1e6 || conversionNormalPre.z > 1e6) printf("########################################There is NaN Value########################################\n");
			//半径
			const float radius = computeRadius(vertex_center.z, normal.z, cameraFocal);

			//写入局部变量
			normal_radius_value.x = normal.x;
			normal_radius_value.y = normal.y;
			normal_radius_value.z = normal.z;
			normal_radius_value.w = radius;
		}//结束检查有效像素的数量
	}

	//Write to the surface
	surf2Dwrite(normal_radius_value, normalRadiusMap, x * sizeof(float4), y);
}

__global__ void SparseSurfelFusion::device::createColorTimeMapKernel(const PtrSize<const uchar3> rawColor, const unsigned int rows, const unsigned int cols, const float initTime, const float CameraID, cudaSurfaceObject_t colorTimeMap)
{
	const unsigned int clip_x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int clip_y = threadIdx.y + blockIdx.y * blockDim.y;
	if (clip_x >= cols || clip_y >= rows) return;

	//从这里开始，对raw_rgb_img的访问应该在范围内
	const unsigned int boundary_clip = CLIP_BOUNDARY;
	const unsigned int raw_x = clip_x + boundary_clip;
	const unsigned int raw_y = clip_y + boundary_clip;
	const unsigned int raw_flatten = raw_x + raw_y * (cols + 2 * boundary_clip);
	const uchar3 raw_pixel = rawColor[raw_flatten];
	//printf("raw_pixel=(%d,%d,%d) \n",raw_pixel.x, raw_pixel.y, raw_pixel.z);
	const float encoded_pixel = float_encode_rgb(raw_pixel);
	//printf("encoded_pixel=%f\n", encoded_pixel);
	//构造结果并存储它
	const float4 color_time_value = make_float4(encoded_pixel, CameraID + 0.5f, initTime, initTime);
	surf2Dwrite(color_time_value, colorTimeMap, clip_x * sizeof(float4), clip_y);
	//if((clip_x%3==0)&&(clip_y%3==0)){
	//	//printf("color_time_value=%f %f %f %f\n", color_time_value.x, color_time_value.y, color_time_value.z, color_time_value.w);
	//	uchar3 rgb;
	//	float_decode_rgb(color_time_value.x,rgb);
	//	printf("%u %u %u\n", rgb.x, rgb.y, rgb.z);

	//}
}

__global__ void SparseSurfelFusion::device::markValidDepthPixelKernel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, PtrSize<char> validIndicator)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;

	// 将二维图像转成一维数组存储，标志位也是一维数组，标志位数组的index是可以反过来确定二维图像的哪个像素点坐标的
	const unsigned int flatten_idx = x + cols * y;
	char valid = 0;
	const unsigned short depth_value = tex2D<unsigned short>(depthImage, x, y);
	if (depth_value > 0) {
		valid = 1;
	}

	// 写入输出(存储哪些深度信息是有效的，记录index)
	validIndicator[flatten_idx] = valid;
}

__global__ void SparseSurfelFusion::device::markValidDepthPixelKernel(cudaTextureObject_t depthImage, cudaTextureObject_t foregroundMask, cudaTextureObject_t normalMap, const unsigned int rows, const unsigned int cols, PtrSize<char> validIndicator)
{
	const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= cols || y >= rows) return;

	// 将二维图像转成一维数组存储，标志位也是一维数组，标志位数组的index是可以反过来确定二维图像的哪个像素点坐标的
	const unsigned int flatten_idx = x + cols * y;
	char valid = 0;
	const unsigned short depth_value = tex2D<unsigned short>(depthImage, x, y);
	const unsigned char mask_value = tex2D<unsigned char>(foregroundMask, x, y);
	const float4 normal = tex2D<float4>(normalMap, x, y);
	if (depth_value > 0 && mask_value > 0 && normal.x != 0 && normal.y != 0 && normal.z != 0) {	// 并且法线有值
		valid = 1;
	}

	// 写入输出(存储哪些深度信息是有效的，记录index)
	validIndicator[flatten_idx] = valid;
}

__global__ void SparseSurfelFusion::device::markValidMatchedPointsKernel(DeviceArrayView<float4> MatchedPointsPairs, const unsigned int pairsNum, PtrSize<char> validIndicator)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= pairsNum) return;

	char valid;
	const float4 previous = MatchedPointsPairs[idx];
	const float4 current = MatchedPointsPairs[2 * idx];

	if ((previous.x == 0 && previous.y == 0 && previous.z == 0) || (current.x == 0 && current.y == 0 && current.z == 0)) {
		valid = 0;
	}
	else {
		valid = 1;
	}
	validIndicator[idx] = valid;
}

__global__ void SparseSurfelFusion::device::collectDepthSurfelKernel(
	cudaTextureObject_t vertexConfidenceMap, 
	cudaTextureObject_t normalRadiusMap, 
	cudaTextureObject_t colorTimeMap, 
	const PtrSize<const int> selectedArray, 
	const unsigned int rows, 
	const unsigned int cols,
	const unsigned int CameraID,
	PtrSize<DepthSurfel> validDepthSurfel
)
{
	const unsigned int selected_idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (selected_idx >= selectedArray.size) return;
	const unsigned int idx = selectedArray[selected_idx];
	const unsigned int x = idx % cols;
	const unsigned int y = idx / cols;

	// 构造输出DepthSurfel
	DepthSurfel surfel;
	surfel.pixelCoordinate.x() = x;
	surfel.pixelCoordinate.y() = y;
	surfel.VertexAndConfidence = tex2D<float4>(vertexConfidenceMap, x, y);
	surfel.NormalAndRadius = tex2D<float4>(normalRadiusMap, x, y);
	surfel.ColorAndTime = tex2D<float4>(colorTimeMap, x, y);
	surfel.ColorAndTime.y = CameraID + 0.5f;
	surfel.isMerged = false;	// 初始都未被融合过
	surfel.CameraID = CameraID;
	// 写入输出数组
	validDepthSurfel[selected_idx] = surfel;
}

__global__ void SparseSurfelFusion::device::collectMatchedPointsKernel(cudaTextureObject_t previousTexture, cudaTextureObject_t currentTexture, DeviceArrayView<PixelCoordinate> pointsPairsCoor, const unsigned int totalPointsNum, DeviceArrayHandle<float4> matchedPoints)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < totalPointsNum) {
		matchedPoints[idx] =  tex2D<float4>(previousTexture, pointsPairsCoor[idx].x(), pointsPairsCoor[idx].y());
		matchedPoints[2 * idx] = tex2D<float4>(currentTexture, pointsPairsCoor[2 * idx].x(), pointsPairsCoor[2 * idx].y());
	}
}

__global__ void SparseSurfelFusion::device::collectValidMatchedPointsKernel(DeviceArrayView<float4> rawMatchedPoints, const PtrSize<const int> selectedArray, const unsigned int validPairsNum, DeviceArrayHandle<float4> validMatchedPoints)
{
	const unsigned int selectedIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if (selectedIdx >= validPairsNum) return;
	const unsigned int previousIdx = selectedArray[selectedIdx];
	const unsigned int currentIdx = selectedArray[2 * selectedIdx];

	validMatchedPoints[selectedIdx] = rawMatchedPoints[previousIdx];
	validMatchedPoints[2 * selectedIdx] = rawMatchedPoints[currentIdx];
}




void SparseSurfelFusion::clipFilterDepthImage(cudaTextureObject_t rawDepth, const unsigned int clipImageRows, const unsigned int clipImageCols, const unsigned int clipNear, const unsigned clipFar, cudaSurfaceObject_t filterDepth, cudaStream_t stream)
{
	//滤波器参数
	const float sigmaS = Constants::kFilterSigmaS;
	const float sigmaR = Constants::kFilterSigmaR;
	const float sigmaSInverseSquare = 1.0f / (sigmaS * sigmaS);
	const float sigmaRInverseSquare = 1.0f / (sigmaR * sigmaR);

	//调用内核
	dim3 block(16, 16);
	dim3 grid(divUp(clipImageCols, block.x), divUp(clipImageRows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::clipFilterDepthKernel << <grid, block, 0, stream >> > (rawDepth, clipImageRows, clipImageCols, clipNear, clipFar, sigmaSInverseSquare, sigmaRInverseSquare, filterDepth);
}

void SparseSurfelFusion::clipNormalizeColorImage(const DeviceArray<uchar3>& rawColor, unsigned int clipRows, unsigned int clipCols, cudaSurfaceObject_t clipColor, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(clipCols, block.x), divUp(clipRows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::clipNormalizeColorKernel << <grid, block, 0, stream >> > (rawColor, clipRows, clipCols, clipColor);
}

void SparseSurfelFusion::clipNormalizeColorImage(
	const DeviceArray<uchar3>& rawColor, 
	unsigned int clipRows, 
	unsigned int clipCols, 
	cudaSurfaceObject_t clipColor, 
	cudaSurfaceObject_t grayScale, 
	cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(clipCols, block.x), divUp(clipRows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::clipNormalizeColorKernel << <grid, block, 0, stream >> > (rawColor, clipRows, clipCols, clipColor, grayScale);
}

void SparseSurfelFusion::filterGrayScaleImage(cudaTextureObject_t grayScale, unsigned int rows, unsigned int cols, cudaSurfaceObject_t filteredGrayScale, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::filterCrayScaleImageKernel << <grid, block, 0, stream >> > (grayScale, rows, cols, filteredGrayScale);
}

void SparseSurfelFusion::copyPreviousVertexAndNormal(cudaSurfaceObject_t collectPreviousVertex, cudaSurfaceObject_t collectPreviousNormal, cudaTextureObject_t previousVertexTexture, cudaTextureObject_t previousNormalTexture, const unsigned int rows, const unsigned int cols, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	device::copyPreviousVertexAndNormalKernel << <grid, block, 0, stream >> > (collectPreviousVertex, collectPreviousNormal, previousVertexTexture, previousNormalTexture, rows, cols);
}

void SparseSurfelFusion::createVertexConfidenceMap(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, const IntrinsicInverse intrinsic_inv, cudaSurfaceObject_t vertexConfidenceMap, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::createVertexConfidenceMapKernel << <grid, block, 0, stream >> > (depthImage, rows, cols, intrinsic_inv, vertexConfidenceMap);
}

void SparseSurfelFusion::createNormalRadiusMap(cudaTextureObject_t vertexMap, const unsigned int rows, const unsigned int cols, float cameraFocal, cudaSurfaceObject_t normalRadiusMap, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::createNormalRadiusMapKernel << <grid, block, 0, stream >> > (vertexMap, rows, cols, cameraFocal, normalRadiusMap);
}

void SparseSurfelFusion::createColorTimeMap(const DeviceArray<uchar3> rawColor, const unsigned int rows, const unsigned int cols, const float initTime, const float CameraID, cudaSurfaceObject_t colorTimeMap, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	/*<<<块个数，线程个数，动态分配共享内存，流>>>*/
	device::createColorTimeMapKernel << <grid, block, 0, stream >> > (rawColor, rows, cols, initTime, CameraID, colorTimeMap);
}

void SparseSurfelFusion::markValidDepthPixel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, DeviceArray<char>& validIndicator, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	device::markValidDepthPixelKernel << <grid, block, 0, stream >> > (depthImage, rows, cols, validIndicator);
}

void SparseSurfelFusion::markValidDepthPixel(cudaTextureObject_t depthImage, cudaTextureObject_t foregroundMask, cudaTextureObject_t normalMap, const unsigned int rows, const unsigned int cols, DeviceArray<char>& validIndicator, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	device::markValidDepthPixelKernel << <grid, block, 0, stream >> > (depthImage, foregroundMask, normalMap, rows, cols, validIndicator);
}

void SparseSurfelFusion::markValidMatchedPoints(DeviceArrayView<float4>& MatchedPointsPairs, const unsigned int pairsNum, DeviceArray<char>& validIndicator, cudaStream_t stream)
{
	dim3 block(16);
	dim3 grid(divUp(MatchedPointsPairs.Size(), block.x));
	device::markValidMatchedPointsKernel << <grid, block, 0, stream >> > (MatchedPointsPairs, pairsNum, validIndicator);

}

void SparseSurfelFusion::collectDepthSurfel(cudaTextureObject_t vertexConfidenceMap, cudaTextureObject_t normalRadiusMap, cudaTextureObject_t colorTimeMap, const DeviceArray<int>& selectedArray, const unsigned int rows, const unsigned int cols, const unsigned int CameraID, DeviceArray<DepthSurfel>& validDepthSurfel, cudaStream_t stream)
{
	dim3 block(128);
	dim3 grid(divUp(selectedArray.size(), block.x));
	device::collectDepthSurfelKernel << <grid, block, 0, stream >> > (vertexConfidenceMap, normalRadiusMap, colorTimeMap, selectedArray, rows, cols, CameraID, validDepthSurfel);
}

void SparseSurfelFusion::collectMatchedPoints(cudaTextureObject_t previousTexture, cudaTextureObject_t currentTexture, DeviceArrayView<PixelCoordinate>& pointsPairs, DeviceBufferArray<float4>& matchedPoints, cudaStream_t stream)
{
	const unsigned int totalPairsNum = pointsPairs.Size() / 2;
	DeviceArrayHandle<float4> matchedPointsHandle = matchedPoints.ArrayHandle();	//	暴露指针接口
	dim3 block(8);
	dim3 grid(divUp(totalPairsNum, block.x));
	device::collectMatchedPointsKernel << <grid, block, 0, stream >> > (previousTexture, currentTexture, pointsPairs, totalPairsNum, matchedPointsHandle);
}

void SparseSurfelFusion::collectValidMatchedPoints(DeviceArrayView<float4>& rawMatchedPoints, const DeviceArray<int>& selectedArray, DeviceBufferArray<float4>& validMatchedPoints, cudaStream_t stream)
{
	
	const unsigned int validPairsNum = selectedArray.size() / 2;
	printf("有效匹配点对数 = %d\n", validPairsNum);
	DeviceArrayHandle<float4> validMatchedPointsHandle = validMatchedPoints.ArrayHandle();
	dim3 block(8);
	dim3 grid(divUp(validPairsNum, block.x));
	device::collectValidMatchedPointsKernel << <grid, block, 0, stream >> > (rawMatchedPoints, selectedArray, validPairsNum, validMatchedPointsHandle);
}

void SparseSurfelFusion::mapMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, device::MapMergedSurfelInterface& mergedSurfel, const unsigned int clipedWidth, const unsigned int clipedHeight, cudaStream_t stream)
{
	dim3 block(64);
	dim3 grid(divUp(validSurfelArray.Size(), block.x));
	device::mapMergedDepthSurfelKernel << <grid, block, 0, stream >> > (validSurfelArray, mergedSurfel, validSurfelArray.Size(), clipedWidth, clipedHeight);
}

void SparseSurfelFusion::clearMapSurfel(const unsigned int clipedWidth, const unsigned int clipedHeight, device::MapMergedSurfelInterface& mergedSurfel, cudaStream_t stream)
{
	dim3 block(16,16);
	dim3 grid(divUp(clipedWidth, block.x), divUp(clipedHeight, block.y));
	device::clearMapSurfelKernel << <grid, block, 0, stream >> > (mergedSurfel, clipedWidth, clipedHeight);
}

void SparseSurfelFusion::MultiViewForegroundMaskConstriction(CudaTextureSurface* depthMap, CudaTextureSurface* vertexMap, CudaTextureSurface* normalMap, CudaTextureSurface* colorMap, device::MultiViewMaskInterface& MaskConstrictionInterface, const unsigned int devicesCount, const unsigned int clipedWidth, const unsigned int clipedHeight, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid(divUp(clipedWidth, block.x), divUp(clipedHeight, block.y));
	for (int i = 0; i < devicesCount; i++) {
		device::constrictMultiViewForegroundKernel << <grid, block, 0, stream >> > (depthMap[i].surface, vertexMap[i].surface, normalMap[i].surface, colorMap[i].surface, MaskConstrictionInterface, i, clipedWidth, clipedHeight);
	}
}

