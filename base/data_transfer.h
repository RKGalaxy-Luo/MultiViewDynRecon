#pragma once

#include <base/CommonTypes.h>
#include <base/CommonUtils.h>

#include <base/EncodeUtils.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <render/Renderer.h>



namespace SparseSurfelFusion
{
	namespace device {
		template<typename T>
		__global__ void textureToMap2DKernel(
			cudaTextureObject_t texture,
			PtrStepSize<T> map
		);

		__global__ void textureComponentToMap2DKernel(
			cudaTextureObject_t texture,
			bool xComponent,
			PtrStepSize<float> map
		);

		__global__ void textureToMap2DKernelfloat4(
			cudaTextureObject_t texture,
			PtrStepSize<float4> map
		);

		__global__ void textureToMap2DKernel(
			cudaTextureObject_t texture,
			PtrStepSize<float4> map,
			mat34 SE3
		);

		__global__ void textureToMap2DKernel(
			cudaTextureObject_t texture,
			PtrStepSize<float4> map,
			const DeviceArrayView2D<mat34> Se3Map
		);

		__global__ void textureToMap2DUseSE3andCamera2WorldKernel(
			cudaTextureObject_t texture,
			PtrStepSize<float4> map,
			mat34 SE3,
			mat34 camera2world
		);
		__global__ void textureToMap2DUseSE3andCamera2WorldAndBackKernel(
			cudaTextureObject_t texture,
			PtrStepSize<float4> map,
			mat34 SE3,
			mat34 camera2world,
			mat34 SE3back
		);
		__global__ void indexmapToMap2DKernel(
			cudaTextureObject_t indexmap,
			PtrStepSize<unsigned short> map
		);
		__global__ void debugrefence(
			cudaTextureObject_t indexmap0,
			cudaTextureObject_t indexmap1,
			cudaTextureObject_t referencevertexmap0,
			cudaTextureObject_t referencevertexmap1,
			mat34 SE3,
			DeviceArrayView<float4> CanonicalVertexConfidence
		);

		__global__ void debugfusiondepthsurfelmap(
			cudaTextureObject_t vertex, 
			cudaTextureObject_t normal, 
			cudaTextureObject_t colortime
		);

	}
	//hsg 用于查看indexmap
	void debugIndexMap(cudaTextureObject_t indexmap0, cudaTextureObject_t indexmap1, cudaTextureObject_t indexmap2, cv::String name);
	void debugmask(cudaTextureObject_t mask0, cv::String name);
	void debugmask(cudaTextureObject_t mask0, cudaTextureObject_t mask1, cudaTextureObject_t mask2, cv::String name);

	void debugwarpedvectexmap(DeviceArray2D<unsigned short> &image_gpu);
	void debugfusionDepthSurfelMap(cudaTextureObject_t vertex, cudaTextureObject_t normal, cudaTextureObject_t colortime, cudaStream_t stream = 0);

	/* Download the image from GPU memory to CPU memory
	 */
	cv::Mat downloadmaskImage(const DeviceArray2D<unsigned short>& image_gpu);
	cv::Mat downloadDepthImage(const DeviceArray2D<unsigned short>& image_gpu);
	cv::Mat downloadDepthImage(cudaTextureObject_t image_gpu);
	cv::Mat downloadRGBImage(
		const DeviceArray<uchar3>& image_gpu,
		const unsigned rows, const unsigned cols
	);

	//The rgb texture is in float4
	cv::Mat downloadNormalizeRGBImage(const DeviceArray2D<float4>& rgb_img);
	cv::Mat downloadNormalizeRGBImage(cudaTextureObject_t rgb_img);
	cv::Mat rgbImageFromColorTimeMap(cudaTextureObject_t color_time_map);
	cv::Mat normalMapForVisualize(cudaTextureObject_t normal_map);

	//The segmentation mask texture
	void downloadSegmentationMask(cudaTextureObject_t mask, std::vector<unsigned char>& h_mask);
	cv::Mat downloadRawSegmentationMask(cudaTextureObject_t mask); //uchar texture
	cv::Mat downloadFilteredSegmentationMask(cudaTextureObject_t filteredMask); //float texture

	//The gray scale image
	void downloadGrayScaleImage(cudaTextureObject_t image, cv::Mat& h_image, float scale = 1.0f);

	cv::Mat downloadGradientComponentImage(cudaTextureObject_t image, std::string orientation);

	//The binary meanfield map, the texture contains the
	//mean field probability of the positive label
	void downloadTransferBinaryMeanfield(cudaTextureObject_t meanfield_q, cv::Mat& h_meanfield_uchar);


	/* The point cloud download functions
	 */
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray<float4>& vertex);
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray2D<float4>& vertex_map);
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray2D<float4>& vertex_map, DeviceArrayView<unsigned> indicator);
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray2D<float4>& vertex_map0, const DeviceArray2D<float4>& vertex_map1, DeviceArrayView<ushort3> pixel);
	void downloadPointCloud(const DeviceArray2D<float4>& vertex_map, std::vector<float4>& point_cloud);

	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map);
	//PointCloud3f_Pointer downloadPointCloudUseWorld2Camera(cudaTextureObject_t vertex_map);
	PointCloud3f_Pointer downloadPointCloudUseSE3(cudaTextureObject_t vertex_map, mat34 SE3);
	PointCloud3f_Pointer downloadPointCloudUseSE3Map(cudaTextureObject_t vertex_map, const DeviceArrayView2D<mat34>& Se3Map);
	PointCloud3f_Pointer downloadPointCloudUseSE3AndWorld2Camera(cudaTextureObject_t vertex_map, mat34 SE3,mat34 camera2world);
	PointCloud3f_Pointer downloadPointCloudUseSE3AndWorld2CameraandBack(cudaTextureObject_t vertex_map, mat34 SE3, mat34 camera2world,mat34 SEback);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<unsigned> indicator);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map0, cudaTextureObject_t vertex_map1, DeviceArrayView<ushort3> pixel);
	void downloadPointCloud(cudaTextureObject_t vertex_map, std::vector<float4>& point_cloud);

	void downloadPointNormalCloud(
		const DeviceArray<DepthSurfel>& surfel_array,
		PointCloud3f_Pointer& point_cloud,
		//#ifdef WITH_PCL
		PointCloudNormal_Pointer& normal_cloud,
		//#endif
		const float point_scale = 1000.0f
	);

	void downloadPointNormalCloud(
		const DeviceArrayView<DepthSurfel> surfel_array,
		PointCloud3f_Pointer& point_cloud,
		//#ifdef WITH_PCL
		PointCloudNormal_Pointer& normal_cloud,
		//#endif
		const float point_scale = 1000.0f
	);

	void downloadPointNormalCloud(
		const DeviceArray<DepthSurfel>& surfel_array,
		PointCloud3fRGB_Pointer& point_cloud,
//#ifdef WITH_PCL
		PointCloudNormal_Pointer& normal_cloud,
//#endif
		const float point_scale = 1000.0f
	);


	void hsgseparateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		const unsigned remainnumber,
		const unsigned number0view,
		PointCloud3f_Pointer& fused_cloud,
		PointCloud3f_Pointer& view0,
		PointCloud3f_Pointer& view1
	);
	//Download it with indicator
	void separateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		const DeviceArrayView<unsigned>& indicator,
		PointCloud3f_Pointer& fused_cloud,
		PointCloud3f_Pointer& unfused_cloud
	);
	void separateDownloadPointCloudWithDiffColor(
		const DeviceArrayView<float4>& point_cloud,
		const DeviceArrayView<unsigned>& indicator,
		PointCloud3fRGB_Pointer& fused_cloud,
		PointCloud3fRGB_Pointer& unfused_cloud
	);

	void separateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		unsigned num_remaining_surfels,
		PointCloud3f_Pointer& remaining_cloud,
		PointCloud3f_Pointer& appended_cloud
	);

	/* The normal cloud download functions
	*/
//#ifdef WITH_PCL
	PointCloudNormal_Pointer downloadNormalCloud(const DeviceArray<float4>& normal_cloud);
	PointCloudNormal_Pointer downloadNormalCloud(const DeviceArray2D<float4>& normal_map);
	PointCloudNormal_Pointer downloadNormalCloud(cudaTextureObject_t normal_map);
//#elif defined(WITH_CILANTRO)
//	void downloadNormalCloud(const DeviceArray<float4>& normal_cloud, PointCloudNormal_Pointer& point_cloud);
//	void downloadNormalCloud(const DeviceArray2D<float4>& normal_map, PointCloudNormal_Pointer& point_cloud);
//	void downloadNormalCloud(cudaTextureObject_t normal_map, PointCloudNormal_Pointer& point_cloud);
//#endif

	PointCloud3fRGB_Pointer download2PointCloudWithDiffColor(
		const DeviceArray<float4>& pointcloud0,
		const DeviceArray<float4>& pointcloud1
	);
	/* The colored point cloud download function
	 */
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		const DeviceArray<float4>& vertex_confid,
		const DeviceArray<float4>& color_time
	);
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		cudaTextureObject_t vertex_map,
		cudaTextureObject_t color_time_map,
		bool flip_color = false
	);
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		const DeviceArrayView<DepthSurfel>& vertex,
		const mat34& convert,
		const std::string type = "canonical"
	);


	/* Colorize the point cloud
	 */
	PointCloud3fRGB_Pointer addColorToPointCloud(const PointCloud3f_Pointer& point_cloud, uchar4 rgba);


	/* Query the index map
	 */
	void queryIndexMapFromPixels(cudaTextureObject_t index_map, const DeviceArrayView<ushort4>& pixel_array, DeviceArray<unsigned>& openglID);

	/* Transfer the memory from texture to GPU memory.
	 * Assume ALLOCATED device memory.
	 */
	template<typename T>
	void textureToMap2D(
		cudaTextureObject_t texture,
		DeviceArray2D<T>& map,
		cudaStream_t stream = 0
	);


	void textureComponentToMap2D(
		cudaTextureObject_t texture,
		DeviceArray2D<float>& map,
		std::string orientation,
		cudaStream_t stream = 0
	);

	void textureToMap2Dfloat4(
		cudaTextureObject_t texture,
		DeviceArray2D<float4>& map,
		cudaStream_t stream = 0
	);
	//hsg 
	template<typename T>
	void indexmapToMap2D(
		cudaTextureObject_t indexmap,
		DeviceArray2D<T>& map,
		unsigned max,
		cudaStream_t stream = 0
	);

	void textureToMap2D(
		cudaTextureObject_t texture,
		DeviceArray2D<float4>& map,
		mat34 SE3,
		cudaStream_t stream = 0
	);

	void textureToMap2D(
		cudaTextureObject_t texture,
		DeviceArray2D<float4>& map,
		const DeviceArrayView2D<mat34>& SE3,
		cudaStream_t stream = 0
	);
	void textureToMap2DUseSE3andCamera2World(
		cudaTextureObject_t texture,
		DeviceArray2D<float4>& map,
		mat34 SE3,
		mat34 camera2world,
		cudaStream_t stream = 0
	);
	void textureToMap2DUseSE3andCamera2WorldandBack(
		cudaTextureObject_t texture,
		DeviceArray2D<float4>& map,
		mat34 SE3,
		mat34 camera2world,
		mat34 SE3back,
		cudaStream_t stream = 0
	);


}

