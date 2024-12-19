/*****************************************************************//**
 * \file   ImageProcessByGPU.h
 * \brief  主要涉及一些用GPU处理图像的操作：剪裁，滤波，构建面元属性图等
 * 
 * \author LUO
 * \date   January 29th 2024
 *********************************************************************/
#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <base/Constants.h>
#include <base/GlobalConfigs.h>
#include <base/ColorTypeTransfer.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <base/EncodeUtils.h>
#include <math/VectorUtils.h>
#include <math/MatUtils.h>
#include <base/CommonTypes.h>
#include <visualization/Visualizer.h>
namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief 通过多视角的前景Mask，约束模型边缘深度异常点.
		 */
		struct MultiViewMaskInterface {
			cudaTextureObject_t depthMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t colorMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t foreground[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3Inverse[MAX_CAMERA_COUNT];
			Intrinsic ClipedIntrinsic[MAX_CAMERA_COUNT];
		};

		/**
		 * \brief 将融合的面元反映射到各个Camera的IndexMap中.
		 */
		struct MapMergedSurfelInterface {
			cudaSurfaceObject_t vertex[MAX_CAMERA_COUNT];
			cudaSurfaceObject_t normal[MAX_CAMERA_COUNT];
			cudaSurfaceObject_t color[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3Inverse[MAX_CAMERA_COUNT];
			Intrinsic ClipedIntrinsic[MAX_CAMERA_COUNT];
		};

		__global__ void trandepthdata(
			const DeviceArrayView<DepthSurfel> validSurfelArray,
			float4* buffer
			);


/*******************************************     剪裁RGB和Depth图片     *******************************************/
		/**
		 * \brief 处理深度图剪裁的核函数.
		 *
		 * \param rawDepth 原始深度图(以cudaTextureObject_t类型传入)
		 * \param clipImageRows	剪裁的图像高度
		 * \param clipImageCols	剪裁的图像宽度
		 * \param clipNear 深度图最近的距离
		 * \param clipFar 深度图最远的距离
		 * \param sigmaSInverseSquare 双边滤波中空间距离权重的平方根倒数
		 * \param sigmaRInverseSquare 双边滤波中像素值权重的平方根倒数
		 * \param filterDepth 裁剪完成的深度图(以cudaSurfaceObject_t类型传出)
		 */
		__global__ void clipFilterDepthKernel(cudaTextureObject_t rawDepth, const unsigned int clipImageRows, const unsigned int clipImageCols, const unsigned int clipNear, const unsigned int clipFar, const float sigmaSInverseSquare, const float sigmaRInverseSquare, cudaSurfaceObject_t filterDepth);
	
		
		/**
		 * \brief 将RGB图像剪裁并归一化的核函数.
		 *
		 * \param rawColorImage 原始的RGB图像(以const DeviceArray<uchar3>类型传入：类型自动转换成const PtrSize<const uchar3>)
		 * \param clipRows 剪裁图像的高
		 * \param clipCols 剪裁图像的宽
		 * \param clipColorImage 剪裁后的RGB图像(以cudaSurfaceObject_t类型传出)
		 */
		__global__ void clipNormalizeColorKernel(const PtrSize<const uchar3> rawColorImage, const unsigned int clipRows, const unsigned int clipCols, cudaSurfaceObject_t clipColorImage);

		/**
		 * \brief 将RGB图像剪裁并归一化的核函数.
		 *
		 * \param rawColorImage 原始的RGB图像(以const DeviceArray<uchar3>类型传入：类型自动转换成const PtrSize<const uchar3>)
		 * \param clipRows 剪裁图像的高
		 * \param clipCols 剪裁图像的宽
		 * \param clipColorImage 剪裁后的RGB图像(以cudaSurfaceObject_t类型传出)
		 * \param GrayScaleImage 根据RGB图像获得密度(灰度)图(以cudaSurfaceObject_t类型传出)
		 */
		__global__ void clipNormalizeColorKernel(const PtrSize<const uchar3> rawColorImage, const unsigned int clipRows, const unsigned int clipCols, cudaSurfaceObject_t clipColorImage, cudaSurfaceObject_t GrayScaleImage);
	
		/**
		 * \brief 对灰度图像进行双边滤波的核函数.
		 *
		 * \param grayScaleImage 完成剪裁的灰度图像
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 * \param filteredGrayScaleImage 完成滤波后的灰度图像
		 */
		__global__ void filterCrayScaleImageKernel(cudaTextureObject_t grayScaleImage, unsigned int rows, unsigned int cols, cudaSurfaceObject_t filteredGrayScaleImage);


/*******************************************     根据RGB和Depth图片构建面元属性Map     *******************************************/
		__host__ __device__ __forceinline__ float computeRadius(float depth_value, float normal_z, float focal) {
			const float radius = depth_value * (1000 * 1.414f / focal);
			normal_z = abs(normal_z);
			float radius_n = 2.0f * radius;
			if (normal_z > 0.5) {
				radius_n = radius / normal_z;
			}
			return radius_n;//[mm]
		}
		enum {
			windowLength = 7,
			windowHalfSize = 3,
			windowSize = windowLength * windowLength
		};

		/**
		 * \brief 拷贝前一帧的顶点和法线数据.
		 *
		 * \param collectPreviousVertex 收集前一帧顶点
		 * \param collectPreviousNormal 收集前一帧法线
		 * \param previousVertexTexture 前一帧顶点
		 * \param previousNormalTexture 前一帧法线
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 */
		__global__ void copyPreviousVertexAndNormalKernel(cudaSurfaceObject_t collectPreviousVertex, cudaSurfaceObject_t collectPreviousNormal, cudaTextureObject_t previousVertexTexture, cudaTextureObject_t previousNormalTexture, const unsigned int rows, const unsigned int cols);

		/**
		 * \brief 构造顶点和置信度的Map的核函数.
		 *
		 * \param depthImage 传入深度图(以cudaTextureObject_t形式传入)
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 * \param intrinsicInverse 相机内参的逆，为了求顶点在相机坐标系下的坐标
		 * \param vertexConfidenceMap 传出的顶点及置信度的图(传出以cudaSurfaceObject_t的形式)
		 */
		__global__ void createVertexConfidenceMapKernel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, const IntrinsicInverse intrinsicInverse, cudaSurfaceObject_t vertexConfidenceMap);

		/**
		 * \brief 构造法线和半径的Map的核函数.
		 *
		 * \param vertexMap 完成构造的顶点及置信度的图Map(以cudaTextureObject_t形式传入)
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 * \param cameraFocal 相机焦距
		 * \param normalRadiusMap 传出的法线以及半径的图(传出以cudaSurfaceObject_t的形式)
		 */
		__global__ void createNormalRadiusMapKernel(cudaTextureObject_t vertexMap, const unsigned int rows, const unsigned int cols, float cameraFocal, cudaSurfaceObject_t normalRadiusMap);
	
/*******************************************     根据RGB和上一次看到面元的时刻构建Color-Time Map     *******************************************/

		/**
		 * \brief 创建Color-Time图.
		 *
		 * \param rawColor 原始RGB图像
		 * \param rows 剪裁后的图像高
		 * \param cols 剪裁后的图像宽
		 * \param initTime 初始时刻(第几帧)
		 * \param colorTimeMap Color-Time图
		 */
		__global__ void createColorTimeMapKernel(const PtrSize<const uchar3> rawColor, const unsigned int rows, const unsigned int cols, const float initTime, const float CameraID, cudaSurfaceObject_t colorTimeMap);
	
/*********************************************     构造并选择有效的深度面元     *********************************************/
		/**
		 * \brief 在深度图中选择有效的面元，并对validIndicator这个标志数组赋值.
		 *
		 * \param depthImage 深度纹理
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 * \param validIndicator 有效面元index的标志数组 (DeviceArray可以直接转换成PtrSize类型，转换的原因是DeviceArray不能直接使用[]访问元素)
		 */
		__global__ void markValidDepthPixelKernel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, PtrSize<char> validIndicator);

		/**
		 * \brief 在深度图中选择有效的面元，并对validIndicator这个标志数组赋值.
		 *
		 * \param depthImage 深度纹理
		 * \param foregroundMask 前景掩膜
		 * \param normalMap 法线图
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 * \param validIndicator 有效面元index的标志数组 (DeviceArray可以直接转换成PtrSize类型，转换的原因是DeviceArray不能直接使用[]访问元素)
		 */
		__global__ void markValidDepthPixelKernel(cudaTextureObject_t depthImage, cudaTextureObject_t foregroundMask, cudaTextureObject_t normalMap, const unsigned int rows, const unsigned int cols, PtrSize<char> validIndicator);

		/**
		 * \brief 在匹配点对中选择有效的匹配点，previous = (0,0,0) 或者 current = (0,0,0)的匹配点去掉.
		 * 
		 * \param MatchedPointsPairs 匹配的点对 previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum)
		 * \param pairsNum 匹配对的数量
		 * \param validIndicator 有效面元index的标志数组 (DeviceArray可以直接转换成PtrSize类型，转换的原因是DeviceArray不能直接使用[]访问元素)
		 */
		__global__ void markValidMatchedPointsKernel(DeviceArrayView<float4> MatchedPointsPairs, const unsigned int pairsNum, PtrSize<char> validIndicator);

		/**
		 * \brief 运用cub库中的Flagged函数去将有效的深度面元填充到validDepthSurfel数组中(二维转一维).
		 *
		 * \param vertexConfidenceMap 顶点及置信度Map
		 * \param normalRadiusMap 法线及半径Map
		 * \param colorTimeMap Color-Time Map
		 * \param selectedArray 存储是否有效标志位的数组
		 * \param rows 图像的高
		 * \param cols 图像的宽
		 * \param validDepthSurfel 有效的深度面元数组(传出以DeviceArray<DepthSurfel>类型)
		 */
		__global__ void collectDepthSurfelKernel(cudaTextureObject_t vertexConfidenceMap, cudaTextureObject_t normalRadiusMap, cudaTextureObject_t colorTimeMap, const PtrSize<const int> selectedArray, const unsigned int rows, const unsigned int cols, const unsigned int CameraID, PtrSize<DepthSurfel> validDepthSurfel);
		
		/**
		 * \brief 根据匹配点的坐标，提取匹配点的三维坐标.
		 * 
		 * \param previousTexture 前一个相机所拍到的图片的纹理
		 * \param currentTexture 当前相机所拍到的纹理
		 * \param pointsPairsCoor 匹配点坐标(双数是previous，单数是current)
		 * \param totalPointsNum 整个传入的匹配点数量(必然是双数)
		 * \param matchedPoints 提取出的匹配点(双数是previous，单数是current)
		 */
		__global__ void collectMatchedPointsKernel(cudaTextureObject_t previousTexture, cudaTextureObject_t currentTexture, DeviceArrayView<PixelCoordinate> pointsPairsCoor, const unsigned int totalPointsNum, DeviceArrayHandle<float4> matchedPoints);
	
		/**
		 * \brief 根据原始匹配点以及筛选Array，提取有效的三维匹配点坐标.
		 * 
		 * \param rawMatchedPoints 原始匹配点
		 * \param selectedArray 存储是否有效标志位的数组
		 * \param validPairsNum 有效点对数量
		 * \param validMatchedPoints 【输出】有效的匹配点对
		 */
		__global__ void collectValidMatchedPointsKernel(DeviceArrayView<float4> rawMatchedPoints, const PtrSize<const int> selectedArray, const unsigned int validPairsNum, DeviceArrayHandle<float4> validMatchedPoints);
	

/*******************************************     反映射融合后数据     *******************************************/
		/**
		 * \brief 清空mapfurface用的，每帧都要先清空再用.
		 */
		__global__ void clearMapSurfelKernel(MapMergedSurfelInterface mergedSurfel, const unsigned int clipedWidth, const unsigned int clipedHeight);

		/**
		 * \brief 映射单一视角的深度图，结果用来代替每一帧的深度顶点映射图.
		 */
		__global__ void mapMergedDepthSurfelKernel(const DeviceArrayView<DepthSurfel> validSurfelArray, MapMergedSurfelInterface mergedSurfel, const unsigned int validSurfelNum, const unsigned int clipedWidth, const unsigned int clipedHeight);

/*******************************************     多视角前景限制     *******************************************/

		__global__ void constrictMultiViewForegroundKernel(cudaSurfaceObject_t depthMap, cudaSurfaceObject_t vertexMap, cudaSurfaceObject_t normalMap, cudaSurfaceObject_t colorMap, MultiViewMaskInterface MultiViewInterface, const unsigned int CameraID, const unsigned int clipedWidth, const unsigned int clipedHeight);
	}

/*******************************************     剪裁RGB和Depth图片     *******************************************/

	/**
	 * \brief 在.cu文件中实现深度图裁剪函数.
	 *
	 * \param rawDepth 原始的深度图(以cudaTextureObject_t形式传入)
	 * \param clipImageRows 剪裁的深度高度
	 * \param clipImageCols 剪裁的深度宽度
	 * \param clipNear 深度最近的距离是多少
	 * \param clipFar 深度最远的距离是多少
	 * \param filterDepth 裁剪后的深度图(以cudaSurfaceObject_t形式存)
	 * \param stream cuda流ID
	 */
	void clipFilterDepthImage(cudaTextureObject_t rawDepth, const unsigned int clipImageRows, const unsigned int clipImageCols, const unsigned int clipNear, const unsigned clipFar, cudaSurfaceObject_t filterDepth, cudaStream_t stream);

	/**
	 * \brief 处理并归一化RGB图像.
	 *
	 * \param rawColorImage 原始的Color图像(以DeviceArray<uchar3>的形式传入)
	 * \param clipRows 剪裁图像的高
	 * \param clipCols 剪裁图像的宽
	 * \param clipColorImage 剪裁后的RGB图像(以cudaSurfaceObject_t的形式传出)
	 * \param stream cuda流ID
	 */
	void clipNormalizeColorImage(const DeviceArray<uchar3>& rawColorImage, unsigned int clipRows, unsigned int clipCols, cudaSurfaceObject_t clipColorImage, cudaStream_t stream);

	/**
	 * \brief 处理并归一化RGB图像.
	 *
	 * \param rawColorImage 原始的Color图像(以DeviceArray<uchar3>的形式传入)
	 * \param clipRows 剪裁图像的高
	 * \param clipCols 剪裁图像的宽
	 * \param clipColorImage 剪裁后的RGB图像(以cudaSurfaceObject_t的形式传出)
	 * \param grayScaleImage 获得密度(灰度)图(以cudaSurfaceObject_t的形式传出)
	 * \param stream cuda流ID
	 */
	void clipNormalizeColorImage(const DeviceArray<uchar3>& rawColorImage, unsigned int clipRows, unsigned int clipCols, cudaSurfaceObject_t clipColorImage, cudaSurfaceObject_t grayScaleImage, cudaStream_t stream);

	/**
	 * \brief 剪裁灰度图像与深度图像大小一致.
	 *
	 * \param grayScaleImage 需要双边滤波的灰度图像(以cudaTextureObject_t形式传入)
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param filteredGrayScaleImage 完成双边滤波的灰度(以cudaSurfaceObject_t形式传出)
	 * \param stream cuda流ID
	 */
	void filterGrayScaleImage(cudaTextureObject_t grayScaleImage, unsigned int rows, unsigned int cols, cudaSurfaceObject_t filteredGrayScaleImage, cudaStream_t stream);


/*******************************************     根据RGB和Depth图片构建面元属性Map     *******************************************/

	/**
	 * \brief 拷贝前一帧的顶点和法线数据.
	 * 
	 * \param collectPreviousVertex 收集前一帧顶点
	 * \param collectPreviousNormal 收集前一帧法线
	 * \param previousVertexTexture 前一帧顶点
	 * \param previousNormalTexture 前一帧法线
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param stream cuda流ID
	 */
	void copyPreviousVertexAndNormal(cudaSurfaceObject_t collectPreviousVertex, cudaSurfaceObject_t collectPreviousNormal, cudaTextureObject_t previousVertexTexture, cudaTextureObject_t previousNormalTexture, const unsigned int rows, const unsigned int cols, cudaStream_t stream = 0);

	/**
	 * \brief 构造顶点和置信度的Map(一种抽象的float4纹理：x,y,z是顶点坐标，w是顶点置信度).
	 *
	 * \param depthImage 传入深度图(以cudaTextureObject_t形式传入)
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param intrinsicInverse 相机内参的逆，为了求顶点在相机坐标系下的坐标
	 * \param vertexConfidenceMap 传出的顶点及置信度的图(传出以cudaSurfaceObject_t的形式)
	 * \param stream cuda流ID
	 */
	void createVertexConfidenceMap(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, const IntrinsicInverse intrinsicInverse, cudaSurfaceObject_t vertexConfidenceMap, cudaStream_t stream);

	/**
	 * \brief 构造法线和半径的Map(一种抽象的float4纹理：x,y,z是法线，w是半径).
	 *
	 * \param vertexMap 完成构造的顶点及置信度的图Map(以cudaTextureObject_t形式传入)
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param cameraFocal 相机的光心，用来计算面元半径
	 * \param normalRadiusMap 传出的法线以及半径的图(传出以cudaSurfaceObject_t的形式)
	 * \param stream cuda流ID
	 */
	void createNormalRadiusMap(cudaTextureObject_t vertexMap, const unsigned int rows, const unsigned int cols, float CameraFocal, cudaSurfaceObject_t normalRadiusMap, cudaStream_t stream);

/*******************************************     根据RGB和上一次看到面元的时刻构建Color-Time Map     *******************************************/

	/**
	 * \brief 创建Color-Time图.
	 *
	 * \param rawColor 传入原始图像
	 * \param rows 剪裁后图像的高
	 * \param cols 剪裁后图像的宽
	 * \param initTime 初始化时刻(第几帧)
	 * \param colorTimeMap 传出颜色-上一帧捕捉的时刻[Color-Time图](传出以cudaSurfaceObject_t形式)
	 * \param stream cuda流ID
	 */
	void createColorTimeMap(const DeviceArray<uchar3> rawColor, const unsigned int rows, const unsigned int cols, const float initTime, const float CameraID, cudaSurfaceObject_t colorTimeMap, cudaStream_t stream);


/*********************************************     构造并选择有效的深度面元     *********************************************/
	
	/**
	 * \brief 标记有效的深度像素，将有效深度的index存到validIndicator中，对应的数组的index位置为1，其余为0.
	 *
	 * \param depthImage 深度图像
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param validIndicator 【输出】有效深度像素的指示器(标志位) 
	 * \param stream cuda流ID
	 */
	void markValidDepthPixel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, DeviceArray<char>& validIndicator, cudaStream_t stream);

	/**
	 * \brief 标记有效的深度像素，将有效深度的index存到validIndicator中，对应的数组的index位置为1，其余为0.
	 *
	 * \param depthImage 深度图像
	 * \param foregroundMask 前景掩膜
	 * \param normalMap 法线纹理，如果法线为0则为无效点
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param validIndicator 【输出】有效深度像素的指示器(标志位)
	 * \param stream cuda流ID
	 */
	void markValidDepthPixel(cudaTextureObject_t depthImage, cudaTextureObject_t foregroundMask, cudaTextureObject_t normalMap, const unsigned int rows, const unsigned int cols, DeviceArray<char>& validIndicator, cudaStream_t stream);

	/**
	 * \brief 标记有效的匹配点，将有坐标信息和深度信息的点的index标记1.
	 * 
	 * \param MatchedPointsPairs 需要寻找的点对
	 * \param pairsNum 一共有多少对匹配点
	 * \param validIndicator 【输出】有效深度像素的指示器(标志位) 
	 * \param stream cuda流ID
	 */
	void markValidMatchedPoints(DeviceArrayView<float4>& MatchedPointsPairs, const unsigned int pairsNum, DeviceArray<char>& validIndicator, cudaStream_t stream);

	/**
	 * \brief 收集有效的深度面元.
	 *
	 * \param vertexConfidenceMap 顶点及置信度纹理
	 * \param normalRadiusMap 法线及半径纹理
	 * \param colorTimeMap 颜色及上次见到顶点的时刻
	 * \param selectedArray 记录有效深度像素index的数组
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param CameraID 来自于哪个相机
	 * \param validDepthSurfel 获得的有效的深度面元
	 * \param stream cuda流ID
	 */
	void collectDepthSurfel(cudaTextureObject_t vertexConfidenceMap, cudaTextureObject_t normalRadiusMap, cudaTextureObject_t colorTimeMap, const DeviceArray<int>& selectedArray, const unsigned int rows, const unsigned int cols, const unsigned int CameraID, DeviceArray<DepthSurfel>& validDepthSurfel, cudaStream_t stream);
	
	/**
	 * \brief 收集匹配点对，previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum).
	 *
	 * \param rows 图像的高
	 * \param cols 图像的宽
	 * \param previousTexture 前一个相机的Texture
	 * \param currentTexture 当前相机的Texture
	 * \param pointsPairs 特征点匹配获得的点坐标
	 * \param matchedPoints 【输出】获得的有效的三维点坐标
	 * \param stream cuda流ID
	 */
	void collectMatchedPoints(cudaTextureObject_t previousTexture, cudaTextureObject_t currentTexture, DeviceArrayView<PixelCoordinate>& pointsPairs, DeviceBufferArray<float4>& matchedPoints, cudaStream_t stream);

	/**
	 * \brief 收集有效的匹配点对，previous [0,ValidMatchedPointsNum)   current [ValidMatchedPointsNum, 2 * ValidMatchedPointsNum).
	 * 
	 * \param rawMatchedPoints 原始包含无效点的匹配点
	 * \param selectedArray 筛选的index
	 * \param validPairsNum 有效的点对数量，根据selectedArray / 2来的
	 * \param validMatchedPoints 获得的有效的匹配点
	 * \param stream cuda流ID
	 */
	void collectValidMatchedPoints(DeviceArrayView<float4>& rawMatchedPoints, const DeviceArray<int>& selectedArray, DeviceBufferArray<float4>& validMatchedPoints, cudaStream_t stream);

	/**
	 * \brief 将融合后的深度面元映射到对应的VertexMap，NormalMap，ColorTimeMap上.
	 */
	void mapMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, device::MapMergedSurfelInterface& mergedSurfel, const unsigned int clipedWidth, const unsigned int clipedHeight, cudaStream_t stream = 0);

	/**
	 * \brief 清空上一帧的VertexMap，NormalMap，ColorTimeMap.
	 */
	void clearMapSurfel(const unsigned int clipedWidth, const unsigned int clipedHeight, device::MapMergedSurfelInterface& mergedSurfel, cudaStream_t stream = 0);


/*********************************************     构造并选择有效的深度面元     *********************************************/
	
	/**
	 * \brief 多视角前景Mask约束.
	 *
	 * \param depth 深度纹理
	 * \param vertex 需要约束的VertexMap
	 * \param normal 需要约束的NormalMap
	 * \param color	需要约束的ColorMap
	 * \param MaskConstriction 多视角前景Mask约束参数接口
	 * \param devicesCount 相机数量
	 * \param clipedWidth 剪裁后图像宽
	 * \param clipedHeight 剪裁后图像高
	 * \param stream cuda流
	 */
	void MultiViewForegroundMaskConstriction(CudaTextureSurface* depthMap, CudaTextureSurface* vertexMap, CudaTextureSurface* normalMap, CudaTextureSurface* colorMap, device::MultiViewMaskInterface& MaskConstrictionInterface, const unsigned int devicesCount, const unsigned int clipedWidth, const unsigned int clipedHeight, cudaStream_t stream = 0);
}
