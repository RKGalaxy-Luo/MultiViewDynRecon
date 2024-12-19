/*****************************************************************//**
 * \file   CameraObservation.h
 * \brief  记录摄像头拍摄到的，上传到cuda纹理中的一些图片
 * 
 * \author LUO
 * \date   February 3rd 2024
 *********************************************************************/
#pragma once
#include "CommonTypes.h"
#include <base/DeviceReadWrite/DeviceArrayView.h>

namespace SparseSurfelFusion {
	/**
	 * \brief 结构体用来承载传输数据.
	 */
	struct CameraObservation {
		//根据融合后的深度面元生成的单视角深度面元映射 直接替换掉了vertexConfidenceMap、normalRadiusMap、colorTimeMap这三个
		//以下这三个是未融合原始的深度面元映射，用于ICP刚性对齐的。
		cudaTextureObject_t icpvertexConfidenceMap[MAX_CAMERA_COUNT];
		cudaTextureObject_t icpnormalRadiusMap[MAX_CAMERA_COUNT];
		//colorTimeMap单独保存下
		cudaTextureObject_t rawcolorTimeMap[MAX_CAMERA_COUNT];
		cudaTextureObject_t rawDepthImage[MAX_CAMERA_COUNT];				// 用于可视化的原始深度图像

		// 上一帧的顶点和发现纹理，用于ICP
		cudaTextureObject_t PreviousVertexConfidenceMap[MAX_CAMERA_COUNT];	// 上一帧的顶点纹理
		cudaTextureObject_t PreviousNormalRadiusMap[MAX_CAMERA_COUNT];		// 上一帧的法线纹理

		// 几何成员
		cudaTextureObject_t filteredDepthImage[MAX_CAMERA_COUNT];			// 过滤剪裁后的深度图像
		cudaTextureObject_t vertexConfidenceMap[MAX_CAMERA_COUNT];			// “顶点-置信度”纹理 -> float4
		cudaTextureObject_t normalRadiusMap[MAX_CAMERA_COUNT];				// “法线-半径”纹理 -> float4
		DeviceArrayView<DepthSurfel> validSurfelArray[MAX_CAMERA_COUNT];	// 可视化有效面元数组

		// 颜色成员
		cudaTextureObject_t colorTimeMap[MAX_CAMERA_COUNT];					// “颜色-上一帧出现时刻” 纹理 -> float4
		cudaTextureObject_t normalizedRGBAMap[MAX_CAMERA_COUNT];			// “归一化RGBA” 纹理 -> float4
		cudaTextureObject_t normalizedRGBAPrevious[MAX_CAMERA_COUNT];		// “归一化上一帧RGBA”纹理 -> float4
		cudaTextureObject_t grayScaleMap[MAX_CAMERA_COUNT];					// RGB灰度图  这应该对应着density_map
		cudaTextureObject_t grayScaleGradientMap[MAX_CAMERA_COUNT];			// RGB灰度梯度图  density_gradient_map


		// 前景掩膜
		cudaTextureObject_t foregroundMask[MAX_CAMERA_COUNT];					// 前景掩膜
		cudaTextureObject_t foregroundMaskPrevious[MAX_CAMERA_COUNT];			// 上一帧的掩膜
		cudaTextureObject_t filteredForegroundMask[MAX_CAMERA_COUNT];			// 滤波前景掩膜
		cudaTextureObject_t foregroundMaskGradientMap[MAX_CAMERA_COUNT];		// 前景掩膜梯度图
		cudaTextureObject_t edgeMaskMap[MAX_CAMERA_COUNT];						// 边缘Mask

		// 特征匹配点
		DeviceArrayView<ushort4> correspondencePixelPairs[MAX_CAMERA_COUNT];	// “对应点对”纹理 -> ushort4      A点(x,y)  <=>  B点(z,w)
	
		// 跨镜匹配点
		DeviceArrayView<CrossViewCorrPairs> crossCorrPairs;						// 跨镜匹配的像素点(x, y)是当前视角的点, (z, w)是匹配到的另一个视角的点
		DeviceArrayView2D<ushort4> corrMap[MAX_CAMERA_COUNT];					// 匹配Map

		// 插值点
		DeviceArrayView2D<unsigned char> interpolatedValidValue[MAX_CAMERA_COUNT];	// 标记有效的插值处
		DeviceArrayView2D<float4> interpolatedVertexMap[MAX_CAMERA_COUNT];			// 插值VertexMap
		DeviceArrayView2D<float4> interpolatedNormalMap[MAX_CAMERA_COUNT];			// 插值NormalMap
		DeviceArrayView2D<float4> interpolatedColorMap[MAX_CAMERA_COUNT];			// 插值ColorMap

	};
}
