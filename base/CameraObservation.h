/*****************************************************************//**
 * \file   CameraObservation.h
 * \brief  ��¼����ͷ���㵽�ģ��ϴ���cuda�����е�һЩͼƬ
 * 
 * \author LUO
 * \date   February 3rd 2024
 *********************************************************************/
#pragma once
#include "CommonTypes.h"
#include <base/DeviceReadWrite/DeviceArrayView.h>

namespace SparseSurfelFusion {
	/**
	 * \brief �ṹ���������ش�������.
	 */
	struct CameraObservation {
		//�����ںϺ�������Ԫ���ɵĵ��ӽ������Ԫӳ�� ֱ���滻����vertexConfidenceMap��normalRadiusMap��colorTimeMap������
		//������������δ�ں�ԭʼ�������Ԫӳ�䣬����ICP���Զ���ġ�
		cudaTextureObject_t icpvertexConfidenceMap[MAX_CAMERA_COUNT];
		cudaTextureObject_t icpnormalRadiusMap[MAX_CAMERA_COUNT];
		//colorTimeMap����������
		cudaTextureObject_t rawcolorTimeMap[MAX_CAMERA_COUNT];
		cudaTextureObject_t rawDepthImage[MAX_CAMERA_COUNT];				// ���ڿ��ӻ���ԭʼ���ͼ��

		// ��һ֡�Ķ���ͷ�����������ICP
		cudaTextureObject_t PreviousVertexConfidenceMap[MAX_CAMERA_COUNT];	// ��һ֡�Ķ�������
		cudaTextureObject_t PreviousNormalRadiusMap[MAX_CAMERA_COUNT];		// ��һ֡�ķ�������

		// ���γ�Ա
		cudaTextureObject_t filteredDepthImage[MAX_CAMERA_COUNT];			// ���˼��ú�����ͼ��
		cudaTextureObject_t vertexConfidenceMap[MAX_CAMERA_COUNT];			// ������-���Ŷȡ����� -> float4
		cudaTextureObject_t normalRadiusMap[MAX_CAMERA_COUNT];				// ������-�뾶������ -> float4
		DeviceArrayView<DepthSurfel> validSurfelArray[MAX_CAMERA_COUNT];	// ���ӻ���Ч��Ԫ����

		// ��ɫ��Ա
		cudaTextureObject_t colorTimeMap[MAX_CAMERA_COUNT];					// ����ɫ-��һ֡����ʱ�̡� ���� -> float4
		cudaTextureObject_t normalizedRGBAMap[MAX_CAMERA_COUNT];			// ����һ��RGBA�� ���� -> float4
		cudaTextureObject_t normalizedRGBAPrevious[MAX_CAMERA_COUNT];		// ����һ����һ֡RGBA������ -> float4
		cudaTextureObject_t grayScaleMap[MAX_CAMERA_COUNT];					// RGB�Ҷ�ͼ  ��Ӧ�ö�Ӧ��density_map
		cudaTextureObject_t grayScaleGradientMap[MAX_CAMERA_COUNT];			// RGB�Ҷ��ݶ�ͼ  density_gradient_map


		// ǰ����Ĥ
		cudaTextureObject_t foregroundMask[MAX_CAMERA_COUNT];					// ǰ����Ĥ
		cudaTextureObject_t foregroundMaskPrevious[MAX_CAMERA_COUNT];			// ��һ֡����Ĥ
		cudaTextureObject_t filteredForegroundMask[MAX_CAMERA_COUNT];			// �˲�ǰ����Ĥ
		cudaTextureObject_t foregroundMaskGradientMap[MAX_CAMERA_COUNT];		// ǰ����Ĥ�ݶ�ͼ
		cudaTextureObject_t edgeMaskMap[MAX_CAMERA_COUNT];						// ��ԵMask

		// ����ƥ���
		DeviceArrayView<ushort4> correspondencePixelPairs[MAX_CAMERA_COUNT];	// ����Ӧ��ԡ����� -> ushort4      A��(x,y)  <=>  B��(z,w)
	
		// �羵ƥ���
		DeviceArrayView<CrossViewCorrPairs> crossCorrPairs;						// �羵ƥ������ص�(x, y)�ǵ�ǰ�ӽǵĵ�, (z, w)��ƥ�䵽����һ���ӽǵĵ�
		DeviceArrayView2D<ushort4> corrMap[MAX_CAMERA_COUNT];					// ƥ��Map

		// ��ֵ��
		DeviceArrayView2D<unsigned char> interpolatedValidValue[MAX_CAMERA_COUNT];	// �����Ч�Ĳ�ֵ��
		DeviceArrayView2D<float4> interpolatedVertexMap[MAX_CAMERA_COUNT];			// ��ֵVertexMap
		DeviceArrayView2D<float4> interpolatedNormalMap[MAX_CAMERA_COUNT];			// ��ֵNormalMap
		DeviceArrayView2D<float4> interpolatedColorMap[MAX_CAMERA_COUNT];			// ��ֵColorMap

	};
}
