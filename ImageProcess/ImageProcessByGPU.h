/*****************************************************************//**
 * \file   ImageProcessByGPU.h
 * \brief  ��Ҫ�漰һЩ��GPU����ͼ��Ĳ��������ã��˲���������Ԫ����ͼ��
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
		 * \brief ͨ�����ӽǵ�ǰ��Mask��Լ��ģ�ͱ�Ե����쳣��.
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
		 * \brief ���ںϵ���Ԫ��ӳ�䵽����Camera��IndexMap��.
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


/*******************************************     ����RGB��DepthͼƬ     *******************************************/
		/**
		 * \brief �������ͼ���õĺ˺���.
		 *
		 * \param rawDepth ԭʼ���ͼ(��cudaTextureObject_t���ʹ���)
		 * \param clipImageRows	���õ�ͼ��߶�
		 * \param clipImageCols	���õ�ͼ����
		 * \param clipNear ���ͼ����ľ���
		 * \param clipFar ���ͼ��Զ�ľ���
		 * \param sigmaSInverseSquare ˫���˲��пռ����Ȩ�ص�ƽ��������
		 * \param sigmaRInverseSquare ˫���˲�������ֵȨ�ص�ƽ��������
		 * \param filterDepth �ü���ɵ����ͼ(��cudaSurfaceObject_t���ʹ���)
		 */
		__global__ void clipFilterDepthKernel(cudaTextureObject_t rawDepth, const unsigned int clipImageRows, const unsigned int clipImageCols, const unsigned int clipNear, const unsigned int clipFar, const float sigmaSInverseSquare, const float sigmaRInverseSquare, cudaSurfaceObject_t filterDepth);
	
		
		/**
		 * \brief ��RGBͼ����ò���һ���ĺ˺���.
		 *
		 * \param rawColorImage ԭʼ��RGBͼ��(��const DeviceArray<uchar3>���ʹ��룺�����Զ�ת����const PtrSize<const uchar3>)
		 * \param clipRows ����ͼ��ĸ�
		 * \param clipCols ����ͼ��Ŀ�
		 * \param clipColorImage ���ú��RGBͼ��(��cudaSurfaceObject_t���ʹ���)
		 */
		__global__ void clipNormalizeColorKernel(const PtrSize<const uchar3> rawColorImage, const unsigned int clipRows, const unsigned int clipCols, cudaSurfaceObject_t clipColorImage);

		/**
		 * \brief ��RGBͼ����ò���һ���ĺ˺���.
		 *
		 * \param rawColorImage ԭʼ��RGBͼ��(��const DeviceArray<uchar3>���ʹ��룺�����Զ�ת����const PtrSize<const uchar3>)
		 * \param clipRows ����ͼ��ĸ�
		 * \param clipCols ����ͼ��Ŀ�
		 * \param clipColorImage ���ú��RGBͼ��(��cudaSurfaceObject_t���ʹ���)
		 * \param GrayScaleImage ����RGBͼ�����ܶ�(�Ҷ�)ͼ(��cudaSurfaceObject_t���ʹ���)
		 */
		__global__ void clipNormalizeColorKernel(const PtrSize<const uchar3> rawColorImage, const unsigned int clipRows, const unsigned int clipCols, cudaSurfaceObject_t clipColorImage, cudaSurfaceObject_t GrayScaleImage);
	
		/**
		 * \brief �ԻҶ�ͼ�����˫���˲��ĺ˺���.
		 *
		 * \param grayScaleImage ��ɼ��õĻҶ�ͼ��
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 * \param filteredGrayScaleImage ����˲���ĻҶ�ͼ��
		 */
		__global__ void filterCrayScaleImageKernel(cudaTextureObject_t grayScaleImage, unsigned int rows, unsigned int cols, cudaSurfaceObject_t filteredGrayScaleImage);


/*******************************************     ����RGB��DepthͼƬ������Ԫ����Map     *******************************************/
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
		 * \brief ����ǰһ֡�Ķ���ͷ�������.
		 *
		 * \param collectPreviousVertex �ռ�ǰһ֡����
		 * \param collectPreviousNormal �ռ�ǰһ֡����
		 * \param previousVertexTexture ǰһ֡����
		 * \param previousNormalTexture ǰһ֡����
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 */
		__global__ void copyPreviousVertexAndNormalKernel(cudaSurfaceObject_t collectPreviousVertex, cudaSurfaceObject_t collectPreviousNormal, cudaTextureObject_t previousVertexTexture, cudaTextureObject_t previousNormalTexture, const unsigned int rows, const unsigned int cols);

		/**
		 * \brief ���춥������Ŷȵ�Map�ĺ˺���.
		 *
		 * \param depthImage �������ͼ(��cudaTextureObject_t��ʽ����)
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 * \param intrinsicInverse ����ڲε��棬Ϊ���󶥵����������ϵ�µ�����
		 * \param vertexConfidenceMap �����Ķ��㼰���Ŷȵ�ͼ(������cudaSurfaceObject_t����ʽ)
		 */
		__global__ void createVertexConfidenceMapKernel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, const IntrinsicInverse intrinsicInverse, cudaSurfaceObject_t vertexConfidenceMap);

		/**
		 * \brief ���취�ߺͰ뾶��Map�ĺ˺���.
		 *
		 * \param vertexMap ��ɹ���Ķ��㼰���Ŷȵ�ͼMap(��cudaTextureObject_t��ʽ����)
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 * \param cameraFocal �������
		 * \param normalRadiusMap �����ķ����Լ��뾶��ͼ(������cudaSurfaceObject_t����ʽ)
		 */
		__global__ void createNormalRadiusMapKernel(cudaTextureObject_t vertexMap, const unsigned int rows, const unsigned int cols, float cameraFocal, cudaSurfaceObject_t normalRadiusMap);
	
/*******************************************     ����RGB����һ�ο�����Ԫ��ʱ�̹���Color-Time Map     *******************************************/

		/**
		 * \brief ����Color-Timeͼ.
		 *
		 * \param rawColor ԭʼRGBͼ��
		 * \param rows ���ú��ͼ���
		 * \param cols ���ú��ͼ���
		 * \param initTime ��ʼʱ��(�ڼ�֡)
		 * \param colorTimeMap Color-Timeͼ
		 */
		__global__ void createColorTimeMapKernel(const PtrSize<const uchar3> rawColor, const unsigned int rows, const unsigned int cols, const float initTime, const float CameraID, cudaSurfaceObject_t colorTimeMap);
	
/*********************************************     ���첢ѡ����Ч�������Ԫ     *********************************************/
		/**
		 * \brief �����ͼ��ѡ����Ч����Ԫ������validIndicator�����־���鸳ֵ.
		 *
		 * \param depthImage �������
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 * \param validIndicator ��Ч��Ԫindex�ı�־���� (DeviceArray����ֱ��ת����PtrSize���ͣ�ת����ԭ����DeviceArray����ֱ��ʹ��[]����Ԫ��)
		 */
		__global__ void markValidDepthPixelKernel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, PtrSize<char> validIndicator);

		/**
		 * \brief �����ͼ��ѡ����Ч����Ԫ������validIndicator�����־���鸳ֵ.
		 *
		 * \param depthImage �������
		 * \param foregroundMask ǰ����Ĥ
		 * \param normalMap ����ͼ
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 * \param validIndicator ��Ч��Ԫindex�ı�־���� (DeviceArray����ֱ��ת����PtrSize���ͣ�ת����ԭ����DeviceArray����ֱ��ʹ��[]����Ԫ��)
		 */
		__global__ void markValidDepthPixelKernel(cudaTextureObject_t depthImage, cudaTextureObject_t foregroundMask, cudaTextureObject_t normalMap, const unsigned int rows, const unsigned int cols, PtrSize<char> validIndicator);

		/**
		 * \brief ��ƥ������ѡ����Ч��ƥ��㣬previous = (0,0,0) ���� current = (0,0,0)��ƥ���ȥ��.
		 * 
		 * \param MatchedPointsPairs ƥ��ĵ�� previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum)
		 * \param pairsNum ƥ��Ե�����
		 * \param validIndicator ��Ч��Ԫindex�ı�־���� (DeviceArray����ֱ��ת����PtrSize���ͣ�ת����ԭ����DeviceArray����ֱ��ʹ��[]����Ԫ��)
		 */
		__global__ void markValidMatchedPointsKernel(DeviceArrayView<float4> MatchedPointsPairs, const unsigned int pairsNum, PtrSize<char> validIndicator);

		/**
		 * \brief ����cub���е�Flagged����ȥ����Ч�������Ԫ��䵽validDepthSurfel������(��άתһά).
		 *
		 * \param vertexConfidenceMap ���㼰���Ŷ�Map
		 * \param normalRadiusMap ���߼��뾶Map
		 * \param colorTimeMap Color-Time Map
		 * \param selectedArray �洢�Ƿ���Ч��־λ������
		 * \param rows ͼ��ĸ�
		 * \param cols ͼ��Ŀ�
		 * \param validDepthSurfel ��Ч�������Ԫ����(������DeviceArray<DepthSurfel>����)
		 */
		__global__ void collectDepthSurfelKernel(cudaTextureObject_t vertexConfidenceMap, cudaTextureObject_t normalRadiusMap, cudaTextureObject_t colorTimeMap, const PtrSize<const int> selectedArray, const unsigned int rows, const unsigned int cols, const unsigned int CameraID, PtrSize<DepthSurfel> validDepthSurfel);
		
		/**
		 * \brief ����ƥ�������꣬��ȡƥ������ά����.
		 * 
		 * \param previousTexture ǰһ��������ĵ���ͼƬ������
		 * \param currentTexture ��ǰ������ĵ�������
		 * \param pointsPairsCoor ƥ�������(˫����previous��������current)
		 * \param totalPointsNum ���������ƥ�������(��Ȼ��˫��)
		 * \param matchedPoints ��ȡ����ƥ���(˫����previous��������current)
		 */
		__global__ void collectMatchedPointsKernel(cudaTextureObject_t previousTexture, cudaTextureObject_t currentTexture, DeviceArrayView<PixelCoordinate> pointsPairsCoor, const unsigned int totalPointsNum, DeviceArrayHandle<float4> matchedPoints);
	
		/**
		 * \brief ����ԭʼƥ����Լ�ɸѡArray����ȡ��Ч����άƥ�������.
		 * 
		 * \param rawMatchedPoints ԭʼƥ���
		 * \param selectedArray �洢�Ƿ���Ч��־λ������
		 * \param validPairsNum ��Ч�������
		 * \param validMatchedPoints ���������Ч��ƥ����
		 */
		__global__ void collectValidMatchedPointsKernel(DeviceArrayView<float4> rawMatchedPoints, const PtrSize<const int> selectedArray, const unsigned int validPairsNum, DeviceArrayHandle<float4> validMatchedPoints);
	

/*******************************************     ��ӳ���ںϺ�����     *******************************************/
		/**
		 * \brief ���mapfurface�õģ�ÿ֡��Ҫ���������.
		 */
		__global__ void clearMapSurfelKernel(MapMergedSurfelInterface mergedSurfel, const unsigned int clipedWidth, const unsigned int clipedHeight);

		/**
		 * \brief ӳ�䵥һ�ӽǵ����ͼ�������������ÿһ֡����ȶ���ӳ��ͼ.
		 */
		__global__ void mapMergedDepthSurfelKernel(const DeviceArrayView<DepthSurfel> validSurfelArray, MapMergedSurfelInterface mergedSurfel, const unsigned int validSurfelNum, const unsigned int clipedWidth, const unsigned int clipedHeight);

/*******************************************     ���ӽ�ǰ������     *******************************************/

		__global__ void constrictMultiViewForegroundKernel(cudaSurfaceObject_t depthMap, cudaSurfaceObject_t vertexMap, cudaSurfaceObject_t normalMap, cudaSurfaceObject_t colorMap, MultiViewMaskInterface MultiViewInterface, const unsigned int CameraID, const unsigned int clipedWidth, const unsigned int clipedHeight);
	}

/*******************************************     ����RGB��DepthͼƬ     *******************************************/

	/**
	 * \brief ��.cu�ļ���ʵ�����ͼ�ü�����.
	 *
	 * \param rawDepth ԭʼ�����ͼ(��cudaTextureObject_t��ʽ����)
	 * \param clipImageRows ���õ���ȸ߶�
	 * \param clipImageCols ���õ���ȿ��
	 * \param clipNear �������ľ����Ƕ���
	 * \param clipFar �����Զ�ľ����Ƕ���
	 * \param filterDepth �ü�������ͼ(��cudaSurfaceObject_t��ʽ��)
	 * \param stream cuda��ID
	 */
	void clipFilterDepthImage(cudaTextureObject_t rawDepth, const unsigned int clipImageRows, const unsigned int clipImageCols, const unsigned int clipNear, const unsigned clipFar, cudaSurfaceObject_t filterDepth, cudaStream_t stream);

	/**
	 * \brief ������һ��RGBͼ��.
	 *
	 * \param rawColorImage ԭʼ��Colorͼ��(��DeviceArray<uchar3>����ʽ����)
	 * \param clipRows ����ͼ��ĸ�
	 * \param clipCols ����ͼ��Ŀ�
	 * \param clipColorImage ���ú��RGBͼ��(��cudaSurfaceObject_t����ʽ����)
	 * \param stream cuda��ID
	 */
	void clipNormalizeColorImage(const DeviceArray<uchar3>& rawColorImage, unsigned int clipRows, unsigned int clipCols, cudaSurfaceObject_t clipColorImage, cudaStream_t stream);

	/**
	 * \brief ������һ��RGBͼ��.
	 *
	 * \param rawColorImage ԭʼ��Colorͼ��(��DeviceArray<uchar3>����ʽ����)
	 * \param clipRows ����ͼ��ĸ�
	 * \param clipCols ����ͼ��Ŀ�
	 * \param clipColorImage ���ú��RGBͼ��(��cudaSurfaceObject_t����ʽ����)
	 * \param grayScaleImage ����ܶ�(�Ҷ�)ͼ(��cudaSurfaceObject_t����ʽ����)
	 * \param stream cuda��ID
	 */
	void clipNormalizeColorImage(const DeviceArray<uchar3>& rawColorImage, unsigned int clipRows, unsigned int clipCols, cudaSurfaceObject_t clipColorImage, cudaSurfaceObject_t grayScaleImage, cudaStream_t stream);

	/**
	 * \brief ���ûҶ�ͼ�������ͼ���Сһ��.
	 *
	 * \param grayScaleImage ��Ҫ˫���˲��ĻҶ�ͼ��(��cudaTextureObject_t��ʽ����)
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param filteredGrayScaleImage ���˫���˲��ĻҶ�(��cudaSurfaceObject_t��ʽ����)
	 * \param stream cuda��ID
	 */
	void filterGrayScaleImage(cudaTextureObject_t grayScaleImage, unsigned int rows, unsigned int cols, cudaSurfaceObject_t filteredGrayScaleImage, cudaStream_t stream);


/*******************************************     ����RGB��DepthͼƬ������Ԫ����Map     *******************************************/

	/**
	 * \brief ����ǰһ֡�Ķ���ͷ�������.
	 * 
	 * \param collectPreviousVertex �ռ�ǰһ֡����
	 * \param collectPreviousNormal �ռ�ǰһ֡����
	 * \param previousVertexTexture ǰһ֡����
	 * \param previousNormalTexture ǰһ֡����
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param stream cuda��ID
	 */
	void copyPreviousVertexAndNormal(cudaSurfaceObject_t collectPreviousVertex, cudaSurfaceObject_t collectPreviousNormal, cudaTextureObject_t previousVertexTexture, cudaTextureObject_t previousNormalTexture, const unsigned int rows, const unsigned int cols, cudaStream_t stream = 0);

	/**
	 * \brief ���춥������Ŷȵ�Map(һ�ֳ����float4����x,y,z�Ƕ������꣬w�Ƕ������Ŷ�).
	 *
	 * \param depthImage �������ͼ(��cudaTextureObject_t��ʽ����)
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param intrinsicInverse ����ڲε��棬Ϊ���󶥵����������ϵ�µ�����
	 * \param vertexConfidenceMap �����Ķ��㼰���Ŷȵ�ͼ(������cudaSurfaceObject_t����ʽ)
	 * \param stream cuda��ID
	 */
	void createVertexConfidenceMap(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, const IntrinsicInverse intrinsicInverse, cudaSurfaceObject_t vertexConfidenceMap, cudaStream_t stream);

	/**
	 * \brief ���취�ߺͰ뾶��Map(һ�ֳ����float4����x,y,z�Ƿ��ߣ�w�ǰ뾶).
	 *
	 * \param vertexMap ��ɹ���Ķ��㼰���Ŷȵ�ͼMap(��cudaTextureObject_t��ʽ����)
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param cameraFocal ����Ĺ��ģ�����������Ԫ�뾶
	 * \param normalRadiusMap �����ķ����Լ��뾶��ͼ(������cudaSurfaceObject_t����ʽ)
	 * \param stream cuda��ID
	 */
	void createNormalRadiusMap(cudaTextureObject_t vertexMap, const unsigned int rows, const unsigned int cols, float CameraFocal, cudaSurfaceObject_t normalRadiusMap, cudaStream_t stream);

/*******************************************     ����RGB����һ�ο�����Ԫ��ʱ�̹���Color-Time Map     *******************************************/

	/**
	 * \brief ����Color-Timeͼ.
	 *
	 * \param rawColor ����ԭʼͼ��
	 * \param rows ���ú�ͼ��ĸ�
	 * \param cols ���ú�ͼ��Ŀ�
	 * \param initTime ��ʼ��ʱ��(�ڼ�֡)
	 * \param colorTimeMap ������ɫ-��һ֡��׽��ʱ��[Color-Timeͼ](������cudaSurfaceObject_t��ʽ)
	 * \param stream cuda��ID
	 */
	void createColorTimeMap(const DeviceArray<uchar3> rawColor, const unsigned int rows, const unsigned int cols, const float initTime, const float CameraID, cudaSurfaceObject_t colorTimeMap, cudaStream_t stream);


/*********************************************     ���첢ѡ����Ч�������Ԫ     *********************************************/
	
	/**
	 * \brief �����Ч��������أ�����Ч��ȵ�index�浽validIndicator�У���Ӧ�������indexλ��Ϊ1������Ϊ0.
	 *
	 * \param depthImage ���ͼ��
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param validIndicator ���������Ч������ص�ָʾ��(��־λ) 
	 * \param stream cuda��ID
	 */
	void markValidDepthPixel(cudaTextureObject_t depthImage, const unsigned int rows, const unsigned int cols, DeviceArray<char>& validIndicator, cudaStream_t stream);

	/**
	 * \brief �����Ч��������أ�����Ч��ȵ�index�浽validIndicator�У���Ӧ�������indexλ��Ϊ1������Ϊ0.
	 *
	 * \param depthImage ���ͼ��
	 * \param foregroundMask ǰ����Ĥ
	 * \param normalMap ���������������Ϊ0��Ϊ��Ч��
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param validIndicator ���������Ч������ص�ָʾ��(��־λ)
	 * \param stream cuda��ID
	 */
	void markValidDepthPixel(cudaTextureObject_t depthImage, cudaTextureObject_t foregroundMask, cudaTextureObject_t normalMap, const unsigned int rows, const unsigned int cols, DeviceArray<char>& validIndicator, cudaStream_t stream);

	/**
	 * \brief �����Ч��ƥ��㣬����������Ϣ�������Ϣ�ĵ��index���1.
	 * 
	 * \param MatchedPointsPairs ��ҪѰ�ҵĵ��
	 * \param pairsNum һ���ж��ٶ�ƥ���
	 * \param validIndicator ���������Ч������ص�ָʾ��(��־λ) 
	 * \param stream cuda��ID
	 */
	void markValidMatchedPoints(DeviceArrayView<float4>& MatchedPointsPairs, const unsigned int pairsNum, DeviceArray<char>& validIndicator, cudaStream_t stream);

	/**
	 * \brief �ռ���Ч�������Ԫ.
	 *
	 * \param vertexConfidenceMap ���㼰���Ŷ�����
	 * \param normalRadiusMap ���߼��뾶����
	 * \param colorTimeMap ��ɫ���ϴμ��������ʱ��
	 * \param selectedArray ��¼��Ч�������index������
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param CameraID �������ĸ����
	 * \param validDepthSurfel ��õ���Ч�������Ԫ
	 * \param stream cuda��ID
	 */
	void collectDepthSurfel(cudaTextureObject_t vertexConfidenceMap, cudaTextureObject_t normalRadiusMap, cudaTextureObject_t colorTimeMap, const DeviceArray<int>& selectedArray, const unsigned int rows, const unsigned int cols, const unsigned int CameraID, DeviceArray<DepthSurfel>& validDepthSurfel, cudaStream_t stream);
	
	/**
	 * \brief �ռ�ƥ���ԣ�previous [0,MatchedPointsNum)   current [MatchedPointsNum, 2 * MatchedPointsNum).
	 *
	 * \param rows ͼ��ĸ�
	 * \param cols ͼ��Ŀ�
	 * \param previousTexture ǰһ�������Texture
	 * \param currentTexture ��ǰ�����Texture
	 * \param pointsPairs ������ƥ���õĵ�����
	 * \param matchedPoints ���������õ���Ч����ά������
	 * \param stream cuda��ID
	 */
	void collectMatchedPoints(cudaTextureObject_t previousTexture, cudaTextureObject_t currentTexture, DeviceArrayView<PixelCoordinate>& pointsPairs, DeviceBufferArray<float4>& matchedPoints, cudaStream_t stream);

	/**
	 * \brief �ռ���Ч��ƥ���ԣ�previous [0,ValidMatchedPointsNum)   current [ValidMatchedPointsNum, 2 * ValidMatchedPointsNum).
	 * 
	 * \param rawMatchedPoints ԭʼ������Ч���ƥ���
	 * \param selectedArray ɸѡ��index
	 * \param validPairsNum ��Ч�ĵ������������selectedArray / 2����
	 * \param validMatchedPoints ��õ���Ч��ƥ���
	 * \param stream cuda��ID
	 */
	void collectValidMatchedPoints(DeviceArrayView<float4>& rawMatchedPoints, const DeviceArray<int>& selectedArray, DeviceBufferArray<float4>& validMatchedPoints, cudaStream_t stream);

	/**
	 * \brief ���ںϺ�������Ԫӳ�䵽��Ӧ��VertexMap��NormalMap��ColorTimeMap��.
	 */
	void mapMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, device::MapMergedSurfelInterface& mergedSurfel, const unsigned int clipedWidth, const unsigned int clipedHeight, cudaStream_t stream = 0);

	/**
	 * \brief �����һ֡��VertexMap��NormalMap��ColorTimeMap.
	 */
	void clearMapSurfel(const unsigned int clipedWidth, const unsigned int clipedHeight, device::MapMergedSurfelInterface& mergedSurfel, cudaStream_t stream = 0);


/*********************************************     ���첢ѡ����Ч�������Ԫ     *********************************************/
	
	/**
	 * \brief ���ӽ�ǰ��MaskԼ��.
	 *
	 * \param depth �������
	 * \param vertex ��ҪԼ����VertexMap
	 * \param normal ��ҪԼ����NormalMap
	 * \param color	��ҪԼ����ColorMap
	 * \param MaskConstriction ���ӽ�ǰ��MaskԼ�������ӿ�
	 * \param devicesCount �������
	 * \param clipedWidth ���ú�ͼ���
	 * \param clipedHeight ���ú�ͼ���
	 * \param stream cuda��
	 */
	void MultiViewForegroundMaskConstriction(CudaTextureSurface* depthMap, CudaTextureSurface* vertexMap, CudaTextureSurface* normalMap, CudaTextureSurface* colorMap, device::MultiViewMaskInterface& MaskConstrictionInterface, const unsigned int devicesCount, const unsigned int clipedWidth, const unsigned int clipedHeight, cudaStream_t stream = 0);
}
