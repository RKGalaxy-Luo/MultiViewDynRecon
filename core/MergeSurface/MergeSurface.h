/*****************************************************************//**
 * \file   MergeSurface.h
 * \brief  ʹ�ò�ֵ��������ķ�ʽ�ں���������
 * 
 * \author LUOJIAXUAN
 * \date   June 14th 2024
 *********************************************************************/
#pragma once

#include <chrono>

#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <math/MatUtils.h>
#include <base/EncodeUtils.h>
#include <base/Constants.h>
#include <base/CommonTypes.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
namespace SparseSurfelFusion {

	// ��¼����ڲ�����Լ�ID
	struct CameraParameters {
		Intrinsic intrinsic;	// ����ڲ�
		mat34 SE3;				// �������ڻ�׼�����λ��
		short ID;				// ���ID
	};

	namespace device {
		enum {
			ReduceThreadsPerBlock = 128
		};

		/**
		 * \brief ���㽫������ͶӰ����ֵ���ص����ص�����.
		 * 
		 * \param Camera_1_SE3 ���1����ڻ�׼�����λ��
		 * \param Camera_1_intrinsic ���1���ڲε�Inverse
		 * \param Camera_2_SE3 ���2����ڻ�׼�����λ�˵�
		 * \param Camera_2_intrinsic ���2���ڲ�Inverse
		 * \param InterprolateCameraSE3 ��ֵ�������ڻ�׼�����λ��
		 * \param InterprolateCameraIntrinsic ��ֵ������ڲ�
		 * \param clipedCols ��ֵ�������ƽ��Ŀ�
		 * \param clipedRows ��ֵ�������ƽ��ĸ�
		 * \param totalSurfelNum ���������õ������Ԫ������
		 * \param Camera_1_DepthSurfels ���1���������ϵ����Ԫ
		 * \param Camera_2_DepthSurfels ���2���������ϵ����Ԫ
		 * \param OverlappingIndex ��õ�ǰ�ص���Ԫ��ͳ��ʱ��˳��
		 * \param overlappingSurfelsCountMap �ص�����ͳ��һά����
		 */
		__global__ void CountOverlappingSurfelsKernel(const mat34 Camera_1_SE3, const Intrinsic Camera_1_intrinsic, const mat34 Camera_2_SE3, const Intrinsic Camera_2_intrinsic, const mat34 InterprolateCameraSE3, const Intrinsic InterprolateCameraIntrinsic, const unsigned int clipedCols, const unsigned int clipedRows, const unsigned int Camera_1_SurfelsNums, const unsigned int totalSurfelNum, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, unsigned short* OverlappingIndex, unsigned int* overlappingSurfelsCountMap, ushort2* surfelProjectedPixelPos);

		/**
		 * \brief ��Լ�������ص��������.
		 * 
		 * \param overlappingSurfelsCountMap �ص���Ԫ����������
		 * \param clipedImageSize ����ͼ���С
		 * \param MaxCountData ����������¼
		 */
		__global__ void reduceMaxOverlappingSurfelCountKernel(const unsigned int* overlappingSurfelsCountMap, const unsigned int clipedImageSize, unsigned int* MaxCountData);

		/**
		 * \brief �ҵ��ص�����������.
		 * 
		 * \param maxCountDataHost ����ÿ��gird������ֵ
		 * \param gridNum grid������
		 * \param MaxValue ���ֵ
		 */
		__host__ void findMaxValueOfOverlappingCount(const unsigned int* maxCountDataHost, const unsigned int gridNum, unsigned int& MaxValue);

		/**
		 * \brief �ռ������ش��ص���surfel.
		 */
		__global__ void CollectOverlappingSurfelInPixel(const DepthSurfel* Camera_1_DepthSurfels, const DepthSurfel* Camera_2_DepthSurfels, const unsigned int* overlappingSurfelsCountMap, const unsigned short* OverlappingOrderMap, const ushort2* surfelProjectedPixelPos, const unsigned int Camera_1_SurfelCount, const unsigned int totalSurfels, const unsigned int clipedCols, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, unsigned int* SurfelIndexArray);


		/**
		 * \brief ����TSDF���¹����ص����Surfelλ��.
		 */
		__global__ void CalculateOverlappingSurfelTSDF(const CameraParameters Camera_1_Para, const CameraParameters Camera_2_Para, const CameraParameters Camera_inter_Para, const float maxMergeSquaredDistance, const unsigned int* overlappingSurfelsCountMap, const unsigned int* SurfelIndexArray, const unsigned int Camera_1_SurfelNum, const unsigned int clipedImageSize, const unsigned int MaxOverlappingSurfelNum, DepthSurfel* MergeArray, float* MergeSurfelTSDFValue, DepthSurfel* Camera_1_DepthSurfels, DepthSurfel* Camera_2_DepthSurfels, bool* markValidMergedSurfel);

		/**
		 * \brief ������������ƽ����ֵ.
		 */
		__forceinline__ __device__ float CalculateSquaredDistance(const float4& v1, const float4& v2);

		/**
		 * \brief ����TSDF��ֵ.
		 */
		__forceinline__ __device__ float CalculateTSDFValue(float zeroPoint, float otherPoint);

		/**
		 * \brief ����RGB����.
		 */
		__forceinline__ __device__ void DecodeFloatRGB(const float RGBCode, uchar3& color);

		/**
		 * \brief ��һ������.
		 */
		__forceinline__ __device__ float3 NormalizeNormal(float& nx, float& ny, float& nz);

		/**
		 * \brief ���û�в����ںϵĵ㣬������Щ��ת����Canonical����.
		 */
		__global__ void MarkAndConvertValidNotMergedSurfeltoCanonical(const DepthSurfel* SurfelsArray, const CameraParameters cameraPara, const unsigned int SurfelsNums, DepthSurfel* SurfelsCanonical, bool* markValidNotMergedFlag);

	}
	class MergeSurface
	{
	public:
		/**
			* \brief ������Ҫ�ںϵ�������棬�����0Ϊ������������1Ϊ�����.
			* 
			* \param CameraID_0 ���0
			* \param CameraID_1 ���1
			* \param intrinsics ����������ú���ڲξ���
			*/
		MergeSurface(const Intrinsic* intrinsics);
		~MergeSurface();
		using Ptr = std::shared_ptr<MergeSurface>;

		void MergeAllSurfaces(DeviceBufferArray<DepthSurfel>* depthSurfelsView);

		/**
		 * \brief ����ںϺ����Ԫ����ǰ������Ԫ����Canonical��.
		 * 
		 * \return �ںϺ��Canonical����Ԫ��ֻ����
		 */
		DeviceArrayView<DepthSurfel> GetMergedSurfelsView() { return MergedDepthSurfels.ArrayView(); }

		/**
		 * \brief ����ںϺ����Ԫ����ǰ������Ԫ����Canonical��.
		 *
		 * \return �ںϺ��Canonical����Ԫ���ɶ�д��
		 */
		DeviceArray<DepthSurfel> GetMergedSurfelArray() { return MergedDepthSurfels.Array(); }

	private:
		const unsigned int CamerasCount = MAX_CAMERA_COUNT;				// �������
		const unsigned int ClipedImageSize = CLIP_HEIGHT * CLIP_WIDTH;	// ����ͼƬ��С
		const unsigned int clipedCols = CLIP_WIDTH;
		const unsigned int clipedRows = CLIP_HEIGHT;
		DeviceBufferArray<DepthSurfel> MergedDepthSurfels;				// �ںϺ�������Ԫ����Ԫ��Canonical��
		CameraParameters CameraParameter[2 * MAX_CAMERA_COUNT];			// ��¼����ڲ�����Լ�ID����ʵ�����ż��������ֵ�����������
		DeviceBufferArray<unsigned int> OverlappingSurfelsCountMap;		// ��¼�ص��������(ʹ��shortò�ƻ�Ӱ��ԭ�Ӳ���������)
		DeviceBufferArray<unsigned short> OverlappingOrderMap;			// ��¼�����ӽǵ���Ԫ�����ص�ʱ��˳��[0, 65535]
		DeviceBufferArray<ushort2> surfelProjectedPixelPos;				// ��¼��ǰ��Ԫ��ͶӰ���˲�ֵ������ĸ����ص�
		cudaStream_t stream;											// cuda��


		/**
		 * \brief �������TSDF�ںϣ������ںϺ����Ԫ����MergedDepthSurfels.
		 * 
		 * \param Camera_1_Para ���1�����
		 * \param Camera_2_Para ���2�����
		 * \param Camera_Inter_Para ��ֵ��������
		 * \param Camera_1_DepthSurfels �������1�������Ԫ
		 * \param Camera_2_DepthSurfels �������1�������Ԫ
		 * \param stream cuda��
		 */
		void CalculateTSDFMergedSurfels(const CameraParameters Camera_1_Para, const CameraParameters Camera_2_Para, const CameraParameters Camera_Inter_Para, DeviceArray<DepthSurfel> Camera_1_DepthSurfels, DeviceArray<DepthSurfel> Camera_2_DepthSurfels, cudaStream_t stream = 0);

		/**
		 * \brief �ռ�û�в����ںϵ���Ԫ.
		 * 
		 * \param depthSurfels �����Ԫ
		 * \param stream cuda��
		 */
		void CollectNotMergedSurfels(DeviceBufferArray<DepthSurfel>* depthSurfels, cudaStream_t stream = 0);
	};
	
}


