#pragma once
/*****************************************************************//**
 * \file   CrossViewMatchingInterpolation.h
 * \brief  ��Կ羵ƥ�����б�Ե��ֵ
 * 
 * 
 * \author LUO
 * \date   December 2024
 *********************************************************************/
#include "CrossViewEdgeCorrespondence.h"

#include <math/DualQuaternion/DualQuaternion.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>

#include <visualization/Visualizer.h>

namespace SparseSurfelFusion {
	namespace device {
		struct CrossViewInterpolateInput {
			cudaTextureObject_t VertexMap[MAX_CAMERA_COUNT];				// vertexMap
			cudaTextureObject_t NormalMap[MAX_CAMERA_COUNT];				// normalMap����
			cudaTextureObject_t ColorMap[MAX_CAMERA_COUNT];					// colorMap����
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];						// �ӽ�λ��
			mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];					// �ӽ�λ��
			Intrinsic intrinsic[MAX_CAMERA_COUNT];							// ����ڲ�
		};

		struct CrossViewInterpolateOutput {
			PtrStepSize<float4> interVertexMap[MAX_CAMERA_COUNT];			// ��ֵ���vertexMap
			PtrStepSize<float4> interNormalMap[MAX_CAMERA_COUNT];			// ��ֵ���NormalMap
			PtrStepSize<float4> interColorMap[MAX_CAMERA_COUNT];			// ��ֵ���ColorMap

			PtrStepSize<unsigned int> mutexFlag[MAX_CAMERA_COUNT];		// ��ֹ�����ֵ��ͶӰ��ͬһ�����ص���ɷô����
			PtrStepSize<unsigned char> markInterValue[MAX_CAMERA_COUNT];	// �����Ч�Ĳ�ֵ��λ��
		};

		struct CorrectTextureIO {
			// ��ֵ����
			DeviceArrayView2D<float4> interVertexMap[MAX_CAMERA_COUNT];			// ��ֵ���vertexMap
			DeviceArrayView2D<float4> interNormalMap[MAX_CAMERA_COUNT];			// ��ֵ���NormalMap
			DeviceArrayView2D<float4> interColorMap[MAX_CAMERA_COUNT];			// ��ֵ���ColorMap

			DeviceArrayView2D<unsigned int> mutexFlag[MAX_CAMERA_COUNT];		// ��ֹ�����ֵ��ͶӰ��ͬһ�����ص���ɷô����
			DeviceArrayView2D<unsigned char> markInterValue[MAX_CAMERA_COUNT];	// �����Ч�Ĳ�ֵ��λ��

			// �۲�����
			cudaTextureObject_t VertexTextureMap[MAX_CAMERA_COUNT];			// ��ǰ֡�۲�ֻ����VertexMap
			cudaTextureObject_t NormalTextureMap[MAX_CAMERA_COUNT];			// ��ǰ֡�۲�ֻ����NormalMap
			cudaTextureObject_t ColorTextureMap[MAX_CAMERA_COUNT];			// ��ǰ֡�۲�ֻ����ColorMap

			cudaSurfaceObject_t VertexSurfaceMap[MAX_CAMERA_COUNT];			// ��ǰ֡�۲��д���VertexMap
			cudaSurfaceObject_t NormalSurfaceMap[MAX_CAMERA_COUNT]; 		// ��ǰ֡�۲��д���NormalMap
			cudaSurfaceObject_t ColorSurfaceMap[MAX_CAMERA_COUNT];			// ��ǰ֡�۲��д���ColorMap
		};

		/**
		 * \brief ɳ�������˶�.
		 */
		__device__ mat34 ScrewInterpolationMat(DualQuaternion dq, float ratio);

		/**
		 * \brief �羵��Ԫ��ֵ.
		 */
		__global__ void SurfelsInterpolationKernel(CrossViewInterpolateInput input, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, const unsigned int Cols, const unsigned int Rows, const unsigned int pairsNum, const float disThreshold, CrossViewInterpolateOutput output);
	
		/**
		 * \brief У���۲��Texture.
		 */
		__global__ void CorrectObservedTextureKernel(CorrectTextureIO io, const unsigned int Cols, const unsigned int Rows, const unsigned int CameraNum);
	}

	class CrossViewMatchingInterpolation
	{
	public:
		using Ptr = std::shared_ptr<CrossViewMatchingInterpolation>;

		CrossViewMatchingInterpolation(Intrinsic* intrinsicArray, mat34* initialPoseArray);

		~CrossViewMatchingInterpolation();

		struct CrossViewInterpolationInput {
			cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];				// vertexMap����
			cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];				// normalMap����
			cudaTextureObject_t colorMap[MAX_CAMERA_COUNT];					// colorMap����
		};



	private:
		const unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int ImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int ImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageSize = FRAME_HEIGHT * FRAME_WIDTH;
		const unsigned int clipedImageSize = ImageRowsCliped * ImageColsCliped;

		DeviceArray2D<float4> interVertexMap[MAX_CAMERA_COUNT];			// ��ֵ���vertexMap
		DeviceArray2D<float4> interNormalMap[MAX_CAMERA_COUNT];			// ��ֵ���NormalMap
		DeviceArray2D<float4> interColorMap [MAX_CAMERA_COUNT];			// ��ֵ���ColorMap

		DeviceArray2D<unsigned int> mutexFlag[MAX_CAMERA_COUNT];		// ��ֹ�����ֵ��ͶӰ��ͬһ�����ص���ɷô����
		DeviceArray2D<unsigned char> markInterValue[MAX_CAMERA_COUNT];	// �����Ч�Ĳ�ֵ��λ��

		device::CrossViewInterpolateInput interpolationInput;			// �����ֵ�������
		device::CrossViewInterpolateOutput interpolationOutput;			// ��ֵ���Ӧ���
		device::CorrectTextureIO correctedIO;							// �޸Ĺ۲�֡��IO

		DeviceArrayView<CrossViewCorrPairs> crossCorrPairs;				// ����Ŀ羵ƥ���
		
	public:
		/**
		 * \brief �羵��ֵ����.
		 * 
		 * \param input ����
		 * \param crossPairs �羵ƥ���
		 */
		void SetCrossViewInterpolationInput(const CrossViewInterpolationInput& input, DeviceArrayView<CrossViewCorrPairs> crossPairs);

		/**
		 * \brief �羵ƥ���ֵSurfels.
		 * 
		 * \param stream cuda��
		 */
		void CrossViewInterpolateSurfels(cudaStream_t stream = 0);

		/**
		 * \brief У��VertexMap, NormalMap, ColorMap, ����ֵ�������ȥ.
		 * 
		 * \param VertexMap ����Map
		 * \param NormalMap ����Map
		 * \param ColorMap ��ɫMap
		 * \param stream cuda��
		 */
		void CorrectVertexNormalColorTexture(CudaTextureSurface* VertexMap, CudaTextureSurface* NormalMap, CudaTextureSurface* ColorMap, cudaStream_t stream = 0);
	
		/**
		 * \brief ���ĳ��ƽ��Ĳ�ֵVertexMap.
		 * 
		 * \param CameraID ����ӽ�ID
		 * \return ƽ��Ĳ�ֵVertexMap
		 */
		DeviceArrayView2D<float4> GetInterpolatedVertexMap(const unsigned int CameraID) { return correctedIO.interVertexMap[CameraID]; }
		
		/**
		 * \brief ���ĳ��ƽ��Ĳ�ֵNormalMap.
		 * 
		 * \param CameraID ����ӽ�ID
		 * \return ƽ��Ĳ�ֵNormalMap
		 */
		DeviceArrayView2D<float4> GetInterpolatedNormalMap(const unsigned int CameraID) { return correctedIO.interNormalMap[CameraID]; }

		/**
		 * \brief ���ĳ��ƽ��Ĳ�ֵColorMap.
		 * 
		 * \param CameraID ����ӽ�ID
		 * \return ƽ��Ĳ�ֵColorMap
		 */
		DeviceArrayView2D<float4> GetInterpolatedColorMap(const unsigned int CameraID) { return correctedIO.interColorMap[CameraID]; }

		/**
		 * \brief �����Ч�Ĳ�ֵ���Map.
		 * 
		 * \param CameraID ����ӽ�ID
		 * \return ��Ч�Ĳ�ֵ���Map
		 */
		DeviceArrayView2D<uchar> GetValidInterpolatedMarkMap(const unsigned int CameraID) { return correctedIO.markInterValue[CameraID]; }
	
	private:
		/**
		 * \brief ���ó�ʼֵ.
		 * 
		 * \param stream cuda��
		 */
		void SetInitialValue(cudaStream_t stream = 0);
	};
}


