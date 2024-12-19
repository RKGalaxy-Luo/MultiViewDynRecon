/*****************************************************************//**
 * \file   CrossViewEdgeCorrespondence.h
 * \brief  �羵��Եƥ��
 * 
 * \author LUO
 * \date   November 2024
 *********************************************************************/
#pragma once
#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/Constants.h>
#include <base/GlobalConfigs.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <core/AlgorithmTypes.h>
#include <math/MatUtils.h>
#include <memory>

#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>

namespace SparseSurfelFusion {
	namespace device {

		enum {
			CrossViewSearchSize = 20,
			CrossViewSearchStep = 2,
			OpticalSearchSize = 3,
			OpticalSearchStep = 1
		};

		struct CrossViewCorrInput {
			cudaTextureObject_t VertexMap[MAX_CAMERA_COUNT];				// vertexMap
			cudaTextureObject_t EdgeMask[MAX_CAMERA_COUNT];					// ǰ����ԵMask
			cudaTextureObject_t RawClipedMask[MAX_CAMERA_COUNT];			// ԭʼǰ��Mask(�ü���δ��̬ѧ����)
			DeviceArrayView2D<ushort4> CorrPairsMap[MAX_CAMERA_COUNT];		// ƥ����Map(û�ҵ������ص�ΪOXFFFF)
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];						// �ӽ�λ��
			mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];					// �ӽ�λ��
			Intrinsic intrinsic[MAX_CAMERA_COUNT];							// ����ڲ�
		};

		/**
		 * \brief �Բ�ͬ�ӽ�ƥ����������.
		 */
		__host__ __device__ __forceinline__ unsigned int EncodeCrossViewPixelPair(unsigned int x, unsigned int y, unsigned int view);

		/**
		 * \brief �ڱ�Ե��������Ŀ羵ƥ����.
		 */
		__device__ __forceinline__ bool GetClosetCrossViewPairs(CrossViewCorrInput& input, const unsigned int& view, const unsigned int& nextView, const unsigned int& lastView, const unsigned int& x, const unsigned int& y, const unsigned int& clipedCols, const unsigned int& clipedRows, const float& pairsSquDisThreshold, CrossViewCorrPairs& crossPairs);
		 
		/**
		 * \brief ����Ч��Pair����Ѱ����Ч�Ĺ���.
		 */
		__device__ __forceinline__ bool GetTrackedCrossViewPairs(CrossViewCorrInput& input, const CrossViewCorrPairs& closestPairs, CrossViewCorrPairs& trackedPairs);

		/**
		 * \brief �����Ч�Ŀ羵ƥ���.
		 */
		__global__ void MarkValidPairPixels(CrossViewCorrInput input, const unsigned int clipedCols, const unsigned int clipedRows, const unsigned int clipedImageSize, const unsigned int cameraNum, const float pairsSquDisThreshold, CrossViewCorrPairs* corrPairs, unsigned char* markValidEdgePixel);

		/**
		 * \brief ����羵�ӽǵ���������.
		 */
		__global__ void EncodeDiffViewPixelCoor(DeviceArrayView<CrossViewCorrPairs> validCorrPairs, const unsigned int PairsNum, unsigned int* diffViewCoorKey);

		/**
		 * \brief ����羵�ӽǵ���������.
		 */
		__global__ void EncodeSameViewPixelCoor(DeviceArrayView<CrossViewCorrPairs> validCorrPairs, const unsigned int PairsNum, unsigned int* sameViewCoorKey);

		/**
		 * \brief ��ǲ�ͬ��Key��ʼλ��.
		 */
		__global__ void LabelSortedPixelCoor(PtrSize<unsigned int> sortedKey, const unsigned int PairsNum, unsigned int* corrPairsLabel);
	
		/**
		 * \brief ��ù���ͬһ��ƥ����Pairs��Offset.
		 */
		__global__ void ComputePairsArrayOffset(const unsigned int* pairsLabel, const unsigned int* prefixSumLabel, const unsigned int PairsNum, const unsigned int uniquePairsNum, unsigned int* offset);

		/**
		 * \brief ɸ��pair�������Ӧ�����������Pair��Ϊ���յ�һ��һPairs.
		 */
		__global__ void SelectUniqueCrossViewMatchingPairs(DeviceArrayView<CrossViewCorrPairs> sortedPairs, CrossViewCorrInput input, const unsigned int* offset, const unsigned int uniquePairsNum, CrossViewCorrPairs* uniquePairs);

		/**
		 * \brief ɸѡ���ܱ�ƥ���׷�ݵĵ��.
		 */
		__global__ void SelectCrossViewBackTracingPairs(DeviceArrayView<CrossViewCorrPairs> uniquePairs, CrossViewCorrInput input, const unsigned int uniquePairsNum, unsigned char* markBackTracingPairs);
	}

	// �����ڱ�Ե���ӽǵ�ƥ��
	class CrossViewEdgeCorrespondence
	{
	public:
		using Ptr = std::shared_ptr<CrossViewEdgeCorrespondence>;

		CrossViewEdgeCorrespondence(Intrinsic* intrinsicArray, mat34* cameraPoseArray);

		~CrossViewEdgeCorrespondence();

		// �羵ƥ�������Package
		struct CrossViewMatchingInput {
			cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];				// vertexMap����
			cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];				// normalMap����
			cudaTextureObject_t colorMap[MAX_CAMERA_COUNT];					// colorMap����
			cudaTextureObject_t edgeMask[MAX_CAMERA_COUNT];					// ��ԵMask����
			cudaTextureObject_t rawClipedMask[MAX_CAMERA_COUNT];			// ���ú��Mask
			DeviceArrayView2D<ushort4> matchedPairsMap[MAX_CAMERA_COUNT];	// Mask�ϵ�ƥ���
		};



	private:
		const unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int ImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int ImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageSize = FRAME_HEIGHT * FRAME_WIDTH;
		const unsigned int clipedImageSize = ImageRowsCliped * ImageColsCliped;

		device::CrossViewCorrInput crossCorrInput;								// ������ڼ����Ե�羵��ƥ���

		DeviceBufferArray<CrossViewCorrPairs> corrPairs;						// ����Ǽ�¼�羳ƥ��ĵ����Դ: (x, y)�ǵ�ǰ�ӽ�ƥ��pixel, (z, w)�ǿ羵��Ѱ���ӽ�ƥ��pixel
		DeviceBufferArray<unsigned char> markValidPairs;						// �����Ч�ı�ԵPairs
		DeviceBufferArray<CrossViewCorrPairs> validCorrPairs;					// ɸѡ�õ����ʵı�ԵPairs(����һ�Զ�)

		DeviceBufferArray<unsigned int> viewCoorKey;							// �����¼��ͬView���������(����z, w)
		KeyValueSort<unsigned int, CrossViewCorrPairs> corrPairsSort;			// ���ݱ���������Pairs
		DeviceBufferArray<unsigned int> corrPairsLabel;							// ��Ƕ��һ�ĵ�
		PrefixSum prefixSum;
		DeviceBufferArray<unsigned int> sortedPairsOffset;						// ����ͬһ��ƥ����Pairs��Offset	
		DeviceBufferArray<CrossViewCorrPairs> uniqueCrossMatchingPairs;			// һ��һ�羵ƥ���

		DeviceBufferArray<CrossViewCorrPairs> uniqueCrossViewBackTracingPairs;	// ����ǿ羵ƥ������ͨ������һ֡ƥ�����ݵĵ��
		DeviceBufferArray<unsigned char> markValidBackTracingPairs;				// �����Ч�ı�ԵPairs

		unsigned int validCrossViewPairsNum = 0;								// ��¼��Ч�Ŀ羵ƥ���
		unsigned int UniqueMatchingNum = 0;										// һ��һ�羵ƥ������
		unsigned int UniqueBackTracingPairsNum = 0;								// �ܱ�ƥ���׷�ݵ�ƥ����
		/**
		 * \brief �����ڴ�.
		 * 
		 */
		void AllocateBuffer();

		/**
		 * \brief �ͷ��ڴ�.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief ѡ����Ч�Ŀ羵ƥ���.
		 * 
		 * \param stream cuda��
		 */
		void SelectValidCrossViewPairs(cudaStream_t stream = 0);

		/**
		 * \brief ɾ��View.x���ظ���ƥ��.
		 *
		 * \param stream cuda��
		 */
		void FilterSameViewRepetitiveMatching(cudaStream_t stream = 0);

		/**
		 * \brief ɾ��View.y���ظ���ƥ��.
		 * 
		 * \param stream cuda��
		 */
		void FilterDiffViewRepetitiveMatching(cudaStream_t stream = 0);

		/**
		 * \brief ɾ�����һƥ��.
		 * 
		 * \param stream cuda��
		 */
		void FilterSeveralForOneMatchingPairs(cudaStream_t stream = 0);

		/**
		 * \brief ɸѡ���ܱ��������ݵĵ�.
		 * 
		 * \param stream
		 */
		void SelectCrossViewBackTracingPairs(cudaStream_t stream = 0);

	public:
		/**
		 * \brief Ѱ�ҿ羵ƥ��������.
		 * 
		 * \param vertexMap 
		 * \param edgeMask 
		 * \param edgeMatchedPairs 
		 */
		void SetCrossViewMatchingInput(const CrossViewMatchingInput& input);
		/**
		 * \brief Ѱ�ҿ羵ƥ���.
		 * 
		 * \param stream cuda��
		 */
		void FindCrossViewEdgeMatchedPairs(cudaStream_t stream = 0);

		/**
		 * \brief ����羵ƥ���.
		 *
		 * \return �羵ƥ���
		 */
		DeviceArrayView<CrossViewCorrPairs> GetCrossViewMatchingCorrPairs() { return validCorrPairs.ArrayView(); }

		/**
		 * \brief ����羵ƥ���.
		 * 
		 * \return �羵ƥ���
		 */
		DeviceArrayView<CrossViewCorrPairs> GetCrossViewMatchingUniqueCorrPairs() { return uniqueCrossMatchingPairs.ArrayView(); }

		/**
		 * \brief ����羵����ƥ����.
		 * 
		 * \return �羵����ƥ����
		 */
		DeviceArrayView<CrossViewCorrPairs> GetCrossViewBackTracingCorrPairs() { return uniqueCrossViewBackTracingPairs.ArrayView(); }
	};
}


