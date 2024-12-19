/*****************************************************************//**
 * \file   PatchColliderRGBCorrespondence.cpp
 * \brief  ����RGBͼ�񹹽�ɭ�֣���Ѱƥ���
 *
 * \author LUO
 * \date   March 2024
 *********************************************************************/
#pragma once
#include <base/Constants.h>
#include <base/CommonTypes.h>
#include <base/FileOperation/Stream.h>
#include <base/FileOperation/Serializer.h>
#include <base/FileOperation/BinaryFileStream.h>
#include "base/data_transfer.h"

#include <core/AlgorithmTypes.h>

#include "PatchColliderForest.h"
#include "PatchColliderBuildFeature.h"
namespace SparseSurfelFusion {
	namespace device {

		enum {
			CorrMapHalfWindow = 4
		};

		/**
		 * \brief ����ѵ���õ�ɭ�֣�������ɭ�ֲ������ṹ����Patch�õ���feature�ֱ��·ŵ���ͬ���ӽڵ�(ĩ�˽ڵ�)���������Patch��ͬһ��ĩ���ӽڵ㣬���ж���Ϊƥ���.
		 * 
		 * \param ����Patch����õ�������DCT��ֵ(��ʾ���Patch������)
		 * \param ����ѵ���õ�ɭ�֣�ͨ��ɭ���е����������Patch���䵽ÿ������ͬ��ŵ��ӽڵ�(ĩ�˽ڵ�)
		 */
		template<int FeatureDim, int TreesNum> 
		__device__ __forceinline__ unsigned int SearchGPCForest(const GPCPatchFeature<FeatureDim>& feature, const typename PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice& forest);

		/**
		 * \brief �������ص�������й�������patch���������ر��룬����isPrevious������������λ������.
		 * 
		 * \param ���ص�x����
		 * \param ���ص�y����
		 * \param �Ƿ���ǰһ��ͼƬ��true���������λ��0��false���������λ��1
		 */
		__host__ __device__ __forceinline__ unsigned int EncodePixelPair(const int rgbX, const int rgbY, const bool isPrevious);

		/**
		 * \brief ���������ݽ��룬����ֵ��־λ�ж��Ƿ���Previous.
		 * 
		 * \param encode ��Ҫ���������
		 * \param rgbX �����õ�Patch��������x����
		 * \param rgbY �����õ�Patch��������y����
		 * \param isPrevious ��Previous(true)����Current(false)
		 */
		__host__ __device__ __forceinline__ void DecodePixelPair(const unsigned int encode, int& rgbX, int& rgbY, bool& isPrevious);

		/**
		 * \brief ����RGBͼ����������GPC��ײ��ֵ��.
		 * 
		 * \param rgb_0 ǰһ֡ͼƬ
		 * \param rgb_1 ��ǰ֡ͼƬ
		 * \param forest ����GPU�д洢��ɭ�֡��ļ����м��س����ġ�
		 * \param stride ����ÿһ��Patch���ļ������Stride
		 * \param keyValueRows ��ֵ��ӳ��ͼ�ĸ�
		 * \param keyValueCols ��ֵ��ӳ��ͼ�Ŀ�
		 * \param keys �������������ϣ����Ҷ�ӽڵ�����
		 * \param values �������ֵ����������ض�
		 */
		template<int PatchHalfSize, int TreesNum>
		__global__ void buildColliderKeyValueKernel(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, const typename PatchColliderForest<18, TreesNum>::GPCForestDevice forest, const int stride, const int keyValueRows, const int keyValueCols, unsigned int* keys, unsigned int* values);

		/**
		 * \brief �������кõ�Ҷ�ӵ�Key��Value��ͬʱ����ǰ��Mask���ж��Ƿ���ǰ�����֣�ֻƥ��ǰ�������еĵ㣬��������㣬��CandidateIndicator�д˴�λ��Ϊ1������Ϊ0.
		 * 
		 * \param SortedTreeLeafKey ���кõ�Ҷ�Ӽ�Key
		 * \param SortedPixelValue ���кõ�Ҷ��ֵValue
		 * \param preForeground ��һ֡ǰ������
		 * \param foreground ǰ��Mask������
		 * \param CandidateIndicator �Ƿ�������Ҫ���ƥ���ָʾ���������ӦIndexΪ1������Ϊ0
		 */
		__global__ void markCorrespondenceCandidateKernel(const PtrSize<const unsigned int> SortedTreeLeafKey, const unsigned int* SortedPixelValue, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, unsigned int* CandidateIndicator);

		/**
		 * \brief �ֻ���Ч�ĺ�ѡƥ���ĺ˺���������PixelPairArray.
		 * 
		 * \param CandidateIndicator ��ѡ���ָʾ����Ϊ1��ʱ���ʾ�������CurrentͼƬ�ϵ�ǰ���㣬�����������Previous��Ҳ�ҵ���Ψһһ��ƥ�����
		 * \param SortedPixelValue ���кõ�Patch���ĵ�����ı���
		 * \param PrefixSumIndicator ǰ��ͣ�Ϊ�˽�ͬһ��������ʾ��ƥ��㲢��д���Ӧ������
		 * \param PixelPairArray �������ƥ�������  -->  (x, y) Previousͼ���ƥ��㣬(z, w)Currentͼ���ƥ���
		 */
		__global__ void collectCandidatePixelPairKernel(const PtrSize<const unsigned int> CandidateIndicator, const unsigned int* SortedPixelValue, const unsigned int* PrefixSumIndicator, ushort4* PixelPairArray, PtrStepSize<ushort4> CorrMap);
	}

	class PatchColliderRGBCorrespondence
	{
	public:
		/**
		 * \brief GPC�㷨����.
		 */
		enum GPCParameters {
			PatchRadius					= 10,				// patch�뾶
			PatchClip					= PatchRadius,		// patch���ô�С
			PatchStride					= 2,				// patch��������
			FeatureDim					= 18,				// ����ά��
			TreesNum					= 5,				// ��������
			MaxSearchLevel				= 16,				// �����������
			MaxCorrespondencePairsNum	= 40000				// �������ƥ��Ե�����
		};
	private:
		unsigned int colorRows = 0;							// ���ú�RGBͼ��ĸ�
		unsigned int colorCols = 0;							// ���ú�RGBͼ��Ŀ�
		unsigned int clipedImageSize = 0;					// ���ú�ͼƬ��С
		cudaTextureObject_t rgbPrevious;					// ��һ֡��RGB
		cudaTextureObject_t rgbCurrent;						// ��ǰ֡��RGB
		cudaTextureObject_t foregroundCurrent;				// ��ǰ֡��ǰ��Mask
		cudaTextureObject_t currEdgeMask;					// ��ǰ֡ǰ����ԵMask
		cudaTextureObject_t foregroundPrevious;				// ��ǰ֡��ǰ��Mask
		cudaTextureObject_t previousVertexMap;				// ��һ֡��vertexMap
		cudaTextureObject_t currentVertexMap;				// ��ǰ֡��vertexMap
		cudaTextureObject_t previousNormalMap;				// ��һ֡��normalMap
		cudaTextureObject_t currentNormalMap;				// ��ǰ֡��normalMap


		unsigned int CameraID;
		unsigned int FrameIndex;

		// ԭʼRGBͼ����
		unsigned int KeyValueMapRows;						// ��ֵ��ӳ��ͼ�ĸ�
		unsigned int KeyValueMapCols;						// ��ֵ��ӳ��ͼ�Ŀ�

		// ɭ�ֶ���
		PatchColliderForest<GPCParameters::FeatureDim, GPCParameters::TreesNum> Forest;

		// ��Ԫ�ؽ�������key�ǹ�ϣҶ��������value�Ǳ�������ض�
		KeyValueSort<unsigned int, unsigned int> CollideSort;	
		DeviceArray<unsigned int> TreeLeafKey;				// ��ϣҶ������
		DeviceArray<unsigned int> PixelValue;				// ���ر����

		// ��Ч�����ض�ָʾ�����������Ч�ģ����ڶ�Ӧindex�±�ע
		DeviceArray<unsigned int> CandidatePixelPairsIndicator;
		
		PrefixSum Prefixsum;								// ������Чָʾ����ѹ��ƥ����ֵ(����Чֵȥ��)�ķ���
		unsigned int* CandidatePairsNumPagelock;			// CPU�л��һ���ж��ٺ�ѡ��ƥ���ԣ�ָ���ַ��ľ��ǵ������

		DeviceBufferArray<ushort4> CorrespondencePixels;	// ����������� (x, y) Previousͼ���ƥ��㣬(z, w)Currentͼ���ƥ���

		DeviceArray2D<ushort4> sparseCorrPairsMap;		// ��¼���ƥ����Se3Map

	public:
		using Ptr = std::shared_ptr<PatchColliderRGBCorrespondence>;

		PatchColliderRGBCorrespondence() = default;
		~PatchColliderRGBCorrespondence() = default;

		/**
		 * \brief ����GPC�㷨�����ڴ���Դ棬������GPC�ļ�.
		 * 
		 * \param imageRows ͼ��ĸ�
		 * \param imageCols ͼ��Ŀ�
		 */
		void AllocateBuffer(unsigned int imageRows, unsigned int imageCols);

		/**
		 * \brief �ͷ�GPC�㷨�����ڴ�.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief ����ͼ������.
		 * 
		 * \param color_0 ǰһ֡��ͼ��
		 * \param color_1 ��ǰ֡��ͼ��
		 * \param preForeground ��һ֡ǰ��Mask
		 * \param foreground ��ǰ֡��ǰ��Mask
		 */
		void SetInputImage(cudaTextureObject_t color_0, cudaTextureObject_t color_1, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t edgeMask, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, const unsigned int cameraID, const unsigned int frameIdx);

		/**
		 * \brief Ѱ��ƥ������ص�.
		 * 
		 * \param stream CUDA��ID
		 */
		void FindCorrespondence(cudaStream_t stream);

		/**
		 * \brief ��ƥ������ض�.
		 * 
		 * \return ���ض�
		 */
		DeviceArray<ushort4> CorrespondedPixelPairs() const;

		/**
		 * \brief ��ñ�Ե���ƥ���.
		 * 
		 * \return ��Ե���ƥ���Map.
		 */
		DeviceArrayView2D<ushort4> getCorrespondencePairsMap() { return DeviceArrayView2D<ushort4>(sparseCorrPairsMap); }
	};

}

