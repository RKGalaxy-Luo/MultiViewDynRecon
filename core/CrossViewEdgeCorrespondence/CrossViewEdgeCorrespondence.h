/*****************************************************************//**
 * \file   CrossViewEdgeCorrespondence.h
 * \brief  跨镜边缘匹配
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
			cudaTextureObject_t EdgeMask[MAX_CAMERA_COUNT];					// 前景边缘Mask
			cudaTextureObject_t RawClipedMask[MAX_CAMERA_COUNT];			// 原始前景Mask(裁剪但未形态学处理)
			DeviceArrayView2D<ushort4> CorrPairsMap[MAX_CAMERA_COUNT];		// 匹配点对Map(没找到的像素点为OXFFFF)
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];						// 视角位姿
			mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];					// 视角位姿
			Intrinsic intrinsic[MAX_CAMERA_COUNT];							// 相机内参
		};

		/**
		 * \brief 对不同视角匹配点坐标编码.
		 */
		__host__ __device__ __forceinline__ unsigned int EncodeCrossViewPixelPair(unsigned int x, unsigned int y, unsigned int view);

		/**
		 * \brief 在边缘处找最近的跨镜匹配点对.
		 */
		__device__ __forceinline__ bool GetClosetCrossViewPairs(CrossViewCorrInput& input, const unsigned int& view, const unsigned int& nextView, const unsigned int& lastView, const unsigned int& x, const unsigned int& y, const unsigned int& clipedCols, const unsigned int& clipedRows, const float& pairsSquDisThreshold, CrossViewCorrPairs& crossPairs);
		 
		/**
		 * \brief 在有效的Pair附近寻找有效的光流.
		 */
		__device__ __forceinline__ bool GetTrackedCrossViewPairs(CrossViewCorrInput& input, const CrossViewCorrPairs& closestPairs, CrossViewCorrPairs& trackedPairs);

		/**
		 * \brief 获得有效的跨镜匹配点.
		 */
		__global__ void MarkValidPairPixels(CrossViewCorrInput input, const unsigned int clipedCols, const unsigned int clipedRows, const unsigned int clipedImageSize, const unsigned int cameraNum, const float pairsSquDisThreshold, CrossViewCorrPairs* corrPairs, unsigned char* markValidEdgePixel);

		/**
		 * \brief 编码跨镜视角的像素坐标.
		 */
		__global__ void EncodeDiffViewPixelCoor(DeviceArrayView<CrossViewCorrPairs> validCorrPairs, const unsigned int PairsNum, unsigned int* diffViewCoorKey);

		/**
		 * \brief 编码跨镜视角的像素坐标.
		 */
		__global__ void EncodeSameViewPixelCoor(DeviceArrayView<CrossViewCorrPairs> validCorrPairs, const unsigned int PairsNum, unsigned int* sameViewCoorKey);

		/**
		 * \brief 标记不同的Key起始位置.
		 */
		__global__ void LabelSortedPixelCoor(PtrSize<unsigned int> sortedKey, const unsigned int PairsNum, unsigned int* corrPairsLabel);
	
		/**
		 * \brief 获得共享同一个匹配点的Pairs的Offset.
		 */
		__global__ void ComputePairsArrayOffset(const unsigned int* pairsLabel, const unsigned int* prefixSumLabel, const unsigned int PairsNum, const unsigned int uniquePairsNum, unsigned int* offset);

		/**
		 * \brief 筛出pair两个点对应最近的那两个Pair作为最终的一对一Pairs.
		 */
		__global__ void SelectUniqueCrossViewMatchingPairs(DeviceArrayView<CrossViewCorrPairs> sortedPairs, CrossViewCorrInput input, const unsigned int* offset, const unsigned int uniquePairsNum, CrossViewCorrPairs* uniquePairs);

		/**
		 * \brief 筛选出能被匹配点追溯的点对.
		 */
		__global__ void SelectCrossViewBackTracingPairs(DeviceArrayView<CrossViewCorrPairs> uniquePairs, CrossViewCorrInput input, const unsigned int uniquePairsNum, unsigned char* markBackTracingPairs);
	}

	// 处理在边缘跨视角的匹配
	class CrossViewEdgeCorrespondence
	{
	public:
		using Ptr = std::shared_ptr<CrossViewEdgeCorrespondence>;

		CrossViewEdgeCorrespondence(Intrinsic* intrinsicArray, mat34* cameraPoseArray);

		~CrossViewEdgeCorrespondence();

		// 跨镜匹配的输入Package
		struct CrossViewMatchingInput {
			cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];				// vertexMap数组
			cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];				// normalMap数组
			cudaTextureObject_t colorMap[MAX_CAMERA_COUNT];					// colorMap数组
			cudaTextureObject_t edgeMask[MAX_CAMERA_COUNT];					// 边缘Mask数组
			cudaTextureObject_t rawClipedMask[MAX_CAMERA_COUNT];			// 剪裁后的Mask
			DeviceArrayView2D<ushort4> matchedPairsMap[MAX_CAMERA_COUNT];	// Mask上的匹配点
		};



	private:
		const unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int ImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
		const unsigned int ImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		const unsigned int rawImageSize = FRAME_HEIGHT * FRAME_WIDTH;
		const unsigned int clipedImageSize = ImageRowsCliped * ImageColsCliped;

		device::CrossViewCorrInput crossCorrInput;								// 打包用于计算边缘跨镜的匹配点

		DeviceBufferArray<CrossViewCorrPairs> corrPairs;						// 这个是记录跨境匹配的点的来源: (x, y)是当前视角匹配pixel, (z, w)是跨镜搜寻的视角匹配pixel
		DeviceBufferArray<unsigned char> markValidPairs;						// 标记有效的边缘Pairs
		DeviceBufferArray<CrossViewCorrPairs> validCorrPairs;					// 筛选得到合适的边缘Pairs(存在一对多)

		DeviceBufferArray<unsigned int> viewCoorKey;							// 这个记录不同View的坐标编码(编码z, w)
		KeyValueSort<unsigned int, CrossViewCorrPairs> corrPairsSort;			// 根据编码升序排Pairs
		DeviceBufferArray<unsigned int> corrPairsLabel;							// 标记多对一的点
		PrefixSum prefixSum;
		DeviceBufferArray<unsigned int> sortedPairsOffset;						// 共享同一个匹配点的Pairs的Offset	
		DeviceBufferArray<CrossViewCorrPairs> uniqueCrossMatchingPairs;			// 一对一跨镜匹配点

		DeviceBufferArray<CrossViewCorrPairs> uniqueCrossViewBackTracingPairs;	// 这个是跨镜匹配点均能通过与上一帧匹配点回溯的点对
		DeviceBufferArray<unsigned char> markValidBackTracingPairs;				// 标记有效的边缘Pairs

		unsigned int validCrossViewPairsNum = 0;								// 记录有效的跨镜匹配点
		unsigned int UniqueMatchingNum = 0;										// 一对一跨镜匹配数量
		unsigned int UniqueBackTracingPairsNum = 0;								// 能被匹配点追溯的匹配点对
		/**
		 * \brief 分配内存.
		 * 
		 */
		void AllocateBuffer();

		/**
		 * \brief 释放内存.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief 选择有效的跨镜匹配点.
		 * 
		 * \param stream cuda流
		 */
		void SelectValidCrossViewPairs(cudaStream_t stream = 0);

		/**
		 * \brief 删除View.x下重复的匹配.
		 *
		 * \param stream cuda流
		 */
		void FilterSameViewRepetitiveMatching(cudaStream_t stream = 0);

		/**
		 * \brief 删除View.y下重复的匹配.
		 * 
		 * \param stream cuda流
		 */
		void FilterDiffViewRepetitiveMatching(cudaStream_t stream = 0);

		/**
		 * \brief 删除多对一匹配.
		 * 
		 * \param stream cuda流
		 */
		void FilterSeveralForOneMatchingPairs(cudaStream_t stream = 0);

		/**
		 * \brief 筛选出能被光流回溯的点.
		 * 
		 * \param stream
		 */
		void SelectCrossViewBackTracingPairs(cudaStream_t stream = 0);

	public:
		/**
		 * \brief 寻找跨镜匹配点的输入.
		 * 
		 * \param vertexMap 
		 * \param edgeMask 
		 * \param edgeMatchedPairs 
		 */
		void SetCrossViewMatchingInput(const CrossViewMatchingInput& input);
		/**
		 * \brief 寻找跨镜匹配点.
		 * 
		 * \param stream cuda流
		 */
		void FindCrossViewEdgeMatchedPairs(cudaStream_t stream = 0);

		/**
		 * \brief 输出跨镜匹配点.
		 *
		 * \return 跨镜匹配点
		 */
		DeviceArrayView<CrossViewCorrPairs> GetCrossViewMatchingCorrPairs() { return validCorrPairs.ArrayView(); }

		/**
		 * \brief 输出跨镜匹配点.
		 * 
		 * \return 跨镜匹配点
		 */
		DeviceArrayView<CrossViewCorrPairs> GetCrossViewMatchingUniqueCorrPairs() { return uniqueCrossMatchingPairs.ArrayView(); }

		/**
		 * \brief 输出跨镜回溯匹配点对.
		 * 
		 * \return 跨镜回溯匹配点对
		 */
		DeviceArrayView<CrossViewCorrPairs> GetCrossViewBackTracingCorrPairs() { return uniqueCrossViewBackTracingPairs.ArrayView(); }
	};
}


