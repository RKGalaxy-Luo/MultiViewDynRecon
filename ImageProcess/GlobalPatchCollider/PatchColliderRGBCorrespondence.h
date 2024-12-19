/*****************************************************************//**
 * \file   PatchColliderRGBCorrespondence.cpp
 * \brief  根据RGB图像构建森林，找寻匹配点
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
		 * \brief 传入训练好的森林，并根据森林参数及结构，将Patch得到的feature分别下放到不同的子节点(末端节点)，如果两个Patch在同一个末端子节点，则判断其为匹配点.
		 * 
		 * \param 传入Patch计算得到的特征DCT的值(表示这个Patch的特征)
		 * \param 传入训练好的森林，通过森林中的树，将这个Patch分配到每个树不同标号的子节点(末端节点)
		 */
		template<int FeatureDim, int TreesNum> 
		__device__ __forceinline__ unsigned int SearchGPCForest(const GPCPatchFeature<FeatureDim>& feature, const typename PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice& forest);

		/**
		 * \brief 根据像素点坐标进行构建，将patch的中心像素编码，根据isPrevious的情况决定最高位的数字.
		 * 
		 * \param 像素点x坐标
		 * \param 像素点y坐标
		 * \param 是否是前一张图片，true：编码最高位是0，false：编码最高位是1
		 */
		__host__ __device__ __forceinline__ unsigned int EncodePixelPair(const int rgbX, const int rgbY, const bool isPrevious);

		/**
		 * \brief 将编码数据解码，并赋值标志位判断是否是Previous.
		 * 
		 * \param encode 需要解码的数据
		 * \param rgbX 解码获得的Patch中心像素x坐标
		 * \param rgbY 解码获得的Patch中心像素y坐标
		 * \param isPrevious 是Previous(true)还是Current(false)
		 */
		__host__ __device__ __forceinline__ void DecodePixelPair(const unsigned int encode, int& rgbX, int& rgbY, bool& isPrevious);

		/**
		 * \brief 根据RGB图像纹理，构建GPC碰撞键值对.
		 * 
		 * \param rgb_0 前一帧图片
		 * \param rgb_1 当前帧图片
		 * \param forest 传入GPU中存储的森林【文件流中加载出来的】
		 * \param stride 步长每一个Patch中心间隔多少Stride
		 * \param keyValueRows 键值对映射图的高
		 * \param keyValueCols 键值对映射图的宽
		 * \param keys 【输出】键：哈希编码叶子节点索引
		 * \param values 【输出】值：编码的像素对
		 */
		template<int PatchHalfSize, int TreesNum>
		__global__ void buildColliderKeyValueKernel(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, const typename PatchColliderForest<18, TreesNum>::GPCForestDevice forest, const int stride, const int keyValueRows, const int keyValueCols, unsigned int* keys, unsigned int* values);

		/**
		 * \brief 传入排列好的叶子的Key和Value，同时传入前景Mask，判断是否是前景部分，只匹配前景部分中的点，如果均满足，则CandidateIndicator中此处位置为1，否则为0.
		 * 
		 * \param SortedTreeLeafKey 排列好的叶子键Key
		 * \param SortedPixelValue 排列好的叶子值Value
		 * \param preForeground 上一帧前景纹理
		 * \param foreground 前景Mask的纹理
		 * \param CandidateIndicator 是否是满足要求的匹配点指示器，是则对应Index为1，否则为0
		 */
		__global__ void markCorrespondenceCandidateKernel(const PtrSize<const unsigned int> SortedTreeLeafKey, const unsigned int* SortedPixelValue, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, unsigned int* CandidateIndicator);

		/**
		 * \brief 手机有效的候选匹配点的核函数，存入PixelPairArray.
		 * 
		 * \param CandidateIndicator 候选点的指示器，为1的时候表示：这个是Current图片上的前景点，并且这个点在Previous中也找到了唯一一个匹配对象
		 * \param SortedPixelValue 排列好的Patch中心点坐标的编码
		 * \param PrefixSumIndicator 前序和，为了将同一个特征表示的匹配点并行写入对应的数组
		 * \param PixelPairArray 【输出】匹配点数组  -->  (x, y) Previous图像的匹配点，(z, w)Current图像的匹配点
		 */
		__global__ void collectCandidatePixelPairKernel(const PtrSize<const unsigned int> CandidateIndicator, const unsigned int* SortedPixelValue, const unsigned int* PrefixSumIndicator, ushort4* PixelPairArray, PtrStepSize<ushort4> CorrMap);
	}

	class PatchColliderRGBCorrespondence
	{
	public:
		/**
		 * \brief GPC算法参数.
		 */
		enum GPCParameters {
			PatchRadius					= 10,				// patch半径
			PatchClip					= PatchRadius,		// patch剪裁大小
			PatchStride					= 2,				// patch滑动步长
			FeatureDim					= 18,				// 特征维度
			TreesNum					= 5,				// 树的数量
			MaxSearchLevel				= 16,				// 最大搜索层数
			MaxCorrespondencePairsNum	= 40000				// 最大特征匹配对的数量
		};
	private:
		unsigned int colorRows = 0;							// 剪裁后RGB图像的高
		unsigned int colorCols = 0;							// 剪裁后RGB图像的宽
		unsigned int clipedImageSize = 0;					// 剪裁后图片大小
		cudaTextureObject_t rgbPrevious;					// 上一帧的RGB
		cudaTextureObject_t rgbCurrent;						// 当前帧的RGB
		cudaTextureObject_t foregroundCurrent;				// 当前帧的前景Mask
		cudaTextureObject_t currEdgeMask;					// 当前帧前景边缘Mask
		cudaTextureObject_t foregroundPrevious;				// 当前帧的前景Mask
		cudaTextureObject_t previousVertexMap;				// 上一帧的vertexMap
		cudaTextureObject_t currentVertexMap;				// 当前帧的vertexMap
		cudaTextureObject_t previousNormalMap;				// 上一帧的normalMap
		cudaTextureObject_t currentNormalMap;				// 当前帧的normalMap


		unsigned int CameraID;
		unsigned int FrameIndex;

		// 原始RGB图像上
		unsigned int KeyValueMapRows;						// 键值对映射图的高
		unsigned int KeyValueMapCols;						// 键值对映射图的宽

		// 森林对象
		PatchColliderForest<GPCParameters::FeatureDim, GPCParameters::TreesNum> Forest;

		// 对元素进行排序：key是哈希叶子索引，value是编码的像素对
		KeyValueSort<unsigned int, unsigned int> CollideSort;	
		DeviceArray<unsigned int> TreeLeafKey;				// 哈希叶子索引
		DeviceArray<unsigned int> PixelValue;				// 像素编码对

		// 有效的像素对指示器，如果是有效的，则在对应index下标注
		DeviceArray<unsigned int> CandidatePixelPairsIndicator;
		
		PrefixSum Prefixsum;								// 根据有效指示器，压缩匹配点对值(将无效值去除)的方法
		unsigned int* CandidatePairsNumPagelock;			// CPU中获得一共有多少候选的匹配点对，指针地址存的就是点对数量

		DeviceBufferArray<ushort4> CorrespondencePixels;	// 相关联的像素 (x, y) Previous图像的匹配点，(z, w)Current图像的匹配点

		DeviceArray2D<ushort4> sparseCorrPairsMap;		// 记录相关匹配点的Se3Map

	public:
		using Ptr = std::shared_ptr<PatchColliderRGBCorrespondence>;

		PatchColliderRGBCorrespondence() = default;
		~PatchColliderRGBCorrespondence() = default;

		/**
		 * \brief 分配GPC算法所需内存和显存，并加载GPC文件.
		 * 
		 * \param imageRows 图像的高
		 * \param imageCols 图像的宽
		 */
		void AllocateBuffer(unsigned int imageRows, unsigned int imageCols);

		/**
		 * \brief 释放GPC算法运行内存.
		 * 
		 */
		void ReleaseBuffer();

		/**
		 * \brief 设置图像输入.
		 * 
		 * \param color_0 前一帧的图像
		 * \param color_1 当前帧的图像
		 * \param preForeground 上一帧前景Mask
		 * \param foreground 当前帧的前景Mask
		 */
		void SetInputImage(cudaTextureObject_t color_0, cudaTextureObject_t color_1, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t edgeMask, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, const unsigned int cameraID, const unsigned int frameIdx);

		/**
		 * \brief 寻找匹配的像素点.
		 * 
		 * \param stream CUDA流ID
		 */
		void FindCorrespondence(cudaStream_t stream);

		/**
		 * \brief 相匹配的像素对.
		 * 
		 * \return 像素对
		 */
		DeviceArray<ushort4> CorrespondedPixelPairs() const;

		/**
		 * \brief 获得边缘相关匹配点.
		 * 
		 * \return 边缘相关匹配点Map.
		 */
		DeviceArrayView2D<ushort4> getCorrespondencePairsMap() { return DeviceArrayView2D<ushort4>(sparseCorrPairsMap); }
	};

}

