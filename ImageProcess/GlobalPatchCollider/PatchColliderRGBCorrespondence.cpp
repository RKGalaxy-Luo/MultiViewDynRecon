/*****************************************************************//**
 * \file   PatchColliderRGBCorrespondence.cpp
 * \brief  根据RGB图像构建森林，找寻匹配点
 * 
 * \author LUO
 * \date   March 2024
 *********************************************************************/
#include "PatchColliderRGBCorrespondence.h"

void SparseSurfelFusion::PatchColliderRGBCorrespondence::AllocateBuffer(unsigned int imageRows, unsigned int imageCols)
{
	std::string GPCPathStr = std::string(GPC_MODEL_PATH);
	BinaryFileStream InFileStream(GPCPathStr.c_str(), BinaryFileStream::FileOperationMode::ReadOnly);
	Forest.Load(&InFileStream);
	Forest.UploadToDevice();
	Forest.UpdateSearchLevel(GPCParameters::MaxSearchLevel);	// 只搜索到16层？

	colorRows = imageRows;
	colorCols = imageCols;
	clipedImageSize = imageRows * imageCols;
	KeyValueMapRows = (colorRows - PatchClip * 2) / PatchStride;
	KeyValueMapCols = (colorCols - PatchClip * 2) / PatchStride;
	const unsigned int KeyValueMapSize = KeyValueMapRows * KeyValueMapCols;

	// rgbPrevious和rgbCurrent都有键值对
	TreeLeafKey.create(2 * KeyValueMapSize);
	PixelValue.create(2 * KeyValueMapSize);
	CollideSort.AllocateBuffer(2 * KeyValueMapSize);
	// 标记有效的匹配点对
	CandidatePixelPairsIndicator.create(2 * KeyValueMapSize);
	// 用于Prefixsum和压缩有效数据的缓冲区
	Prefixsum.AllocateBuffer(CandidatePixelPairsIndicator.size());
	CHECKCUDA(cudaMallocHost((void**)(&CandidatePairsNumPagelock), sizeof(unsigned int)));

	CorrespondencePixels.AllocateBuffer(MaxCorrespondencePairsNum);
	sparseCorrPairsMap.create(colorRows, colorCols);
}

void SparseSurfelFusion::PatchColliderRGBCorrespondence::ReleaseBuffer()
{
	TreeLeafKey.release();
	PixelValue.release();
	CandidatePixelPairsIndicator.release();
	sparseCorrPairsMap.release();
	CHECKCUDA(cudaFreeHost(CandidatePairsNumPagelock));
}

void SparseSurfelFusion::PatchColliderRGBCorrespondence::SetInputImage(cudaTextureObject_t color_0, cudaTextureObject_t color_1, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t edgeMask, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, const unsigned int cameraID, const unsigned int frameIdx)
{
	rgbPrevious = color_0;
	rgbCurrent = color_1;
	foregroundPrevious = preForeground;
	foregroundCurrent = foreground;
	currEdgeMask = edgeMask;

	previousVertexMap = preVertex;
	currentVertexMap = currVertex;

	previousNormalMap = preNormal;
	currentNormalMap = currNormal;

	CameraID = cameraID;
	FrameIndex = frameIdx;
}



SparseSurfelFusion::DeviceArray<ushort4> SparseSurfelFusion::PatchColliderRGBCorrespondence::CorrespondedPixelPairs() const
{
	return CorrespondencePixels.Array();
}
