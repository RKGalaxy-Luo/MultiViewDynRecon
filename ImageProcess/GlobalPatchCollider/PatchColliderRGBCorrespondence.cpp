/*****************************************************************//**
 * \file   PatchColliderRGBCorrespondence.cpp
 * \brief  ����RGBͼ�񹹽�ɭ�֣���Ѱƥ���
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
	Forest.UpdateSearchLevel(GPCParameters::MaxSearchLevel);	// ֻ������16�㣿

	colorRows = imageRows;
	colorCols = imageCols;
	clipedImageSize = imageRows * imageCols;
	KeyValueMapRows = (colorRows - PatchClip * 2) / PatchStride;
	KeyValueMapCols = (colorCols - PatchClip * 2) / PatchStride;
	const unsigned int KeyValueMapSize = KeyValueMapRows * KeyValueMapCols;

	// rgbPrevious��rgbCurrent���м�ֵ��
	TreeLeafKey.create(2 * KeyValueMapSize);
	PixelValue.create(2 * KeyValueMapSize);
	CollideSort.AllocateBuffer(2 * KeyValueMapSize);
	// �����Ч��ƥ����
	CandidatePixelPairsIndicator.create(2 * KeyValueMapSize);
	// ����Prefixsum��ѹ����Ч���ݵĻ�����
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
