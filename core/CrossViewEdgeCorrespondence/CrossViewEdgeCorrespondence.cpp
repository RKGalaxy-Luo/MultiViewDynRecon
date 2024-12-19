#include "CrossViewEdgeCorrespondence.h"

SparseSurfelFusion::CrossViewEdgeCorrespondence::CrossViewEdgeCorrespondence(Intrinsic* intrinsicArray, mat34* cameraPoseArray)
{
	for (int i = 0; i < devicesCount; i++) {
		crossCorrInput.intrinsic[i] = intrinsicArray[i];
		crossCorrInput.InitialCameraSE3[i] = cameraPoseArray[i];
		crossCorrInput.InitialCameraSE3Inv[i] = cameraPoseArray[i].inverse();
	}
	AllocateBuffer();
}

SparseSurfelFusion::CrossViewEdgeCorrespondence::~CrossViewEdgeCorrespondence()
{
	ReleaseBuffer();
}



void SparseSurfelFusion::CrossViewEdgeCorrespondence::ReleaseBuffer()
{
	corrPairs.ReleaseBuffer();
	markValidPairs.ReleaseBuffer();
	validCorrPairs.ReleaseBuffer();
	corrPairsLabel.ReleaseBuffer();
	viewCoorKey.ReleaseBuffer();
	sortedPairsOffset.ReleaseBuffer();
	uniqueCrossMatchingPairs.ReleaseBuffer();
	uniqueCrossViewBackTracingPairs.ReleaseBuffer();
	markValidBackTracingPairs.ReleaseBuffer();
}

void SparseSurfelFusion::CrossViewEdgeCorrespondence::SetCrossViewMatchingInput(const CrossViewMatchingInput& input)
{
	for (int i = 0; i < devicesCount; i++) {
		crossCorrInput.VertexMap[i] = input.vertexMap[i];
		crossCorrInput.EdgeMask[i] = input.edgeMask[i];
		crossCorrInput.CorrPairsMap[i] = input.matchedPairsMap[i];
		crossCorrInput.RawClipedMask[i] = input.rawClipedMask[i];
	}
}
