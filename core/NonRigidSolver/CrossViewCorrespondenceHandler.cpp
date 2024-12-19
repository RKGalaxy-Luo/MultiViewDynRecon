#include "CrossViewCorrespondenceHandler.h"

void SparseSurfelFusion::CrossViewCorrespondenceHandler::AllocateBuffer()
{
	validCrossCorrIndicator.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	validCorrPrefixSum.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	GeometryCrossCorrPairs.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);
	ObservedCrossPairs.AllocateBuffer(Constants::MaxCrossViewMatchedPairs);

	validTargetVertex.AllocateBuffer(Constants::MaxCrossViewMatchedPairs * 2);
	validCanonicalVertex.AllocateBuffer(Constants::MaxCrossViewMatchedPairs * 2);
	validVertexKnn.AllocateBuffer(Constants::MaxCrossViewMatchedPairs * 2);
	validKnnWeight.AllocateBuffer(Constants::MaxCrossViewMatchedPairs * 2);

	validWarpedVertex.AllocateBuffer(Constants::MaxCrossViewMatchedPairs * 2);
}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::ReleaseBuffer()
{
	validCrossCorrIndicator.ReleaseBuffer();
	GeometryCrossCorrPairs.ReleaseBuffer();
	ObservedCrossPairs.ReleaseBuffer();

	validTargetVertex.ReleaseBuffer();
	validCanonicalVertex.ReleaseBuffer();
	validVertexKnn.ReleaseBuffer();
	validKnnWeight.ReleaseBuffer();

	validWarpedVertex.ReleaseBuffer();
}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::SetInputs(DeviceArrayView<DualQuaternion> nodeSe3, DeviceArray2D<KNNAndWeight>* knnMap, cudaTextureObject_t* vertexMap, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, DeviceArrayView2D<ushort4>* corrMap, Renderer::SolverMaps* solverMaps, mat34* world2camera, mat34* InitialCameraSE3)
{
	NodeSe3 = nodeSe3;
	observedCrossViewCorrInterface.crossCorrPairs = crossCorrPairs;
	for (int i = 0; i < devicesCount; i++) {
		observedCrossViewCorrInterface.depthVertexMap[i] = vertexMap[i];
		observedCrossViewCorrInterface.corrMap[i] = corrMap[i];

		geometryCrossViewCorrInterface.canonicalVertexMap[i] = solverMaps[i].reference_vertex_map;
		geometryCrossViewCorrInterface.liveVertexMap[i] = solverMaps[i].warp_vertex_map;
		geometryCrossViewCorrInterface.indexMap[i] = solverMaps[i].index_map;
		geometryCrossViewCorrInterface.knnMap[i] = knnMap[i];
		geometryCrossViewCorrInterface.camera2World[i] = world2camera[i].inverse();
		geometryCrossViewCorrInterface.initialCameraSE3[i] = InitialCameraSE3[i];
	}
}

void SparseSurfelFusion::CrossViewCorrespondenceHandler::UpdateNodeSE3(DeviceArrayView<DualQuaternion> nodeSe3)
{
	FUNCTION_CHECK(NodeSe3.Size() == nodeSe3.Size());
	NodeSe3 = nodeSe3;
}


