/*****************************************************************//**
 * \file   FrameProcessor.cpp
 * \brief  ��Ҫ����Ϊ�����㷨��ȡͼ���Լ��������
 * 
 * \author LUO
 * \date   January 26th 2024
 *********************************************************************/
#include "FrameProcessor.h"

SparseSurfelFusion::FrameProcessor::FrameProcessor(std::shared_ptr<ThreadPool> threadPool) : pool(threadPool)
{
	numThreadsCompleted.store(0);	// ����ֵ
	frame = std::make_shared<GetFrameFromCamera>(pool);		// ��ȡͼ��

	deviceCount = frame->getCameraCount();						// ����������

	configParser = std::make_shared<ConfigParser>(frame);		// ��������ֵ��configParser
	surfelsProcessor = std::make_shared<SurfelsProcessor>();	// ������Ԫ������
	for (int i = 0; i < deviceCount; i++) {						// ���ÿ��������ڲ�
		rawColorIntrinsic[i] = frame->getColorIntrinsic(i);
		rawDepthIntrinsic[i] = frame->getDepthIntrinsic(i);
		clipColorIntrinsic[i] = frame->getClipColorIntrinsic(i);
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
		InitialCameraSE3Inv[i] = InitialCameraSE3[i].inverse();
	}
	//mergeSurface = std::make_shared<MergeSurface>(clipColorIntrinsic);

	allocateImageHostBuffer();				// ����ͼ����Host�ϵ���ҳ�ڴ�
	allocateDepthTexture();					// ����GPU�ϵ���������ڴ�
	allocateColorTexture();					// ����GPU�ϵ�RGB�����ڴ�
	allocateSurfelAttributeTexture();		// ����GPU����Ԫ������ص������ڴ�
	allocateColorTimeTexture();				// ����GPU���й���Ԫ��ɫ�Լ���һ�ο�����Ԫ��ʱ��(�ڵڼ�֡)
	allocateValidSurfelSelectionBuffer();	// ������Ч��Ԫ��ѡ��������
	allocateForegroundSegmentationBuffer();	// ����ǰ���ָ��������ڴ�
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	allocateOpticalFlowBuffer();			// �������������ڴ�
#else
	allocateFeatureCorrespondenceBuffer();	// ����Ѱ��ϡ����������ڴ�
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	allocateCrossViewMatchingBuffer();		// ����羵ƥ��㻺��
	allocateGradientMapBuffer();			// �����ݶ�ͼ�����ڴ�
	initProcessorStream();					// ��ʼ����

		//����ӳ���ںϺ�������Ԫ���ڴ�
	for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
		createFloat4TextureSurface(CLIP_HEIGHT, CLIP_WIDTH, icp_vertex_confidence[i]);
		createFloat4TextureSurface(CLIP_HEIGHT, CLIP_WIDTH, icp_normal_radius[i]);
		createFloat4TextureSurface(CLIP_HEIGHT, CLIP_WIDTH, icp_color_time[i]);
	}


}

SparseSurfelFusion::FrameProcessor::~FrameProcessor()
{
	releaseImageHostBuffer();				// �ͷ�ͼ����Host�ϵ���ҳ�ڴ�
	releaseDepthTexture();					// �ͷ�GPU�ϵ���������ڴ�
	releaseColorTexture();					// �ͷ�GPU�ϵ�RGB�����ڴ�
	releaseSurfelAttributeTexture();		// �ͷ�GPU����Ԫ������ص������ڴ�
	releaseColorTimeTexture();				// �ͷ�GPU���й�ColorTime��ص�����
	releaseValidSurfelSelectionBuffer();	// �ͷ���Ч��Ԫ��ѡ��������
	releaseForegroundSegmentationBuffer();	// �ͷ�ǰ���ָ��������ڴ�
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	releaseOpticalFlowBuffer();				// �ͷż���������ڴ�
#else
	releaseFeatureCorrespondenceBuffer();	// �ͷ�Ѱ��ϡ����������ڴ�
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	releaseCrossViewMatchingBuffer();		// �ͷſ羵ƥ����ڴ�
	releaseGradientMapBuffer();				// �ͷ��ݶ�ͼ�����ڴ�

	releaseProcessorStream();				// �ͷ���

		//�ͷ�texture��Դ����ֹ�ڴ�й¶
	for (int i = 0; i < MAX_CAMERA_COUNT; i++)
	{
		releaseTextureCollect(icp_vertex_confidence[i]);
		releaseTextureCollect(icp_normal_radius[i]);
		releaseTextureCollect(icp_color_time[i]);
	}

}

bool SparseSurfelFusion::FrameProcessor::CamerasReady()
{
	return frame->CamerasReady();
}

void SparseSurfelFusion::FrameProcessor::showAllImages()
{
	// ��ȡ����ʼ���е�ʱ���
	for (int i = 0; i < deviceCount; i++) {
		cv::Mat currentColor = frame->GetColorImage(i);
		cv::Mat currentDepth = frame->GetDepthImage(i);
		cv::imshow(frame->GetColorWindowName(i), currentColor);
		cv::imshow(frame->GetDepthWindowName(i), currentDepth);
	}
}

cv::Mat SparseSurfelFusion::FrameProcessor::getCurrentColorImage(int CameraID)
{
	return frame->GetColorImage(CameraID);
}

cv::Mat SparseSurfelFusion::FrameProcessor::getCurrentDepthImage(int CameraID)
{
	return frame->GetDepthImage(CameraID);
}

void SparseSurfelFusion::FrameProcessor::ProcessCurrentCameraImageTask(const unsigned int i, CameraObservation& observation)
{

	int streamIdx = i * A_IMAGEPROCESSOR_NEED_CUDA_STREAM;	// ��������Index
	FetchFrame(FrameIndex, i);
/**************************** 2��Camera ƽ������2ms - 5ms (ÿ�����ʹ��һ��������color��depth��ʹ��2���������ٶ�һ��)****************************/
	UploadRawColorImageToGPU(ProcessorStream[streamIdx], i);						// ��ȡԭʼ��RGB��Ϣ
	SegmentForeground(ProcessorStream[streamIdx + 1], i);							// �ָ�ǰ�����ڲ�����ͬ�������·�ͬ�������ָ���ͼ�����н��С�
	UploadDepthImageToGPU(ProcessorStream[streamIdx], i);							// ��ȡ�����Ϣ
	ClipFilterDepthImage(ProcessorStream[streamIdx], i);							// ���ò������ͼ����˫���˲�
	ClipNormalizeColorImage(ProcessorStream[streamIdx], i);							// ����RGBͼ�񣬻�ûҶ�ͼ�����ԻҶ�ͼ����
/**************************** 2��Camera ƽ������40us - 60us (ÿ�����ʹ��һ�����������ɶ��㼰���Ŷ�Map�����߼��뾶Map)****************************/
	//����֡ʱ��ֱ�Ӱ�ԭʼӳ��д��icp_��
	buildVertexConfidenceMap(ProcessorStream[streamIdx], i);						// �������㼰���Ŷ�Map
	//��ЩҲҪ�ĳ���icp_vertex
	buildNormalRadiusMap(ProcessorStream[streamIdx], i);							// �������߼��뾶Map
	//����Ҳ��
	buildColorTimeMap(FrameIndex, ProcessorStream[streamIdx], i);					// ����Color-Timeͼ
	SyncAndConvertForeground2Texture(ProcessorStream[streamIdx + 1], i);			// ͬ���ָ����������ָ���ת��Texture�洢���������ʹ�÷ָ���ĵط���
/**************************** FindCorrespondenceѰ��ƥ��㿪�������ʱ40ms��ռ�ܴ���ʱ����80% ****************************/
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	computeOpticalFlow(ProcessorStream[streamIdx + 1], i);							// ����3D����
#else
	FindCorrespondence(ProcessorStream[streamIdx + 1], i);							// (�˴�����)Ѱ��ƥ��ĵ㡾ʹ�õ���Foreground��
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

	ComputeGradientMap(ProcessorStream[streamIdx], i);								// ����ǰ��mask�ݶȺ�grayscaleͼ���ݶ�

	// ������ɺ�֪ͨ���߳�
	numThreadsCompleted.fetch_add(1);

}

SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::FrameProcessor::ProcessFirstFrame(size_t frameIndex, RigidSolver::Ptr rigidSolver)
{

	//֮��ÿ֡�ںϵ�ʱ��ʹ�ã������൱�ڳ�ʼ��
	FrameIndex = frameIndex;
	for (int i = 0; i < deviceCount; i++) {
		int streamIdx = i * A_IMAGEPROCESSOR_NEED_CUDA_STREAM;	// ��������Index,ÿ���������3����
		FetchFrame(FrameIndex, i);
/**************************** 2��Camera ƽ������2ms - 5ms (ÿ�����ʹ��һ��������color��depth��ʹ��2���������ٶ�һ��)****************************/
		UploadRawColorImageToGPU(ProcessorStream[streamIdx], i);						// ��ȡԭʼ��RGB��Ϣ
		SegmentForeground(ProcessorStream[streamIdx + 1], i);							// �ָ�ǰ�����ָ���ͼ����ͬʱ���С�
		UploadDepthImageToGPU(ProcessorStream[streamIdx], i);							// ��ȡ�����Ϣ
		ClipFilterDepthImage(ProcessorStream[streamIdx], i);							// ���ò������ͼ����˫���˲�
		ClipNormalizeColorImage(ProcessorStream[streamIdx], i);							// ����RGBͼ�񣬻�ûҶ�ͼ�����ԻҶ�ͼ����
/**************************** 2��Camera ƽ������40us - 60us (ÿ�����ʹ��һ�����������ɶ��㼰���Ŷ�Map�����߼��뾶Map)****************************/
		buildVertexConfidenceMap(ProcessorStream[streamIdx], i);						// �������㼰���Ŷ�Map
		buildNormalRadiusMap(ProcessorStream[streamIdx], i);							// �������߼��뾶Map
		buildColorTimeMap(FrameIndex, ProcessorStream[streamIdx], i);					// ����Color-Timeͼ
		SyncAndConvertForeground2Texture(ProcessorStream[streamIdx + 1], i);			// ͬ���ָ����������ָ���ת��Texture�洢���������ʹ�÷ָ���ĵط���
		rigidSolver->setCamerasInitialSE3(i);
	}
	syncAllProcessorStream();			// ͬ��������

/**************************** ������ӽ�ǰ��Լ���ӿڲ������������Texture����Լ�� ****************************/
	device::MultiViewMaskInterface MaskConstriction;
	for (int i = 0; i < deviceCount; i++) {
		MaskConstriction.depthMap[i] = clipedDepthCUDATexture[i].texture;
		MaskConstriction.vertexMap[i] = icp_vertex_confidence[i].texture;
		MaskConstriction.normalMap[i] = icp_normal_radius[i].texture;
		MaskConstriction.colorMap[i] = icp_color_time[i].texture;
		MaskConstriction.foreground[i] = getForegroundMaskTexture(i);
		MaskConstriction.InitialCameraSE3[i] = InitialCameraSE3[i];
		MaskConstriction.InitialCameraSE3Inverse[i] = InitialCameraSE3Inv[i];
		MaskConstriction.ClipedIntrinsic[i] = clipColorIntrinsic[i];
	}
#ifdef REBUILD_WITHOUT_BACKGROUND

	// �����ڲ�����ͬ��
	MultiViewForegroundMaskConstriction(clipedDepthCUDATexture, vertexConfidenceTexture, normalRadiusTexture, colorTimeTexture, MaskConstriction, deviceCount, rawImageColsCliped, rawImageRowsCliped);
#endif // REBUILD_WITHOUT_BACKGROUND
	//Visualizer::DrawRawSegmentMask(0, MaskConstriction.foreground[0]);
	//Visualizer::DrawRawSegmentMask(1, MaskConstriction.foreground[1]); 
	//Visualizer::DrawRawSegmentMask(2, MaskConstriction.foreground[2]);
	//Visualizer::DrawPointCloud(vertexConfidenceTexture[1].texture);
	for (int i = 0; i < deviceCount; i++) {
		collectValidDepthSurfel(ProcessorStream[i], i);
	}
	syncAllProcessorStream();			// ͬ��������

/**************************** ������ӽ�ǰ��Լ���ӿڲ������������Texture����Լ�� ****************************/
	// ���������
	surfelsProcessor->MergeDenseSurfels(deviceCount, depthSurfelView, InitialCameraSE3, preAlignedSurfel);

	//�������δ�ںϵ�����
	preAlignedSurfelView = preAlignedSurfel.ArrayView();
	//���ӻ�δ�ںϵ�����
	//Visualizer::DrawPointCloudWithNormal(preAlignedSurfel.Array());

	syncAllProcessorStream();			// ͬ��������

	//�����ںϲ���
	//mergeSurface->MergeAllSurfaces(depthSurfel);

	// ���ӻ��ںϺ������
	//Visualizer::DrawPointCloudWithNormal(mergeSurface->GetMergedSurfelArray());

#ifdef DEBUG_RUNNING_INFO
	//printf("�ںϺ�ĵ����� = %lld ��\n", mergeSurface->GetMergedSurfelArray().size());
#endif // DEBUG_RUNNING_INFO

	//�ںϵ�����
	return preAlignedSurfelView;
}

void SparseSurfelFusion::FrameProcessor::ProcessCurrentFrame(CameraObservation& observation, size_t frameIndex)
{

	memset(&observation, 0, sizeof(observation));				// ��ʼ��observation�����ڶ��߳��г�ʼ����ֹ�ô����
	FrameIndex = frameIndex;

	// ÿ���߳�ִ�ж��Camera�����������ܲ���
	for (int i = 0; i < deviceCount; i++) {
		// �����õķ�ʽ����observation����ֵ���ݵķ�ʽ������������
		pool->AddTask([=, &observation]() { ProcessCurrentCameraImageTask(i, observation); });
	}
	while (numThreadsCompleted.load() != deviceCount) {
		std::this_thread::yield();		// ���̵߳ȴ�
	}
	numThreadsCompleted.store(0);		// ��ռ�����־λ
	// �ȴ������߳����
	syncAllProcessorStream();			// ͬ��������


/**************************** ������ӽ�ǰ��Լ���ӿڲ������������Texture����Լ�� ****************************/
	device::MultiViewMaskInterface MaskConstriction;
	for (int i = 0; i < deviceCount; i++) {
		MaskConstriction.depthMap[i] = clipedDepthCUDATexture[i].texture;
		MaskConstriction.vertexMap[i] = icp_vertex_confidence[i].texture;
		MaskConstriction.normalMap[i] = icp_normal_radius[i].texture;
		MaskConstriction.colorMap[i] = icp_color_time[i].texture;
		MaskConstriction.foreground[i] = getForegroundMaskTexture(i);
		MaskConstriction.InitialCameraSE3[i] = InitialCameraSE3[i];
		MaskConstriction.InitialCameraSE3Inverse[i] = InitialCameraSE3Inv[i];
		MaskConstriction.ClipedIntrinsic[i] = clipColorIntrinsic[i];
	}
#ifdef REBUILD_WITHOUT_BACKGROUND
	// �����ڲ�����ͬ��
	MultiViewForegroundMaskConstriction(clipedDepthCUDATexture, vertexConfidenceTexture, normalRadiusTexture, colorTimeTexture, MaskConstriction, deviceCount, rawImageColsCliped, rawImageRowsCliped, ProcessorStream[0]);
#endif // REBUILD_WITHOUT_BACKGROUND
/**************************** ������ӽ�ǰ��Լ���ӿڲ������������Texture����Լ�� ****************************/

/**************************** ���ӽ�Ѱ��ƥ��㲢��ֵ ****************************/
	CrossViewEdgeCorrespondence::CrossViewMatchingInput input;
	CrossViewMatchingInterpolation::CrossViewInterpolationInput interInput;
	for (int i = 0; i < deviceCount; i++) {
		// �羳ƥ���
		input.vertexMap[i] = vertexConfidenceTexture[i].texture;
		input.normalMap[i] = normalRadiusTexture[i].texture;
		input.colorMap[i] = colorTimeTexture[i].texture;
		input.edgeMask[i] = foregroundSegmenter[i]->getForegroundEdgeMaskTexture();
		input.rawClipedMask[i] = foregroundSegmenter[i]->getClipedMaskTexture();
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
		input.matchedPairsMap[i] = opticalFlow[i]->getCorrespondencePairsMap();
#else
		input.matchedPairsMap[i] = featureCorrespondenceFinder[i]->getCorrespondencePairsMap();
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
		// ���ӽǲ�ֵ��
		interInput.vertexMap[i] = vertexConfidenceTexture[i].texture;
		interInput.normalMap[i] = normalRadiusTexture[i].texture;
		interInput.colorMap[i] = colorTimeTexture[i].texture;
	}
	CrossViewMatching->SetCrossViewMatchingInput(input);
	CrossViewMatching->FindCrossViewEdgeMatchedPairs(ProcessorStream[0]);
	observation.crossCorrPairs = CrossViewMatching->GetCrossViewBackTracingCorrPairs();

	CrossViewInterpolation->SetCrossViewInterpolationInput(interInput, CrossViewMatching->GetCrossViewMatchingUniqueCorrPairs());
	CrossViewInterpolation->CrossViewInterpolateSurfels(ProcessorStream[0]);
/**************************** ���ӽ�Ѱ��ƥ��㲢��ֵ ****************************/

	syncAllProcessorStream();			// ͬ��������

	for (int i = 0; i < deviceCount; i++) {
		//����Ҳ����
		collectValidDepthSurfel(ProcessorStream[i], i);							// �ռ���Ч�������Ԫ���洢��depthSurfel[]��
		//�����ںϵ�map
		CollectObservationParameters(ProcessorStream[i], observation, i);		// �ռ�����۲�Ĳ���
	}
	syncAllProcessorStream();			// ͬ��������

	//Visualizer::DrawRawSegmentMask(0, foregroundSegmenter[0]->getForegroundEdgeMaskTexture());
	//Visualizer::DrawRawSegmentMask(1, foregroundSegmenter[1]->getForegroundEdgeMaskTexture());
	//Visualizer::DrawRawSegmentMask(2, foregroundSegmenter[2]->getForegroundEdgeMaskTexture());
	// ���������
	surfelsProcessor->MergeDenseSurfels(deviceCount, depthSurfelView, InitialCameraSE3, preAlignedSurfel);
	//�������δ�ںϵ�����
	preAlignedSurfelView = preAlignedSurfel.ArrayView();


	//if (frameIndex == 95) {
	//	// ���ƿ��ӽǵ�ƥ���
	//	Visualizer::DrawCrossCorrPairs(
	//		preAlignedSurfel.Array(),
	//		icp_vertex_confidence[0].texture, InitialCameraSE3[0],
	//		icp_vertex_confidence[1].texture, InitialCameraSE3[1],
	//		icp_vertex_confidence[2].texture, InitialCameraSE3[2],
	//		CrossViewMatching->GetCrossViewBackTracingCorrPairs()
	//	);

	//	//Visualizer::DrawPointCloud(vertexConfidenceTexture[0].texture);
	//	//Visualizer::DrawPointCloud(vertexConfidenceTexture[1].texture);
	//	//Visualizer::DrawPointCloud(vertexConfidenceTexture[2].texture);

		//Visualizer::DrawInterpolatedSurfels(
		//	preAlignedSurfel.Array(),
		//	observation.interpolatedVertexMap[0], observation.interpolatedValidValue[0], InitialCameraSE3[0],
		//	observation.interpolatedVertexMap[1], observation.interpolatedValidValue[1], InitialCameraSE3[1],
		//	observation.interpolatedVertexMap[2], observation.interpolatedValidValue[2], InitialCameraSE3[2]
		//);
	//}





	//if (frameIndex == 34) {
		//Visualizer::DrawImagePairCorrespondence(clipedNormalizeColorImagePrevious[0].texture, clipedNormalizeColorImage[0].texture, getCorrespondencePixelPair(0));
	//}

	//Visualizer::DrawGradientMap(observation.grayScaleGradientMap[0], "X");
	//Visualizer::DrawGradientMap(observation.grayScaleGradientMap[0], "Y");

	//Visualizer::DrawRawSegmentMask(0, observation.foregroundMask[0]);
	//Visualizer::DrawRawSegmentMask(1, observation.foregroundMask[1]); 
	//Visualizer::DrawRawSegmentMask(2, observation.foregroundMask[2]);

	//Visualizer::DrawFilteredSegmentMask(observation.filteredForegroundMask[1]);
	//Visualizer::DrawFilteredSegmentMask(observation.grayScaleMap[1]);

	//mergeSurface->MergeAllSurfaces(depthSurfel);

#ifdef DEBUG_RUNNING_INFO
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	printf("OpticalFlow�ҵ���ƥ������ص����(%lld + %lld + %lld) = %lld \n", observation.correspondencePixelPairs[0].Size(), observation.correspondencePixelPairs[1].Size(), observation.correspondencePixelPairs[2].Size(), observation.correspondencePixelPairs[0].Size() + observation.correspondencePixelPairs[1].Size() + observation.correspondencePixelPairs[2].Size());
#else
	//printf("GPC�ҵ���ƥ������ص����(%lld + %lld + %lld) = %lld \n", observation.correspondencePixelPairs[0].Size(), observation.correspondencePixelPairs[1].Size(), observation.correspondencePixelPairs[2].Size(), observation.correspondencePixelPairs[0].Size() + observation.correspondencePixelPairs[1].Size() + observation.correspondencePixelPairs[2].Size());
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS	printf("�ںϺ�ĵ�������%lld��\n", mergeSurface->GetMergedSurfelsView().Size());
#endif // DEBUG_RUNNING_INFO

	//if (frameIndex % 10 == 0) {
	//	Visualizer::DrawPointCloudWithNormal(mergeSurface->GetMergedSurfelArray());
	//}
	//device::MapMergedSurfelInterface mergeSurfelInterface;
	//for (int i = 0; i < deviceCount; i++) {
	//	mergeSurfelInterface.vertex[i] = vertexConfidenceTexture[i].surface;
	//	mergeSurfelInterface.normal[i] = normalRadiusTexture[i].surface;
	//	mergeSurfelInterface.color[i] = colorTimeTexture[i].surface;
	//	mergeSurfelInterface.ClipedIntrinsic[i] = clipColorIntrinsic[i];
	//	mergeSurfelInterface.InitialCameraSE3Inverse[i] = InitialCameraSE3Inv[i];
	//}

	//ֱ��cudaӳ���
	//clearMapSurfel(rawImageColsCliped, rawImageRowsCliped, mergeSurfelInterface, ProcessorStream[0]);
	//mapMergedDepthSurfel(mergeSurface->GetMergedSurfelsView(), mergeSurfelInterface, rawImageColsCliped, rawImageRowsCliped, ProcessorStream[0]);
	CHECKCUDA(cudaStreamSynchronize(ProcessorStream[0]));
	
	//CollectMergeObservationParameters(observation);

	// �鿴ƥ���
	//if (frameIndex == 600){
	//	//debugmask(foregroundSegmenter[0]->getPreviousMaskTexture(), foregroundSegmenter[1]->getPreviousMaskTexture(), foregroundSegmenter[2]->getPreviousMaskTexture(), "PreviousImage");
	//	//debugmask(foregroundSegmenter[0]->getClipedMaskTexture(), foregroundSegmenter[1]->getClipedMaskTexture(), foregroundSegmenter[2]->getClipedMaskTexture(), "CurrentImage");
	//	Visualizer::DrawImagePairCorrespondence(clipedNormalizeColorImagePrevious[1].texture, clipedNormalizeColorImage[1].texture, getCorrespondencePixelPair(1));
	//}
	
	//return mergeSurface->GetMergedSurfelsView();
}

void SparseSurfelFusion::FrameProcessor::CollectObservationParameters(cudaStream_t stream, CameraObservation& observation, const unsigned int CameraID)
{
	//����ԭ�������Ԫӳ�� ����Ҳ����
	observation.icpvertexConfidenceMap[CameraID] = getBackgroundVertexConfidenceTexture(CameraID);
	observation.icpnormalRadiusMap[CameraID] = getBackgroundNormalRadiusTexture(CameraID);
	// ���ڿ��ӻ���ԭʼ���ͼ��
	observation.rawcolorTimeMap[CameraID] = getBackgroundColorTimeTexture(CameraID);
	observation.rawDepthImage[CameraID] = getRawDepthCUDATexture(CameraID);

	// ��õ�ǰ֡����Ϣ
	observation.vertexConfidenceMap[CameraID] = getVertexConfidenceTexture(CameraID);
	observation.normalRadiusMap[CameraID] = getNormalRadiusTexture(CameraID);
	observation.colorTimeMap[CameraID] = getColorTimeTexture(CameraID);

	// ��ȡ��һ֡����Ϣ
	observation.PreviousVertexConfidenceMap[CameraID] = getPreviousVertexConfidenceTexture(CameraID);
	observation.PreviousNormalRadiusMap[CameraID] = getPreviousNormalRadiusTexture(CameraID);

	//Reference���е�Geometryͼ��
	observation.filteredDepthImage[CameraID] = getClipedDepthCUDATexture(CameraID);


	//color ͼ�Լ��ܶ�ͼ�Լ�Color Time
	observation.normalizedRGBAMap[CameraID] = getClipedNormalizeColorImageTexture(CameraID);
	observation.normalizedRGBAPrevious[CameraID] = getClipedNormalizeColorImagePreviousTexture(CameraID);
	observation.grayScaleMap[CameraID] = getGrayScaleImageTexture(CameraID);
	observation.grayScaleGradientMap[CameraID] = getGrayScaleGradientTexture(CameraID);

	//ǰ��mask
	observation.foregroundMask[CameraID] = getForegroundMaskTexture(CameraID);
	observation.filteredForegroundMask[CameraID] = getFilteredForegroundMaskTexture(CameraID);
	observation.foregroundMaskGradientMap[CameraID] = getForegroundMaskGradientTexture(CameraID);
	observation.foregroundMaskPrevious[CameraID] = getPreviousForegroundMaskTexture(CameraID);
	observation.edgeMaskMap[CameraID] = getCurrentForegroundEdgeMaskTexture(CameraID);

#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	const auto& opticalflowpixelpair = opticalFlow[CameraID]->getCorrespondencePixelPair();
	observation.correspondencePixelPairs[CameraID] = DeviceArrayView<ushort4>(opticalflowpixelpair.ptr(), opticalflowpixelpair.size());
	observation.corrMap[CameraID] = opticalFlow[CameraID]->getCorrespondencePairsMap();
#else
	//������Ľڵ��
	const DeviceArray<ushort4>& pixel_pair_array = getCorrespondencePixelPair(CameraID);
	observation.correspondencePixelPairs[CameraID] = DeviceArrayView<ushort4>(pixel_pair_array.ptr(), pixel_pair_array.size());
	observation.corrMap[CameraID] = featureCorrespondenceFinder[CameraID]->getCorrespondencePairsMap();
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

	observation.interpolatedValidValue[CameraID] = CrossViewInterpolation->GetValidInterpolatedMarkMap(CameraID);
	observation.interpolatedVertexMap[CameraID] = CrossViewInterpolation->GetInterpolatedVertexMap(CameraID);
	observation.interpolatedNormalMap[CameraID] = CrossViewInterpolation->GetInterpolatedNormalMap(CameraID);
	observation.interpolatedColorMap[CameraID] = CrossViewInterpolation->GetInterpolatedColorMap(CameraID);
}

void SparseSurfelFusion::FrameProcessor::CollectMergeObservationParameters(CameraObservation& observation)
{
	//���������뵥��������ռ���
	for (int CameraID = 0; CameraID < deviceCount; CameraID++) {
		observation.vertexConfidenceMap[CameraID] = getVertexConfidenceTexture(CameraID);
		observation.normalRadiusMap[CameraID] = getNormalRadiusTexture(CameraID);
		observation.colorTimeMap[CameraID] = getColorTimeTexture(CameraID);
	}
}

void SparseSurfelFusion::FrameProcessor::initProcessorStream()
{
	//������
	for (int i = 0; i < TotalProcessingStreams; i++) {
		CHECKCUDA(cudaStreamCreate(&ProcessorStream[i]));
	}
}

void SparseSurfelFusion::FrameProcessor::releaseProcessorStream()
{
	for (int i = 0; i < TotalProcessingStreams; i++) {
		CHECKCUDA(cudaStreamDestroy(ProcessorStream[i]));
		ProcessorStream[i] = 0;
	}
}

void SparseSurfelFusion::FrameProcessor::syncAllProcessorStream()
{
	for (int i = 0; i < TotalProcessingStreams; i++) {
		CHECKCUDA(cudaStreamSynchronize(ProcessorStream[i]));
	}
}

void SparseSurfelFusion::FrameProcessor::allocateImageHostBuffer()
{
	//���ȷ��仺����
	unsigned int rawImageSize = rawImageRows * rawImageCols; // ÿһ�������ͼ��Ҫ��
	for (int i = 0; i < deviceCount; i++) {
		CHECKCUDA(cudaMallocHost(&depthBufferPagelock[i], sizeof(unsigned short) * rawImageSize));
		CHECKCUDA(cudaMallocHost(&depthBufferPagelockPrevious[i], sizeof(unsigned short) * rawImageSize));
		CHECKCUDA(cudaMallocHost(&colorBufferPagelock[i], sizeof(uchar4) * rawImageSize));
		CHECKCUDA(cudaMallocHost(&colorBufferPagelockPrevious[i], sizeof(uchar4) * rawImageSize));
		CHECKCUDA(cudaDeviceSynchronize());
	}
}

void SparseSurfelFusion::FrameProcessor::releaseImageHostBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		CHECKCUDA(cudaFreeHost(depthBufferPagelock[i]));
		CHECKCUDA(cudaFreeHost(depthBufferPagelockPrevious[i]));
		CHECKCUDA(cudaFreeHost(colorBufferPagelock[i]));
		CHECKCUDA(cudaFreeHost(colorBufferPagelockPrevious[i]));
		CHECKCUDA(cudaDeviceSynchronize());
	}
}

void SparseSurfelFusion::FrameProcessor::GetOfflineData(std::vector<cv::Mat>* ColorOffline, std::vector<cv::Mat>* DepthOffline, std::string suffixPath, bool intoBuffer)
{
	this->intoBuffer = intoBuffer;
	this->dataSuffixPath = suffixPath;
	if (intoBuffer == true) {
		ColorOfflineData = ColorOffline;
		DepthOfflineData = DepthOffline;
	}
}

void SparseSurfelFusion::FrameProcessor::FetchFrame(size_t frameIndex, const unsigned int CameraID)
{
	std::string prefixPath = DATA_PATH_PREFIX;

	if (intoBuffer == true) {
		if (frameIndex == beginIndex) {
			previousColorImage[CameraID] = ColorOfflineData[CameraID][frameIndex];
			previousDepthImage[CameraID] = DepthOfflineData[CameraID][frameIndex];
			colorImage[CameraID] = previousColorImage[CameraID];
			depthImage[CameraID] = previousDepthImage[CameraID];

		}
		else {
			previousColorImage[CameraID] = colorImage[CameraID];	// ��һ֡�����ȴ洢
			previousDepthImage[CameraID] = depthImage[CameraID];	// ��һ֡�����ȴ洢
			colorImage[CameraID] = ColorOfflineData[CameraID][frameIndex];// �洢��ǰ֡
			depthImage[CameraID] = DepthOfflineData[CameraID][frameIndex];// �洢��ǰ֡
		}
	}
	else {
		std::string path;
		path = prefixPath + dataSuffixPath + "/Camera_" + std::to_string(CameraID);
		std::stringstream ss;
		ss << std::setw(5) << std::setfill('0') << frameIndex;
		std::string indexString;
		ss >> indexString;
		cv::String colorFrameName = cv::String(std::string("color_frame_") + indexString + std::string(".png"));
		cv::String depthFrameName = cv::String(std::string("depth_frame_") + indexString + std::string(".png"));

		if (frameIndex == beginIndex) {
			previousColorImage[CameraID] = cv::imread(path + "/color/" + colorFrameName, cv::IMREAD_ANYCOLOR);
			previousDepthImage[CameraID] = cv::imread(path + "/depth/" + depthFrameName, cv::IMREAD_ANYDEPTH);
			colorImage[CameraID] = previousColorImage[CameraID];
			depthImage[CameraID] = previousDepthImage[CameraID];
		}
		else {
			previousColorImage[CameraID] = colorImage[CameraID].clone();	// ��һ֡�����ȴ洢
			previousDepthImage[CameraID] = depthImage[CameraID].clone();	// ��һ֡�����ȴ洢
			colorImage[CameraID] = cv::imread(path + "/color/" + colorFrameName, cv::IMREAD_ANYCOLOR);
			depthImage[CameraID] = cv::imread(path + "/depth/" + depthFrameName, cv::IMREAD_ANYDEPTH);

			//cv::imshow("Pre", previousColorImage[CameraID]);
			//cv::imshow("Curr", colorImage[CameraID]);
			//cv::waitKey(10000000);
		}
	}
	if (colorImage[CameraID].empty()) LOGGING(FATAL) << "colorImageͼƬ��ȡʧ��";
	if (depthImage[CameraID].empty()) LOGGING(FATAL) << "depthImageͼƬ��ȡʧ��";
	if (previousColorImage[CameraID].empty()) LOGGING(FATAL) << "previousColorImageͼƬ��ȡʧ��";



	memcpy(colorBufferPagelock[CameraID], colorImage[CameraID].data, sizeof(uchar3) * rawImageRows * rawImageCols);
	memcpy(colorBufferPagelockPrevious[CameraID], previousColorImage[CameraID].data, sizeof(uchar3) * rawImageRows * rawImageCols);
	memcpy(depthBufferPagelock[CameraID], depthImage[CameraID].data, sizeof(unsigned short) * rawImageRows * rawImageCols);
	memcpy(depthBufferPagelockPrevious[CameraID], previousDepthImage[CameraID].data, sizeof(unsigned short) * rawImageRows * rawImageCols);
}

void SparseSurfelFusion::FrameProcessor::UploadDepthImageToGPU(cudaStream_t stream, int CameraID)
{
	void* ptr = rawDepthImage[CameraID].ptr();		// ���RGBͼ����GPU�е�ַ����������ҳ�ڴ��п�������
	// �첽������rawColorImage[CameraID]�����õ�GPU�ĵ�ַ
	CHECKCUDA(cudaMemcpyAsync(ptr, depthBufferPagelock[CameraID], sizeof(unsigned short) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));
	ptr = rawDepthImagePrevious[CameraID].ptr();	// ���ǰһ֡ͼ����GPU�еĵ�ַ����������ҳ�ڴ��п�������
	CHECKCUDA(cudaMemcpyAsync(ptr, depthBufferPagelockPrevious[CameraID], sizeof(unsigned short) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));
	// ����ҳ�ڴ��е�2D�����첽������cuda��
	CHECKCUDA(cudaMemcpy2DToArrayAsync(rawDepthCUDATexture[CameraID].cudaArray, 0, 0, depthBufferPagelock[CameraID], sizeof(unsigned short) * rawImageCols, sizeof(unsigned short) * rawImageCols, rawImageRows, cudaMemcpyHostToDevice, stream));

}

void SparseSurfelFusion::FrameProcessor::ClipFilterDepthImage(cudaStream_t stream, int CameraID)
{
	clipFilterDepthImage(rawDepthCUDATexture[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, clipNear, clipFar, clipedDepthCUDATexture[CameraID].surface, stream);
}

void SparseSurfelFusion::FrameProcessor::allocateDepthTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		rawDepthImage[i].create(size_t(rawImageCols) * rawImageRows);
		rawDepthImagePrevious[i].create(size_t(rawImageCols) * rawImageRows);
		// ԭʼͼ��Ӧ��ʹ��rawImageRows,rawImageCols
		createDepthTextureSurface(rawImageRows, rawImageCols, rawDepthCUDATexture[i]);
		// ���˺��ͼ��Ӧ��ʹ��rawImageRowsCliped,rawImageColsCliped
		createDepthTextureSurface(rawImageRowsCliped, rawImageColsCliped, clipedDepthCUDATexture[i]);
	}

}

void SparseSurfelFusion::FrameProcessor::releaseDepthTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		rawDepthImage[i].release();
		rawDepthImagePrevious[i].release();
		releaseTextureCollect(rawDepthCUDATexture[i]);
		releaseTextureCollect(clipedDepthCUDATexture[i]);
	}
}

void SparseSurfelFusion::FrameProcessor::UploadRawColorImageToGPU(cudaStream_t stream, int CameraID)
{
	void* ptr = rawColorImage[CameraID].ptr();		// ���RGBͼ����GPU�е�ַ����������ҳ�ڴ��п�������
	// �첽������rawColorImage[CameraID]�����õ�GPU�ĵ�ַ
	CHECKCUDA(cudaMemcpyAsync(ptr, colorBufferPagelock[CameraID], sizeof(uchar3) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));
	ptr = rawColorImagePrevious[CameraID].ptr();	// ���ǰһ֡ͼ����GPU�еĵ�ַ����������ҳ�ڴ��п�������
	CHECKCUDA(cudaMemcpyAsync(ptr, colorBufferPagelockPrevious[CameraID], sizeof(uchar3) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));

}

void SparseSurfelFusion::FrameProcessor::ClipNormalizeColorImage(cudaStream_t stream, int CameraID)
{
	// ���ò�ɫͼ��
	clipNormalizeColorImage(rawColorImage[CameraID], rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImage[CameraID].surface, GrayScaleImage[CameraID].surface, stream);
	// ���ûҶ�ͼ��
	filterGrayScaleImage(GrayScaleImage[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, GrayScaleImageFiltered[CameraID].surface, stream);
	// ����ǰһ֡RGBͼ��
	clipNormalizeColorImage(rawColorImagePrevious[CameraID], rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImagePrevious[CameraID].surface, stream);
}

void SparseSurfelFusion::FrameProcessor::allocateColorTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		rawColorImage[i].create(size_t(rawImageCols) * rawImageRows);												// ��ԭʼcolorͼ�����GPU�ڴ�
		rawColorImagePrevious[i].create(size_t(rawImageCols) * rawImageRows);										// ��ԭʼPrevious colorͼ�����GPU�ڴ�
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImage[i]);			// ��clipedNormalizeColorImage[i]����Float4���͵�����(��RGB)
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImagePrevious[i]);	// ��clipedNormalizeColorImagePrevious[i]����Float4���͵�����(��RGB)
		createFloat1TextureSurface(rawImageRowsCliped, rawImageColsCliped, GrayScaleImage[i]);						// ��GrayScaleImage[i]����Float1���͵�����(��Ҷ�ͼ)
		createFloat1TextureSurface(rawImageRowsCliped, rawImageColsCliped, GrayScaleImageFiltered[i]);				// ��GrayScaleImageFiltered[i]����Float1���͵�����(��Ҷ�ͼ)
	}
}

void SparseSurfelFusion::FrameProcessor::releaseColorTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		rawColorImage[i].release();									// �ͷ�rawColorImage��GPU�е��ڴ�
		rawColorImagePrevious[i].release();							// �ͷ�rawColorImagePrevious��GPU�е��ڴ�
		releaseTextureCollect(clipedNormalizeColorImage[i]);		// �ͷ�clipedNormalizeColorImage��GPU�д洢�����������ڴ�
		releaseTextureCollect(clipedNormalizeColorImagePrevious[i]);// �ͷ�clipedNormalizeColorImagePrevious��GPU�д洢�����������ڴ�
		releaseTextureCollect(GrayScaleImage[i]);					// �ͷ�GrayScaleImage��GPU�д洢�����������ڴ�
		releaseTextureCollect(GrayScaleImageFiltered[i]);			// �ͷ�GrayScaleImageFiltered��GPU�д洢�����������ڴ�
	}
}

void SparseSurfelFusion::FrameProcessor::allocateSurfelAttributeTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, previousVertexConfidenceTexture[i]);
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, previousNormalRadiusTexture[i]);
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, vertexConfidenceTexture[i]);
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, normalRadiusTexture[i]);
	}
}

void SparseSurfelFusion::FrameProcessor::releaseSurfelAttributeTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		releaseTextureCollect(previousVertexConfidenceTexture[i]);
		releaseTextureCollect(previousNormalRadiusTexture[i]);
		releaseTextureCollect(vertexConfidenceTexture[i]);
		releaseTextureCollect(normalRadiusTexture[i]);
	}
}

void SparseSurfelFusion::FrameProcessor::buildVertexConfidenceMap(cudaStream_t stream, int CameraID)
{
	//��һ֡��
	copyPreviousVertexAndNormal(
		previousVertexConfidenceTexture[CameraID].surface,
		previousNormalRadiusTexture[CameraID].surface,
		icp_vertex_confidence[CameraID].texture,
		icp_normal_radius[CameraID].texture,
		rawImageRowsCliped, 
		rawImageColsCliped
	);//����һ֡�����ݴ�textureд��suface��
	const IntrinsicInverse clipIntrinsicInverse = inverse(clipColorIntrinsic[CameraID]);
	//��һ֡�Ĳü����ÿһ��������ص��Ӧ���������ϵ�µ���ά�㼰�����Ŷȣ�float4��
	createVertexConfidenceMap(clipedDepthCUDATexture[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, clipIntrinsicInverse, icp_vertex_confidence[CameraID].surface, stream);
}

void SparseSurfelFusion::FrameProcessor::buildNormalRadiusMap(cudaStream_t stream, int CameraID)
{
	createNormalRadiusMap(icp_vertex_confidence[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, (clipColorIntrinsic[CameraID].focal_x + clipColorIntrinsic[CameraID].focal_x) / 2.0f, icp_normal_radius[CameraID].surface, stream);
}

void SparseSurfelFusion::FrameProcessor::allocateColorTimeTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, colorTimeTexture[i]);
	}
}

void SparseSurfelFusion::FrameProcessor::releaseColorTimeTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		releaseTextureCollect(colorTimeTexture[i]);
	}
}

void SparseSurfelFusion::FrameProcessor::buildColorTimeMap(size_t frameIdx, cudaStream_t stream, int CameraID)
{
	const float initTime = float(FrameIndex);
	createColorTimeMap(rawColorImage[CameraID], rawImageRowsCliped, rawImageColsCliped, initTime, CameraID, icp_color_time[CameraID].surface, stream);
}

void SparseSurfelFusion::FrameProcessor::allocateValidSurfelSelectionBuffer()
{
	const unsigned int pixelsNum = rawImageColsCliped * rawImageRowsCliped;
	for (int i = 0; i < deviceCount; i++) {
		validDepthPixelSelector[i].AllocateAndInit(pixelsNum);
		depthSurfel[i].AllocateBuffer(pixelsNum);
		subsampleSparseVertice[i].AllocateBuffer(Constants::maxSubsampledSparsePointsNum);
	}

	preAlignedSurfel.AllocateBuffer(size_t(Constants::maxSurfelsNum) * MAX_CAMERA_COUNT);
}

void SparseSurfelFusion::FrameProcessor::releaseValidSurfelSelectionBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		depthSurfel[i].ReleaseBuffer();
		subsampleSparseVertice[i].ReleaseBuffer();
	}

}

void SparseSurfelFusion::FrameProcessor::collectValidDepthSurfel(cudaStream_t stream, int CameraID)
{
	//��������
	const unsigned int num_pixels = rawImageColsCliped * rawImageRowsCliped;
	DeviceArray<char> validIndicator = DeviceArray<char>(validDepthPixelSelector[CameraID].selectIndicatorBuffer.ptr(), num_pixels);
	//markValidDepthPixel(clipedDepthCUDATexture[CameraID].texture, foregroundSegmenter[CameraID]->getClipedMaskTexture(), icp_normal_radius[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, validDepthPixelSelector[CameraID].selectIndicatorBuffer, stream);
	markValidDepthPixel(clipedDepthCUDATexture[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, validDepthPixelSelector[CameraID].selectIndicatorBuffer, stream);

	//ɸѡ��Ч��������ص�
	validDepthPixelSelector[CameraID].Select(validIndicator, stream);

	//�������
	const size_t selected_surfel_size = validDepthPixelSelector[CameraID].validSelectedIndex.size();
	depthSurfel[CameraID].ResizeArrayOrException(selected_surfel_size);

	//��������
	DeviceArray<DepthSurfel> valid_surfel_array = depthSurfel[CameraID].Array();//��ַ����
	collectDepthSurfel(icp_vertex_confidence[CameraID].texture, icp_normal_radius[CameraID].texture, icp_color_time[CameraID].texture, validDepthPixelSelector[CameraID].validSelectedIndex, rawImageRowsCliped, rawImageColsCliped, CameraID, valid_surfel_array, stream);
	depthSurfelView[CameraID] = DeviceArrayView<DepthSurfel>(valid_surfel_array.ptr(), valid_surfel_array.size());
}


void SparseSurfelFusion::FrameProcessor::alignedDenseValidDepthSurfel(RigidSolver::Ptr solver, cudaStream_t stream, int CameraID)
{
	unsigned int offset = 0;								// ����preAlignedSurfel��ƫ��
	size_t totalDenseSurfelNum = depthSurfelView[0].Size();	// �ܹ�����preAlignedSurfel����Array�ռ�
	// �������Canonical�ռ��ƫ��
	for (int i = 0; i < CameraID + 1; i++) {
		if (i == 0) continue;
		totalDenseSurfelNum += depthSurfelView[i].Size();
		offset += depthSurfelView[i - 1].Size();
	}
	preAlignedSurfel.ResizeArrayOrException(totalDenseSurfelNum);

	solver->mergeDenseSurfelToCanonicalField(preAlignedSurfel, depthSurfelView[CameraID], CameraID, offset, stream);

	preAlignedSurfelView = preAlignedSurfel.ArrayView();
}

void SparseSurfelFusion::FrameProcessor::allocateForegroundSegmentationBuffer()
{
	auto start = std::chrono::high_resolution_clock::now();// ��ȡ����ʼʱ���

	float inferMemorySize = 0;
	for (int i = 0; i < deviceCount; i++) {
		size_t memorySize;
		foregroundSegmenter[i] = std::make_shared<ForegroundSegmenter>(memorySize);
		foregroundSegmenter[i]->AllocateInferBuffer();
		inferMemorySize += memorySize * 1.0f;
	}
	auto end = std::chrono::high_resolution_clock::now();// ��ȡ�������ʱ���
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// �����������ʱ�䣨���룩
	std::cout << "����RVMģ�͹��ķ��Դ��С��" << inferMemorySize / (1024.0f * 1024.0f) << "MB     ��ʱ��" << duration / 1000.0f << " ��" << std::endl;
}

void SparseSurfelFusion::FrameProcessor::releaseForegroundSegmentationBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		foregroundSegmenter[i]->ReleaseInferBuffer();
	}
}

void SparseSurfelFusion::FrameProcessor::SegmentForeground(cudaStream_t stream, const unsigned int CameraID)
{
	foregroundSegmenter[CameraID]->InferMask(stream, rawColorImage[CameraID]);
}

void SparseSurfelFusion::FrameProcessor::SyncAndConvertForeground2Texture(cudaStream_t stream, const unsigned int CameraID)
{
	foregroundSegmenter[CameraID]->SyncAndConvert2Texture(stream);
}

void SparseSurfelFusion::FrameProcessor::allocateFeatureCorrespondenceBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		featureCorrespondenceFinder[i] = std::make_shared<PatchColliderRGBCorrespondence>();
		featureCorrespondenceFinder[i]->AllocateBuffer(rawImageRowsCliped, rawImageColsCliped);
	}

}

void SparseSurfelFusion::FrameProcessor::releaseFeatureCorrespondenceBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		featureCorrespondenceFinder[i]->ReleaseBuffer();
	}
}

void SparseSurfelFusion::FrameProcessor::FindCorrespondence(cudaStream_t stream, const unsigned int CameraID)
{
	cudaTextureObject_t rgbPrevious = getClipedNormalizeColorImagePreviousTexture(CameraID);
	cudaTextureObject_t rgbCurrent = getClipedNormalizeColorImageTexture(CameraID);
	cudaTextureObject_t foregroundPrevious = getPreviousForegroundMaskTexture(CameraID);
	cudaTextureObject_t foregroundCurrent = getForegroundMaskTexture(CameraID);
	cudaTextureObject_t curEdgeMask = getCurrentForegroundEdgeMaskTexture(CameraID);	// ��ǰ֡ǰ����ԵMask

	cudaTextureObject_t previousVertexMap = getPreviousVertexConfidenceTexture(CameraID);
	cudaTextureObject_t currentVertexMap = getBackgroundVertexConfidenceTexture(CameraID);
					    
	cudaTextureObject_t previousNormalMap = getPreviousNormalRadiusTexture(CameraID);
	cudaTextureObject_t currentNormalMap = getBackgroundNormalRadiusTexture(CameraID);
	featureCorrespondenceFinder[CameraID]->SetInputImage(rgbPrevious, rgbCurrent, foregroundPrevious, foregroundCurrent, curEdgeMask, previousVertexMap, currentVertexMap, previousNormalMap, currentNormalMap, CameraID, FrameIndex);
	featureCorrespondenceFinder[CameraID]->FindCorrespondence(stream);
}

void SparseSurfelFusion::FrameProcessor::allocateOpticalFlowBuffer()
{
	auto start = std::chrono::high_resolution_clock::now();// ��ȡ����ʼʱ���
	float inferMemorySize = 0;
	for (int i = 0; i < deviceCount; i++) {
		size_t memorySize;
		opticalFlow[i] = std::make_shared<OpticalFlow>(memorySize);
		opticalFlow[i]->AllocateInferBuffer();
		inferMemorySize += memorySize * 1.0f;
	}
	auto end = std::chrono::high_resolution_clock::now();// ��ȡ�������ʱ���
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// �����������ʱ�䣨���룩

	std::cout << "����GMFlowģ�͹��ķ��Դ��С��" << inferMemorySize / (1024.0f * 1024.0f) << "MB     ��ʱ��" << duration / 1000.0f << " ��" << std::endl;

}

void SparseSurfelFusion::FrameProcessor::releaseOpticalFlowBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		opticalFlow[i]->ReleaseInferBuffer();
	}
}

void SparseSurfelFusion::FrameProcessor::computeOpticalFlow(cudaStream_t stream, const unsigned int CameraID)
{
	auto start = std::chrono::high_resolution_clock::now();								// ��ȡ����ʼʱ���
	
	cudaTextureObject_t preForeground = getPreviousForegroundMaskTexture(CameraID);		// ǰһ֡��ǰ��
	cudaTextureObject_t curForeground = getForegroundMaskTexture(CameraID);				// ��ǰ֡��ǰ��
	cudaTextureObject_t curEdgeMask = getCurrentForegroundEdgeMaskTexture(CameraID);	// ��ǰ֡ǰ����ԵMask

	cudaTextureObject_t preVertexMap = getPreviousVertexConfidenceTexture(CameraID);	// ǰһ֡��VertexMap
	cudaTextureObject_t curVertexMap = getBackgroundVertexConfidenceTexture(CameraID);	// ��ǰ֡��VertexMap
	cudaTextureObject_t preNormalexMap = getPreviousNormalRadiusTexture(CameraID);		// ǰһ֡��VertexMap
	cudaTextureObject_t curNormalMap = getBackgroundNormalRadiusTexture(CameraID);		// ��ǰ֡��VertexMap
	opticalFlow[CameraID]->setForegroundAndVertexMapTexture(preForeground, curForeground, preVertexMap, curVertexMap, preNormalexMap, curNormalMap, curEdgeMask, InitialCameraSE3[CameraID]);
	opticalFlow[CameraID]->InferOpticalFlow(rawColorImagePrevious[CameraID], rawDepthImagePrevious[CameraID], rawColorImage[CameraID], rawDepthImage[CameraID], stream);
	
	auto end = std::chrono::high_resolution_clock::now();// ��ȡ�������ʱ���
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// �����������ʱ�䣨���룩
#ifdef DEBUG_RUNNING_INFO
	std::cout << "�����������" << duration << " ����" << std::endl;
#endif // DEBUG_RUNNING_INFO

}

void SparseSurfelFusion::FrameProcessor::allocateCrossViewMatchingBuffer()
{
	CrossViewMatching = std::make_shared<CrossViewEdgeCorrespondence>(rawColorIntrinsic, InitialCameraSE3);
	CrossViewInterpolation = std::make_shared<CrossViewMatchingInterpolation>(rawColorIntrinsic, InitialCameraSE3);
}

void SparseSurfelFusion::FrameProcessor::releaseCrossViewMatchingBuffer()
{

}

void SparseSurfelFusion::FrameProcessor::allocateGradientMapBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		imageGradient = std::make_shared<ImageGradient>();
		createFloat2TextureSurface(rawImageRowsCliped, rawImageColsCliped, ForegroundMaskGradientMap[i]);
		createFloat2TextureSurface(rawImageRowsCliped, rawImageColsCliped, GrayscaleGradientMap[i]);
	}

}

void SparseSurfelFusion::FrameProcessor::releaseGradientMapBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		releaseTextureCollect(ForegroundMaskGradientMap[i]);
		releaseTextureCollect(GrayscaleGradientMap[i]);
	}
}

void SparseSurfelFusion::FrameProcessor::ComputeGradientMap(cudaStream_t stream, const unsigned int CameraID)
{
	cudaTextureObject_t filteredForegroundMask = getFilteredForegroundMaskTexture(CameraID);
	cudaTextureObject_t grayscaleMask = GrayScaleImage[CameraID].texture;

	imageGradient->computeDensityForegroundMaskGradient(filteredForegroundMask, grayscaleMask, rawImageRowsCliped, rawImageColsCliped, ForegroundMaskGradientMap[CameraID].surface, GrayscaleGradientMap[CameraID].surface);
}

void SparseSurfelFusion::FrameProcessor::setInput(vector<string>* colorpath, vector<string>* depthpath)
{
	for (int i = 0; i < deviceCount; i++)
	{
		ColorOfflinePath[i] = colorpath[i];
		DepthOfflinePath[i] = depthpath[i];
	}
	
}


