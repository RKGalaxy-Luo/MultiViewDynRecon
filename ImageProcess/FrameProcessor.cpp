/*****************************************************************//**
 * \file   FrameProcessor.cpp
 * \brief  主要负责为整体算法获取图像以及相机参数
 * 
 * \author LUO
 * \date   January 26th 2024
 *********************************************************************/
#include "FrameProcessor.h"

SparseSurfelFusion::FrameProcessor::FrameProcessor(std::shared_ptr<ThreadPool> threadPool) : pool(threadPool)
{
	numThreadsCompleted.store(0);	// 赋初值
	frame = std::make_shared<GetFrameFromCamera>(pool);		// 获取图像

	deviceCount = frame->getCameraCount();						// 获得相机个数

	configParser = std::make_shared<ConfigParser>(frame);		// 将参数赋值给configParser
	surfelsProcessor = std::make_shared<SurfelsProcessor>();	// 声明面元处理器
	for (int i = 0; i < deviceCount; i++) {						// 获得每个相机的内参
		rawColorIntrinsic[i] = frame->getColorIntrinsic(i);
		rawDepthIntrinsic[i] = frame->getDepthIntrinsic(i);
		clipColorIntrinsic[i] = frame->getClipColorIntrinsic(i);
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
		InitialCameraSE3Inv[i] = InitialCameraSE3[i].inverse();
	}
	//mergeSurface = std::make_shared<MergeSurface>(clipColorIntrinsic);

	allocateImageHostBuffer();				// 分配图像在Host上的锁页内存
	allocateDepthTexture();					// 分配GPU上的深度纹理内存
	allocateColorTexture();					// 分配GPU上的RGB纹理内存
	allocateSurfelAttributeTexture();		// 分配GPU上面元属性相关的纹理内存
	allocateColorTimeTexture();				// 分配GPU上有关面元颜色以及上一次看到面元的时刻(在第几帧)
	allocateValidSurfelSelectionBuffer();	// 分配有效面元的选择器缓存
	allocateForegroundSegmentationBuffer();	// 分配前景分割推理器内存
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	allocateOpticalFlowBuffer();			// 分配计算光流的内存
#else
	allocateFeatureCorrespondenceBuffer();	// 分配寻找稀疏特征点的内存
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	allocateCrossViewMatchingBuffer();		// 分配跨镜匹配点缓存
	allocateGradientMapBuffer();			// 分配梯度图纹理内存
	initProcessorStream();					// 初始化流

		//分配映射融合后的深度面元的内存
	for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
		createFloat4TextureSurface(CLIP_HEIGHT, CLIP_WIDTH, icp_vertex_confidence[i]);
		createFloat4TextureSurface(CLIP_HEIGHT, CLIP_WIDTH, icp_normal_radius[i]);
		createFloat4TextureSurface(CLIP_HEIGHT, CLIP_WIDTH, icp_color_time[i]);
	}


}

SparseSurfelFusion::FrameProcessor::~FrameProcessor()
{
	releaseImageHostBuffer();				// 释放图像在Host上的锁页内存
	releaseDepthTexture();					// 释放GPU上的深度纹理内存
	releaseColorTexture();					// 释放GPU上的RGB纹理内存
	releaseSurfelAttributeTexture();		// 释放GPU上面元属性相关的纹理内存
	releaseColorTimeTexture();				// 释放GPU上有关ColorTime相关的纹理
	releaseValidSurfelSelectionBuffer();	// 释放有效面元的选择器缓存
	releaseForegroundSegmentationBuffer();	// 释放前景分割推理器内存
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	releaseOpticalFlowBuffer();				// 释放计算光流的内存
#else
	releaseFeatureCorrespondenceBuffer();	// 释放寻找稀疏特征点的内存
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	releaseCrossViewMatchingBuffer();		// 释放跨镜匹配点内存
	releaseGradientMapBuffer();				// 释放梯度图纹理内存

	releaseProcessorStream();				// 释放流

		//释放texture资源，防止内存泄露
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
	// 获取程序开始运行的时间点
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

	int streamIdx = i * A_IMAGEPROCESSOR_NEED_CUDA_STREAM;	// 分配流的Index
	FetchFrame(FrameIndex, i);
/**************************** 2个Camera 平均花费2ms - 5ms (每个相机使用一个流处理color和depth与使用2个流处理速度一致)****************************/
	UploadRawColorImageToGPU(ProcessorStream[streamIdx], i);						// 获取原始的RGB信息
	SegmentForeground(ProcessorStream[streamIdx + 1], i);							// 分割前景【内部流不同步，在下方同步，将分割与图像处理并行进行】
	UploadDepthImageToGPU(ProcessorStream[streamIdx], i);							// 获取深度信息
	ClipFilterDepthImage(ProcessorStream[streamIdx], i);							// 剪裁并对深度图进行双边滤波
	ClipNormalizeColorImage(ProcessorStream[streamIdx], i);							// 剪裁RGB图像，获得灰度图，并对灰度图剪裁
/**************************** 2个Camera 平均花费40us - 60us (每个相机使用一个流处理生成顶点及置信度Map、法线及半径Map)****************************/
	//处理帧时，直接把原始映射写到icp_里
	buildVertexConfidenceMap(ProcessorStream[streamIdx], i);						// 构建顶点及置信度Map
	//这些也要改成用icp_vertex
	buildNormalRadiusMap(ProcessorStream[streamIdx], i);							// 构建法线及半径Map
	//这里也改
	buildColorTimeMap(FrameIndex, ProcessorStream[streamIdx], i);					// 构建Color-Time图
	SyncAndConvertForeground2Texture(ProcessorStream[streamIdx + 1], i);			// 同步分割流，并将分割流转成Texture存储【下面才有使用分割背景的地方】
/**************************** FindCorrespondence寻找匹配点开销最大，用时40ms，占总处理时长的80% ****************************/
#ifdef USE_OPTICALFLOW_CORRESPONDENCE_PAIRS
	computeOpticalFlow(ProcessorStream[streamIdx + 1], i);							// 计算3D光流
#else
	FindCorrespondence(ProcessorStream[streamIdx + 1], i);							// (此处阻塞)寻找匹配的点【使用到了Foreground】
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS

	ComputeGradientMap(ProcessorStream[streamIdx], i);								// 计算前景mask梯度和grayscale图的梯度

	// 任务完成后通知主线程
	numThreadsCompleted.fetch_add(1);

}

SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DepthSurfel> SparseSurfelFusion::FrameProcessor::ProcessFirstFrame(size_t frameIndex, RigidSolver::Ptr rigidSolver)
{

	//之后每帧融合的时候使用，这里相当于初始化
	FrameIndex = frameIndex;
	for (int i = 0; i < deviceCount; i++) {
		int streamIdx = i * A_IMAGEPROCESSOR_NEED_CUDA_STREAM;	// 分配流的Index,每个相机分配3个流
		FetchFrame(FrameIndex, i);
/**************************** 2个Camera 平均花费2ms - 5ms (每个相机使用一个流处理color和depth与使用2个流处理速度一致)****************************/
		UploadRawColorImageToGPU(ProcessorStream[streamIdx], i);						// 获取原始的RGB信息
		SegmentForeground(ProcessorStream[streamIdx + 1], i);							// 分割前景【分割与图像处理同时进行】
		UploadDepthImageToGPU(ProcessorStream[streamIdx], i);							// 获取深度信息
		ClipFilterDepthImage(ProcessorStream[streamIdx], i);							// 剪裁并对深度图进行双边滤波
		ClipNormalizeColorImage(ProcessorStream[streamIdx], i);							// 剪裁RGB图像，获得灰度图，并对灰度图剪裁
/**************************** 2个Camera 平均花费40us - 60us (每个相机使用一个流处理生成顶点及置信度Map、法线及半径Map)****************************/
		buildVertexConfidenceMap(ProcessorStream[streamIdx], i);						// 构建顶点及置信度Map
		buildNormalRadiusMap(ProcessorStream[streamIdx], i);							// 构建法线及半径Map
		buildColorTimeMap(FrameIndex, ProcessorStream[streamIdx], i);					// 构建Color-Time图
		SyncAndConvertForeground2Texture(ProcessorStream[streamIdx + 1], i);			// 同步分割流，并将分割流转成Texture存储【下面才有使用分割背景的地方】
		rigidSolver->setCamerasInitialSE3(i);
	}
	syncAllProcessorStream();			// 同步所有流

/**************************** 构造多视角前景约束接口并对上述构造的Texture进行约束 ****************************/
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

	// 函数内部阻塞同步
	MultiViewForegroundMaskConstriction(clipedDepthCUDATexture, vertexConfidenceTexture, normalRadiusTexture, colorTimeTexture, MaskConstriction, deviceCount, rawImageColsCliped, rawImageRowsCliped);
#endif // REBUILD_WITHOUT_BACKGROUND
	//Visualizer::DrawRawSegmentMask(0, MaskConstriction.foreground[0]);
	//Visualizer::DrawRawSegmentMask(1, MaskConstriction.foreground[1]); 
	//Visualizer::DrawRawSegmentMask(2, MaskConstriction.foreground[2]);
	//Visualizer::DrawPointCloud(vertexConfidenceTexture[1].texture);
	for (int i = 0; i < deviceCount; i++) {
		collectValidDepthSurfel(ProcessorStream[i], i);
	}
	syncAllProcessorStream();			// 同步所有流

/**************************** 构造多视角前景约束接口并对上述构造的Texture进行约束 ****************************/
	// 对齐的数据
	surfelsProcessor->MergeDenseSurfels(deviceCount, depthSurfelView, InitialCameraSE3, preAlignedSurfel);

	//这个放着未融合的数据
	preAlignedSurfelView = preAlignedSurfel.ArrayView();
	//可视化未融合的数据
	//Visualizer::DrawPointCloudWithNormal(preAlignedSurfel.Array());

	syncAllProcessorStream();			// 同步所有流

	//这是融合操作
	//mergeSurface->MergeAllSurfaces(depthSurfel);

	// 可视化融合后的数据
	//Visualizer::DrawPointCloudWithNormal(mergeSurface->GetMergedSurfelArray());

#ifdef DEBUG_RUNNING_INFO
	//printf("融合后的点数量 = %lld 个\n", mergeSurface->GetMergedSurfelArray().size());
#endif // DEBUG_RUNNING_INFO

	//融合的数据
	return preAlignedSurfelView;
}

void SparseSurfelFusion::FrameProcessor::ProcessCurrentFrame(CameraObservation& observation, size_t frameIndex)
{

	memset(&observation, 0, sizeof(observation));				// 初始化observation，不在多线程中初始化防止访存错误
	FrameIndex = frameIndex;

	// 每个线程执行多个Camera流，尽最大可能并行
	for (int i = 0; i < deviceCount; i++) {
		// 以引用的方式捕获observation，以值传递的方式捕获其他变量
		pool->AddTask([=, &observation]() { ProcessCurrentCameraImageTask(i, observation); });
	}
	while (numThreadsCompleted.load() != deviceCount) {
		std::this_thread::yield();		// 主线程等待
	}
	numThreadsCompleted.store(0);		// 清空计数标志位
	// 等待两个线程完成
	syncAllProcessorStream();			// 同步所有流


/**************************** 构造多视角前景约束接口并对上述构造的Texture进行约束 ****************************/
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
	// 函数内部阻塞同步
	MultiViewForegroundMaskConstriction(clipedDepthCUDATexture, vertexConfidenceTexture, normalRadiusTexture, colorTimeTexture, MaskConstriction, deviceCount, rawImageColsCliped, rawImageRowsCliped, ProcessorStream[0]);
#endif // REBUILD_WITHOUT_BACKGROUND
/**************************** 构造多视角前景约束接口并对上述构造的Texture进行约束 ****************************/

/**************************** 跨视角寻找匹配点并插值 ****************************/
	CrossViewEdgeCorrespondence::CrossViewMatchingInput input;
	CrossViewMatchingInterpolation::CrossViewInterpolationInput interInput;
	for (int i = 0; i < deviceCount; i++) {
		// 跨境匹配点
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
		// 跨视角插值点
		interInput.vertexMap[i] = vertexConfidenceTexture[i].texture;
		interInput.normalMap[i] = normalRadiusTexture[i].texture;
		interInput.colorMap[i] = colorTimeTexture[i].texture;
	}
	CrossViewMatching->SetCrossViewMatchingInput(input);
	CrossViewMatching->FindCrossViewEdgeMatchedPairs(ProcessorStream[0]);
	observation.crossCorrPairs = CrossViewMatching->GetCrossViewBackTracingCorrPairs();

	CrossViewInterpolation->SetCrossViewInterpolationInput(interInput, CrossViewMatching->GetCrossViewMatchingUniqueCorrPairs());
	CrossViewInterpolation->CrossViewInterpolateSurfels(ProcessorStream[0]);
/**************************** 跨视角寻找匹配点并插值 ****************************/

	syncAllProcessorStream();			// 同步所有流

	for (int i = 0; i < deviceCount; i++) {
		//这里也改了
		collectValidDepthSurfel(ProcessorStream[i], i);							// 收集有效的深度面元并存储到depthSurfel[]中
		//加上融合的map
		CollectObservationParameters(ProcessorStream[i], observation, i);		// 收集相机观察的参数
	}
	syncAllProcessorStream();			// 同步所有流

	//Visualizer::DrawRawSegmentMask(0, foregroundSegmenter[0]->getForegroundEdgeMaskTexture());
	//Visualizer::DrawRawSegmentMask(1, foregroundSegmenter[1]->getForegroundEdgeMaskTexture());
	//Visualizer::DrawRawSegmentMask(2, foregroundSegmenter[2]->getForegroundEdgeMaskTexture());
	// 对齐的数据
	surfelsProcessor->MergeDenseSurfels(deviceCount, depthSurfelView, InitialCameraSE3, preAlignedSurfel);
	//这个放着未融合的数据
	preAlignedSurfelView = preAlignedSurfel.ArrayView();


	//if (frameIndex == 95) {
	//	// 绘制跨视角的匹配点
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
	printf("OpticalFlow找到的匹配的像素点个数(%lld + %lld + %lld) = %lld \n", observation.correspondencePixelPairs[0].Size(), observation.correspondencePixelPairs[1].Size(), observation.correspondencePixelPairs[2].Size(), observation.correspondencePixelPairs[0].Size() + observation.correspondencePixelPairs[1].Size() + observation.correspondencePixelPairs[2].Size());
#else
	//printf("GPC找到的匹配的像素点个数(%lld + %lld + %lld) = %lld \n", observation.correspondencePixelPairs[0].Size(), observation.correspondencePixelPairs[1].Size(), observation.correspondencePixelPairs[2].Size(), observation.correspondencePixelPairs[0].Size() + observation.correspondencePixelPairs[1].Size() + observation.correspondencePixelPairs[2].Size());
#endif // USE_OPTICALFLOW_CORRESPONDENCE_PAIRS	printf("融合后的点数量：%lld个\n", mergeSurface->GetMergedSurfelsView().Size());
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

	//直接cuda映射版
	//clearMapSurfel(rawImageColsCliped, rawImageRowsCliped, mergeSurfelInterface, ProcessorStream[0]);
	//mapMergedDepthSurfel(mergeSurface->GetMergedSurfelsView(), mergeSurfelInterface, rawImageColsCliped, rawImageRowsCliped, ProcessorStream[0]);
	CHECKCUDA(cudaStreamSynchronize(ProcessorStream[0]));
	
	//CollectMergeObservationParameters(observation);

	// 查看匹配点
	//if (frameIndex == 600){
	//	//debugmask(foregroundSegmenter[0]->getPreviousMaskTexture(), foregroundSegmenter[1]->getPreviousMaskTexture(), foregroundSegmenter[2]->getPreviousMaskTexture(), "PreviousImage");
	//	//debugmask(foregroundSegmenter[0]->getClipedMaskTexture(), foregroundSegmenter[1]->getClipedMaskTexture(), foregroundSegmenter[2]->getClipedMaskTexture(), "CurrentImage");
	//	Visualizer::DrawImagePairCorrespondence(clipedNormalizeColorImagePrevious[1].texture, clipedNormalizeColorImage[1].texture, getCorrespondencePixelPair(1));
	//}
	
	//return mergeSurface->GetMergedSurfelsView();
}

void SparseSurfelFusion::FrameProcessor::CollectObservationParameters(cudaStream_t stream, CameraObservation& observation, const unsigned int CameraID)
{
	//保存原生深度面元映射 函数也改了
	observation.icpvertexConfidenceMap[CameraID] = getBackgroundVertexConfidenceTexture(CameraID);
	observation.icpnormalRadiusMap[CameraID] = getBackgroundNormalRadiusTexture(CameraID);
	// 用于可视化的原始深度图像
	observation.rawcolorTimeMap[CameraID] = getBackgroundColorTimeTexture(CameraID);
	observation.rawDepthImage[CameraID] = getRawDepthCUDATexture(CameraID);

	// 获得当前帧的信息
	observation.vertexConfidenceMap[CameraID] = getVertexConfidenceTexture(CameraID);
	observation.normalRadiusMap[CameraID] = getNormalRadiusTexture(CameraID);
	observation.colorTimeMap[CameraID] = getColorTimeTexture(CameraID);

	// 获取上一帧的信息
	observation.PreviousVertexConfidenceMap[CameraID] = getPreviousVertexConfidenceTexture(CameraID);
	observation.PreviousNormalRadiusMap[CameraID] = getPreviousNormalRadiusTexture(CameraID);

	//Reference域中的Geometry图像
	observation.filteredDepthImage[CameraID] = getClipedDepthCUDATexture(CameraID);


	//color 图以及密度图以及Color Time
	observation.normalizedRGBAMap[CameraID] = getClipedNormalizeColorImageTexture(CameraID);
	observation.normalizedRGBAPrevious[CameraID] = getClipedNormalizeColorImagePreviousTexture(CameraID);
	observation.grayScaleMap[CameraID] = getGrayScaleImageTexture(CameraID);
	observation.grayScaleGradientMap[CameraID] = getGrayScaleGradientTexture(CameraID);

	//前景mask
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
	//相关联的节点对
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
	//这三个必须单独在最后收集。
	for (int CameraID = 0; CameraID < deviceCount; CameraID++) {
		observation.vertexConfidenceMap[CameraID] = getVertexConfidenceTexture(CameraID);
		observation.normalRadiusMap[CameraID] = getNormalRadiusTexture(CameraID);
		observation.colorTimeMap[CameraID] = getColorTimeTexture(CameraID);
	}
}

void SparseSurfelFusion::FrameProcessor::initProcessorStream()
{
	//创建流
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
	//首先分配缓冲区
	unsigned int rawImageSize = rawImageRows * rawImageCols; // 每一个相机的图像都要存
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
			previousColorImage[CameraID] = colorImage[CameraID];	// 上一帧数据先存储
			previousDepthImage[CameraID] = depthImage[CameraID];	// 上一帧数据先存储
			colorImage[CameraID] = ColorOfflineData[CameraID][frameIndex];// 存储当前帧
			depthImage[CameraID] = DepthOfflineData[CameraID][frameIndex];// 存储当前帧
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
			previousColorImage[CameraID] = colorImage[CameraID].clone();	// 上一帧数据先存储
			previousDepthImage[CameraID] = depthImage[CameraID].clone();	// 上一帧数据先存储
			colorImage[CameraID] = cv::imread(path + "/color/" + colorFrameName, cv::IMREAD_ANYCOLOR);
			depthImage[CameraID] = cv::imread(path + "/depth/" + depthFrameName, cv::IMREAD_ANYDEPTH);

			//cv::imshow("Pre", previousColorImage[CameraID]);
			//cv::imshow("Curr", colorImage[CameraID]);
			//cv::waitKey(10000000);
		}
	}
	if (colorImage[CameraID].empty()) LOGGING(FATAL) << "colorImage图片读取失败";
	if (depthImage[CameraID].empty()) LOGGING(FATAL) << "depthImage图片读取失败";
	if (previousColorImage[CameraID].empty()) LOGGING(FATAL) << "previousColorImage图片读取失败";



	memcpy(colorBufferPagelock[CameraID], colorImage[CameraID].data, sizeof(uchar3) * rawImageRows * rawImageCols);
	memcpy(colorBufferPagelockPrevious[CameraID], previousColorImage[CameraID].data, sizeof(uchar3) * rawImageRows * rawImageCols);
	memcpy(depthBufferPagelock[CameraID], depthImage[CameraID].data, sizeof(unsigned short) * rawImageRows * rawImageCols);
	memcpy(depthBufferPagelockPrevious[CameraID], previousDepthImage[CameraID].data, sizeof(unsigned short) * rawImageRows * rawImageCols);
}

void SparseSurfelFusion::FrameProcessor::UploadDepthImageToGPU(cudaStream_t stream, int CameraID)
{
	void* ptr = rawDepthImage[CameraID].ptr();		// 获得RGB图像在GPU中地址，后续从锁页内存中拷入其中
	// 异步拷贝到rawColorImage[CameraID]中设置的GPU的地址
	CHECKCUDA(cudaMemcpyAsync(ptr, depthBufferPagelock[CameraID], sizeof(unsigned short) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));
	ptr = rawDepthImagePrevious[CameraID].ptr();	// 获得前一帧图像在GPU中的地址，后续从锁页内存中拷入其中
	CHECKCUDA(cudaMemcpyAsync(ptr, depthBufferPagelockPrevious[CameraID], sizeof(unsigned short) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));
	// 将锁页内存中的2D数据异步拷贝到cuda中
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
		// 原始图像应该使用rawImageRows,rawImageCols
		createDepthTextureSurface(rawImageRows, rawImageCols, rawDepthCUDATexture[i]);
		// 过滤后的图像应该使用rawImageRowsCliped,rawImageColsCliped
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
	void* ptr = rawColorImage[CameraID].ptr();		// 获得RGB图像在GPU中地址，后续从锁页内存中拷入其中
	// 异步拷贝到rawColorImage[CameraID]中设置的GPU的地址
	CHECKCUDA(cudaMemcpyAsync(ptr, colorBufferPagelock[CameraID], sizeof(uchar3) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));
	ptr = rawColorImagePrevious[CameraID].ptr();	// 获得前一帧图像在GPU中的地址，后续从锁页内存中拷入其中
	CHECKCUDA(cudaMemcpyAsync(ptr, colorBufferPagelockPrevious[CameraID], sizeof(uchar3) * rawImageCols * rawImageRows, cudaMemcpyHostToDevice, stream));

}

void SparseSurfelFusion::FrameProcessor::ClipNormalizeColorImage(cudaStream_t stream, int CameraID)
{
	// 剪裁彩色图像
	clipNormalizeColorImage(rawColorImage[CameraID], rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImage[CameraID].surface, GrayScaleImage[CameraID].surface, stream);
	// 剪裁灰度图像
	filterGrayScaleImage(GrayScaleImage[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, GrayScaleImageFiltered[CameraID].surface, stream);
	// 剪裁前一帧RGB图像
	clipNormalizeColorImage(rawColorImagePrevious[CameraID], rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImagePrevious[CameraID].surface, stream);
}

void SparseSurfelFusion::FrameProcessor::allocateColorTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		rawColorImage[i].create(size_t(rawImageCols) * rawImageRows);												// 给原始color图像分配GPU内存
		rawColorImagePrevious[i].create(size_t(rawImageCols) * rawImageRows);										// 给原始Previous color图像分配GPU内存
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImage[i]);			// 给clipedNormalizeColorImage[i]分配Float4类型的纹理(存RGB)
		createFloat4TextureSurface(rawImageRowsCliped, rawImageColsCliped, clipedNormalizeColorImagePrevious[i]);	// 给clipedNormalizeColorImagePrevious[i]分配Float4类型的纹理(存RGB)
		createFloat1TextureSurface(rawImageRowsCliped, rawImageColsCliped, GrayScaleImage[i]);						// 给GrayScaleImage[i]分配Float1类型的纹理(存灰度图)
		createFloat1TextureSurface(rawImageRowsCliped, rawImageColsCliped, GrayScaleImageFiltered[i]);				// 给GrayScaleImageFiltered[i]分配Float1类型的纹理(存灰度图)
	}
}

void SparseSurfelFusion::FrameProcessor::releaseColorTexture()
{
	for (int i = 0; i < deviceCount; i++) {
		rawColorImage[i].release();									// 释放rawColorImage在GPU中的内存
		rawColorImagePrevious[i].release();							// 释放rawColorImagePrevious在GPU中的内存
		releaseTextureCollect(clipedNormalizeColorImage[i]);		// 释放clipedNormalizeColorImage在GPU中存储的纹理及表面内存
		releaseTextureCollect(clipedNormalizeColorImagePrevious[i]);// 释放clipedNormalizeColorImagePrevious在GPU中存储的纹理及表面内存
		releaseTextureCollect(GrayScaleImage[i]);					// 释放GrayScaleImage在GPU中存储的纹理及表面内存
		releaseTextureCollect(GrayScaleImageFiltered[i]);			// 释放GrayScaleImageFiltered在GPU中存储的纹理及表面内存
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
	//上一帧的
	copyPreviousVertexAndNormal(
		previousVertexConfidenceTexture[CameraID].surface,
		previousNormalRadiusTexture[CameraID].surface,
		icp_vertex_confidence[CameraID].texture,
		icp_normal_radius[CameraID].texture,
		rawImageRowsCliped, 
		rawImageColsCliped
	);//把上一帧的数据从texture写到suface里
	const IntrinsicInverse clipIntrinsicInverse = inverse(clipColorIntrinsic[CameraID]);
	//这一帧的裁剪后的每一个深度像素点对应的相机坐标系下的三维点及其置信度（float4）
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
	//构造数组
	const unsigned int num_pixels = rawImageColsCliped * rawImageRowsCliped;
	DeviceArray<char> validIndicator = DeviceArray<char>(validDepthPixelSelector[CameraID].selectIndicatorBuffer.ptr(), num_pixels);
	//markValidDepthPixel(clipedDepthCUDATexture[CameraID].texture, foregroundSegmenter[CameraID]->getClipedMaskTexture(), icp_normal_radius[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, validDepthPixelSelector[CameraID].selectIndicatorBuffer, stream);
	markValidDepthPixel(clipedDepthCUDATexture[CameraID].texture, rawImageRowsCliped, rawImageColsCliped, validDepthPixelSelector[CameraID].selectIndicatorBuffer, stream);

	//筛选有效的深度像素点
	validDepthPixelSelector[CameraID].Select(validIndicator, stream);

	//构造输出
	const size_t selected_surfel_size = validDepthPixelSelector[CameraID].validSelectedIndex.size();
	depthSurfel[CameraID].ResizeArrayOrException(selected_surfel_size);

	//参数汇总
	DeviceArray<DepthSurfel> valid_surfel_array = depthSurfel[CameraID].Array();//地址传递
	collectDepthSurfel(icp_vertex_confidence[CameraID].texture, icp_normal_radius[CameraID].texture, icp_color_time[CameraID].texture, validDepthPixelSelector[CameraID].validSelectedIndex, rawImageRowsCliped, rawImageColsCliped, CameraID, valid_surfel_array, stream);
	depthSurfelView[CameraID] = DeviceArrayView<DepthSurfel>(valid_surfel_array.ptr(), valid_surfel_array.size());
}


void SparseSurfelFusion::FrameProcessor::alignedDenseValidDepthSurfel(RigidSolver::Ptr solver, cudaStream_t stream, int CameraID)
{
	unsigned int offset = 0;								// 存入preAlignedSurfel的偏差
	size_t totalDenseSurfelNum = depthSurfelView[0].Size();	// 总共开辟preAlignedSurfel多大的Array空间
	// 计算存入Canonical空间的偏差
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
	auto start = std::chrono::high_resolution_clock::now();// 获取程序开始时间点

	float inferMemorySize = 0;
	for (int i = 0; i < deviceCount; i++) {
		size_t memorySize;
		foregroundSegmenter[i] = std::make_shared<ForegroundSegmenter>(memorySize);
		foregroundSegmenter[i]->AllocateInferBuffer();
		inferMemorySize += memorySize * 1.0f;
	}
	auto end = std::chrono::high_resolution_clock::now();// 获取程序结束时间点
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// 计算程序运行时间（毫秒）
	std::cout << "推理RVM模型共耗费显存大小：" << inferMemorySize / (1024.0f * 1024.0f) << "MB     用时：" << duration / 1000.0f << " 秒" << std::endl;
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
	cudaTextureObject_t curEdgeMask = getCurrentForegroundEdgeMaskTexture(CameraID);	// 当前帧前景边缘Mask

	cudaTextureObject_t previousVertexMap = getPreviousVertexConfidenceTexture(CameraID);
	cudaTextureObject_t currentVertexMap = getBackgroundVertexConfidenceTexture(CameraID);
					    
	cudaTextureObject_t previousNormalMap = getPreviousNormalRadiusTexture(CameraID);
	cudaTextureObject_t currentNormalMap = getBackgroundNormalRadiusTexture(CameraID);
	featureCorrespondenceFinder[CameraID]->SetInputImage(rgbPrevious, rgbCurrent, foregroundPrevious, foregroundCurrent, curEdgeMask, previousVertexMap, currentVertexMap, previousNormalMap, currentNormalMap, CameraID, FrameIndex);
	featureCorrespondenceFinder[CameraID]->FindCorrespondence(stream);
}

void SparseSurfelFusion::FrameProcessor::allocateOpticalFlowBuffer()
{
	auto start = std::chrono::high_resolution_clock::now();// 获取程序开始时间点
	float inferMemorySize = 0;
	for (int i = 0; i < deviceCount; i++) {
		size_t memorySize;
		opticalFlow[i] = std::make_shared<OpticalFlow>(memorySize);
		opticalFlow[i]->AllocateInferBuffer();
		inferMemorySize += memorySize * 1.0f;
	}
	auto end = std::chrono::high_resolution_clock::now();// 获取程序结束时间点
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// 计算程序运行时间（毫秒）

	std::cout << "推理GMFlow模型共耗费显存大小：" << inferMemorySize / (1024.0f * 1024.0f) << "MB     用时：" << duration / 1000.0f << " 秒" << std::endl;

}

void SparseSurfelFusion::FrameProcessor::releaseOpticalFlowBuffer()
{
	for (int i = 0; i < deviceCount; i++) {
		opticalFlow[i]->ReleaseInferBuffer();
	}
}

void SparseSurfelFusion::FrameProcessor::computeOpticalFlow(cudaStream_t stream, const unsigned int CameraID)
{
	auto start = std::chrono::high_resolution_clock::now();								// 获取程序开始时间点
	
	cudaTextureObject_t preForeground = getPreviousForegroundMaskTexture(CameraID);		// 前一帧的前景
	cudaTextureObject_t curForeground = getForegroundMaskTexture(CameraID);				// 当前帧的前景
	cudaTextureObject_t curEdgeMask = getCurrentForegroundEdgeMaskTexture(CameraID);	// 当前帧前景边缘Mask

	cudaTextureObject_t preVertexMap = getPreviousVertexConfidenceTexture(CameraID);	// 前一帧的VertexMap
	cudaTextureObject_t curVertexMap = getBackgroundVertexConfidenceTexture(CameraID);	// 当前帧的VertexMap
	cudaTextureObject_t preNormalexMap = getPreviousNormalRadiusTexture(CameraID);		// 前一帧的VertexMap
	cudaTextureObject_t curNormalMap = getBackgroundNormalRadiusTexture(CameraID);		// 当前帧的VertexMap
	opticalFlow[CameraID]->setForegroundAndVertexMapTexture(preForeground, curForeground, preVertexMap, curVertexMap, preNormalexMap, curNormalMap, curEdgeMask, InitialCameraSE3[CameraID]);
	opticalFlow[CameraID]->InferOpticalFlow(rawColorImagePrevious[CameraID], rawDepthImagePrevious[CameraID], rawColorImage[CameraID], rawDepthImage[CameraID], stream);
	
	auto end = std::chrono::high_resolution_clock::now();// 获取程序结束时间点
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();// 计算程序运行时间（毫秒）
#ifdef DEBUG_RUNNING_INFO
	std::cout << "计算光流处理：" << duration << " 毫秒" << std::endl;
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


