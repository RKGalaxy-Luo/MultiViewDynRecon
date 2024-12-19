/*****************************************************************//**
 * \file   Renderer.cpp
 * \brief  渲染器对象，主要是显示和渲染
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#include "Renderer.h"

SparseSurfelFusion::Renderer::Renderer(int imageRows, int imageCols, ConfigParser::Ptr config) : imageHeight(imageRows), imageWidth(imageCols), fusionImageHeight(imageRows * Constants::superSampleScale), fusionImageWidth(imageCols * Constants::superSampleScale), FusionMapScale(Constants::superSampleScale)
{
	if (!glfwInit()) {
		LOGGING(FATAL) << "GLFW加载失败！";
	}
	deviceCount = config->getDeviceCount();
	for (int i = 0; i < deviceCount; i++) {
		if (i == 0) { 
			rendererIntrinsic = config->getClipColorIntrinsic(i);
			widthHeightMaxMinDepth = make_float4(imageWidth, imageHeight, config->getMaxDepthMeter(), config->getMinDepthMeter());
		}
		ACameraRendererIntrinsic[i] = config->getClipColorIntrinsic(i);
		InitialCameraSE3[i] = Constants::GetInitialCameraSE3(i);
	}

	// 初始化
	initGLFW();
	initClearValues();
	initVertexBufferObjects();
	initMapRenderVAO();
	initFrameRenderBuffers();
	initShaders();
	initFilteredSolverMapAndFusionMap();
}

SparseSurfelFusion::Renderer::~Renderer()
{	
	freeVertexBufferObjects();
	freeFrameRenderBuffers();
	releaseFilteredSolverMapAndFusionMap();
}


void SparseSurfelFusion::Renderer::initGLFW()
{
	// opengl上下文，GLFW版本为4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// 默认的framebuffer属性
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE); // 窗口大小不可调整

	// 窗口的设置
	GLFW_Window = glfwCreateWindow(imageWidth, imageHeight, "Windows", NULL, NULL);

	if (GLFW_Window == NULL) {
		LOGGING(FATAL) << "未正确创建GLFW窗口！";
	}

	// 创建新的当前上下文
	glfwMakeContextCurrent(GLFW_Window);

	// 初始化glad
	if (!gladLoadGL()) {
		LOGGING(FATAL) << "glad未正确初始化";
	}

	// 开启深度测试, 禁用“面剔除”功能
	glEnable(GL_DEPTH_TEST);				// 启用深度测试后，OpenGL会在绘制像素之前，根据它们的深度值进行比较，并只绘制深度测试通过的像素，从而产生正确的渲染效果
	glDepthFunc(GL_LESS);					// 用于深度测试，它决定哪些片段（像素）应该被显示，哪些应该被丢弃，基于它们的深度值
	glDisable(GL_CULL_FACE);				// 意味着OpenGL将渲染所有的三角形面，而不管它们的顶点顺序，不管其是否被遮挡
	glEnable(GL_PROGRAM_POINT_SIZE);		// 调用 glEnable(GL_PROGRAM_POINT_SIZE) 函数会启用程序控制的点大小功能，允许您在着色器程序中使用内置变量 gl_PointSize 来控制点的大小
	//glEnable(GL_POINT_SPRITE);			// (现在已经无需打开了，被设置成永久打开的了)调用 glEnable(GL_POINT_SPRITE) 函数会启用点精灵功能，允许您在绘制点时使用点精灵纹理
	// 点精灵是一种渲染技术，用于在渲染点时以纹理的形式呈现每个点。它允许您将一个图像或纹理应用于每个点，从而实现各种效果，如粒子系统、精灵效果等。

	for (int i = 0; i < deviceCount; i++) {
		//fusiondepthVBO[i].GLFW_Window = GLFW_Window;		// 需要在调用OpenGL资源的时候链接上下文
		surfelGeometryVBOs[i][0].GLFW_Window = GLFW_Window;		// 需要在调用OpenGL资源的时候链接上下文
		surfelGeometryVBOs[i][1].GLFW_Window = GLFW_Window;		// 需要在调用OpenGL资源的时候链接上下文
	}
}

void SparseSurfelFusion::Renderer::initFilteredSolverMapAndFusionMap()
{
	initSolverMapsFilteredIndexMap();
	initFusionMapsFilteredIndexMap();
}

void SparseSurfelFusion::Renderer::releaseFilteredSolverMapAndFusionMap()
{
	releaseSolverMapsFilteredIndexMap();
	releaseFusionMapsFilteredIndexMap();
}

void SparseSurfelFusion::Renderer::initSolverMapsFilteredIndexMap()
{
	for (int i = 0; i < deviceCount; i++) {
		createUnsignedTextureSurface(imageHeight, imageWidth, solverMapFilteredIndexMap[i]);
	}
	singleViewTotalSurfels.create(deviceCount);
	singleViewFilteredSurfels.create(deviceCount);
}

void SparseSurfelFusion::Renderer::releaseSolverMapsFilteredIndexMap()
{
	for (int i = 0; i < deviceCount; i++) {
		releaseTextureCollect(solverMapFilteredIndexMap[i]);
	}
	singleViewTotalSurfels.release();
	singleViewFilteredSurfels.release();
}

void SparseSurfelFusion::Renderer::initFusionMapsFilteredIndexMap()
{
	for (int i = 0; i < deviceCount; i++) {
		createUnsignedTextureSurface(fusionImageHeight, fusionImageWidth, fusionMapFilteredIndexMap[i]);
	}
}

void SparseSurfelFusion::Renderer::releaseFusionMapsFilteredIndexMap()
{
	for (int i = 0; i < deviceCount; i++) {
		releaseTextureCollect(fusionMapFilteredIndexMap[i]);
	}
}

void SparseSurfelFusion::Renderer::initClearValues()
{
	clearValues.initialize();
}

void SparseSurfelFusion::Renderer::initVertexBufferObjects()
{
	for (int i = 0; i < deviceCount; i++) {
		initializeGLSurfelGeometry(surfelGeometryVBOs[i][0]);
		initializeGLSurfelGeometry(surfelGeometryVBOs[i][1]);
	}
}

void SparseSurfelFusion::Renderer::freeVertexBufferObjects()
{
	for (int i = 0; i < deviceCount; i++) {
		releaseGLSurfelGeometry(surfelGeometryVBOs[i][0]);
		releaseGLSurfelGeometry(surfelGeometryVBOs[i][1]);
	}
}


void SparseSurfelFusion::Renderer::MapSolverMapsToCuda(Renderer::SolverMaps* maps, cudaStream_t stream) {
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	for (int i = 0; i < deviceCount; i++) {
		solverMapBuffers[i].mapToCuda(
			maps[i].reference_vertex_map,
			maps[i].reference_normal_map,
			maps[i].warp_vertex_map,		//没找到这俩warp的值在哪里赋的
			maps[i].warp_normal_map,		//应该是live里的模型点
			maps[i].index_map,
			maps[i].normalized_rgb_map,
			stream
		);
	}
}

void SparseSurfelFusion::Renderer::UnmapSolverMapsFromCuda(cudaStream_t stream) {
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	for (int i = 0; i < deviceCount; i++) {
		solverMapBuffers[i].unmapFromCuda(stream);
	}
}




void SparseSurfelFusion::Renderer::MapSurfelGeometryToCUDA(int idx, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], int cameraID, cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	idx = idx % 2;
	if (cameraID < 0 || cameraID >= deviceCount) {
		for (int i = 0; i < deviceCount; i++) {
			surfelGeometryVBOs[i][idx].mapToCuda(*(geometry[i][idx]), stream);
		}
	}
	else {
		surfelGeometryVBOs[cameraID][idx].mapToCuda(*(geometry[cameraID][idx]), stream);
	}
}

void SparseSurfelFusion::Renderer::MapSurfelGeometryToCUDA(int idx, int cameraID, cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	idx = idx % 2;
	if (cameraID < 0 || cameraID >= deviceCount) {
		for (int i = 0; i < deviceCount; i++) {
			surfelGeometryVBOs[i][idx].mapToCuda(stream);
		}
	}
	else {
		surfelGeometryVBOs[cameraID][idx].mapToCuda(stream);
	}

}

void SparseSurfelFusion::Renderer::UnmapSurfelGeometryFromCUDA(int idx, int cameraID, cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	idx = idx % 2;
	if (cameraID < 0 || cameraID >= deviceCount) {
		for (int i = 0; i < deviceCount; i++) {
			surfelGeometryVBOs[i][idx].unmapFromCuda(stream);
		}
	}
	else {
		surfelGeometryVBOs[cameraID][idx].unmapFromCuda(stream);
	}
}

void SparseSurfelFusion::Renderer::initMapRenderVAO()
{
	for (int i = 0; i < deviceCount; i++) {
		buildFusionMapVAO(surfelGeometryVBOs[i][0], fusionMapVAO[i][0]);
		buildFusionMapVAO(surfelGeometryVBOs[i][1], fusionMapVAO[i][1]);

		buildSolverMapVAO(surfelGeometryVBOs[i][0], solverMapVAO[i][0]);
		buildSolverMapVAO(surfelGeometryVBOs[i][1], solverMapVAO[i][1]);

		buildCanonicalGeometryVAO(surfelGeometryVBOs[i][0], canonicalGeometryVAO[0]);
		buildCanonicalGeometryVAO(surfelGeometryVBOs[i][1], canonicalGeometryVAO[1]);

		buildLiveGeometryVAO(surfelGeometryVBOs[i][0], liveGeometryVAO[0]);
		buildLiveGeometryVAO(surfelGeometryVBOs[i][1], liveGeometryVAO[1]);
	}
}

void SparseSurfelFusion::Renderer::initFrameRenderBuffers()
{
	for (int i = 0; i < deviceCount; i++) {
		fusionMapBuffers[i].initialize(fusionImageWidth, fusionImageHeight);
		solverMapBuffers[i].initialize(imageWidth, imageHeight);
		//fusionDepthSurfelBuffers[i].initialize(imageWidth, imageHeight);
	}
	visualizationDrawBuffers.initialize(imageWidth, imageHeight);
}

void SparseSurfelFusion::Renderer::freeFrameRenderBuffers()
{
	for (int i = 0; i < deviceCount; i++) {
		fusionMapBuffers[i].release();
		solverMapBuffers[i].release();
		//fusionDepthSurfelBuffers[i].release();
	}
	visualizationDrawBuffers.release();
}

void SparseSurfelFusion::Renderer::MapFusionMapsToCuda(FusionMaps* maps, cudaStream_t stream) {
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	for (int i = 0; i < deviceCount; i++) {
		fusionMapBuffers[i].mapToCuda(
			maps[i].warp_vertex_map,
			maps[i].warp_normal_map,
			maps[i].index_map,
			maps[i].color_time_map,
			stream
		);
	}
}

void SparseSurfelFusion::Renderer::UnmapFusionMapsFromCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文
	for (int i = 0; i < deviceCount; i++) {
		fusionMapBuffers[i].unmapFromCuda(stream);
	}
}
