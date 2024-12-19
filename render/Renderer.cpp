/*****************************************************************//**
 * \file   Renderer.cpp
 * \brief  ��Ⱦ��������Ҫ����ʾ����Ⱦ
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#include "Renderer.h"

SparseSurfelFusion::Renderer::Renderer(int imageRows, int imageCols, ConfigParser::Ptr config) : imageHeight(imageRows), imageWidth(imageCols), fusionImageHeight(imageRows * Constants::superSampleScale), fusionImageWidth(imageCols * Constants::superSampleScale), FusionMapScale(Constants::superSampleScale)
{
	if (!glfwInit()) {
		LOGGING(FATAL) << "GLFW����ʧ�ܣ�";
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

	// ��ʼ��
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
	// opengl�����ģ�GLFW�汾Ϊ4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Ĭ�ϵ�framebuffer����
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE); // ���ڴ�С���ɵ���

	// ���ڵ�����
	GLFW_Window = glfwCreateWindow(imageWidth, imageHeight, "Windows", NULL, NULL);

	if (GLFW_Window == NULL) {
		LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�";
	}

	// �����µĵ�ǰ������
	glfwMakeContextCurrent(GLFW_Window);

	// ��ʼ��glad
	if (!gladLoadGL()) {
		LOGGING(FATAL) << "gladδ��ȷ��ʼ��";
	}

	// ������Ȳ���, ���á����޳�������
	glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
	glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
	glDisable(GL_CULL_FACE);				// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
	glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С
	//glEnable(GL_POINT_SPRITE);			// (�����Ѿ�������ˣ������ó����ô򿪵���)���� glEnable(GL_POINT_SPRITE) ���������õ㾫�鹦�ܣ��������ڻ��Ƶ�ʱʹ�õ㾫������
	// �㾫����һ����Ⱦ��������������Ⱦ��ʱ���������ʽ����ÿ���㡣����������һ��ͼ�������Ӧ����ÿ���㣬�Ӷ�ʵ�ָ���Ч����������ϵͳ������Ч���ȡ�

	for (int i = 0; i < deviceCount; i++) {
		//fusiondepthVBO[i].GLFW_Window = GLFW_Window;		// ��Ҫ�ڵ���OpenGL��Դ��ʱ������������
		surfelGeometryVBOs[i][0].GLFW_Window = GLFW_Window;		// ��Ҫ�ڵ���OpenGL��Դ��ʱ������������
		surfelGeometryVBOs[i][1].GLFW_Window = GLFW_Window;		// ��Ҫ�ڵ���OpenGL��Դ��ʱ������������
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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
	for (int i = 0; i < deviceCount; i++) {
		solverMapBuffers[i].mapToCuda(
			maps[i].reference_vertex_map,
			maps[i].reference_normal_map,
			maps[i].warp_vertex_map,		//û�ҵ�����warp��ֵ�����︳��
			maps[i].warp_normal_map,		//Ӧ����live���ģ�͵�
			maps[i].index_map,
			maps[i].normalized_rgb_map,
			stream
		);
	}
}

void SparseSurfelFusion::Renderer::UnmapSolverMapsFromCuda(cudaStream_t stream) {
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
	for (int i = 0; i < deviceCount; i++) {
		solverMapBuffers[i].unmapFromCuda(stream);
	}
}




void SparseSurfelFusion::Renderer::MapSurfelGeometryToCUDA(int idx, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], int cameraID, cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
	for (int i = 0; i < deviceCount; i++) {
		fusionMapBuffers[i].unmapFromCuda(stream);
	}
}
