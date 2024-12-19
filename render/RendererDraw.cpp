/*****************************************************************//**
 * \file   RendererDraw.cpp
 * \brief  ���벢��ʼ�������Ժ��ͼ��������ɫ�����򣬱�����Ⱦ��ͼƬ��������Ⱦ���ͼƬ
 * 
 * \author LUO
 * \date   February 23rd 2024
 *********************************************************************/
#include "Renderer.h"
SparseSurfelFusion::Camera camera(glm::vec3(0.0f, 0.0f, -4.0f));
float lastX = 600.0 / 2.0f;
float lastY = 360.0 / 2.0f;
bool firstMouse = true;
// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;
//��ü��̵��¼�
void processInput(GLFWwindow* window)
{
	//�����̵ġ�ESCAPE�����£���ʹ��glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS
	//�����̵ġ�ESCAPE��û�а��£�glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_RELEASE
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		// ͨ��glfwSetwindowShouldCloseʹ�ð�WindowShouldClose��������Ϊ true�ķ����ر�GLFW��
		// ��һ��whileѭ����������⽫��ʧ�ܣ����򽫻�رա�
		glfwSetWindowShouldClose(window, true);
	}

	// ��������ͷ�ӽ�
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::AXISROTATION_X, deltaTime);//��X����ת
	if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::AXISROTATION_Y, deltaTime);//��Y����ת
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
		camera.ProcessKeyboard(SparseSurfelFusion::Camera::CameraMovement::AXISROTATION_Z, deltaTime);//��Z����ת
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);															// �����ӿ�
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouseCallback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void SparseSurfelFusion::Renderer::initProcessingShaders()
{
	glfwMakeContextCurrent(GLFW_Window);	// �뵱ǰ����������������
	//����fusionMap����ɫ��
	const std::string fusion_map_vert_path = SHADER_PATH_PREFIX + std::string("FusionMap.vert");
	const std::string fusion_map_frag_path = SHADER_PATH_PREFIX + std::string("FusionMap.frag");
	fusionMapShader.Compile(fusion_map_vert_path, fusion_map_frag_path);

	const std::string fusion_map_indexmap_vert_path = SHADER_PATH_PREFIX + std::string("FusionMapIndexMap.vert");
	const std::string fusion_map_indexmap_frag_path = SHADER_PATH_PREFIX + std::string("FusionMapIndexMap.frag");
	fusionMapShaderIndexMap.Compile(fusion_map_indexmap_vert_path, fusion_map_indexmap_frag_path);

	//����solverMap����ɫ��
	const std::string solver_map_frag_path = SHADER_PATH_PREFIX + std::string("SolverMap.frag");

	//�Ƿ�ʹ���ܼ������ӳ��
#if defined(USE_DENSE_SOLVER_MAPS)
	const std::string solver_map_recent_vert_path = SHADER_PATH_PREFIX + std::string("SolverMapSized.vert");
#else
	const std::string solver_map_recent_vert_path = SHADER_PATH_PREFIX + std::string("SolverMap.vert");
#endif
	solverMapShader.Compile(solver_map_recent_vert_path, solver_map_frag_path);

	const std::string solverMapShaderIndexmap_vert_path = SHADER_PATH_PREFIX + std::string("SolverMapIndexMap.vert");
	const std::string solverMapShaderIndexmap_frag_path = SHADER_PATH_PREFIX + std::string("SolverMapIndexMap.frag");
	solverMapShaderIndexMap.Compile(solverMapShaderIndexmap_vert_path, solverMapShaderIndexmap_frag_path);
}

void SparseSurfelFusion::Renderer::initVisualizationShaders()
{
	glfwMakeContextCurrent(GLFW_Window);	// �뵱ǰ����������������
	// ���ߣ�phong��Ӱ��albedo��ɫ��Ƭ����ɫ��
	const std::string normal_map_frag_path = SHADER_PATH_PREFIX + std::string("NormalAsColor.frag");
	const std::string phong_shading_path = SHADER_PATH_PREFIX + std::string("PhongColor.frag");
	const std::string albedo_color_path = SHADER_PATH_PREFIX + std::string("AlbedoColor.frag");

	// Canonical���Live�����Ԫ���εĶ�����ɫ��
	const std::string geometry_vert_path = SHADER_PATH_PREFIX + std::string("Geometry.vert");
	// ����û������۲����ɫ��
	visualizationShaders.normalMap.Compile(geometry_vert_path, normal_map_frag_path);
	visualizationShaders.phongMap.Compile(geometry_vert_path, phong_shading_path);
	visualizationShaders.albedoMap.Compile(geometry_vert_path, albedo_color_path);


	const std::string coorVertSystemPath = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.vert");
	const std::string coorFragSystemPath = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.frag");
	visualizationShaders.coordinateSystem.Compile(coorVertSystemPath, coorFragSystemPath);
	glGenVertexArrays(1, &(coordinateSystemVAO));	// ����VAO
	glGenBuffers(1, &(coordinateSystemVBO));		// ����VBO
	const unsigned int Num = sizeof(box) / sizeof(box[0]);
	std::vector<float> pvalues;			// ������	
	for (int i = 0; i < Num; i++) {
		pvalues.push_back(box[i][0]);
		pvalues.push_back(box[i][1]);
		pvalues.push_back(box[i][2]);
	}
	glBindVertexArray(coordinateSystemVAO);
	glBindBuffer(GL_ARRAY_BUFFER, coordinateSystemVBO);
	GLsizei bufferSize = sizeof(GLfloat) * pvalues.size();		// float���ݵ�����
	glBufferData(GL_ARRAY_BUFFER, bufferSize, pvalues.data(), GL_STATIC_DRAW);	// ��̬���ƣ�Ŀǰֻ���ȿ��ٸ���С

	// λ��
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// ��ɫ
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO
}

void SparseSurfelFusion::Renderer::initShaders()
{
	initProcessingShaders();
	initVisualizationShaders();
}

void SparseSurfelFusion::Renderer::drawVisualizationMap(GLShaderProgram& shader, GLuint geometry_vao, unsigned num_vertex, int current_time, const Matrix4f& world2camera, bool with_recent_observation)
{
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

	//����͸�Ӿ���
	glm::mat4 projection = glm::perspective(glm::radians(30.0f), (float)imageWidth / (float)imageHeight, 0.1f, 100.0f);
	float radius = 3.5f;
	float camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
	float camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
	glm::mat4 view = glm::lookAt(glm::vec3(camX, 0.0f, camZ + 0.9f), glm::vec3(0.0f, 0.0f, 0.9f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));// ����ģ�Ͳ���ת


	//��С��ͼ��Ĵ�С
	glBindFramebuffer(GL_FRAMEBUFFER, visualizationDrawBuffers.visualizationMapFBO);
	glViewport(0, 0, imageWidth, imageHeight);

	//����ɫ��
	shader.BindProgram();

	//ʹ���ṩ��vao
	glBindVertexArray(geometry_vao);

	//�����Ⱦ�ڴ����
	glClearBufferfv(GL_COLOR, 0, clearValues.visualizeRGBAClear);
	glClearBufferfv(GL_DEPTH, 0, &(clearValues.zBufferClear));

	//����ͳһֵ
	shader.SetUniformMatrix("world2camera", world2camera);
	shader.SetUniformVector("intrinsic", rendererIntrinsic);

	//solver map�ĵ�ǰʱ��
	const float4 width_height_maxdepth_currtime = make_float4(widthHeightMaxMinDepth.x, widthHeightMaxMinDepth.y, widthHeightMaxMinDepth.z, current_time);
	shader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);

	//ʱ����ֵȡ��������
	float2 confid_time_threshold = make_float2(Constants::kStableSurfelConfidenceThreshold, Constants::kRenderingRecentTimeThreshold);
	if (!with_recent_observation) {
		confid_time_threshold.y = -1.0f; //���û�й۲쵽�κ���Ԫ
	}

	//Hand in to shader
	shader.SetUniformVector("confid_time_threshold", confid_time_threshold);

	shader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	shader.setUniformMat4(std::string("view"), view);
	shader.setUniformMat4(std::string("model"), model);

	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);


	glBindVertexArray(0);
	shader.UnbindProgram();


	//visualizationShaders.coordinateSystem.BindProgram();
	////��С��ͼ��Ĵ�С
	//glBindFramebuffer(GL_FRAMEBUFFER, visualizationDrawBuffers.visualizationMapFBO);
	//visualizationShaders.coordinateSystem.setUniformMat4(std::string("projection"), projection);
	//visualizationShaders.coordinateSystem.setUniformMat4(std::string("view"), view);
	//visualizationShaders.coordinateSystem.setUniformMat4(std::string("model"), model);
	//glBindVertexArray(coordinateSystemVAO); // ��VAO�����

	//glLineWidth(3.0f);
	//glDrawArrays(GL_LINES, 0, 34);	// box��54��Ԫ�أ������߶�

	//// �����
	//glBindVertexArray(0);
	//visualizationShaders.coordinateSystem.UnbindProgram();

	//Cleanup code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SparseSurfelFusion::Renderer::ShowLiveNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent)
{
	const GLuint geometry_vao = liveGeometryVAO[vao_idx];
	drawVisualizationMap(visualizationShaders.normalMap, geometry_vao, num_vertex, current_time, world2camera, with_recent);
	visualizationDrawBuffers.show(GLOfflineVisualizationFrameRenderBufferObjects::LiveNormalMap);
}

void SparseSurfelFusion::Renderer::ShowLiveAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent)
{
	const GLuint geometry_vao = liveGeometryVAO[vao_idx];
	drawVisualizationMap(visualizationShaders.albedoMap, geometry_vao, num_vertex, current_time, world2camera, with_recent);
	visualizationDrawBuffers.show(GLOfflineVisualizationFrameRenderBufferObjects::LiveAlbedoMap);
}

void SparseSurfelFusion::Renderer::ShowLivePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent)
{
	const GLuint geometry_vao = liveGeometryVAO[vao_idx];
	drawVisualizationMap(visualizationShaders.phongMap, geometry_vao, num_vertex, current_time, world2camera, with_recent);
	visualizationDrawBuffers.show(GLOfflineVisualizationFrameRenderBufferObjects::LivePhongMap);
}

void SparseSurfelFusion::Renderer::ShowReferenceNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent)
{
	const GLuint geometry_vao = canonicalGeometryVAO[vao_idx];
	drawVisualizationMap(visualizationShaders.normalMap, geometry_vao, num_vertex, current_time, world2camera, with_recent);
	visualizationDrawBuffers.show(GLOfflineVisualizationFrameRenderBufferObjects::CanonicalNormalMap);
}

void SparseSurfelFusion::Renderer::ShowReferenceAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent)
{
	const GLuint geometry_vao = canonicalGeometryVAO[vao_idx];
	drawVisualizationMap(visualizationShaders.albedoMap, geometry_vao, num_vertex, current_time, world2camera, with_recent);
	visualizationDrawBuffers.show(GLOfflineVisualizationFrameRenderBufferObjects::CanonicalAlbedoMap);
}

void SparseSurfelFusion::Renderer::ShowReferencePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent)
{
	const GLuint geometry_vao = canonicalGeometryVAO[vao_idx];
	drawVisualizationMap(visualizationShaders.phongMap, geometry_vao, num_vertex, current_time, world2camera, with_recent);
	visualizationDrawBuffers.show(GLOfflineVisualizationFrameRenderBufferObjects::CanonicalPhongMap);
}

void SparseSurfelFusion::Renderer::DrawFusionMap(const unsigned int num_vertex, int vao_idx, const mat34* world2camera)
{
	for (int i = 0; i < deviceCount; i++) {
		if (i == 0) DrawZeroViewFusionMap(num_vertex, vao_idx, toEigen(world2camera[i]));
		else {
			DrawOtherViewsFusionMaps(i, num_vertex, vao_idx, toEigen(world2camera[i]));
		}
	}
}

void SparseSurfelFusion::Renderer::DrawZeroViewFusionMap(const unsigned int num_vertex, int vao_idx, const Matrix4f& world2camera)
{
	//Bind the shader
	fusionMapShader.BindProgram();

	//The vao/vbo for the rendering
	vao_idx = vao_idx % 2;
	glBindVertexArray(fusionMapVAO[0][vao_idx]);

	//The framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, fusionMapBuffers[0].fusionMapFBO);
	glViewport(0, 0, fusionImageWidth, fusionImageHeight);

	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, clearValues.vertexMapClear);
	glClearBufferfv(GL_COLOR, 1, clearValues.normalMapClear);
	glClearBufferuiv(GL_COLOR, 2, &(clearValues.indexMapClear));
	glClearBufferfv(GL_COLOR, 3, clearValues.colorTimeClear);
	glClearBufferfv(GL_DEPTH, 0, &(clearValues.zBufferClear));

	//Set the uniform values
	fusionMapShader.SetUniformMatrix("world2camera", world2camera);
	fusionMapShader.SetUniformVector("intrinsic", ACameraRendererIntrinsic[0]);
	fusionMapShader.SetUniformVector("width_height_maxdepth", widthHeightMaxMinDepth);

	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	//Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	fusionMapShader.UnbindProgram();
}

void SparseSurfelFusion::Renderer::DrawOtherViewsFusionMaps(const unsigned int CameraID, const unsigned int num_vertex, int vao_idx, const Matrix4f& world2camera)
{
	//Bind the shader
	fusionMapShaderIndexMap.BindProgram();

	//The vao/vbo for the rendering
	vao_idx = vao_idx % 2;
	glBindVertexArray(fusionMapVAO[CameraID][vao_idx]);

	//The framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, fusionMapBuffers[CameraID].fusionMapFBO);
	glViewport(0, 0, fusionImageWidth, fusionImageHeight);

	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, clearValues.vertexMapClear);
	glClearBufferfv(GL_COLOR, 1, clearValues.normalMapClear);
	glClearBufferuiv(GL_COLOR, 2, &(clearValues.indexMapClear));
	glClearBufferfv(GL_COLOR, 3, clearValues.colorTimeClear);
	glClearBufferfv(GL_DEPTH, 0, &(clearValues.zBufferClear));

	//Set the uniform values
	fusionMapShaderIndexMap.SetUniformMatrix("world2camera", world2camera);
	fusionMapShaderIndexMap.SetUniformMatrix("initialCameraSE3Inverse", toEigen(InitialCameraSE3[CameraID].inverse()));
	fusionMapShaderIndexMap.SetUniformVector("intrinsic", ACameraRendererIntrinsic[CameraID]);
	fusionMapShaderIndexMap.SetUniformVector("width_height_maxdepth", widthHeightMaxMinDepth);

	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	//Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	fusionMapShaderIndexMap.UnbindProgram();
}

void SparseSurfelFusion::Renderer::DrawSolverMapsWithRecentObservation(
	unsigned num_vertex,
	int vao_idx,
	int current_time,
	const Matrix4f* world2camera
) {
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
	for (int i = 0; i < deviceCount; i++) {
		if (i == 0) drawZeroViewSolverMapsIndexMap(num_vertex, vao_idx, current_time, world2camera[i], true);
		else drawOtherViewsSolverMapsIndexMap(i, num_vertex, vao_idx, current_time, world2camera[i], true);
	}
	
	
}

void SparseSurfelFusion::Renderer::DrawSolverMapsConfidentObservation(
	unsigned num_vertex,
	int vao_idx,
	int current_time, // Not used
	const Matrix4f* world2camera
) {
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������
	for (int i = 0; i < deviceCount; i++) {
		if (i == 0) drawZeroViewSolverMapsIndexMap(num_vertex, vao_idx, current_time, world2camera[i], false);
		else drawOtherViewsSolverMapsIndexMap(i, num_vertex, vao_idx, current_time, world2camera[i], false);
	}
}


void SparseSurfelFusion::Renderer::drawZeroViewSolverMapsIndexMap(const unsigned int num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent_observation)
{
	//Bind the shader
	solverMapShader.BindProgram();

	//Normalize the vao index
	vao_idx = vao_idx % 2;
	glBindVertexArray(solverMapVAO[0][vao_idx]);

	//The size is image rows/cols
	glBindFramebuffer(GL_FRAMEBUFFER, solverMapBuffers[0].solverMapFBO);
	glViewport(0, 0, imageWidth, imageHeight);

	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, clearValues.vertexMapClear);
	glClearBufferfv(GL_COLOR, 1, clearValues.normalMapClear);
	glClearBufferfv(GL_COLOR, 2, clearValues.vertexMapClear);
	glClearBufferfv(GL_COLOR, 3, clearValues.normalMapClear);
	glClearBufferuiv(GL_COLOR, 4, &(clearValues.indexMapClear));
	glClearBufferfv(GL_COLOR, 5, clearValues.solverRGBAClear);
	glClearBufferfv(GL_DEPTH, 0, &(clearValues.zBufferClear));

	// ����Ҫ��������indexmap����Ҫ������ɫ����
	//Set uniform values
	solverMapShader.SetUniformMatrix("world2camera", world2camera);
	solverMapShader.SetUniformVector("intrinsic", ACameraRendererIntrinsic[0]);

	//The current time of the solver maps
	float4 width_height_maxdepth_currtime = make_float4(widthHeightMaxMinDepth.x, widthHeightMaxMinDepth.y, widthHeightMaxMinDepth.z, current_time);
	solverMapShader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);

	//The time threshold depend on input
	float2 confid_time_threshold = make_float2(SparseSurfelFusion::Constants::kStableSurfelConfidenceThreshold, SparseSurfelFusion::Constants::kRenderingRecentTimeThreshold);
	if (!with_recent_observation) {
		confid_time_threshold.y = -1.0f; //Do not pass any surfel due to recent observed
	}

	solverMapShader.SetUniformVector("confid_time_threshold", confid_time_threshold);

	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	//Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	solverMapShader.UnbindProgram();
}

//1�������draw
void SparseSurfelFusion::Renderer::drawOtherViewsSolverMapsIndexMap(
	const unsigned int CameraID,
	const unsigned int num_vertex,
	int vao_idx,
	int current_time,
	const Matrix4f& world2camera,
	bool with_recent_observation
) {
	//Bind the shader
	solverMapShaderIndexMap.BindProgram();

	//Normalize the vao index
	vao_idx = vao_idx % 2;
	glBindVertexArray(solverMapVAO[CameraID][vao_idx]);


	//The size is image rows/cols
	glBindFramebuffer(GL_FRAMEBUFFER, solverMapBuffers[CameraID].solverMapFBO);
	glViewport(0, 0, imageWidth, imageHeight);

	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, clearValues.vertexMapClear);
	glClearBufferfv(GL_COLOR, 1, clearValues.normalMapClear);
	glClearBufferfv(GL_COLOR, 2, clearValues.vertexMapClear);
	glClearBufferfv(GL_COLOR, 3, clearValues.normalMapClear);
	glClearBufferuiv(GL_COLOR, 4, &(clearValues.indexMapClear));
	glClearBufferfv(GL_COLOR, 5, clearValues.solverRGBAClear);
	glClearBufferfv(GL_DEPTH, 0, &(clearValues.zBufferClear));

	solverMapShaderIndexMap.SetUniformMatrix("world2camera", world2camera);
	solverMapShaderIndexMap.SetUniformMatrix("initialCameraSE3Inverse", toEigen(InitialCameraSE3[CameraID].inverse()));
	solverMapShaderIndexMap.SetUniformVector("intrinsic", ACameraRendererIntrinsic[CameraID]);

	//The current time of the solver maps
	float4 width_height_maxdepth_currtime = make_float4(widthHeightMaxMinDepth.x, widthHeightMaxMinDepth.y, widthHeightMaxMinDepth.z, current_time);
	solverMapShaderIndexMap.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);

	//The time threshold depend on input
	float2 confid_time_threshold = make_float2(Constants::kStableSurfelConfidenceThreshold, SparseSurfelFusion::Constants::kRenderingRecentTimeThreshold);
	if (!with_recent_observation) {
		confid_time_threshold.y = -1.0f; //Do not pass any surfel due to recent observed
	}

	solverMapShaderIndexMap.SetUniformVector("confid_time_threshold", confid_time_threshold);

	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	//Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	solverMapShaderIndexMap.UnbindProgram();
}

