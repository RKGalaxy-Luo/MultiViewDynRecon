/*****************************************************************//**
 * \file   DynamicallyDrawPoints.cpp
 * \brief  ��̬���Ƶ�ķ���
 * 
 * \author LUOJIAXUAN
 * \date   June 2024
 *********************************************************************/
#include "DynamicallyDrawPoints.h"

std::atomic<bool> CheckEscPress;

 // �����ص�����
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE){
		if (key == GLFW_KEY_ESCAPE) {
			CheckEscPress.store(true);	// ֻ���ͷŲ���λ
		}
	}
}

SparseSurfelFusion::DynamicallyDrawPoints::DynamicallyDrawPoints()
{
	initialDynamicRendererAndAllocateBuffer();
}

SparseSurfelFusion::DynamicallyDrawPoints::~DynamicallyDrawPoints()
{
	RenderedFusedPoints.ReleaseBuffer();
	perBlockMaxPoint.DeviceArray().release();
	perBlockMinPoint.DeviceArray().release();

	// ע��OpenGL������ע����cuda�������е���Դ
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaVBOResources));
	glDeleteVertexArrays(1, &FusedLivePointsVAO);
	glDeleteBuffers(1, &FusedLivePointsVBO);
	glDeleteBuffers(1, &FusedLivePointsIBO);
}

void SparseSurfelFusion::DynamicallyDrawPoints::initialDynamicRendererAndAllocateBuffer()
{
	RenderedFusedPoints.AllocateBuffer(MAX_SURFEL_COUNT);
	perBlockMaxPoint.AllocateBuffer(divUp(MAX_SURFEL_COUNT, device::CudaThreadsPerBlock));
	perBlockMinPoint.AllocateBuffer(divUp(MAX_SURFEL_COUNT, device::CudaThreadsPerBlock));

	if (!glfwInit()) {
		LOGGING(FATAL) << "GLFW����ʧ�ܣ�";
	}

	// ��ʼ��glad
	if (!gladLoadGL()) {
		LOGGING(FATAL) << "GLAD ��ʼ��ʧ��!";
	}

	// opengl�����ģ�GLFW�汾Ϊ4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Ĭ�ϵ�framebuffer����
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE);		// ���ڿɼ�
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);	// ���ڴ�С���ɵ���

	// ���ڵ�����
	std::string windowsName = std::string("FusedLivePoints");
	LiveWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowsName.c_str(), NULL, NULL);
	if (LiveWindow == NULL) {
		LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�";
	}
	//else std::cout << "���� " + windowsName + " ������ɣ�" << std::endl;

	glfwMakeContextCurrent(LiveWindow);	// �뵱ǰ����������������
	CheckEscPress.store(false);			// ��ʼ��
	// ���ð����ص�
	glfwSetKeyCallback(LiveWindow, keyCallback);

	// ������Ȳ���, ���á����޳�������
	glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
	glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
	glDisable(GL_CULL_FACE);				// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
	glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С

	// �鿴�ںϺ��Shader�Ķ�����ɫ��
	const std::string live_points_vert_path = SHADER_PATH_PREFIX + std::string("LivePointsVertexShader.vert");
	// �鿴�ںϺ��Shader��Ƭ����ɫ��
	const std::string live_points_frag_path = SHADER_PATH_PREFIX + std::string("LivePointsFragmentShader.frag");

	FusedLivePointsShader.Compile(live_points_vert_path, live_points_frag_path);

	registerCudaResources();	// ע��cuda��Դ

	initialCoordinateSystem();	// ��ʼ��OpenGL������

}

void SparseSurfelFusion::DynamicallyDrawPoints::registerCudaResources()
{
	glfwMakeContextCurrent(LiveWindow);	// �뵱ǰ����������������

/*********************   ���ɲ���VAO������   *********************/
	glGenVertexArrays(1, &FusedLivePointsVAO);	// ����VAO
	glBindVertexArray(FusedLivePointsVAO);		// ��VAO
/*********************   ��RenderedFusedPoints��һ������������(������ GPU �ϴ洢�͹������ݵ� OpenGL ����)   *********************/
	// ����1�����ɻ���������
	// ����2���������ַ�����������Ա㽫���ɵı�ʶ���洢�ڸñ�����
	glGenBuffers(1, &FusedLivePointsVBO);
/*******   ��֮ǰ���ɵĻ���������󶨵��������Ի�����Ŀ�� GL_ARRAY_BUFFER �ϣ�������һ����С���ڴ棬���ڴ洢������������   *******/
	// ����1��GL_ARRAY_BUFFER �ǻ�����Ŀ�꣬��ʾ����������������ڴ洢������������
	// ����2��֮ǰ���ɵĻ���������ı�ʶ����ͨ������������ã����û����������� GL_ARRAY_BUFFER Ŀ����а�
	glBindBuffer(GL_ARRAY_BUFFER, FusedLivePointsVBO);
	// ����1��GL_ARRAY_BUFFER ��ʾ��ǰ�󶨵Ļ�����������һ���������Ի�����
	// ����2��6 * sizeof(float) * Constants::maxSurfelsNum ��ʾҪ������ڴ��С��Coordinate + RGB = 6 * float
	// ����3��NULL ��ʾ��ʱ���ṩ����
	// ����4��GL_DYNAMIC_DRAW ��ʾ������������󽫱�Ƶ���޸ĺ�ʹ��
	glBufferData(GL_ARRAY_BUFFER, sizeof(Renderer::RenderedPoints) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
/*******************    ����VAO������    *******************/
	// λ��
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedPoints), (void*)(0 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// ��ɫ
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedPoints), (void*)(3 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO
/*******   ����Ҫ��CUDA��OpenGL֮��������ݽ���ʱ������ʹ��cudaGraphicsGLRegisterBuffer������OpenGL����������ע��ΪCUDAͼ����Դ   *******/
/*******************   �������Ϳ����� CUDA ��������ʹ�� CUDA API �����ʺͲ����û��������󣬶�������ʽ�ؽ������ݿ���   *******************/
	// ����1������ע���CUDAͼ����Դ���������OpenGL�л����������Ӧ��cuda��Դ�������ֱ�Ӷ����cuda��Դ���в�����
	// ����2��Ҫע��� OpenGL ����������ı�ʶ�����������ʶ����Ӧ��OpenGL��bufferע�ᵽcuda�����ġ�
	// ����3����ѡ����������ָ��ע���־����ʾע�� CUDA ͼ����Դʱ��ʹ���κα�־����ע�� CUDA ͼ����Դʱ��Ӧ���ض�����Ϊ���޸�Ĭ����Ϊ��
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&cudaVBOResources, FusedLivePointsVBO, cudaGraphicsRegisterFlagsNone));
}

void SparseSurfelFusion::DynamicallyDrawPoints::initialCoordinateSystem()
{
	glfwMakeContextCurrent(LiveWindow);	// �뵱ǰ����������������

	std::vector<float> pvalues;			// ������	
	// Live���еĵ㣬�鿴�м���̵Ķ�����ɫ��
	const std::string coordinate_vert_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.vert");
	// Live���еĵ㣬�鿴�м���̵�Ƭ����ɫ��
	const std::string coordinate_frag_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.frag");
	coordinateShader.Compile(coordinate_vert_path, coordinate_frag_path);
	glGenVertexArrays(1, &(coordinateSystemVAO));	// ����VAO
	glGenBuffers(1, &(coordinateSystemVBO));		// ����VBO
	const unsigned int Num = sizeof(box) / sizeof(box[0]);
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

void SparseSurfelFusion::DynamicallyDrawPoints::DrawLiveFusedPoints(DeviceArrayView<float4> SolvedPointsCoor, DeviceArrayView<float4> SolvedPointsColor, cudaStream_t stream)
{
	glfwMakeContextCurrent(LiveWindow);	// �����µĵ�ǰ������

	// �����Χ��
	float3 MaxPoint = make_float3(float(-1e6), float(-1e6), float(-1e6));
	float3 MinPoint = make_float3(float(1e6), float(1e6), float(1e6));
	adjustModelPosition(SolvedPointsCoor, stream);
	getBoundingBox(RenderedFusedPoints.ArrayView(), MaxPoint, MinPoint, stream);
	// �����ܵ�ȫ����������һ���޲�������ɫ������ֵ����RenderedFusedPoints
	adjustPointsCoordinateAndColor(SolvedPointsColor, MaxPoint, MinPoint, stream);
	mapToCuda(stream);	// ��ý�OpenGL��Դָ�룬���ԭ��VBO����������Ⱦ�����ݴ���VBO
	while (true) {
		if (CheckEscPress.load() == true) {				// ����Ƿ���ESC
			CheckEscPress.store(false);					// ��ʼ��
			break; 
		}	
		clearWindow();									// �����Ļ
		drawLivePoints(view, projection, model);		// ���Ƶ�
		drawCoordinateSystem(view, projection, model);	// ��������ϵ
		swapBufferAndCatchEvent();						// ˫���岢��׽�¼�
	}
	unmapFromCuda(stream);
}

void SparseSurfelFusion::DynamicallyDrawPoints::DrawLiveFusedPoints(DeviceArrayView<DepthSurfel> SolvedPoints, bool CheckPeriod, cudaStream_t stream)
{
	glfwMakeContextCurrent(LiveWindow);	// �����µĵ�ǰ������
	CheckSpecificFrame = CheckPeriod;
	// �����Χ��
	float3 MaxPoint = make_float3(float(-1e6), float(-1e6), float(-1e6));
	float3 MinPoint = make_float3(float(1e6), float(1e6), float(1e6));
	//printf("SolvedPointsCount = %d\n", SolvedPoints.Size());
	adjustModelPosition(SolvedPoints, stream);
	getBoundingBox(RenderedFusedPoints.ArrayView(), MaxPoint, MinPoint, stream);
	adjustPointsCoordinateAndColor(SolvedPoints, MaxPoint, MinPoint, stream);

	mapToCuda(stream);	// ��ý�OpenGL��Դָ�룬���ԭ��VBO����������Ⱦ�����ݴ���VBO
	if (CheckSpecificFrame) {
		while (true) {
			if (CheckEscPress.load() == true) {				// ����Ƿ���ESC
				CheckEscPress.store(false);					// ��ʼ��
				CheckSpecificFrame = false;
				break;
			}
			clearWindow();		// �����Ļ
			drawLivePoints(view, projection, model);		// ���Ƶ�
			drawCoordinateSystem(view, projection, model);	// ��������ϵ
			swapBufferAndCatchEvent();						// ˫���岢��׽�¼�
		}
	}
	else {
		clearWindow();		// �����Ļ
		drawLivePoints(view, projection, model);		// ���Ƶ�
		drawCoordinateSystem(view, projection, model);	// ��������ϵ
		swapBufferAndCatchEvent();						// ˫���岢��׽�¼�
	}

	unmapFromCuda(stream);
}

void SparseSurfelFusion::DynamicallyDrawPoints::mapToCuda(cudaStream_t stream)
{
	const size_t validSurfelsNum = RenderedFusedPoints.ArraySize();		// ��Ч����Ԫ����

///** ֱ�Ӱ�FusedLivePointsVBO�������е�������� **/
//	glBindBuffer(GL_ARRAY_BUFFER, FusedLivePointsVBO);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(Renderer::RenderedPoints) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);

/** ʹ��cudaGraphicsMapResources��������Դӳ�䵽������ **/
	// ����1��Ҫӳ��� CUDA ͼ����Դ��������
	// ����2��ָ�� CUDA ͼ����Դ�����ָ�롣
	// ����3����ѡ����������ָ��Ҫ������ִ��ӳ������� CUDA ����Ĭ��ֵΪ 0����ʾʹ��Ĭ������
	CHECKCUDA(cudaGraphicsMapResources(1, &cudaVBOResources, stream));	//����ӳ����Դ

	// ���buffer
	void* ptr;			// ���ڻ�ȡcuda��Դ�ĵ�ַ(�ظ�ʹ��)
	size_t bufferSize;	// ���ڻ�ȡcuda��Դbuffer�Ĵ�С
	// ���OpenGL�ϵ���Դָ��
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&ptr, &bufferSize, cudaVBOResources));

	// ֱ�ӽ����ݿ�������Դָ����
	CHECKCUDA(cudaMemcpyAsync(ptr, RenderedFusedPoints.Ptr(), sizeof(Renderer::RenderedPoints) * validSurfelsNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::DynamicallyDrawPoints::unmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(1, &cudaVBOResources, stream));
}

void SparseSurfelFusion::DynamicallyDrawPoints::clearWindow()
{
	// ������glClearColor�����������Ļ���õ���ɫ
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //RGBA
	// ͨ������glClear�����������Ļ����ɫ���壬������һ������λ(Buffer Bit)��ָ��Ҫ��յĻ��壬���ܵĻ���λ��GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// ����ͬʱ�����Ȼ�����!(�������Ȼ�����������ͼ��)
}

void SparseSurfelFusion::DynamicallyDrawPoints::drawLivePoints(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{

	const unsigned int totalPointsNum = RenderedFusedPoints.ArraySize();
	FusedLivePointsShader.BindProgram();

	//����͸�Ӿ���
	projection = glm::perspective(glm::radians(30.0f), (float)(WINDOW_WIDTH) / (float)(WINDOW_HEIGHT), 0.1f, 100.0f);
	FusedLivePointsShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�

	if (CheckSpecificFrame) {
		camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
		camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
		view = glm::lookAt(glm::vec3(camX, 0.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else {
		view = glm::lookAt(glm::vec3(camX, 0.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	FusedLivePointsShader.setUniformMat4(std::string("view"), view);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//������(1,1,1)��ת
	FusedLivePointsShader.setUniformMat4(std::string("model"), model);

	glBindVertexArray(FusedLivePointsVAO); // ��VAO�����
	glDrawArrays(GL_POINTS, 0, totalPointsNum);
	glBindVertexArray(0);	// �����
	FusedLivePointsShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyDrawPoints::drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	// ��������ϵ
	coordinateShader.BindProgram();	// ���������shader
	coordinateShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	coordinateShader.setUniformMat4(std::string("view"), view);
	coordinateShader.setUniformMat4(std::string("model"), model);
	glBindVertexArray(coordinateSystemVAO); // ��VAO�����

	glLineWidth(3.0f);
	glDrawArrays(GL_LINES, 0, 34);	// box��54��Ԫ�أ������߶�

	// �����
	glBindVertexArray(0);
	coordinateShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyDrawPoints::swapBufferAndCatchEvent()
{
	// �����ύ����ɫ���壨����һ��������GLFW����ÿһ��������ɫֵ�Ĵ󻺳壩��������һ�����б��������ƣ����ҽ�����Ϊ�����ʾ����Ļ��
	glfwSwapBuffers(LiveWindow);
	glfwPollEvents();
}
