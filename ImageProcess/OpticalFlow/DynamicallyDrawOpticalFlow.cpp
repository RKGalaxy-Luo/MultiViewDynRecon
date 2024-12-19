#include "DynamicallyDrawOpticalFlow.h"

SparseSurfelFusion::DynamicallyDrawOpticalFlow::DynamicallyDrawOpticalFlow()
{
	if (!glfwInit()) {
		LOGGING(FATAL) << "GLFW����ʧ�ܣ�";
	}

	// opengl�����ģ�GLFW�汾Ϊ4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Ĭ�ϵ�framebuffer����
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE);		// ���ڲ��ɼ�
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);	// ���ڴ�С���ɵ���

	// ���ڵ�����
	std::string windowsName = std::string("OpticalFlowWindow");
	OpticalFlowWindow = glfwCreateWindow(WindowWidth, WindowHeight, windowsName.c_str(), NULL, NULL);
	if (OpticalFlowWindow == NULL) {
		LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�";
	}
	//else std::cout << "���� " + windowsName + " ������ɣ�" << std::endl;

	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	// ��ʼ��glad
	if (!gladLoadGL()) {
		LOGGING(FATAL) << "GLAD ��ʼ��ʧ��!";
	}

	// ������Ȳ���, ���á����޳�������
	glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
	glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
	glDisable(GL_CULL_FACE);				// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
	glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С

	const std::string vertexPath = SHADER_PATH_PREFIX + std::string("OpticalFlowShader.vert");
	const std::string fragmentPath = SHADER_PATH_PREFIX + std::string("OpticalFlowShader.frag");
	OpticalFlowShader.Compile(vertexPath, fragmentPath);

	const std::string LivePointsVertexPath = SHADER_PATH_PREFIX + std::string("LivePointsVertexShader.vert");
	const std::string LivePointsFragmentPath = SHADER_PATH_PREFIX + std::string("LivePointsFragmentShader.frag");
	ColorVertexShader.Compile(LivePointsVertexPath, LivePointsFragmentPath);

	allocateBuffer();

	registerFlowCudaResources();
	registerVertexCudaResources();

	initialCoordinateSystem();
}

SparseSurfelFusion::DynamicallyDrawOpticalFlow::~DynamicallyDrawOpticalFlow()
{
	// ע��OpenGL������ע����cuda�������е���Դ
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaVBOResources[0]));
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaVBOResources[1]));

	glDeleteVertexArrays(1, &OpticalFlowVAO);
	glDeleteBuffers(1, &OpticalFlowVBO);
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::registerFlowCudaResources()
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	/*********************   ���ɲ���VAO������   *********************/
	glGenVertexArrays(1, &OpticalFlowVAO);	// ����VAO
	glBindVertexArray(OpticalFlowVAO);		// ��VAO
/*********************   ��RenderedFusedPoints��һ������������(������ GPU �ϴ洢�͹������ݵ� OpenGL ����)   *********************/
	// ����1�����ɻ���������
	// ����2���������ַ�����������Ա㽫���ɵı�ʶ���洢�ڸñ�����
	glGenBuffers(1, &OpticalFlowVBO);
/*******   ��֮ǰ���ɵĻ���������󶨵��������Ի�����Ŀ�� GL_ARRAY_BUFFER �ϣ�������һ����С���ڴ棬���ڴ洢������������   *******/
	// ����1��GL_ARRAY_BUFFER �ǻ�����Ŀ�꣬��ʾ����������������ڴ洢������������
	// ����2��֮ǰ���ɵĻ���������ı�ʶ����ͨ������������ã����û����������� GL_ARRAY_BUFFER Ŀ����а�
	glBindBuffer(GL_ARRAY_BUFFER, OpticalFlowVBO);
	// ����1��GL_ARRAY_BUFFER ��ʾ��ǰ�󶨵Ļ�����������һ���������Ի�����
	// ����2��6 * sizeof(float) * Constants::maxSurfelsNum ��ʾҪ������ڴ��С��Coordinate + RGB = 6 * float
	// ����3��NULL ��ʾ��ʱ���ṩ����
	// ����4��GL_DYNAMIC_DRAW ��ʾ������������󽫱�Ƶ���޸ĺ�ʹ��
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 2 * ImageSize, NULL, GL_DYNAMIC_DRAW);
/*******************    ����VAO������    *******************/
	// λ��
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO
/*******   ����Ҫ��CUDA��OpenGL֮��������ݽ���ʱ������ʹ��cudaGraphicsGLRegisterBuffer������OpenGL����������ע��ΪCUDAͼ����Դ   *******/
/*******************   �������Ϳ����� CUDA ��������ʹ�� CUDA API �����ʺͲ����û��������󣬶�������ʽ�ؽ������ݿ���   *******************/
	// ����1������ע���CUDAͼ����Դ���������OpenGL�л����������Ӧ��cuda��Դ�������ֱ�Ӷ����cuda��Դ���в�����
	// ����2��Ҫע��� OpenGL ����������ı�ʶ�����������ʶ����Ӧ��OpenGL��bufferע�ᵽcuda�����ġ�
	// ����3����ѡ����������ָ��ע���־����ʾע�� CUDA ͼ����Դʱ��ʹ���κα�־����ע�� CUDA ͼ����Դʱ��Ӧ���ض�����Ϊ���޸�Ĭ����Ϊ��
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&cudaVBOResources[0], OpticalFlowVBO, cudaGraphicsRegisterFlagsNone));
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::registerVertexCudaResources()
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	glGenVertexArrays(1, &ColorVertexVAO);	// ����VAO
	glBindVertexArray(ColorVertexVAO);		// ��VAO

	glGenBuffers(1, &ColorVertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, ColorVertexVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(ColorVertex) * 2 * ImageSize, NULL, GL_DYNAMIC_DRAW);

	// λ��
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// ��ɫ
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));	// ����VAO������
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO

	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&cudaVBOResources[1], ColorVertexVBO, cudaGraphicsRegisterFlagsNone));
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::initialCoordinateSystem()
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

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

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::allocateBuffer()
{

}



void SparseSurfelFusion::DynamicallyDrawOpticalFlow::OpticalFlowMapToCuda(float3* validFlow, cudaStream_t stream)
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

///** ֱ�Ӱ�FusedLivePointsVBO�������е�������� **/
//	glBindBuffer(GL_ARRAY_BUFFER, OpticalFlowVBO);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * ImageSize * 2, NULL, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);

/** ʹ��cudaGraphicsMapResources��������Դӳ�䵽������ **/
	// ����1��Ҫӳ��� CUDA ͼ����Դ��������
	// ����2��ָ�� CUDA ͼ����Դ�����ָ�롣
	// ����3����ѡ����������ָ��Ҫ������ִ��ӳ������� CUDA ����Ĭ��ֵΪ 0����ʾʹ��Ĭ������
	CHECKCUDA(cudaGraphicsMapResources(1, &cudaVBOResources[0], stream));	//����ӳ����Դ

	// ���buffer
	void* ptr;			// ���ڻ�ȡcuda��Դ�ĵ�ַ(�ظ�ʹ��)
	size_t bufferSize;	// ���ڻ�ȡcuda��Դbuffer�Ĵ�С
	// ���OpenGL�ϵ���Դָ��
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&ptr, &bufferSize, cudaVBOResources[0]));
	// ֱ�ӽ����ݿ�������Դָ����
	CHECKCUDA(cudaMemcpyAsync(ptr, validFlow, sizeof(float3) * ValidNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::LivePointsMapToCuda(ColorVertex* validVertex, cudaStream_t stream)
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	CHECKCUDA(cudaGraphicsMapResources(1, &cudaVBOResources[1], stream));	//����ӳ����Դ
	// ���buffer
	void* ptr;			// ���ڻ�ȡcuda��Դ�ĵ�ַ(�ظ�ʹ��)
	size_t bufferSize;	// ���ڻ�ȡcuda��Դbuffer�Ĵ�С
	// ���OpenGL�ϵ���Դָ��
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&ptr, &bufferSize, cudaVBOResources[1]));
	// ֱ�ӽ����ݿ�������Դָ����
	CHECKCUDA(cudaMemcpyAsync(ptr, validVertex, sizeof(ColorVertex) * ValidNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::UnmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(1, &cudaVBOResources[0], stream));
	CHECKCUDA(cudaGraphicsUnmapResources(1, &cudaVBOResources[1], stream));
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::drawOpticalFlow(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	OpticalFlowShader.BindProgram();

	//����͸�Ӿ���
	projection = glm::perspective(glm::radians(30.0f), (float)WindowWidth / (float)WindowHeight, 0.1f, 100.0f);
	OpticalFlowShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	//float radius = 2.5f;//����ͷ�Ƶİ뾶
	//float camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
	//float camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
	//view = glm::lookAt(glm::vec3(camX, 3.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	view = glm::lookAt(glm::vec3(1.5f, 1.5f, 1.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	OpticalFlowShader.setUniformMat4(std::string("view"), view);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//������(1,1,1)��ת
	OpticalFlowShader.setUniformMat4(std::string("model"), model);

	glBindVertexArray(OpticalFlowVAO); // ��VAO�����
	glLineWidth(3.0f);
	glDrawArrays(GL_LINES, 0, ValidNum);
	glBindVertexArray(0);	// �����
	OpticalFlowShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::drawColorVertex(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	ColorVertexShader.BindProgram();
	ColorVertexShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	ColorVertexShader.setUniformMat4(std::string("view"), view);
	ColorVertexShader.setUniformMat4(std::string("model"), model);
	glBindVertexArray(ColorVertexVAO); // ��VAO�����

	glPointSize(3.0f);
	glDrawArrays(GL_POINTS, 0, ValidNum);	// box��54��Ԫ�أ������߶�

	// �����
	glBindVertexArray(0);
	ColorVertexShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	// ��������ϵ
	coordinateShader.BindProgram();	// ���������shader
	coordinateShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	coordinateShader.setUniformMat4(std::string("view"), view);
	coordinateShader.setUniformMat4(std::string("model"), model);
	glBindVertexArray(coordinateSystemVAO); // ��VAO�����

	glLineWidth(2.0f);
	glDrawArrays(GL_LINES, 0, 34);	// box��54��Ԫ�أ������߶�

	// �����
	glBindVertexArray(0);
	coordinateShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::clearWindow()
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	// ������glClearColor�����������Ļ���õ���ɫ
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f); //RGBA
	// ͨ������glClear�����������Ļ����ɫ���壬������һ������λ(Buffer Bit)��ָ��Ҫ��յĻ��壬���ܵĻ���λ��GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// ����ͬʱ�����Ȼ�����!(�������Ȼ�����������ͼ��)
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::swapBufferAndCatchEvent()
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������

	// �����ύ����ɫ���壨����һ��������GLFW����ÿһ��������ɫֵ�Ĵ󻺳壩��������һ�����б��������ƣ����ҽ�����Ϊ�����ʾ����Ļ��
	glfwSwapBuffers(OpticalFlowWindow);
	glfwPollEvents();
}

void SparseSurfelFusion::DynamicallyDrawOpticalFlow::imshow(float3* validFlow, ColorVertex* validColorVertex, const unsigned int validFlowNum, cudaStream_t stream)
{
	glfwMakeContextCurrent(OpticalFlowWindow);	// �뵱ǰ����������������
	ValidNum = validFlowNum;
	// �����Χ��
	adjustModelPosition(validFlow, validColorVertex, stream);
	adjustPointsCoordinate(validFlow, validColorVertex, stream);
	OpticalFlowMapToCuda(validFlow, stream);	// ��ý�OpenGL��Դָ�룬���ԭ��VBO����������Ⱦ�����ݴ���VBO
	LivePointsMapToCuda(validColorVertex, stream);
	clearWindow();
	drawOpticalFlow(view, projection, model);				// ���ƹ���
	drawColorVertex(view, projection, model);				// ����RGB����
	drawCoordinateSystem(view, projection, model);			// ��������ϵ
	swapBufferAndCatchEvent();								// ˫���岢��׽�¼�
	UnmapFromCuda(stream);
}

