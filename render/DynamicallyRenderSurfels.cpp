#include "DynamicallyRenderSurfels.h"

std::atomic<bool> CheckNextPress;

// �����ص�����
void RenderKeyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		if (key == GLFW_KEY_RIGHT) {
			CheckNextPress.store(true);	// ֻ���ͷŲ���λ
		}
	}
}

#pragma pack(2)//Ӱ���ˡ����롱������ʵ��ǰ�� sizeof(BITMAPFILEHEADER) �Ĳ��

typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned long  DWORD;
typedef long    LONG;

//BMP�ļ�ͷ��
struct BITMAPFILEHEADER
{
	WORD  bfType;		//�ļ����ͱ�ʶ������ΪASCII�롰BM��
	DWORD bfSize;		//�ļ��ĳߴ磬��byteΪ��λ
	WORD  bfReserved1;	//�����֣�����Ϊ0
	WORD  bfReserved2;	//�����֣�����Ϊ0
	DWORD bfOffBits;	//һ����byteΪ��λ��ƫ�ƣ���BITMAPFILEHEADER�ṹ�忪ʼ��λͼ����
};

//BMP��Ϣͷ��
struct BITMAPINFOHEADER
{
	DWORD biSize;			//����ṹ��ĳߴ�
	LONG  biWidth;			//λͼ�Ŀ��
	LONG  biHeight;			//λͼ�ĳ���
	WORD  biPlanes;			//The number of planes for the target device. This value must be set to 1.
	WORD  biBitCount;		//һ�������м�λ
	DWORD biCompression;    //0����ѹ����1��RLE8��2��RLE4
	DWORD biSizeImage;      //4�ֽڶ����ͼ�����ݴ�С
	LONG  biXPelsPerMeter;  //������/�ױ�ʾ��ˮƽ�ֱ���
	LONG  biYPelsPerMeter;  //������/�ױ�ʾ�Ĵ�ֱ�ֱ���
	DWORD biClrUsed;        //ʵ��ʹ�õĵ�ɫ����������0��ʹ�����еĵ�ɫ������
	DWORD biClrImportant;	//��Ҫ�ĵ�ɫ����������0�����еĵ�ɫ����������Ҫ
};

//һ�����ص���ɫ��Ϣ
struct RGBColor
{
	char B;		//��
	char G;		//��
	char R;		//��
};

//����ɫ����д��һ��BMP�ļ���
//FileName:�ļ���
//ColorBuffer:��ɫ����
//ImageWidth:ͼ����
//ImageHeight:ͼ�񳤶�
void WriteBMP(const char* FileName, RGBColor* ColorBuffer, int ImageWidth, int ImageHeight)
{
	//��ɫ�����ܳߴ磺
	const int ColorBufferSize = ImageHeight * ImageWidth * sizeof(RGBColor);

	//�ļ�ͷ
	BITMAPFILEHEADER fileHeader;
	fileHeader.bfType = 0x4D42;	//0x42��'B'��0x4D��'M'
	fileHeader.bfReserved1 = 0;
	fileHeader.bfReserved2 = 0;
	fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + ColorBufferSize;
	fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	//��Ϣͷ
	BITMAPINFOHEADER bitmapHeader = { 0 };
	bitmapHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapHeader.biHeight = ImageHeight;
	bitmapHeader.biWidth = ImageWidth;
	bitmapHeader.biPlanes = 1;
	bitmapHeader.biBitCount = 24;
	bitmapHeader.biSizeImage = ColorBufferSize;
	bitmapHeader.biCompression = 0; //BI_RGB


	FILE* fp;//�ļ�ָ��

	//���ļ���û���򴴽���
	fopen_s(&fp, FileName, "wb");

	//д���ļ�ͷ����Ϣͷ
	fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&bitmapHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	//д����ɫ����
	fwrite(ColorBuffer, ColorBufferSize, 1, fp);

	fclose(fp);
}

SparseSurfelFusion::DynamicallyRenderSurfels::DynamicallyRenderSurfels(RenderSurfelsType renderType, Intrinsic intrinsic)
{
	rendererIntrinsic = intrinsic;
	confidenceTimeThreshold = make_float2(Constants::kStableSurfelConfidenceThreshold, Constants::kRenderingRecentTimeThreshold);
	InitialDynamicRendererAndAllocateBuffer(renderType);
}

SparseSurfelFusion::DynamicallyRenderSurfels::~DynamicallyRenderSurfels()
{
	RenderedSurfels.ReleaseBuffer();
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaVBOResources));	// ע��OpenGL������ע����cuda�������е���Դ
	glDeleteVertexArrays(1, &LiveSurfelsVAO);
	glDeleteBuffers(1, &LiveSurfelsVBO);
	glDeleteBuffers(1, &LiveSurfelsIBO);
}

void SparseSurfelFusion::DynamicallyRenderSurfels::InitialDynamicRendererAndAllocateBuffer(RenderSurfelsType renderType)
{
	RenderType = renderType;
	RenderedSurfels.AllocateBuffer(MAX_SURFEL_COUNT);
	if (!glfwInit()) { LOGGING(FATAL) << "GLFW����ʧ�ܣ�"; }
	if (!gladLoadGL()) { LOGGING(FATAL) << "GLAD ��ʼ��ʧ��!"; }	// ��ʼ��glad
	
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
	std::string windowName;
	std::string liveSurfelsVertPath;
	std::string liveSurfelsFragPath;
	switch (renderType)
	{
	case RenderSurfelsType::Albedo:
		windowName = "AlbedoRenderedMap";
		SurfelsWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowName.c_str(), NULL, NULL);
		if (SurfelsWindow == NULL) { LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�"; }
		glfwMakeContextCurrent(SurfelsWindow);	// �뵱ǰ����������������
		CheckNextPress.store(false);			// ��ʼ��
		// ���ð����ص�
		glfwSetKeyCallback(SurfelsWindow, RenderKeyCallBack);
		// ������Ȳ���, ���á����޳�������
		glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
		glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
		glDisable(GL_CULL_FACE);				// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
		glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С
		liveSurfelsVertPath = SHADER_PATH_PREFIX + std::string("RenderSurfelsGeometry.vert");	// �鿴�ںϺ��Shader�Ķ�����ɫ��
		liveSurfelsFragPath = SHADER_PATH_PREFIX + std::string("AlbedoColor.frag");// �鿴�ںϺ��Shader��Ƭ����ɫ��
		LiveSurfelsShader.Compile(liveSurfelsVertPath, liveSurfelsFragPath);
		break;
	case RenderSurfelsType::Phong:
		 windowName = "PhongRenderedMap";
		SurfelsWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowName.c_str(), NULL, NULL);
		if (SurfelsWindow == NULL) { LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�"; }
		glfwMakeContextCurrent(SurfelsWindow);	// �뵱ǰ����������������
		CheckNextPress.store(false);			// ��ʼ��
		// ���ð����ص�
		glfwSetKeyCallback(SurfelsWindow, RenderKeyCallBack);
		// ������Ȳ���, ���á����޳�������
		glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
		glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
		glDisable(GL_CULL_FACE);				// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
		glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С
		liveSurfelsVertPath = SHADER_PATH_PREFIX + std::string("RenderSurfelsGeometry.vert");	// �鿴�ںϺ��Shader�Ķ�����ɫ��
		liveSurfelsFragPath = SHADER_PATH_PREFIX + std::string("PhongColor.frag");// �鿴�ںϺ��Shader��Ƭ����ɫ��
		LiveSurfelsShader.Compile(liveSurfelsVertPath, liveSurfelsFragPath);
		break;
	case RenderSurfelsType::Normal:
		 windowName = "NormalRenderedMap";
		SurfelsWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowName.c_str(), NULL, NULL);
		if (SurfelsWindow == NULL) { LOGGING(FATAL) << "δ��ȷ����GLFW���ڣ�"; }
		glfwMakeContextCurrent(SurfelsWindow);	// �뵱ǰ����������������
		CheckNextPress.store(false);			// ��ʼ��
		// ���ð����ص�
		glfwSetKeyCallback(SurfelsWindow, RenderKeyCallBack);
		// ������Ȳ���, ���á����޳�������
		glEnable(GL_DEPTH_TEST);				// ������Ȳ��Ժ�OpenGL���ڻ�������֮ǰ���������ǵ����ֵ���бȽϣ���ֻ������Ȳ���ͨ�������أ��Ӷ�������ȷ����ȾЧ��
		glDepthFunc(GL_LESS);					// ������Ȳ��ԣ���������ЩƬ�Σ����أ�Ӧ�ñ���ʾ����ЩӦ�ñ��������������ǵ����ֵ
		glDisable(GL_CULL_FACE);				// ��ζ��OpenGL����Ⱦ���е��������棬���������ǵĶ���˳�򣬲������Ƿ��ڵ�
		glEnable(GL_PROGRAM_POINT_SIZE);		// ���� glEnable(GL_PROGRAM_POINT_SIZE) ���������ó�����Ƶĵ��С���ܣ�����������ɫ��������ʹ�����ñ��� gl_PointSize �����Ƶ�Ĵ�С
		liveSurfelsVertPath = SHADER_PATH_PREFIX + std::string("RenderSurfelsGeometry.vert");	// �鿴�ںϺ��Shader�Ķ�����ɫ��
		liveSurfelsFragPath = SHADER_PATH_PREFIX + std::string("NormalAsColor.frag");// �鿴�ںϺ��Shader��Ƭ����ɫ��
		LiveSurfelsShader.Compile(liveSurfelsVertPath, liveSurfelsFragPath);
		break;
	default:
		LOGGING(FATAL) << "ѡ��δ������Ⱦ��ʽ";
		break;
	}

	RegisterCudaResources();	// ע��cuda��Դ

	InitialCoordinateSystem();	// ��ʼ��OpenGL������
}

void SparseSurfelFusion::DynamicallyRenderSurfels::RegisterCudaResources()
{
	glfwMakeContextCurrent(SurfelsWindow);	// �뵱ǰ����������������

/*********************   ���ɲ���VAO������   *********************/
	glGenVertexArrays(1, &LiveSurfelsVAO);	// ����VAO
	glBindVertexArray(LiveSurfelsVAO);		// ��VAO
/*********************   ��RenderedFusedPoints��һ������������(������ GPU �ϴ洢�͹������ݵ� OpenGL ����)   *********************/
	// ����1�����ɻ���������
	// ����2���������ַ�����������Ա㽫���ɵı�ʶ���洢�ڸñ�����
	glGenBuffers(1, &LiveSurfelsVBO);
/*******   ��֮ǰ���ɵĻ���������󶨵��������Ի�����Ŀ�� GL_ARRAY_BUFFER �ϣ�������һ����С���ڴ棬���ڴ洢������������   *******/
	// ����1��GL_ARRAY_BUFFER �ǻ�����Ŀ�꣬��ʾ����������������ڴ洢������������
	// ����2��֮ǰ���ɵĻ���������ı�ʶ����ͨ������������ã����û����������� GL_ARRAY_BUFFER Ŀ����а�
	glBindBuffer(GL_ARRAY_BUFFER, LiveSurfelsVBO);
	// ����1��GL_ARRAY_BUFFER ��ʾ��ǰ�󶨵Ļ�����������һ���������Ի�����
	// ����2��sizeof(Renderer::RenderedSurfels) * Constants::maxSurfelsNum ��ʾҪ������ڴ��С
	// ����3��NULL ��ʾ��ʱ���ṩ����
	// ����4��GL_DYNAMIC_DRAW ��ʾ������������󽫱�Ƶ���޸ĺ�ʹ��
	glBufferData(GL_ARRAY_BUFFER, sizeof(Renderer::RenderedSurfels) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
/*******************    ����VAO������    *******************/
	// λ��
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedSurfels), (void*)(0 * sizeof(float4)));	// ����VAO������
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// ����
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedSurfels), (void*)(1 * sizeof(float4)));	// ����VAO������
	glEnableVertexAttribArray(1);	// layout (location = 1)
	// ��ɫ
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedSurfels), (void*)(2 * sizeof(float4)));	// ����VAO������
	glEnableVertexAttribArray(2);	// layout (location = 2)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// ���VBO
	glBindVertexArray(0);						// ���VAO
/*******   ����Ҫ��CUDA��OpenGL֮��������ݽ���ʱ������ʹ��cudaGraphicsGLRegisterBuffer������OpenGL����������ע��ΪCUDAͼ����Դ   *******/
/*******************   �������Ϳ����� CUDA ��������ʹ�� CUDA API �����ʺͲ����û��������󣬶�������ʽ�ؽ������ݿ���   *******************/
	// ����1������ע���CUDAͼ����Դ���������OpenGL�л����������Ӧ��cuda��Դ�������ֱ�Ӷ����cuda��Դ���в�����
	// ����2��Ҫע��� OpenGL ����������ı�ʶ�����������ʶ����Ӧ��OpenGL��bufferע�ᵽcuda�����ġ�
	// ����3����ѡ����������ָ��ע���־����ʾע�� CUDA ͼ����Դʱ��ʹ���κα�־����ע�� CUDA ͼ����Դʱ��Ӧ���ض�����Ϊ���޸�Ĭ����Ϊ��
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&cudaVBOResources, LiveSurfelsVBO, cudaGraphicsRegisterFlagsNone));
}

void SparseSurfelFusion::DynamicallyRenderSurfels::InitialCoordinateSystem()
{
	glfwMakeContextCurrent(SurfelsWindow);	// �뵱ǰ����������������

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

void SparseSurfelFusion::DynamicallyRenderSurfels::MapToCuda(cudaStream_t stream)
{
	const size_t validSurfelsNum = RenderedSurfels.ArraySize();		// ��Ч����Ԫ����

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
	CHECKCUDA(cudaMemcpyAsync(ptr, RenderedSurfels.Ptr(), sizeof(Renderer::RenderedSurfels) * validSurfelsNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::DynamicallyRenderSurfels::UnmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(1, &cudaVBOResources, stream));
}



void SparseSurfelFusion::DynamicallyRenderSurfels::ClearWindow()
{
	// ������glClearColor�����������Ļ���õ���ɫ
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //RGBA
	// ͨ������glClear�����������Ļ����ɫ���壬������һ������λ(Buffer Bit)��ָ��Ҫ��յĻ��壬���ܵĻ���λ��GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// ����ͬʱ�����Ȼ�����!(�������Ȼ�����������ͼ��)
}

void SparseSurfelFusion::DynamicallyRenderSurfels::DrawSurfels(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	const unsigned int totalPointsNum = RenderedSurfels.ArraySize();
	float currentTime = frameIndex;
	LiveSurfelsShader.BindProgram();

	//����͸�Ӿ���
	projection = glm::perspective(glm::radians(30.0f), (float)(WINDOW_WIDTH) / (float)(WINDOW_HEIGHT), 0.1f, 100.0f);
	LiveSurfelsShader.setUniformMat4(std::string("projection"), projection); // ע��:Ŀǰ����ÿ֡����ͶӰ���󣬵�����ͶӰ������ٸı䣬�����������ѭ��֮��������һ�Ρ�
	
	camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
	camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
	view = glm::lookAt(glm::vec3(camX, 0.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	//printf("CamX = %.5f   CamZ = %.5f\n", camX, camZ);// 3.97  2.12

	LiveSurfelsShader.setUniformMat4(std::string("view"), view);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//������(1,1,1)��ת
	LiveSurfelsShader.setUniformMat4(std::string("model"), model);

	LiveSurfelsShader.SetUniformVector("intrinsic", rendererIntrinsic);
	LiveSurfelsShader.SetUniformFloat("currentTime", currentTime);
	LiveSurfelsShader.SetUniformVector("confidenceTimeThreshold", confidenceTimeThreshold);

	glBindVertexArray(LiveSurfelsVAO); // ��VAO�����
	glDrawArrays(GL_POINTS, 0, totalPointsNum);
	glBindVertexArray(0);	// �����
	LiveSurfelsShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyRenderSurfels::DrawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
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

void SparseSurfelFusion::DynamicallyRenderSurfels::SwapBufferAndCatchEvent()
{
	// �����ύ����ɫ���壨����һ��������GLFW����ÿһ��������ɫֵ�Ĵ󻺳壩��������һ�����б��������ƣ����ҽ�����Ϊ�����ʾ����Ļ��
	glfwSwapBuffers(SurfelsWindow);
	glfwPollEvents();
}

void SparseSurfelFusion::DynamicallyRenderSurfels::ScreenShot(const unsigned int frameIdx)
{
	//������ɫ�����ڴ�
	RGBColor* ColorBuffer = new RGBColor[WINDOW_WIDTH * WINDOW_HEIGHT];

	//��ȡ���أ�ע������ĸ�ʽ�� BGR��
	glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, ColorBuffer);

	//������д���ļ�
	std::string path = "E:/Paper_3DReconstruction/Result/RawVideo/";
	std::stringstream ss;
	ss << std::setw(5) << std::setfill('0') << frameIdx;
	std::string frameIndexStr;
	ss >> frameIndexStr;
	frameIndexStr = frameIndexStr + ".bmp";
	std::string imageType;
	if (RenderType == RenderSurfelsType::Normal) imageType = "Normal/";
	else if (RenderType == RenderSurfelsType::Phong) imageType = "Phong/";
	else if (RenderType == RenderSurfelsType::Albedo) imageType = "Albedo/";
	else LOGGING(FATAL) << "Save Bmp File Error!";
	path = path + imageType + frameIndexStr;
	//printf("Path = %s\n", path.c_str());
	WriteBMP(path.c_str(), ColorBuffer, WINDOW_WIDTH, WINDOW_HEIGHT);

	//������ɫ�����ڴ�
	delete[] ColorBuffer;
}

void SparseSurfelFusion::DynamicallyRenderSurfels::DrawRenderedSurfels(const DeviceArrayView<DepthSurfel>& surfels, const unsigned int frameIdx, cudaStream_t stream)
{
	frameIndex = frameIdx;
	glfwMakeContextCurrent(SurfelsWindow);								// �����µĵ�ǰ������
	AdjustModelPosition(surfels, stream);								// ��������Surfelsλ�ã�ʹ����Ⱦģ��Ϊ��
	AdjustSurfelsCoordinateAndColor(surfels, Center, MaxEdge, stream);	// ��һ��ģ������
	MapToCuda(stream);													// ��ý�OpenGL��Դָ�룬���ԭ��VBO����������Ⱦ�����ݴ���VBO
	//while (true) {
	//	if (CheckNextPress.load() == true) {				// ����Ƿ���ESC
	//		CheckNextPress.store(false);					// ��ʼ��
	//		break;
	//	}
	//	ClearWindow();														// �����Ļ
	//	DrawSurfels(view, projection, model);								// ���Ƶ�
	//	//DrawCoordinateSystem(view, projection, model);						// ��������ϵ
	//	SwapBufferAndCatchEvent();											// ˫���岢��׽�¼�
	//}
	ClearWindow();														// �����Ļ
	DrawSurfels(view, projection, model);								// ���Ƶ�
	//DrawCoordinateSystem(view, projection, model);						// ��������ϵ
	SwapBufferAndCatchEvent();											// ˫���岢��׽�¼�
	//ScreenShot(frameIdx);

	UnmapFromCuda(stream);												// �����

}

