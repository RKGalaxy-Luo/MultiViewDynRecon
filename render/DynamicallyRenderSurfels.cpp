#include "DynamicallyRenderSurfels.h"

std::atomic<bool> CheckNextPress;

// 按键回调函数
void RenderKeyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		if (key == GLFW_KEY_RIGHT) {
			CheckNextPress.store(true);	// 只有释放才置位
		}
	}
}

#pragma pack(2)//影响了“对齐”。可以实验前后 sizeof(BITMAPFILEHEADER) 的差别

typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned long  DWORD;
typedef long    LONG;

//BMP文件头：
struct BITMAPFILEHEADER
{
	WORD  bfType;		//文件类型标识，必须为ASCII码“BM”
	DWORD bfSize;		//文件的尺寸，以byte为单位
	WORD  bfReserved1;	//保留字，必须为0
	WORD  bfReserved2;	//保留字，必须为0
	DWORD bfOffBits;	//一个以byte为单位的偏移，从BITMAPFILEHEADER结构体开始到位图数据
};

//BMP信息头：
struct BITMAPINFOHEADER
{
	DWORD biSize;			//这个结构体的尺寸
	LONG  biWidth;			//位图的宽度
	LONG  biHeight;			//位图的长度
	WORD  biPlanes;			//The number of planes for the target device. This value must be set to 1.
	WORD  biBitCount;		//一个像素有几位
	DWORD biCompression;    //0：不压缩，1：RLE8，2：RLE4
	DWORD biSizeImage;      //4字节对齐的图像数据大小
	LONG  biXPelsPerMeter;  //用象素/米表示的水平分辨率
	LONG  biYPelsPerMeter;  //用象素/米表示的垂直分辨率
	DWORD biClrUsed;        //实际使用的调色板索引数，0：使用所有的调色板索引
	DWORD biClrImportant;	//重要的调色板索引数，0：所有的调色板索引都重要
};

//一个像素的颜色信息
struct RGBColor
{
	char B;		//蓝
	char G;		//绿
	char R;		//红
};

//将颜色数据写到一个BMP文件中
//FileName:文件名
//ColorBuffer:颜色数据
//ImageWidth:图像宽度
//ImageHeight:图像长度
void WriteBMP(const char* FileName, RGBColor* ColorBuffer, int ImageWidth, int ImageHeight)
{
	//颜色数据总尺寸：
	const int ColorBufferSize = ImageHeight * ImageWidth * sizeof(RGBColor);

	//文件头
	BITMAPFILEHEADER fileHeader;
	fileHeader.bfType = 0x4D42;	//0x42是'B'；0x4D是'M'
	fileHeader.bfReserved1 = 0;
	fileHeader.bfReserved2 = 0;
	fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + ColorBufferSize;
	fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	//信息头
	BITMAPINFOHEADER bitmapHeader = { 0 };
	bitmapHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapHeader.biHeight = ImageHeight;
	bitmapHeader.biWidth = ImageWidth;
	bitmapHeader.biPlanes = 1;
	bitmapHeader.biBitCount = 24;
	bitmapHeader.biSizeImage = ColorBufferSize;
	bitmapHeader.biCompression = 0; //BI_RGB


	FILE* fp;//文件指针

	//打开文件（没有则创建）
	fopen_s(&fp, FileName, "wb");

	//写入文件头和信息头
	fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&bitmapHeader, sizeof(BITMAPINFOHEADER), 1, fp);
	//写入颜色数据
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
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaVBOResources));	// 注销OpenGL缓冲区注册在cuda上下文中的资源
	glDeleteVertexArrays(1, &LiveSurfelsVAO);
	glDeleteBuffers(1, &LiveSurfelsVBO);
	glDeleteBuffers(1, &LiveSurfelsIBO);
}

void SparseSurfelFusion::DynamicallyRenderSurfels::InitialDynamicRendererAndAllocateBuffer(RenderSurfelsType renderType)
{
	RenderType = renderType;
	RenderedSurfels.AllocateBuffer(MAX_SURFEL_COUNT);
	if (!glfwInit()) { LOGGING(FATAL) << "GLFW加载失败！"; }
	if (!gladLoadGL()) { LOGGING(FATAL) << "GLAD 初始化失败!"; }	// 初始化glad
	
	// opengl上下文，GLFW版本为4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// 默认的framebuffer属性
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE);		// 窗口可见
	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_STEREO, GL_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);	// 窗口大小不可调整

	// 窗口的设置
	std::string windowName;
	std::string liveSurfelsVertPath;
	std::string liveSurfelsFragPath;
	switch (renderType)
	{
	case RenderSurfelsType::Albedo:
		windowName = "AlbedoRenderedMap";
		SurfelsWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowName.c_str(), NULL, NULL);
		if (SurfelsWindow == NULL) { LOGGING(FATAL) << "未正确创建GLFW窗口！"; }
		glfwMakeContextCurrent(SurfelsWindow);	// 与当前窗口做上下文链接
		CheckNextPress.store(false);			// 初始化
		// 设置按键回调
		glfwSetKeyCallback(SurfelsWindow, RenderKeyCallBack);
		// 开启深度测试, 禁用“面剔除”功能
		glEnable(GL_DEPTH_TEST);				// 启用深度测试后，OpenGL会在绘制像素之前，根据它们的深度值进行比较，并只绘制深度测试通过的像素，从而产生正确的渲染效果
		glDepthFunc(GL_LESS);					// 用于深度测试，它决定哪些片段（像素）应该被显示，哪些应该被丢弃，基于它们的深度值
		glDisable(GL_CULL_FACE);				// 意味着OpenGL将渲染所有的三角形面，而不管它们的顶点顺序，不管其是否被遮挡
		glEnable(GL_PROGRAM_POINT_SIZE);		// 调用 glEnable(GL_PROGRAM_POINT_SIZE) 函数会启用程序控制的点大小功能，允许您在着色器程序中使用内置变量 gl_PointSize 来控制点的大小
		liveSurfelsVertPath = SHADER_PATH_PREFIX + std::string("RenderSurfelsGeometry.vert");	// 查看融合后点Shader的顶点着色器
		liveSurfelsFragPath = SHADER_PATH_PREFIX + std::string("AlbedoColor.frag");// 查看融合后点Shader的片段着色器
		LiveSurfelsShader.Compile(liveSurfelsVertPath, liveSurfelsFragPath);
		break;
	case RenderSurfelsType::Phong:
		 windowName = "PhongRenderedMap";
		SurfelsWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowName.c_str(), NULL, NULL);
		if (SurfelsWindow == NULL) { LOGGING(FATAL) << "未正确创建GLFW窗口！"; }
		glfwMakeContextCurrent(SurfelsWindow);	// 与当前窗口做上下文链接
		CheckNextPress.store(false);			// 初始化
		// 设置按键回调
		glfwSetKeyCallback(SurfelsWindow, RenderKeyCallBack);
		// 开启深度测试, 禁用“面剔除”功能
		glEnable(GL_DEPTH_TEST);				// 启用深度测试后，OpenGL会在绘制像素之前，根据它们的深度值进行比较，并只绘制深度测试通过的像素，从而产生正确的渲染效果
		glDepthFunc(GL_LESS);					// 用于深度测试，它决定哪些片段（像素）应该被显示，哪些应该被丢弃，基于它们的深度值
		glDisable(GL_CULL_FACE);				// 意味着OpenGL将渲染所有的三角形面，而不管它们的顶点顺序，不管其是否被遮挡
		glEnable(GL_PROGRAM_POINT_SIZE);		// 调用 glEnable(GL_PROGRAM_POINT_SIZE) 函数会启用程序控制的点大小功能，允许您在着色器程序中使用内置变量 gl_PointSize 来控制点的大小
		liveSurfelsVertPath = SHADER_PATH_PREFIX + std::string("RenderSurfelsGeometry.vert");	// 查看融合后点Shader的顶点着色器
		liveSurfelsFragPath = SHADER_PATH_PREFIX + std::string("PhongColor.frag");// 查看融合后点Shader的片段着色器
		LiveSurfelsShader.Compile(liveSurfelsVertPath, liveSurfelsFragPath);
		break;
	case RenderSurfelsType::Normal:
		 windowName = "NormalRenderedMap";
		SurfelsWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowName.c_str(), NULL, NULL);
		if (SurfelsWindow == NULL) { LOGGING(FATAL) << "未正确创建GLFW窗口！"; }
		glfwMakeContextCurrent(SurfelsWindow);	// 与当前窗口做上下文链接
		CheckNextPress.store(false);			// 初始化
		// 设置按键回调
		glfwSetKeyCallback(SurfelsWindow, RenderKeyCallBack);
		// 开启深度测试, 禁用“面剔除”功能
		glEnable(GL_DEPTH_TEST);				// 启用深度测试后，OpenGL会在绘制像素之前，根据它们的深度值进行比较，并只绘制深度测试通过的像素，从而产生正确的渲染效果
		glDepthFunc(GL_LESS);					// 用于深度测试，它决定哪些片段（像素）应该被显示，哪些应该被丢弃，基于它们的深度值
		glDisable(GL_CULL_FACE);				// 意味着OpenGL将渲染所有的三角形面，而不管它们的顶点顺序，不管其是否被遮挡
		glEnable(GL_PROGRAM_POINT_SIZE);		// 调用 glEnable(GL_PROGRAM_POINT_SIZE) 函数会启用程序控制的点大小功能，允许您在着色器程序中使用内置变量 gl_PointSize 来控制点的大小
		liveSurfelsVertPath = SHADER_PATH_PREFIX + std::string("RenderSurfelsGeometry.vert");	// 查看融合后点Shader的顶点着色器
		liveSurfelsFragPath = SHADER_PATH_PREFIX + std::string("NormalAsColor.frag");// 查看融合后点Shader的片段着色器
		LiveSurfelsShader.Compile(liveSurfelsVertPath, liveSurfelsFragPath);
		break;
	default:
		LOGGING(FATAL) << "选择未定义渲染方式";
		break;
	}

	RegisterCudaResources();	// 注册cuda资源

	InitialCoordinateSystem();	// 初始化OpenGL的坐标
}

void SparseSurfelFusion::DynamicallyRenderSurfels::RegisterCudaResources()
{
	glfwMakeContextCurrent(SurfelsWindow);	// 与当前窗口做上下文链接

/*********************   生成并绑定VAO解释器   *********************/
	glGenVertexArrays(1, &LiveSurfelsVAO);	// 生成VAO
	glBindVertexArray(LiveSurfelsVAO);		// 绑定VAO
/*********************   将RenderedFusedPoints绑定一个缓冲区对象(用于在 GPU 上存储和管理数据的 OpenGL 对象)   *********************/
	// 参数1：生成缓冲区数量
	// 参数2：将对象地址传给函数，以便将生成的标识符存储在该变量中
	glGenBuffers(1, &LiveSurfelsVBO);
/*******   将之前生成的缓冲区对象绑定到顶点属性缓冲区目标 GL_ARRAY_BUFFER 上，并分配一定大小的内存，用于存储顶点属性数据   *******/
	// 参数1：GL_ARRAY_BUFFER 是缓冲区目标，表示这个缓冲区对象将用于存储顶点属性数据
	// 参数2：之前生成的缓冲区对象的标识符，通过这个函数调用，将该缓冲区对象与 GL_ARRAY_BUFFER 目标进行绑定
	glBindBuffer(GL_ARRAY_BUFFER, LiveSurfelsVBO);
	// 参数1：GL_ARRAY_BUFFER 表示当前绑定的缓冲区对象是一个顶点属性缓冲区
	// 参数2：sizeof(Renderer::RenderedSurfels) * Constants::maxSurfelsNum 表示要分配的内存大小
	// 参数3：NULL 表示暂时不提供数据
	// 参数4：GL_DYNAMIC_DRAW 表示这个缓冲区对象将被频繁修改和使用
	glBufferData(GL_ARRAY_BUFFER, sizeof(Renderer::RenderedSurfels) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
/*******************    设置VAO解释器    *******************/
	// 位置
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedSurfels), (void*)(0 * sizeof(float4)));	// 设置VAO解释器
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// 法线
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedSurfels), (void*)(1 * sizeof(float4)));	// 设置VAO解释器
	glEnableVertexAttribArray(1);	// layout (location = 1)
	// 颜色
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedSurfels), (void*)(2 * sizeof(float4)));	// 设置VAO解释器
	glEnableVertexAttribArray(2);	// layout (location = 2)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// 解绑VBO
	glBindVertexArray(0);						// 解绑VAO
/*******   当需要在CUDA和OpenGL之间进行数据交互时，可以使用cudaGraphicsGLRegisterBuffer函数将OpenGL缓冲区对象注册为CUDA图形资源   *******/
/*******************   这样，就可以在 CUDA 上下文中使用 CUDA API 来访问和操作该缓冲区对象，而无需显式地进行数据拷贝   *******************/
	// 参数1：返回注册的CUDA图形资源句柄【即与OpenGL中缓冲区对象对应的cuda资源，后面就直接对这个cuda资源进行操作】
	// 参数2：要注册的 OpenGL 缓冲区对象的标识符【将这个标识符对应的OpenGL的buffer注册到cuda上下文】
	// 参数3：可选参数，用于指定注册标志【表示注册 CUDA 图形资源时不使用任何标志，即注册 CUDA 图形资源时不应用特定的行为或修改默认行为】
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&cudaVBOResources, LiveSurfelsVBO, cudaGraphicsRegisterFlagsNone));
}

void SparseSurfelFusion::DynamicallyRenderSurfels::InitialCoordinateSystem()
{
	glfwMakeContextCurrent(SurfelsWindow);	// 与当前窗口做上下文链接

	std::vector<float> pvalues;			// 点坐标	
	// Live域中的点，查看中间过程的顶点着色器
	const std::string coordinate_vert_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.vert");
	// Live域中的点，查看中间过程的片段着色器
	const std::string coordinate_frag_path = SHADER_PATH_PREFIX + std::string("CoordinateSystemShader.frag");
	coordinateShader.Compile(coordinate_vert_path, coordinate_frag_path);
	glGenVertexArrays(1, &(coordinateSystemVAO));	// 生成VAO
	glGenBuffers(1, &(coordinateSystemVBO));		// 生成VBO
	const unsigned int Num = sizeof(box) / sizeof(box[0]);
	for (int i = 0; i < Num; i++) {
		pvalues.push_back(box[i][0]);
		pvalues.push_back(box[i][1]);
		pvalues.push_back(box[i][2]);
	}
	glBindVertexArray(coordinateSystemVAO);
	glBindBuffer(GL_ARRAY_BUFFER, coordinateSystemVBO);
	GLsizei bufferSize = sizeof(GLfloat) * pvalues.size();		// float数据的数量
	glBufferData(GL_ARRAY_BUFFER, bufferSize, pvalues.data(), GL_STATIC_DRAW);	// 动态绘制，目前只是先开辟个大小

	// 位置
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0 * sizeof(GLfloat)));	// 设置VAO解释器
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// 颜色
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));	// 设置VAO解释器
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// 解绑VBO
	glBindVertexArray(0);						// 解绑VAO
}

void SparseSurfelFusion::DynamicallyRenderSurfels::MapToCuda(cudaStream_t stream)
{
	const size_t validSurfelsNum = RenderedSurfels.ArraySize();		// 有效的面元数量

/** 使用cudaGraphicsMapResources函数将资源映射到上下文 **/
	// 参数1：要映射的 CUDA 图形资源的数量。
	// 参数2：指向 CUDA 图形资源数组的指针。
	// 参数3：可选参数，用于指定要在其上执行映射操作的 CUDA 流。默认值为 0，表示使用默认流。
	CHECKCUDA(cudaGraphicsMapResources(1, &cudaVBOResources, stream));	//首先映射资源

	// 获得buffer
	void* ptr;			// 用于获取cuda资源的地址(重复使用)
	size_t bufferSize;	// 用于获取cuda资源buffer的大小
	// 获得OpenGL上的资源指针
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&ptr, &bufferSize, cudaVBOResources));

	// 直接将数据拷贝到资源指针上
	CHECKCUDA(cudaMemcpyAsync(ptr, RenderedSurfels.Ptr(), sizeof(Renderer::RenderedSurfels) * validSurfelsNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::DynamicallyRenderSurfels::UnmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(1, &cudaVBOResources, stream));
}



void SparseSurfelFusion::DynamicallyRenderSurfels::ClearWindow()
{
	// 调用了glClearColor来设置清空屏幕所用的颜色
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //RGBA
	// 通过调用glClear函数来清空屏幕的颜色缓冲，它接受一个缓冲位(Buffer Bit)来指定要清空的缓冲，可能的缓冲位有GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 现在同时清除深度缓冲区!(不清除深度画不出来立体图像)
}

void SparseSurfelFusion::DynamicallyRenderSurfels::DrawSurfels(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	const unsigned int totalPointsNum = RenderedSurfels.ArraySize();
	float currentTime = frameIndex;
	LiveSurfelsShader.BindProgram();

	//设置透视矩阵
	projection = glm::perspective(glm::radians(30.0f), (float)(WINDOW_WIDTH) / (float)(WINDOW_HEIGHT), 0.1f, 100.0f);
	LiveSurfelsShader.setUniformMat4(std::string("projection"), projection); // 注意:目前我们每帧设置投影矩阵，但由于投影矩阵很少改变，所以最好在主循环之外设置它一次。
	
	camX = static_cast<float>(sin(glfwGetTime() * 0.5f) * radius);
	camZ = static_cast<float>(cos(glfwGetTime() * 0.5f) * radius);
	view = glm::lookAt(glm::vec3(camX, 0.0f, camZ), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	//printf("CamX = %.5f   CamZ = %.5f\n", camX, camZ);// 3.97  2.12

	LiveSurfelsShader.setUniformMat4(std::string("view"), view);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//绕向量(1,1,1)旋转
	LiveSurfelsShader.setUniformMat4(std::string("model"), model);

	LiveSurfelsShader.SetUniformVector("intrinsic", rendererIntrinsic);
	LiveSurfelsShader.SetUniformFloat("currentTime", currentTime);
	LiveSurfelsShader.SetUniformVector("confidenceTimeThreshold", confidenceTimeThreshold);

	glBindVertexArray(LiveSurfelsVAO); // 绑定VAO后绘制
	glDrawArrays(GL_POINTS, 0, totalPointsNum);
	glBindVertexArray(0);	// 清除绑定
	LiveSurfelsShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyRenderSurfels::DrawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{
	// 绘制坐标系
	coordinateShader.BindProgram();	// 绑定坐标轴的shader
	coordinateShader.setUniformMat4(std::string("projection"), projection); // 注意:目前我们每帧设置投影矩阵，但由于投影矩阵很少改变，所以最好在主循环之外设置它一次。
	coordinateShader.setUniformMat4(std::string("view"), view);
	coordinateShader.setUniformMat4(std::string("model"), model);
	glBindVertexArray(coordinateSystemVAO); // 绑定VAO后绘制

	glLineWidth(3.0f);
	glDrawArrays(GL_LINES, 0, 34);	// box有54个元素，绘制线段

	// 清除绑定
	glBindVertexArray(0);
	coordinateShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyRenderSurfels::SwapBufferAndCatchEvent()
{
	// 函数会交换颜色缓冲（它是一个储存着GLFW窗口每一个像素颜色值的大缓冲），它在这一迭代中被用来绘制，并且将会作为输出显示在屏幕上
	glfwSwapBuffers(SurfelsWindow);
	glfwPollEvents();
}

void SparseSurfelFusion::DynamicallyRenderSurfels::ScreenShot(const unsigned int frameIdx)
{
	//申请颜色数据内存
	RGBColor* ColorBuffer = new RGBColor[WINDOW_WIDTH * WINDOW_HEIGHT];

	//读取像素（注意这里的格式是 BGR）
	glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, ColorBuffer);

	//将数据写入文件
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

	//清理颜色数据内存
	delete[] ColorBuffer;
}

void SparseSurfelFusion::DynamicallyRenderSurfels::DrawRenderedSurfels(const DeviceArrayView<DepthSurfel>& surfels, const unsigned int frameIdx, cudaStream_t stream)
{
	frameIndex = frameIdx;
	glfwMakeContextCurrent(SurfelsWindow);								// 创建新的当前上下文
	AdjustModelPosition(surfels, stream);								// 调整整体Surfels位置，使得渲染模型为正
	AdjustSurfelsCoordinateAndColor(surfels, Center, MaxEdge, stream);	// 归一化模型坐标
	MapToCuda(stream);													// 获得将OpenGL资源指针，清空原先VBO，并将待渲染点数据传入VBO
	//while (true) {
	//	if (CheckNextPress.load() == true) {				// 检查是否按下ESC
	//		CheckNextPress.store(false);					// 初始化
	//		break;
	//	}
	//	ClearWindow();														// 清空屏幕
	//	DrawSurfels(view, projection, model);								// 绘制点
	//	//DrawCoordinateSystem(view, projection, model);						// 绘制坐标系
	//	SwapBufferAndCatchEvent();											// 双缓冲并捕捉事件
	//}
	ClearWindow();														// 清空屏幕
	DrawSurfels(view, projection, model);								// 绘制点
	//DrawCoordinateSystem(view, projection, model);						// 绘制坐标系
	SwapBufferAndCatchEvent();											// 双缓冲并捕捉事件
	//ScreenShot(frameIdx);

	UnmapFromCuda(stream);												// 解除绑定

}

