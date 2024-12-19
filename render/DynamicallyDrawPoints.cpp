/*****************************************************************//**
 * \file   DynamicallyDrawPoints.cpp
 * \brief  动态绘制点的方法
 * 
 * \author LUOJIAXUAN
 * \date   June 2024
 *********************************************************************/
#include "DynamicallyDrawPoints.h"

std::atomic<bool> CheckEscPress;

 // 按键回调函数
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE){
		if (key == GLFW_KEY_ESCAPE) {
			CheckEscPress.store(true);	// 只有释放才置位
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

	// 注销OpenGL缓冲区注册在cuda上下文中的资源
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
		LOGGING(FATAL) << "GLFW加载失败！";
	}

	// 初始化glad
	if (!gladLoadGL()) {
		LOGGING(FATAL) << "GLAD 初始化失败!";
	}

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
	std::string windowsName = std::string("FusedLivePoints");
	LiveWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, windowsName.c_str(), NULL, NULL);
	if (LiveWindow == NULL) {
		LOGGING(FATAL) << "未正确创建GLFW窗口！";
	}
	//else std::cout << "窗口 " + windowsName + " 创建完成！" << std::endl;

	glfwMakeContextCurrent(LiveWindow);	// 与当前窗口做上下文链接
	CheckEscPress.store(false);			// 初始化
	// 设置按键回调
	glfwSetKeyCallback(LiveWindow, keyCallback);

	// 开启深度测试, 禁用“面剔除”功能
	glEnable(GL_DEPTH_TEST);				// 启用深度测试后，OpenGL会在绘制像素之前，根据它们的深度值进行比较，并只绘制深度测试通过的像素，从而产生正确的渲染效果
	glDepthFunc(GL_LESS);					// 用于深度测试，它决定哪些片段（像素）应该被显示，哪些应该被丢弃，基于它们的深度值
	glDisable(GL_CULL_FACE);				// 意味着OpenGL将渲染所有的三角形面，而不管它们的顶点顺序，不管其是否被遮挡
	glEnable(GL_PROGRAM_POINT_SIZE);		// 调用 glEnable(GL_PROGRAM_POINT_SIZE) 函数会启用程序控制的点大小功能，允许您在着色器程序中使用内置变量 gl_PointSize 来控制点的大小

	// 查看融合后点Shader的顶点着色器
	const std::string live_points_vert_path = SHADER_PATH_PREFIX + std::string("LivePointsVertexShader.vert");
	// 查看融合后点Shader的片段着色器
	const std::string live_points_frag_path = SHADER_PATH_PREFIX + std::string("LivePointsFragmentShader.frag");

	FusedLivePointsShader.Compile(live_points_vert_path, live_points_frag_path);

	registerCudaResources();	// 注册cuda资源

	initialCoordinateSystem();	// 初始化OpenGL的坐标

}

void SparseSurfelFusion::DynamicallyDrawPoints::registerCudaResources()
{
	glfwMakeContextCurrent(LiveWindow);	// 与当前窗口做上下文链接

/*********************   生成并绑定VAO解释器   *********************/
	glGenVertexArrays(1, &FusedLivePointsVAO);	// 生成VAO
	glBindVertexArray(FusedLivePointsVAO);		// 绑定VAO
/*********************   将RenderedFusedPoints绑定一个缓冲区对象(用于在 GPU 上存储和管理数据的 OpenGL 对象)   *********************/
	// 参数1：生成缓冲区数量
	// 参数2：将对象地址传给函数，以便将生成的标识符存储在该变量中
	glGenBuffers(1, &FusedLivePointsVBO);
/*******   将之前生成的缓冲区对象绑定到顶点属性缓冲区目标 GL_ARRAY_BUFFER 上，并分配一定大小的内存，用于存储顶点属性数据   *******/
	// 参数1：GL_ARRAY_BUFFER 是缓冲区目标，表示这个缓冲区对象将用于存储顶点属性数据
	// 参数2：之前生成的缓冲区对象的标识符，通过这个函数调用，将该缓冲区对象与 GL_ARRAY_BUFFER 目标进行绑定
	glBindBuffer(GL_ARRAY_BUFFER, FusedLivePointsVBO);
	// 参数1：GL_ARRAY_BUFFER 表示当前绑定的缓冲区对象是一个顶点属性缓冲区
	// 参数2：6 * sizeof(float) * Constants::maxSurfelsNum 表示要分配的内存大小，Coordinate + RGB = 6 * float
	// 参数3：NULL 表示暂时不提供数据
	// 参数4：GL_DYNAMIC_DRAW 表示这个缓冲区对象将被频繁修改和使用
	glBufferData(GL_ARRAY_BUFFER, sizeof(Renderer::RenderedPoints) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
/*******************    设置VAO解释器    *******************/
	// 位置
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedPoints), (void*)(0 * sizeof(GLfloat)));	// 设置VAO解释器
	glEnableVertexAttribArray(0);	// layout (location = 0)
	// 颜色
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Renderer::RenderedPoints), (void*)(3 * sizeof(GLfloat)));	// 设置VAO解释器
	glEnableVertexAttribArray(1);	// layout (location = 1)

	glBindBuffer(GL_ARRAY_BUFFER, 0);			// 解绑VBO
	glBindVertexArray(0);						// 解绑VAO
/*******   当需要在CUDA和OpenGL之间进行数据交互时，可以使用cudaGraphicsGLRegisterBuffer函数将OpenGL缓冲区对象注册为CUDA图形资源   *******/
/*******************   这样，就可以在 CUDA 上下文中使用 CUDA API 来访问和操作该缓冲区对象，而无需显式地进行数据拷贝   *******************/
	// 参数1：返回注册的CUDA图形资源句柄【即与OpenGL中缓冲区对象对应的cuda资源，后面就直接对这个cuda资源进行操作】
	// 参数2：要注册的 OpenGL 缓冲区对象的标识符【将这个标识符对应的OpenGL的buffer注册到cuda上下文】
	// 参数3：可选参数，用于指定注册标志【表示注册 CUDA 图形资源时不使用任何标志，即注册 CUDA 图形资源时不应用特定的行为或修改默认行为】
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&cudaVBOResources, FusedLivePointsVBO, cudaGraphicsRegisterFlagsNone));
}

void SparseSurfelFusion::DynamicallyDrawPoints::initialCoordinateSystem()
{
	glfwMakeContextCurrent(LiveWindow);	// 与当前窗口做上下文链接

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

void SparseSurfelFusion::DynamicallyDrawPoints::DrawLiveFusedPoints(DeviceArrayView<float4> SolvedPointsCoor, DeviceArrayView<float4> SolvedPointsColor, cudaStream_t stream)
{
	glfwMakeContextCurrent(LiveWindow);	// 创建新的当前上下文

	// 求解点包围盒
	float3 MaxPoint = make_float3(float(-1e6), float(-1e6), float(-1e6));
	float3 MinPoint = make_float3(float(1e6), float(1e6), float(1e6));
	adjustModelPosition(SolvedPointsCoor, stream);
	getBoundingBox(RenderedFusedPoints.ArrayView(), MaxPoint, MinPoint, stream);
	// 将稠密点全部调整到第一象限并解码颜色，并将值赋给RenderedFusedPoints
	adjustPointsCoordinateAndColor(SolvedPointsColor, MaxPoint, MinPoint, stream);
	mapToCuda(stream);	// 获得将OpenGL资源指针，清空原先VBO，并将待渲染点数据传入VBO
	while (true) {
		if (CheckEscPress.load() == true) {				// 检查是否按下ESC
			CheckEscPress.store(false);					// 初始化
			break; 
		}	
		clearWindow();									// 清空屏幕
		drawLivePoints(view, projection, model);		// 绘制点
		drawCoordinateSystem(view, projection, model);	// 绘制坐标系
		swapBufferAndCatchEvent();						// 双缓冲并捕捉事件
	}
	unmapFromCuda(stream);
}

void SparseSurfelFusion::DynamicallyDrawPoints::DrawLiveFusedPoints(DeviceArrayView<DepthSurfel> SolvedPoints, bool CheckPeriod, cudaStream_t stream)
{
	glfwMakeContextCurrent(LiveWindow);	// 创建新的当前上下文
	CheckSpecificFrame = CheckPeriod;
	// 求解点包围盒
	float3 MaxPoint = make_float3(float(-1e6), float(-1e6), float(-1e6));
	float3 MinPoint = make_float3(float(1e6), float(1e6), float(1e6));
	//printf("SolvedPointsCount = %d\n", SolvedPoints.Size());
	adjustModelPosition(SolvedPoints, stream);
	getBoundingBox(RenderedFusedPoints.ArrayView(), MaxPoint, MinPoint, stream);
	adjustPointsCoordinateAndColor(SolvedPoints, MaxPoint, MinPoint, stream);

	mapToCuda(stream);	// 获得将OpenGL资源指针，清空原先VBO，并将待渲染点数据传入VBO
	if (CheckSpecificFrame) {
		while (true) {
			if (CheckEscPress.load() == true) {				// 检查是否按下ESC
				CheckEscPress.store(false);					// 初始化
				CheckSpecificFrame = false;
				break;
			}
			clearWindow();		// 清空屏幕
			drawLivePoints(view, projection, model);		// 绘制点
			drawCoordinateSystem(view, projection, model);	// 绘制坐标系
			swapBufferAndCatchEvent();						// 双缓冲并捕捉事件
		}
	}
	else {
		clearWindow();		// 清空屏幕
		drawLivePoints(view, projection, model);		// 绘制点
		drawCoordinateSystem(view, projection, model);	// 绘制坐标系
		swapBufferAndCatchEvent();						// 双缓冲并捕捉事件
	}

	unmapFromCuda(stream);
}

void SparseSurfelFusion::DynamicallyDrawPoints::mapToCuda(cudaStream_t stream)
{
	const size_t validSurfelsNum = RenderedFusedPoints.ArraySize();		// 有效的面元数量

///** 直接绑定FusedLivePointsVBO并将其中的数据清空 **/
//	glBindBuffer(GL_ARRAY_BUFFER, FusedLivePointsVBO);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(Renderer::RenderedPoints) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);

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
	CHECKCUDA(cudaMemcpyAsync(ptr, RenderedFusedPoints.Ptr(), sizeof(Renderer::RenderedPoints) * validSurfelsNum, cudaMemcpyDeviceToDevice, stream));
	CHECKCUDA(cudaStreamSynchronize(stream));
}

void SparseSurfelFusion::DynamicallyDrawPoints::unmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(1, &cudaVBOResources, stream));
}

void SparseSurfelFusion::DynamicallyDrawPoints::clearWindow()
{
	// 调用了glClearColor来设置清空屏幕所用的颜色
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //RGBA
	// 通过调用glClear函数来清空屏幕的颜色缓冲，它接受一个缓冲位(Buffer Bit)来指定要清空的缓冲，可能的缓冲位有GL_COLOR_BUFFER_BIT
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 现在同时清除深度缓冲区!(不清除深度画不出来立体图像)
}

void SparseSurfelFusion::DynamicallyDrawPoints::drawLivePoints(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
{

	const unsigned int totalPointsNum = RenderedFusedPoints.ArraySize();
	FusedLivePointsShader.BindProgram();

	//设置透视矩阵
	projection = glm::perspective(glm::radians(30.0f), (float)(WINDOW_WIDTH) / (float)(WINDOW_HEIGHT), 0.1f, 100.0f);
	FusedLivePointsShader.setUniformMat4(std::string("projection"), projection); // 注意:目前我们每帧设置投影矩阵，但由于投影矩阵很少改变，所以最好在主循环之外设置它一次。

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
	model = glm::rotate(model, glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));//绕向量(1,1,1)旋转
	FusedLivePointsShader.setUniformMat4(std::string("model"), model);

	glBindVertexArray(FusedLivePointsVAO); // 绑定VAO后绘制
	glDrawArrays(GL_POINTS, 0, totalPointsNum);
	glBindVertexArray(0);	// 清除绑定
	FusedLivePointsShader.UnbindProgram();
}

void SparseSurfelFusion::DynamicallyDrawPoints::drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model)
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

void SparseSurfelFusion::DynamicallyDrawPoints::swapBufferAndCatchEvent()
{
	// 函数会交换颜色缓冲（它是一个储存着GLFW窗口每一个像素颜色值的大缓冲），它在这一迭代中被用来绘制，并且将会作为输出显示在屏幕上
	glfwSwapBuffers(LiveWindow);
	glfwPollEvents();
}
