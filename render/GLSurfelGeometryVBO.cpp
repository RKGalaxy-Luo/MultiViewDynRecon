#include "GLSurfelGeometryVBO.h"

void SparseSurfelFusion::GLSurfelGeometryVBO::initialize()
{
	initializeGLSurfelGeometry(*this);
}

void SparseSurfelFusion::GLSurfelGeometryVBO::release()
{
	releaseGLSurfelGeometry(*this);
}

void SparseSurfelFusion::GLSurfelGeometryVBO::mapToCuda(SurfelGeometry& geometry, cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文

/** 使用cudaGraphicsMapResources函数将资源映射到上下文 **/
	
	// 参数1：要映射的 CUDA 图形资源的数量。
	// 参数2：指向 CUDA 图形资源数组的指针。
	// 参数3：可选参数，用于指定要在其上执行映射操作的 CUDA 流。默认值为 0，表示使用默认流。
	CHECKCUDA(cudaGraphicsMapResources(5, cudaVBOResources, stream));	//首先映射资源

	// 获得buffer
	void* dptr;			// 用于获取cuda资源的地址(重复使用)
	size_t bufferSize;	// 用于获取cuda资源buffer的大小
	const size_t validSurfelsNum = geometry.ValidSurfelsNum();	// 有效的面元数量
/** 在使用cudaGraphicsMapResources函数将资源映射到上下文之后，使用cudaGraphicsResourceGetMappedPointer 函数来获取该资源在设备内存中的指针 **/
	//参考顶点置信度
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[0]));//把cudaVBOResources[0]在设备上的地址丢给了dptr，大小给了buffersize
	geometry.CanonicalVertexConfidence = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);//这里是把CanonicalVertexConfidence连接给了cudaVBOResources[0]

	//参考法线-半径
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[1]));
	geometry.CanonicalNormalRadius = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//实时顶点置信度
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[2]));
	geometry.LiveVertexConfidence = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//实时法线-半径
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[3]));
	geometry.LiveNormalRadius = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//观察时间
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[4]));
	geometry.ColorTime = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);
}

void SparseSurfelFusion::GLSurfelGeometryVBO::mapToCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文

	/** 使用cudaGraphicsMapResources函数将资源映射到上下文 **/
	CHECKCUDA(cudaGraphicsMapResources(5, cudaVBOResources, stream));		// 映射图形资源以供 CUDA 访问
}

void SparseSurfelFusion::GLSurfelGeometryVBO::unmapFromCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文

	/** 用于取消映射已映射的 CUDA 图形资源，使其不再与当前的 CUDA 上下文关联 **/
	CHECKCUDA(cudaGraphicsUnmapResources(5, cudaVBOResources, stream));	// 取消已经映射的cuda上下文资源
	// 函数执行成功后，已映射的资源将与当前的 CUDA 上下文分离，不再可用于 GPU 上的操作。
	// 这样可以确保资源在主机和设备之间的一致性，并释放相关资源的内存
}

void SparseSurfelFusion::initializeGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

/*********************   将surfel的各项参数绑定一个缓冲区对象(用于在 GPU 上存储和管理数据的 OpenGL 对象)   *********************/
	// 参数1：生成缓冲区数量
	// 参数2：将对象地址传给函数，以便将生成的标识符存储在该变量中
	glGenBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glGenBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glGenBuffers(1, &(surfelVBO.LiveVertexConfidence));
	glGenBuffers(1, &(surfelVBO.LiveNormalRadius));
	glGenBuffers(1, &(surfelVBO.ColorTime));

	//glGenBuffers(1, &(surfelVBO.From));

/*******   将之前生成的缓冲区对象绑定到顶点属性缓冲区目标 GL_ARRAY_BUFFER 上，并分配一定大小的内存，用于存储顶点属性数据   *******/
	// 参数1：GL_ARRAY_BUFFER 是缓冲区目标，表示这个缓冲区对象将用于存储顶点属性数据
	// 参数2：之前生成的缓冲区对象的标识符，通过这个函数调用，将该缓冲区对象与 GL_ARRAY_BUFFER 目标进行绑定
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalVertexConfidence);
	// 参数1：GL_ARRAY_BUFFER 表示当前绑定的缓冲区对象是一个顶点属性缓冲区
	// 参数2：4 * sizeof(float) * Constants::maxSurfelsNum 表示要分配的内存大小
	// 参数3：NULL 表示暂时不提供数据
	// 参数4：GL_DYNAMIC_DRAW 表示这个缓冲区对象将被频繁修改和使用
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalNormalRadius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.LiveVertexConfidence);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.LiveNormalRadius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.ColorTime);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);

	//glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.From);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(int) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);

/*******   当需要在CUDA和OpenGL之间进行数据交互时，可以使用cudaGraphicsGLRegisterBuffer函数将OpenGL缓冲区对象注册为CUDA图形资源   *******/
/*******************   这样，就可以在 CUDA 上下文中使用 CUDA API 来访问和操作该缓冲区对象，而无需显式地进行数据拷贝   *******************/
	// 参数1：返回注册的CUDA图形资源句柄【即与OpenGL中缓冲区对象对应的cuda资源，后面就直接对这个cuda资源进行操作】
	// 参数2：要注册的 OpenGL 缓冲区对象的标识符【将这个标识符对应的OpenGL的buffer注册到cuda上下文】
	// 参数3：可选参数，用于指定注册标志【表示注册 CUDA 图形资源时不使用任何标志，即注册 CUDA 图形资源时不应用特定的行为或修改默认行为】
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[0]), surfelVBO.CanonicalVertexConfidence, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[1]), surfelVBO.CanonicalNormalRadius, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[2]), surfelVBO.LiveVertexConfidence, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[3]), surfelVBO.LiveNormalRadius, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[4]), surfelVBO.ColorTime, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::initializeGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	/*********************   将surfel的各项参数绑定一个缓冲区对象(用于在 GPU 上存储和管理数据的 OpenGL 对象)   *********************/
	// 参数1：生成缓冲区数量
	// 参数2：将对象地址传给函数，以便将生成的标识符存储在该变量中
	glGenBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glGenBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glGenBuffers(1, &(surfelVBO.ColorTime));

/*******   将之前生成的缓冲区对象绑定到顶点属性缓冲区目标 GL_ARRAY_BUFFER 上，并分配一定大小的内存，用于存储顶点属性数据   *******/
	// 参数1：GL_ARRAY_BUFFER 是缓冲区目标，表示这个缓冲区对象将用于存储顶点属性数据
	// 参数2：之前生成的缓冲区对象的标识符，通过这个函数调用，将该缓冲区对象与 GL_ARRAY_BUFFER 目标进行绑定
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalVertexConfidence);
	// 参数1：GL_ARRAY_BUFFER 表示当前绑定的缓冲区对象是一个顶点属性缓冲区
	// 参数2：4 * sizeof(float) * Constants::maxSurfelsNum 表示要分配的内存大小
	// 参数3：NULL 表示暂时不提供数据
	// 参数4：GL_DYNAMIC_DRAW 表示这个缓冲区对象将被频繁修改和使用
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalNormalRadius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.ColorTime);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);


/*******   当需要在CUDA和OpenGL之间进行数据交互时，可以使用cudaGraphicsGLRegisterBuffer函数将OpenGL缓冲区对象注册为CUDA图形资源   *******/
/*******************   这样，就可以在 CUDA 上下文中使用 CUDA API 来访问和操作该缓冲区对象，而无需显式地进行数据拷贝   *******************/
	// 参数1：返回注册的CUDA图形资源句柄【即与OpenGL中缓冲区对象对应的cuda资源，后面就直接对这个cuda资源进行操作】
	// 参数2：要注册的 OpenGL 缓冲区对象的标识符【将这个标识符对应的OpenGL的buffer注册到cuda上下文】
	// 参数3：可选参数，用于指定注册标志【表示注册 CUDA 图形资源时不使用任何标志，即注册 CUDA 图形资源时不应用特定的行为或修改默认行为】
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[0]), surfelVBO.CanonicalVertexConfidence, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[1]), surfelVBO.CanonicalNormalRadius, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[2]), surfelVBO.ColorTime, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::releaseGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	// 注销OpenGL缓冲区注册在cuda上下文中的资源
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[0]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[1]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[2]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[3]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[4]));

	// 删除之前声明分配的VBO(顶点缓冲区)
	glDeleteBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glDeleteBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glDeleteBuffers(1, &(surfelVBO.LiveVertexConfidence));
	glDeleteBuffers(1, &(surfelVBO.LiveNormalRadius));
	glDeleteBuffers(1, &(surfelVBO.ColorTime));
}

void SparseSurfelFusion::releaseGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	// 注销OpenGL缓冲区注册在cuda上下文中的资源
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[0]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[1]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[2]));
	// 删除之前声明分配的VBO(顶点缓冲区)
	glDeleteBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glDeleteBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glDeleteBuffers(1, &(surfelVBO.ColorTime));
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::initialize()
{
	initializeGLfusionDepthSurfel(*this);
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::release()
{
	releaseGLfusionDepthSurfel(*this);
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::mapToCuda(FusionDepthGeometry& geometry, cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文

	/** 使用cudaGraphicsMapResources函数将资源映射到上下文 **/

	// 参数1：要映射的 CUDA 图形资源的数量。
	// 参数2：指向 CUDA 图形资源数组的指针。
	// 参数3：可选参数，用于指定要在其上执行映射操作的 CUDA 流。默认值为 0，表示使用默认流。
	CHECKCUDA(cudaGraphicsMapResources(3, cudaVBOResources, stream));	//首先映射资源

	// 获得buffer
	void* dptr;			// 用于获取cuda资源的地址(重复使用)
	size_t bufferSize;	// 用于获取cuda资源buffer的大小
	const size_t validSurfelsNum = geometry.ValidSurfelsNum();	// 有效的面元数量
/** 在使用cudaGraphicsMapResources函数将资源映射到上下文之后，使用cudaGraphicsResourceGetMappedPointer 函数来获取该资源在设备内存中的指针 **/
	//参考顶点置信度
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[0]));//把cudaVBOResources[0]在设备上的地址丢给了dptr，大小给了buffersize
	geometry.CanonicalVertexConfidence = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);//这里是把CanonicalVertexConfidence连接给了cudaVBOResources[0]
	//这样就能借助geometry里的CanonicalVertexConfidence操作cuda上对应的数据了
	//printf("map fusionDepthSurfelGepmetryVBO里： %#x \n", geometry.CanonicalVertexConfidence.ArrayView().RawPtr());

	//参考法线-半径
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[1]));
	geometry.CanonicalNormalRadius = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//观察时间
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[2]));
	geometry.ColorTime = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::mapToCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文

	/** 使用cudaGraphicsMapResources函数将资源映射到上下文 **/

	CHECKCUDA(cudaGraphicsMapResources(3, cudaVBOResources, stream));		// 映射图形资源以供 CUDA 访问
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::unmapFromCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// 绑定调用OpenGL的上下文

	/** 用于取消映射已映射的 CUDA 图形资源，使其不再与当前的 CUDA 上下文关联 **/
	CHECKCUDA(cudaGraphicsUnmapResources(3, cudaVBOResources, stream));	// 取消已经映射的cuda上下文资源
	// 函数执行成功后，已映射的资源将与当前的 CUDA 上下文分离，不再可用于 GPU 上的操作。
	// 这样可以确保资源在主机和设备之间的一致性，并释放相关资源的内存
}
