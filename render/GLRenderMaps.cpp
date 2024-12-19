#include "GLRenderMaps.h"
#include <base/data_transfer.h>

/**
 * \brief 生成默认的(为0)的资源描述符数组.
 * 
 * \return 默认的(为0)的cuda descriptor 的数组
 */
static const cudaResourceDesc& resourceDescCudaArray() {
	static cudaResourceDesc desc;				// cudaResourceDesc 结构体，用于描述 CUDA 资源
	memset(&desc, 0, sizeof(cudaResourceDesc));	// 使用 memset 函数将 desc 结构体清零
	desc.resType = cudaResourceTypeArray;		// 设置 desc.resType 为 cudaResourceTypeArray，表示该资源是一个 CUDA 数组
	return desc;
}
/**
 * \brief 生成默认(为0)的二维纹理描述符.
 * 
 * \return 默认(为0)的二维纹理描述子
 */
static const cudaTextureDesc& textureDescDefault2d() {
	static cudaTextureDesc desc;				// 一个 cudaTextureDesc 结构体，用于描述 CUDA 纹理
	memset(&desc, 0, sizeof(cudaTextureDesc));	// 首先使用 memset 函数将 desc 结构体清零，然后设置一些默认的纹理属性
	desc.addressMode[0] = cudaAddressModeBorder;// desc.addressMode[0] 和 desc.addressMode[1] 设置为 cudaAddressModeBorder，表示纹理坐标超出范围时使用边界颜色
	desc.addressMode[1] = cudaAddressModeBorder;
	desc.filterMode = cudaFilterModePoint;		// desc.filterMode 设置为 cudaFilterModePoint，表示纹理采样时使用最近邻插值
	desc.readMode = cudaReadModeElementType;	// desc.readMode 设置为 cudaReadModeElementType，表示纹理采样时返回的值类型与纹理元素的数据类型相同
	desc.normalizedCoords = 0;					// desc.normalizedCoords 设置为 0，表示纹理坐标是非归一化的
	return desc;
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::initialize(int scaledWidth, int scaledHeight)
{
	// 生成一个FBO对象
	glGenFramebuffers(1, &fusionMapFBO);

/********************** 生成一个渲染缓冲区对象(RBO)，下面表示分配这个帧的渲染缓冲区标识符给GLuint **********************/ 
	// 参数1：生成缓冲区个数
	// 参数2：将缓冲区标识符赋给这个GLuint
	glGenRenderbuffers(1, &liveVertexMap);
	glGenRenderbuffers(1, &liveNormalMap);
	glGenRenderbuffers(1, &colorTimeMap);
	glGenRenderbuffers(1, &depthBuffer);

	glGenRenderbuffers(1, &indexMap);
	//glGenRenderbuffers(1, &flag);

/********************** 生成一个渲染缓冲区对象(RBO)，下面表示为上述渲染缓冲区(RBO)分配数据存储 **********************/
	// 参数1：将渲染缓冲区对象 liveVertexMap 绑定到目标 GL_RENDERBUFFER (RBO)上。这意味着后续的渲染缓冲区操作将应用于 liveVertexMap 对象。
	// 参数2：缓冲区标识符。后续操作都是对当前这个缓冲区而言。
	glBindRenderbuffer(GL_RENDERBUFFER, liveVertexMap);
	// 参数1：表示此时目标是渲染缓冲区(RBO)
	// 参数2：GL_RGBA32F 是渲染缓冲区的内部格式，表示每个像素由四个浮点数（32 位浮点数）组成，分别表示红、绿、蓝和透明度通道。
	// 参数3：渲染缓冲区的宽度
	// 参数4：渲染缓冲区的高度
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, liveNormalMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, colorTimeMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, scaledWidth, scaledHeight);

	//glBindRenderbuffer(GL_RENDERBUFFER, flag);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_R32I, scaledWidth, scaledHeight);

	// 索引映射中的数据必须为unsigned int 
	glBindRenderbuffer(GL_RENDERBUFFER, indexMap); // 将indexMap绑定RBO
	// GL_R32UI   ->   R通道(单通道)32位无符号整型		【GL_LUMINANCE32UI_EXT在OpenGL3.3后就没有了】
	glRenderbufferStorage(GL_RENDERBUFFER, GL_R32UI, scaledWidth, scaledHeight); // 指定这个RBO的存储格式、宽度和高度


/***** 绑定FBO到帧缓冲区，后续默认对当前帧缓冲区进行操作，下述是将渲染缓冲区对象(RBO)附加到当前绑定的帧缓冲对象fusionMapFBO的颜色附件上 *****/
	// 参数1：表示将 FBO 绑定到帧缓冲目标上
	// 参数2：这是要绑定的帧缓冲对象的标识符。fusionMapFBO 是之前通过 glGenFramebuffers 函数生成的 FBO 的标识符
	glBindFramebuffer(GL_FRAMEBUFFER, fusionMapFBO);
	// 参数1：帧缓冲目标，表示当前绑定的帧缓冲对象是目标(默认是上面的FBO)。
	// 参数2：颜色附件的索引，表示将渲染缓冲区对象附加到颜色附件 0 上(数字不同表示附件不同)
	// 参数3：附件类型，表示附件是一个渲染缓冲区(RBO)对象
	// 参数4：附加的渲染缓冲区对象，传递前面生成的标识符
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, liveVertexMap);	// 将帧缓冲区对象绑定到fusionMapFBO
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, liveNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, indexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, colorTimeMap);
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_RENDERBUFFER, flag);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

/****************************** 检查帧缓冲对象的完整性，以确保帧缓冲对象设置正确并可以正常使用 ******************************/
	/** glCheckFramebufferStatus(GL_FRAMEBUFFER)：
	 *  这是一个 OpenGL 函数调用，用于检查当前绑定的帧缓冲对象的完整性。它接受一个参数 GL_FRAMEBUFFER，表示要检查的帧缓冲对象是当前绑定的帧缓冲对象。
	 * 
	 *  GL_FRAMEBUFFER_COMPLETE：
	 *  这是一个常量，表示帧缓冲对象的完整性状态。当帧缓冲对象完整且设置正确时，glCheckFramebufferStatus 函数将返回这个常量。 
	 */
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "面元融合的帧Buffer不完整";
	}

	GLuint drawBuffers[] = { // 指定要配置的帧缓冲对象(FBO)的渲染缓冲区(RBO)附件
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3
	};
	// 设置帧缓冲对象的渲染缓冲区附件
	// 参数1：帧缓冲对象(FBO)附件的数量，即 drawBuffers 数组中的元素个数
	// 参数2：一个指向帧缓冲对象(FBO)附件数组的指针，用于指定要配置的附件
	glDrawBuffers(4, drawBuffers);

	// 解绑帧缓冲区FBO，上述已经将FBO所有设置调整好，并绑定到了标志位为fusionMapFBO的FBO中
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

/**************************************** 将 OpenGL 渲染缓冲区对象（RBO）注册为 CUDA 图像资源 ****************************************/
	// 通过调用 cudaGraphicsGLRegisterImage 函数，将 OpenGL 渲染缓冲区对象 liveVertexMap 注册为 CUDA 图像资源，
	// 并将其句柄存储在 cudaRBOResources[0] 中。这样就可以在 CUDA 中使用这个图像资源进行只读操作
	// 参数1：CUDA 图像资源对象，用于存储与 OpenGL 渲染缓冲区对象关联的 CUDA 图像资源句柄。这是一个输出参数，函数将为其赋值
	// 参数2：要注册的 OpenGL 渲染缓冲区对象(RBO)的句柄
	// 参数3：目标类型，表示要注册的是渲染缓冲区对象
	// 参数4：表示注册的 CUDA 图像资源将被视为只读资源
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[0], liveVertexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[1], liveNormalMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[2], indexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[3], colorTimeMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::release()
{
	// 释放对这些缓冲区的cuda访问，解除CUDA和OpenGL之间的映射关系
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[3]));

	// 释放渲染缓冲对象和帧缓冲对象
	glDeleteRenderbuffers(1, &liveVertexMap);
	glDeleteRenderbuffers(1, &liveNormalMap);
	glDeleteRenderbuffers(1, &indexMap);
	glDeleteRenderbuffers(1, &colorTimeMap);
	glDeleteRenderbuffers(1, &depthBuffer);

	// 清除帧缓冲区对象
	glDeleteFramebuffers(1, &fusionMapFBO);
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::mapToCuda(cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& colorTimeTexture, cudaStream_t stream)
{
	// cudaGraphicsMapResources 函数将先前注册的 CUDA 图像资源 cudaRBOResources 映射到 CUDA 上下文
	// 参数1：要映射的 CUDA 图像资源的数量
	// 参数2：CUDA 图像资源数组，包含先前注册的 CUDA 图像资源
	// 参数3：一个 CUDA 流，用于指定映射操作所在的流
	CHECKCUDA(cudaGraphicsMapResources(4, cudaRBOResources, stream));

	// cudaGraphicsSubResourceGetMappedArray 函数用于获取映射数组，即从映射的 CUDA 图像资源中获取数组句柄
	// 参数1：&(cudaMappedArrays[0]) 是一个指向 CUDA 数组的指针，用于存储获取的数组句柄。这是一个输出参数，函数将为其赋值
	// 残数2：cudaRBOResources[0] 是要获取映射数组的 CUDA 图像资源，将这个数组的句柄赋值给cudaMappedArrays[0]
	// 参数3：要获取的子资源索引
	// 参数4：要获取的数组索引
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[0]), cudaRBOResources[0], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[1]), cudaRBOResources[1], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[2]), cudaRBOResources[2], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[3]), cudaRBOResources[3], 0, 0));

	// 创建纹理
	cudaResourceDesc resourceDesc = resourceDescCudaArray();	// 资源描述符：这里设置了资源描述符默认的属性
	cudaTextureDesc textureDesc = textureDescDefault2d();		// 纹理描述符：这里设置了纹理描述符默认的属性
	resourceDesc.res.array.array = cudaMappedArrays[0];	// 资源描述符中的数组资源用cudaMappedArrays赋值
	// 参数1：&(cudaMappedTexture[0]) 是一个指向 CUDA 纹理对象的指针，用于存储创建的纹理对象。这是一个输出参数，函数将为其赋值
	// 参数2：&resourceDesc 是一个指向 CUDA 资源描述符的指针，描述了要用于纹理的 CUDA 数组
	// 参数3：&textureDesc 是一个指向 CUDA 纹理描述符的指针，描述了纹理的属性
	// 参数4：NULL 是一个可选的 CUDA 配置指针，用于指定纹理对象的配置
	cudaCreateTextureObject(&(cudaMappedTexture[0]), &resourceDesc, &textureDesc, NULL);	// 使用设置好的资源描述符和纹理描述符，创建纹理对象，并存储到cudaMappedTexture[0]中
	resourceDesc.res.array.array = cudaMappedArrays[1];
	cudaCreateTextureObject(&(cudaMappedTexture[1]), &resourceDesc, &textureDesc, NULL);
	resourceDesc.res.array.array = cudaMappedArrays[2];
	cudaCreateTextureObject(&(cudaMappedTexture[2]), &resourceDesc, &textureDesc, NULL);
	resourceDesc.res.array.array = cudaMappedArrays[3];
	cudaCreateTextureObject(&(cudaMappedTexture[3]), &resourceDesc, &textureDesc, NULL);
	

	// 将结果存储到函数传递进来的地址中，引用赋值
	liveVertexTexture = cudaMappedTexture[0];
	liveNormalTexture = cudaMappedTexture[1];
	indexTexture = cudaMappedTexture[2];
	colorTimeTexture = cudaMappedTexture[3];
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::unmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(4, cudaRBOResources, stream));
}

void SparseSurfelFusion::GLSolverMapsFrameRenderBufferObjects::initialize(int width, int height)
{
	// 创建帧缓冲对象FBO
	glGenFramebuffers(1, &solverMapFBO);

	//这个帧的渲染Buffer
	glGenRenderbuffers(1, &canonicalVertexMap);
	glGenRenderbuffers(1, &canonicalNormalMap);
	glGenRenderbuffers(1, &liveVertexMap);
	glGenRenderbuffers(1, &liveNormalMap);
	glGenRenderbuffers(1, &indexMap);
	glGenRenderbuffers(1, &normalizedRGBMap);
	glGenRenderbuffers(1, &depthBuffer);

	//为渲染缓冲区分配数据存储
	glBindRenderbuffer(GL_RENDERBUFFER, canonicalVertexMap);			//0
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, canonicalNormalMap);			//1
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, liveVertexMap);					//2
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, liveNormalMap);					//3
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, normalizedRGBMap);				//4
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);					//5
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);

	//索引映射中的数据必须为unsigned int
	glBindRenderbuffer(GL_RENDERBUFFER, indexMap);						//6
	// GL_R32UI   ->   R通道(单通道)32位无符号整型		【GL_LUMINANCE32UI_EXT在OpenGL3.3后就没有了】
	glRenderbufferStorage(GL_RENDERBUFFER, GL_R32UI, width, height);

	//将渲染缓冲区附加到framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, solverMapFBO);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, canonicalVertexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, canonicalNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, liveVertexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, liveNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_RENDERBUFFER, indexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_RENDERBUFFER, normalizedRGBMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER, depthBuffer);

	//检查framebuffer附件
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "面元融合的帧Buffer不完整";
	}

	GLuint drawBuffers[] = {	// 指定要配置的帧缓冲对象(FBO)的渲染缓冲区(RBO)附件
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3,
		GL_COLOR_ATTACHMENT4,
		GL_COLOR_ATTACHMENT5
	};
	glDrawBuffers(6, drawBuffers);

	// 解绑FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//初始化cuda资源
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[0], canonicalVertexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[1], canonicalNormalMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[2], liveVertexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[3], liveNormalMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[4], indexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[5], normalizedRGBMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::GLSolverMapsFrameRenderBufferObjects::release()
{
	// Release the resource by cuda
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaRBOResources[0]));
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaRBOResources[1]));
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaRBOResources[2]));
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaRBOResources[3]));
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaRBOResources[4]));
	CHECKCUDA(cudaGraphicsUnregisterResource(cudaRBOResources[5]));

	// Now we can release the buffer
	glDeleteRenderbuffers(1, &canonicalVertexMap);
	glDeleteRenderbuffers(1, &canonicalNormalMap);
	glDeleteRenderbuffers(1, &liveVertexMap);
	glDeleteRenderbuffers(1, &liveNormalMap);
	glDeleteRenderbuffers(1, &indexMap);
	glDeleteRenderbuffers(1, &normalizedRGBMap);
	glDeleteRenderbuffers(1, &depthBuffer);

	glDeleteFramebuffers(1, &solverMapFBO);
}

void SparseSurfelFusion::GLSolverMapsFrameRenderBufferObjects::mapToCuda(cudaTextureObject_t& canonicalVertexTexture, cudaTextureObject_t& canonicalNormalTexture, cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalMapTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& normalizedRGBTexture, cudaStream_t stream)
{
	//首先映射资源
	
	CHECKCUDA(cudaGraphicsMapResources(6, cudaRBOResources, stream)); //cuda_rbo_resources应该是有值的

	//cuda数组
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[0]), cudaRBOResources[0], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[1]), cudaRBOResources[1], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[2]), cudaRBOResources[2], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[3]), cudaRBOResources[3], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[4]), cudaRBOResources[4], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[5]), cudaRBOResources[5], 0, 0));

	//创建纹理
	cudaResourceDesc resource_desc = resourceDescCudaArray();
	cudaTextureDesc texture_desc = textureDescDefault2d();
	resource_desc.res.array.array = cudaMappedArrays[0];
	cudaCreateTextureObject(&(cudaMappedTexture[0]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cudaMappedArrays[1];
	cudaCreateTextureObject(&(cudaMappedTexture[1]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cudaMappedArrays[2];
	cudaCreateTextureObject(&(cudaMappedTexture[2]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cudaMappedArrays[3];
	cudaCreateTextureObject(&(cudaMappedTexture[3]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cudaMappedArrays[4];
	cudaCreateTextureObject(&(cudaMappedTexture[4]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cudaMappedArrays[5];
	cudaCreateTextureObject(&(cudaMappedTexture[5]), &resource_desc, &texture_desc, NULL);

	//储存结果
	canonicalVertexTexture = cudaMappedTexture[0];
	canonicalNormalTexture = cudaMappedTexture[1];
	liveVertexTexture = cudaMappedTexture[2];
	liveNormalMapTexture = cudaMappedTexture[3];
	indexTexture = cudaMappedTexture[4];
	normalizedRGBTexture = cudaMappedTexture[5];
}

void SparseSurfelFusion::GLSolverMapsFrameRenderBufferObjects::unmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(6, cudaRBOResources, stream));
}

void SparseSurfelFusion::GLOfflineVisualizationFrameRenderBufferObjects::initialize(int width, int height)
{
	//Generate the framebuffer object
	glGenFramebuffers(1, &visualizationMapFBO);

	//The render buffer for this frame
	glGenRenderbuffers(1, &normalizedRGBARBO);
	glGenRenderbuffers(1, &depthBuffer);

	//Allocate data storage for render buffer
	glBindRenderbuffer(GL_RENDERBUFFER, normalizedRGBARBO);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);

	//Attach the render buffer to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, visualizationMapFBO);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, normalizedRGBARBO);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

	//Check the framebuffer attachment
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "The frame buffer for visualization is not complete";
	}

	//Enable draw-buffers
	GLuint draw_buffers[] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1
	};
	glDrawBuffers(1, draw_buffers);

	//Clean-up
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void SparseSurfelFusion::GLOfflineVisualizationFrameRenderBufferObjects::release()
{
	glDeleteRenderbuffers(1, &normalizedRGBARBO);
	glDeleteRenderbuffers(1, &depthBuffer);

	glDeleteFramebuffers(1, &visualizationMapFBO);
}

void SparseSurfelFusion::GLOfflineVisualizationFrameRenderBufferObjects::save(const std::string& path)
{
	LOGGING(INFO) << "save：离线可视化未实现！";
}

void SparseSurfelFusion::GLOfflineVisualizationFrameRenderBufferObjects::show(RenderType symbol)
{
	//Bind the render buffer object
	glBindRenderbuffer(GL_RENDERBUFFER, normalizedRGBARBO);

	//First query the size of render buffer object
	GLint width, height;
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height);

	//Construct the storage
	cv::Mat rendered_map_cv(height, width, CV_8UC4);
	glBindFramebuffer(GL_FRAMEBUFFER, visualizationMapFBO);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rendered_map_cv.data);

	//Cleanup code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	switch (symbol)
	{
	case LiveNormalMap:
		//cv::namedWindow("Live域Normal图", cv::WINDOW_NORMAL);
		cv::imshow("Live域Normal图", rendered_map_cv);
		break;
	case LiveAlbedoMap:
		//cv::namedWindow("Live域Albedo图", cv::WINDOW_NORMAL);
		cv::imshow("Live域Albedo图", rendered_map_cv);
		break;
	case LivePhongMap:
		//cv::namedWindow("Live域Phong图", cv::WINDOW_NORMAL);
		cv::imshow("Live域Phong图", rendered_map_cv);
		break;
	case CanonicalNormalMap:
		//cv::namedWindow("Reference域Normal图", cv::WINDOW_NORMAL);
		cv::imshow("Reference域Normal图", rendered_map_cv);
		break;
	case CanonicalAlbedoMap:
		//cv::namedWindow("Reference域Albedo图", cv::WINDOW_NORMAL);
		cv::imshow("Reference域Albedo图", rendered_map_cv);
		break;
	case CanonicalPhongMap:
		//cv::namedWindow("Reference域Phong图", cv::WINDOW_NORMAL);
		cv::imshow("Reference域Phong图", rendered_map_cv);
		break;
	default:
		LOGGING(FATAL) << "无法绘制渲染的图";
		break;
	}
	//// 只存Live
	//if (symbol == LiveNormalMap || symbol == LivePhongMap || symbol == LiveAlbedoMap) {
	//	//将数据写入文件
	//	std::string path = "E:/Paper_3DReconstruction/Result/WithoutAdaptiveCorrectionSingleView/";
	//	std::stringstream ss;
	//	ss << std::setw(5) << std::setfill('0') << frameIndex;
	//	std::string frameIndexStr;
	//	ss >> frameIndexStr;
	//	frameIndexStr = frameIndexStr + ".bmp";
	//	std::string imageType;
	//	if (symbol == LiveNormalMap) imageType = "Normal/";
	//	else if (symbol == LivePhongMap) imageType = "Phong/";
	//	else if (symbol == LiveAlbedoMap) imageType = "Albedo/";
	//	else LOGGING(FATAL) << "Save Single View Render Map Error!";
	//	path = path + imageType + frameIndexStr;
	//	cv::Mat currentFrame;
	//	if (rendered_map_cv.channels() == 4) {
	//		cv::cvtColor(rendered_map_cv, currentFrame, cv::COLOR_BGRA2BGR);
	//		std::cout << path << std::endl;
	//	}
	//	cv::imwrite(path, currentFrame);
	//}

}
//*****************用于每帧深度融合*************************
void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::initialize(int scaledWidth, int scaledHeight)
{
	// 生成一个FBO对象
	glGenFramebuffers(1, &fusionDepthSurfelFBO);

	/********************** 生成一个渲染缓冲区对象(RBO)，下面表示分配这个帧的渲染缓冲区标识符给GLuint **********************/
		// 参数1：生成缓冲区个数
		// 参数2：将缓冲区标识符赋给这个GLuint
	glGenRenderbuffers(1, &canonicalVertexMap);
	glGenRenderbuffers(1, &canonicalNormalMap);
	glGenRenderbuffers(1, &colorTimeMap);
	glGenRenderbuffers(1, &depthBuffer);


/********************** 生成一个渲染缓冲区对象(RBO)，下面表示为上述渲染缓冲区(RBO)分配数据存储 **********************/
	// 参数1：将渲染缓冲区对象 liveVertexMap 绑定到目标 GL_RENDERBUFFER (RBO)上。这意味着后续的渲染缓冲区操作将应用于 liveVertexMap 对象。
	// 参数2：缓冲区标识符。后续操作都是对当前这个缓冲区而言。
	glBindRenderbuffer(GL_RENDERBUFFER, canonicalVertexMap);
	// 参数1：表示此时目标是渲染缓冲区(RBO)
	// 参数2：GL_RGBA32F 是渲染缓冲区的内部格式，表示每个像素由四个浮点数（32 位浮点数）组成，分别表示红、绿、蓝和透明度通道。
	// 参数3：渲染缓冲区的宽度
	// 参数4：渲染缓冲区的高度
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, canonicalNormalMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, colorTimeMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, scaledWidth, scaledHeight);


/***** 绑定FBO到帧缓冲区，后续默认对当前帧缓冲区进行操作，下述是将渲染缓冲区对象(RBO)附加到当前绑定的帧缓冲对象fusionMapFBO的颜色附件上 *****/
	// 参数1：表示将 FBO 绑定到帧缓冲目标上
	// 参数2：这是要绑定的帧缓冲对象的标识符。fusionMapFBO 是之前通过 glGenFramebuffers 函数生成的 FBO 的标识符
	glBindFramebuffer(GL_FRAMEBUFFER, fusionDepthSurfelFBO);
	// 参数1：帧缓冲目标，表示当前绑定的帧缓冲对象是目标(默认是上面的FBO)。
	// 参数2：颜色附件的索引，表示将渲染缓冲区对象附加到颜色附件 0 上(数字不同表示附件不同)
	// 参数3：附件类型，表示附件是一个渲染缓冲区(RBO)对象
	// 参数4：附加的渲染缓冲区对象，传递前面生成的标识符
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, canonicalVertexMap);	// 将帧缓冲区对象绑定到fusionMapFBO
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, canonicalNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, colorTimeMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

	/****************************** 检查帧缓冲对象的完整性，以确保帧缓冲对象设置正确并可以正常使用 ******************************/
		/** glCheckFramebufferStatus(GL_FRAMEBUFFER)：
		 *  这是一个 OpenGL 函数调用，用于检查当前绑定的帧缓冲对象的完整性。它接受一个参数 GL_FRAMEBUFFER，表示要检查的帧缓冲对象是当前绑定的帧缓冲对象。
		 *
		 *  GL_FRAMEBUFFER_COMPLETE：
		 *  这是一个常量，表示帧缓冲对象的完整性状态。当帧缓冲对象完整且设置正确时，glCheckFramebufferStatus 函数将返回这个常量。
		 */
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "hsg加的每帧深度面元融合的帧Buffer不完整";
	}

	GLuint drawBuffers[] = { // 指定要配置的帧缓冲对象(FBO)的渲染缓冲区(RBO)附件
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2
	};
	// 设置帧缓冲对象的渲染缓冲区附件
	// 参数1：帧缓冲对象(FBO)附件的数量，即 drawBuffers 数组中的元素个数
	// 参数2：一个指向帧缓冲对象(FBO)附件数组的指针，用于指定要配置的附件
	glDrawBuffers(3, drawBuffers);

	// 解绑帧缓冲区FBO，上述已经将FBO所有设置调整好，并绑定到了标志位为fusionDepthSurfelFBO的FBO中
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/**************************************** 将 OpenGL 渲染缓冲区对象（RBO）注册为 CUDA 图像资源 ****************************************/
		// 通过调用 cudaGraphicsGLRegisterImage 函数，将 OpenGL 渲染缓冲区对象 liveVertexMap 注册为 CUDA 图像资源，
		// 并将其句柄存储在 cudaRBOResources[0] 中。这样就可以在 CUDA 中使用这个图像资源进行只读操作
		// 参数1：CUDA 图像资源对象，用于存储与 OpenGL 渲染缓冲区对象关联的 CUDA 图像资源句柄。这是一个输出参数，函数将为其赋值
		// 参数2：要注册的 OpenGL 渲染缓冲区对象(RBO)的句柄
		// 参数3：目标类型，表示要注册的是渲染缓冲区对象
		// 参数4：表示注册的 CUDA 图像资源将被视为只读资源
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[0], canonicalVertexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[1], canonicalNormalMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[2], colorTimeMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::release()
{
	// 释放对这些缓冲区的cuda访问，解除CUDA和OpenGL之间的映射关系
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[2]));


	// 释放渲染缓冲对象和帧缓冲对象
	glDeleteRenderbuffers(1, &canonicalVertexMap);
	glDeleteRenderbuffers(1, &canonicalNormalMap);
	glDeleteRenderbuffers(1, &colorTimeMap);
	glDeleteRenderbuffers(1, &depthBuffer);

	// 清除帧缓冲区对象
	glDeleteFramebuffers(1, &fusionDepthSurfelFBO);
}

void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::mapToCuda(
	cudaTextureObject_t& canonicalvertextexture,
	cudaTextureObject_t& canonicalnormaltexture,
	cudaTextureObject_t& colorTimeTexture,
	cudaStream_t stream
)
{
	// cudaGraphicsMapResources 函数将先前注册的 CUDA 图像资源 cudaRBOResources 映射到 CUDA 上下文
	// 参数1：要映射的 CUDA 图像资源的数量
	// 参数2：CUDA 图像资源数组，包含先前注册的 CUDA 图像资源
	// 参数3：一个 CUDA 流，用于指定映射操作所在的流
	CHECKCUDA(cudaGraphicsMapResources(3, cudaRBOResources, stream));

	// cudaGraphicsSubResourceGetMappedArray 函数用于获取映射数组，即从映射的 CUDA 图像资源中获取数组句柄
	// 参数1：&(cudaMappedArrays[0]) 是一个指向 CUDA 数组的指针，用于存储获取的数组句柄。这是一个输出参数，函数将为其赋值
	// 残数2：cudaRBOResources[0] 是要获取映射数组的 CUDA 图像资源，将这个数组的句柄赋值给cudaMappedArrays[0]
	// 参数3：要获取的子资源索引
	// 参数4：要获取的数组索引
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[0]), cudaRBOResources[0], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[1]), cudaRBOResources[1], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[2]), cudaRBOResources[2], 0, 0));

	// 创建纹理
	cudaResourceDesc resourceDesc = resourceDescCudaArray();	// 资源描述符：这里设置了资源描述符默认的属性
	cudaTextureDesc textureDesc = textureDescDefault2d();		// 纹理描述符：这里设置了纹理描述符默认的属性
	resourceDesc.res.array.array = cudaMappedArrays[0];	// 资源描述符中的数组资源用cudaMappedArrays赋值
	// 参数1：&(cudaMappedTexture[0]) 是一个指向 CUDA 纹理对象的指针，用于存储创建的纹理对象。这是一个输出参数，函数将为其赋值
	// 参数2：&resourceDesc 是一个指向 CUDA 资源描述符的指针，描述了要用于纹理的 CUDA 数组
	// 参数3：&textureDesc 是一个指向 CUDA 纹理描述符的指针，描述了纹理的属性
	// 参数4：NULL 是一个可选的 CUDA 配置指针，用于指定纹理对象的配置
	cudaCreateTextureObject(&(cudaMappedTexture[0]), &resourceDesc, &textureDesc, NULL);	// 使用设置好的资源描述符和纹理描述符，创建纹理对象，并存储到cudaMappedTexture[0]中
	resourceDesc.res.array.array = cudaMappedArrays[1];
	cudaCreateTextureObject(&(cudaMappedTexture[1]), &resourceDesc, &textureDesc, NULL);
	resourceDesc.res.array.array = cudaMappedArrays[2];
	cudaCreateTextureObject(&(cudaMappedTexture[2]), &resourceDesc, &textureDesc, NULL);


	// 将结果存储到函数传递进来的地址中，引用赋值
	canonicalvertextexture = cudaMappedTexture[0];
	canonicalnormaltexture = cudaMappedTexture[1];
	colorTimeTexture = cudaMappedTexture[2];
}

void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::unmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(3, cudaRBOResources, stream));
}
