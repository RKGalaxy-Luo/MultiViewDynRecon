#include "GLRenderMaps.h"
#include <base/data_transfer.h>

/**
 * \brief ����Ĭ�ϵ�(Ϊ0)����Դ����������.
 * 
 * \return Ĭ�ϵ�(Ϊ0)��cuda descriptor ������
 */
static const cudaResourceDesc& resourceDescCudaArray() {
	static cudaResourceDesc desc;				// cudaResourceDesc �ṹ�壬�������� CUDA ��Դ
	memset(&desc, 0, sizeof(cudaResourceDesc));	// ʹ�� memset ������ desc �ṹ������
	desc.resType = cudaResourceTypeArray;		// ���� desc.resType Ϊ cudaResourceTypeArray����ʾ����Դ��һ�� CUDA ����
	return desc;
}
/**
 * \brief ����Ĭ��(Ϊ0)�Ķ�ά����������.
 * 
 * \return Ĭ��(Ϊ0)�Ķ�ά����������
 */
static const cudaTextureDesc& textureDescDefault2d() {
	static cudaTextureDesc desc;				// һ�� cudaTextureDesc �ṹ�壬�������� CUDA ����
	memset(&desc, 0, sizeof(cudaTextureDesc));	// ����ʹ�� memset ������ desc �ṹ�����㣬Ȼ������һЩĬ�ϵ���������
	desc.addressMode[0] = cudaAddressModeBorder;// desc.addressMode[0] �� desc.addressMode[1] ����Ϊ cudaAddressModeBorder����ʾ�������곬����Χʱʹ�ñ߽���ɫ
	desc.addressMode[1] = cudaAddressModeBorder;
	desc.filterMode = cudaFilterModePoint;		// desc.filterMode ����Ϊ cudaFilterModePoint����ʾ�������ʱʹ������ڲ�ֵ
	desc.readMode = cudaReadModeElementType;	// desc.readMode ����Ϊ cudaReadModeElementType����ʾ�������ʱ���ص�ֵ����������Ԫ�ص�����������ͬ
	desc.normalizedCoords = 0;					// desc.normalizedCoords ����Ϊ 0����ʾ���������Ƿǹ�һ����
	return desc;
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::initialize(int scaledWidth, int scaledHeight)
{
	// ����һ��FBO����
	glGenFramebuffers(1, &fusionMapFBO);

/********************** ����һ����Ⱦ����������(RBO)�������ʾ�������֡����Ⱦ��������ʶ����GLuint **********************/ 
	// ����1�����ɻ���������
	// ����2������������ʶ���������GLuint
	glGenRenderbuffers(1, &liveVertexMap);
	glGenRenderbuffers(1, &liveNormalMap);
	glGenRenderbuffers(1, &colorTimeMap);
	glGenRenderbuffers(1, &depthBuffer);

	glGenRenderbuffers(1, &indexMap);
	//glGenRenderbuffers(1, &flag);

/********************** ����һ����Ⱦ����������(RBO)�������ʾΪ������Ⱦ������(RBO)�������ݴ洢 **********************/
	// ����1������Ⱦ���������� liveVertexMap �󶨵�Ŀ�� GL_RENDERBUFFER (RBO)�ϡ�����ζ�ź�������Ⱦ������������Ӧ���� liveVertexMap ����
	// ����2����������ʶ���������������ǶԵ�ǰ������������ԡ�
	glBindRenderbuffer(GL_RENDERBUFFER, liveVertexMap);
	// ����1����ʾ��ʱĿ������Ⱦ������(RBO)
	// ����2��GL_RGBA32F ����Ⱦ���������ڲ���ʽ����ʾÿ���������ĸ���������32 λ����������ɣ��ֱ��ʾ�졢�̡�����͸����ͨ����
	// ����3����Ⱦ�������Ŀ��
	// ����4����Ⱦ�������ĸ߶�
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, liveNormalMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, colorTimeMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, scaledWidth, scaledHeight);

	//glBindRenderbuffer(GL_RENDERBUFFER, flag);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_R32I, scaledWidth, scaledHeight);

	// ����ӳ���е����ݱ���Ϊunsigned int 
	glBindRenderbuffer(GL_RENDERBUFFER, indexMap); // ��indexMap��RBO
	// GL_R32UI   ->   Rͨ��(��ͨ��)32λ�޷�������		��GL_LUMINANCE32UI_EXT��OpenGL3.3���û���ˡ�
	glRenderbufferStorage(GL_RENDERBUFFER, GL_R32UI, scaledWidth, scaledHeight); // ָ�����RBO�Ĵ洢��ʽ����Ⱥ͸߶�


/***** ��FBO��֡������������Ĭ�϶Ե�ǰ֡���������в����������ǽ���Ⱦ����������(RBO)���ӵ���ǰ�󶨵�֡�������fusionMapFBO����ɫ������ *****/
	// ����1����ʾ�� FBO �󶨵�֡����Ŀ����
	// ����2������Ҫ�󶨵�֡�������ı�ʶ����fusionMapFBO ��֮ǰͨ�� glGenFramebuffers �������ɵ� FBO �ı�ʶ��
	glBindFramebuffer(GL_FRAMEBUFFER, fusionMapFBO);
	// ����1��֡����Ŀ�꣬��ʾ��ǰ�󶨵�֡���������Ŀ��(Ĭ���������FBO)��
	// ����2����ɫ��������������ʾ����Ⱦ���������󸽼ӵ���ɫ���� 0 ��(���ֲ�ͬ��ʾ������ͬ)
	// ����3���������ͣ���ʾ������һ����Ⱦ������(RBO)����
	// ����4�����ӵ���Ⱦ���������󣬴���ǰ�����ɵı�ʶ��
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, liveVertexMap);	// ��֡����������󶨵�fusionMapFBO
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, liveNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, indexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, colorTimeMap);
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_RENDERBUFFER, flag);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

/****************************** ���֡�������������ԣ���ȷ��֡�������������ȷ����������ʹ�� ******************************/
	/** glCheckFramebufferStatus(GL_FRAMEBUFFER)��
	 *  ����һ�� OpenGL �������ã����ڼ�鵱ǰ�󶨵�֡�������������ԡ�������һ������ GL_FRAMEBUFFER����ʾҪ����֡��������ǵ�ǰ�󶨵�֡�������
	 * 
	 *  GL_FRAMEBUFFER_COMPLETE��
	 *  ����һ����������ʾ֡��������������״̬����֡�������������������ȷʱ��glCheckFramebufferStatus ������������������� 
	 */
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "��Ԫ�ںϵ�֡Buffer������";
	}

	GLuint drawBuffers[] = { // ָ��Ҫ���õ�֡�������(FBO)����Ⱦ������(RBO)����
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3
	};
	// ����֡����������Ⱦ����������
	// ����1��֡�������(FBO)�������������� drawBuffers �����е�Ԫ�ظ���
	// ����2��һ��ָ��֡�������(FBO)���������ָ�룬����ָ��Ҫ���õĸ���
	glDrawBuffers(4, drawBuffers);

	// ���֡������FBO�������Ѿ���FBO�������õ����ã����󶨵��˱�־λΪfusionMapFBO��FBO��
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

/**************************************** �� OpenGL ��Ⱦ����������RBO��ע��Ϊ CUDA ͼ����Դ ****************************************/
	// ͨ������ cudaGraphicsGLRegisterImage �������� OpenGL ��Ⱦ���������� liveVertexMap ע��Ϊ CUDA ͼ����Դ��
	// ���������洢�� cudaRBOResources[0] �С������Ϳ����� CUDA ��ʹ�����ͼ����Դ����ֻ������
	// ����1��CUDA ͼ����Դ�������ڴ洢�� OpenGL ��Ⱦ��������������� CUDA ͼ����Դ���������һ�����������������Ϊ�丳ֵ
	// ����2��Ҫע��� OpenGL ��Ⱦ����������(RBO)�ľ��
	// ����3��Ŀ�����ͣ���ʾҪע�������Ⱦ����������
	// ����4����ʾע��� CUDA ͼ����Դ������Ϊֻ����Դ
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[0], liveVertexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[1], liveNormalMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[2], indexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[3], colorTimeMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::release()
{
	// �ͷŶ���Щ��������cuda���ʣ����CUDA��OpenGL֮���ӳ���ϵ
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[3]));

	// �ͷ���Ⱦ��������֡�������
	glDeleteRenderbuffers(1, &liveVertexMap);
	glDeleteRenderbuffers(1, &liveNormalMap);
	glDeleteRenderbuffers(1, &indexMap);
	glDeleteRenderbuffers(1, &colorTimeMap);
	glDeleteRenderbuffers(1, &depthBuffer);

	// ���֡����������
	glDeleteFramebuffers(1, &fusionMapFBO);
}

void SparseSurfelFusion::GLFusionMapsFrameRenderBufferObjects::mapToCuda(cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& colorTimeTexture, cudaStream_t stream)
{
	// cudaGraphicsMapResources ��������ǰע��� CUDA ͼ����Դ cudaRBOResources ӳ�䵽 CUDA ������
	// ����1��Ҫӳ��� CUDA ͼ����Դ������
	// ����2��CUDA ͼ����Դ���飬������ǰע��� CUDA ͼ����Դ
	// ����3��һ�� CUDA ��������ָ��ӳ��������ڵ���
	CHECKCUDA(cudaGraphicsMapResources(4, cudaRBOResources, stream));

	// cudaGraphicsSubResourceGetMappedArray �������ڻ�ȡӳ�����飬����ӳ��� CUDA ͼ����Դ�л�ȡ������
	// ����1��&(cudaMappedArrays[0]) ��һ��ָ�� CUDA �����ָ�룬���ڴ洢��ȡ��������������һ�����������������Ϊ�丳ֵ
	// ����2��cudaRBOResources[0] ��Ҫ��ȡӳ������� CUDA ͼ����Դ�����������ľ����ֵ��cudaMappedArrays[0]
	// ����3��Ҫ��ȡ������Դ����
	// ����4��Ҫ��ȡ����������
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[0]), cudaRBOResources[0], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[1]), cudaRBOResources[1], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[2]), cudaRBOResources[2], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[3]), cudaRBOResources[3], 0, 0));

	// ��������
	cudaResourceDesc resourceDesc = resourceDescCudaArray();	// ��Դ��������������������Դ������Ĭ�ϵ�����
	cudaTextureDesc textureDesc = textureDescDefault2d();		// ��������������������������������Ĭ�ϵ�����
	resourceDesc.res.array.array = cudaMappedArrays[0];	// ��Դ�������е�������Դ��cudaMappedArrays��ֵ
	// ����1��&(cudaMappedTexture[0]) ��һ��ָ�� CUDA ��������ָ�룬���ڴ洢�����������������һ�����������������Ϊ�丳ֵ
	// ����2��&resourceDesc ��һ��ָ�� CUDA ��Դ��������ָ�룬������Ҫ��������� CUDA ����
	// ����3��&textureDesc ��һ��ָ�� CUDA ������������ָ�룬���������������
	// ����4��NULL ��һ����ѡ�� CUDA ����ָ�룬����ָ��������������
	cudaCreateTextureObject(&(cudaMappedTexture[0]), &resourceDesc, &textureDesc, NULL);	// ʹ�����úõ���Դ������������������������������󣬲��洢��cudaMappedTexture[0]��
	resourceDesc.res.array.array = cudaMappedArrays[1];
	cudaCreateTextureObject(&(cudaMappedTexture[1]), &resourceDesc, &textureDesc, NULL);
	resourceDesc.res.array.array = cudaMappedArrays[2];
	cudaCreateTextureObject(&(cudaMappedTexture[2]), &resourceDesc, &textureDesc, NULL);
	resourceDesc.res.array.array = cudaMappedArrays[3];
	cudaCreateTextureObject(&(cudaMappedTexture[3]), &resourceDesc, &textureDesc, NULL);
	

	// ������洢���������ݽ����ĵ�ַ�У����ø�ֵ
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
	// ����֡�������FBO
	glGenFramebuffers(1, &solverMapFBO);

	//���֡����ȾBuffer
	glGenRenderbuffers(1, &canonicalVertexMap);
	glGenRenderbuffers(1, &canonicalNormalMap);
	glGenRenderbuffers(1, &liveVertexMap);
	glGenRenderbuffers(1, &liveNormalMap);
	glGenRenderbuffers(1, &indexMap);
	glGenRenderbuffers(1, &normalizedRGBMap);
	glGenRenderbuffers(1, &depthBuffer);

	//Ϊ��Ⱦ�������������ݴ洢
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

	//����ӳ���е����ݱ���Ϊunsigned int
	glBindRenderbuffer(GL_RENDERBUFFER, indexMap);						//6
	// GL_R32UI   ->   Rͨ��(��ͨ��)32λ�޷�������		��GL_LUMINANCE32UI_EXT��OpenGL3.3���û���ˡ�
	glRenderbufferStorage(GL_RENDERBUFFER, GL_R32UI, width, height);

	//����Ⱦ���������ӵ�framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, solverMapFBO);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, canonicalVertexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, canonicalNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, liveVertexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, liveNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_RENDERBUFFER, indexMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_RENDERBUFFER, normalizedRGBMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER, depthBuffer);

	//���framebuffer����
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "��Ԫ�ںϵ�֡Buffer������";
	}

	GLuint drawBuffers[] = {	// ָ��Ҫ���õ�֡�������(FBO)����Ⱦ������(RBO)����
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3,
		GL_COLOR_ATTACHMENT4,
		GL_COLOR_ATTACHMENT5
	};
	glDrawBuffers(6, drawBuffers);

	// ���FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//��ʼ��cuda��Դ
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
	//����ӳ����Դ
	
	CHECKCUDA(cudaGraphicsMapResources(6, cudaRBOResources, stream)); //cuda_rbo_resourcesӦ������ֵ��

	//cuda����
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[0]), cudaRBOResources[0], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[1]), cudaRBOResources[1], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[2]), cudaRBOResources[2], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[3]), cudaRBOResources[3], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[4]), cudaRBOResources[4], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[5]), cudaRBOResources[5], 0, 0));

	//��������
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

	//������
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
	LOGGING(INFO) << "save�����߿��ӻ�δʵ�֣�";
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
		//cv::namedWindow("Live��Normalͼ", cv::WINDOW_NORMAL);
		cv::imshow("Live��Normalͼ", rendered_map_cv);
		break;
	case LiveAlbedoMap:
		//cv::namedWindow("Live��Albedoͼ", cv::WINDOW_NORMAL);
		cv::imshow("Live��Albedoͼ", rendered_map_cv);
		break;
	case LivePhongMap:
		//cv::namedWindow("Live��Phongͼ", cv::WINDOW_NORMAL);
		cv::imshow("Live��Phongͼ", rendered_map_cv);
		break;
	case CanonicalNormalMap:
		//cv::namedWindow("Reference��Normalͼ", cv::WINDOW_NORMAL);
		cv::imshow("Reference��Normalͼ", rendered_map_cv);
		break;
	case CanonicalAlbedoMap:
		//cv::namedWindow("Reference��Albedoͼ", cv::WINDOW_NORMAL);
		cv::imshow("Reference��Albedoͼ", rendered_map_cv);
		break;
	case CanonicalPhongMap:
		//cv::namedWindow("Reference��Phongͼ", cv::WINDOW_NORMAL);
		cv::imshow("Reference��Phongͼ", rendered_map_cv);
		break;
	default:
		LOGGING(FATAL) << "�޷�������Ⱦ��ͼ";
		break;
	}
	//// ֻ��Live
	//if (symbol == LiveNormalMap || symbol == LivePhongMap || symbol == LiveAlbedoMap) {
	//	//������д���ļ�
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
//*****************����ÿ֡����ں�*************************
void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::initialize(int scaledWidth, int scaledHeight)
{
	// ����һ��FBO����
	glGenFramebuffers(1, &fusionDepthSurfelFBO);

	/********************** ����һ����Ⱦ����������(RBO)�������ʾ�������֡����Ⱦ��������ʶ����GLuint **********************/
		// ����1�����ɻ���������
		// ����2������������ʶ���������GLuint
	glGenRenderbuffers(1, &canonicalVertexMap);
	glGenRenderbuffers(1, &canonicalNormalMap);
	glGenRenderbuffers(1, &colorTimeMap);
	glGenRenderbuffers(1, &depthBuffer);


/********************** ����һ����Ⱦ����������(RBO)�������ʾΪ������Ⱦ������(RBO)�������ݴ洢 **********************/
	// ����1������Ⱦ���������� liveVertexMap �󶨵�Ŀ�� GL_RENDERBUFFER (RBO)�ϡ�����ζ�ź�������Ⱦ������������Ӧ���� liveVertexMap ����
	// ����2����������ʶ���������������ǶԵ�ǰ������������ԡ�
	glBindRenderbuffer(GL_RENDERBUFFER, canonicalVertexMap);
	// ����1����ʾ��ʱĿ������Ⱦ������(RBO)
	// ����2��GL_RGBA32F ����Ⱦ���������ڲ���ʽ����ʾÿ���������ĸ���������32 λ����������ɣ��ֱ��ʾ�졢�̡�����͸����ͨ����
	// ����3����Ⱦ�������Ŀ��
	// ����4����Ⱦ�������ĸ߶�
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, canonicalNormalMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, colorTimeMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaledWidth, scaledHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, scaledWidth, scaledHeight);


/***** ��FBO��֡������������Ĭ�϶Ե�ǰ֡���������в����������ǽ���Ⱦ����������(RBO)���ӵ���ǰ�󶨵�֡�������fusionMapFBO����ɫ������ *****/
	// ����1����ʾ�� FBO �󶨵�֡����Ŀ����
	// ����2������Ҫ�󶨵�֡�������ı�ʶ����fusionMapFBO ��֮ǰͨ�� glGenFramebuffers �������ɵ� FBO �ı�ʶ��
	glBindFramebuffer(GL_FRAMEBUFFER, fusionDepthSurfelFBO);
	// ����1��֡����Ŀ�꣬��ʾ��ǰ�󶨵�֡���������Ŀ��(Ĭ���������FBO)��
	// ����2����ɫ��������������ʾ����Ⱦ���������󸽼ӵ���ɫ���� 0 ��(���ֲ�ͬ��ʾ������ͬ)
	// ����3���������ͣ���ʾ������һ����Ⱦ������(RBO)����
	// ����4�����ӵ���Ⱦ���������󣬴���ǰ�����ɵı�ʶ��
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, canonicalVertexMap);	// ��֡����������󶨵�fusionMapFBO
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, canonicalNormalMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, colorTimeMap);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

	/****************************** ���֡�������������ԣ���ȷ��֡�������������ȷ����������ʹ�� ******************************/
		/** glCheckFramebufferStatus(GL_FRAMEBUFFER)��
		 *  ����һ�� OpenGL �������ã����ڼ�鵱ǰ�󶨵�֡�������������ԡ�������һ������ GL_FRAMEBUFFER����ʾҪ����֡��������ǵ�ǰ�󶨵�֡�������
		 *
		 *  GL_FRAMEBUFFER_COMPLETE��
		 *  ����һ����������ʾ֡��������������״̬����֡�������������������ȷʱ��glCheckFramebufferStatus �������������������
		 */
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOGGING(FATAL) << "hsg�ӵ�ÿ֡�����Ԫ�ںϵ�֡Buffer������";
	}

	GLuint drawBuffers[] = { // ָ��Ҫ���õ�֡�������(FBO)����Ⱦ������(RBO)����
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2
	};
	// ����֡����������Ⱦ����������
	// ����1��֡�������(FBO)�������������� drawBuffers �����е�Ԫ�ظ���
	// ����2��һ��ָ��֡�������(FBO)���������ָ�룬����ָ��Ҫ���õĸ���
	glDrawBuffers(3, drawBuffers);

	// ���֡������FBO�������Ѿ���FBO�������õ����ã����󶨵��˱�־λΪfusionDepthSurfelFBO��FBO��
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/**************************************** �� OpenGL ��Ⱦ����������RBO��ע��Ϊ CUDA ͼ����Դ ****************************************/
		// ͨ������ cudaGraphicsGLRegisterImage �������� OpenGL ��Ⱦ���������� liveVertexMap ע��Ϊ CUDA ͼ����Դ��
		// ���������洢�� cudaRBOResources[0] �С������Ϳ����� CUDA ��ʹ�����ͼ����Դ����ֻ������
		// ����1��CUDA ͼ����Դ�������ڴ洢�� OpenGL ��Ⱦ��������������� CUDA ͼ����Դ���������һ�����������������Ϊ�丳ֵ
		// ����2��Ҫע��� OpenGL ��Ⱦ����������(RBO)�ľ��
		// ����3��Ŀ�����ͣ���ʾҪע�������Ⱦ����������
		// ����4����ʾע��� CUDA ͼ����Դ������Ϊֻ����Դ
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[0], canonicalVertexMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[1], canonicalNormalMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGraphicsGLRegisterImage(&cudaRBOResources[2], colorTimeMap, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::release()
{
	// �ͷŶ���Щ��������cuda���ʣ����CUDA��OpenGL֮���ӳ���ϵ
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cudaRBOResources[2]));


	// �ͷ���Ⱦ��������֡�������
	glDeleteRenderbuffers(1, &canonicalVertexMap);
	glDeleteRenderbuffers(1, &canonicalNormalMap);
	glDeleteRenderbuffers(1, &colorTimeMap);
	glDeleteRenderbuffers(1, &depthBuffer);

	// ���֡����������
	glDeleteFramebuffers(1, &fusionDepthSurfelFBO);
}

void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::mapToCuda(
	cudaTextureObject_t& canonicalvertextexture,
	cudaTextureObject_t& canonicalnormaltexture,
	cudaTextureObject_t& colorTimeTexture,
	cudaStream_t stream
)
{
	// cudaGraphicsMapResources ��������ǰע��� CUDA ͼ����Դ cudaRBOResources ӳ�䵽 CUDA ������
	// ����1��Ҫӳ��� CUDA ͼ����Դ������
	// ����2��CUDA ͼ����Դ���飬������ǰע��� CUDA ͼ����Դ
	// ����3��һ�� CUDA ��������ָ��ӳ��������ڵ���
	CHECKCUDA(cudaGraphicsMapResources(3, cudaRBOResources, stream));

	// cudaGraphicsSubResourceGetMappedArray �������ڻ�ȡӳ�����飬����ӳ��� CUDA ͼ����Դ�л�ȡ������
	// ����1��&(cudaMappedArrays[0]) ��һ��ָ�� CUDA �����ָ�룬���ڴ洢��ȡ��������������һ�����������������Ϊ�丳ֵ
	// ����2��cudaRBOResources[0] ��Ҫ��ȡӳ������� CUDA ͼ����Դ�����������ľ����ֵ��cudaMappedArrays[0]
	// ����3��Ҫ��ȡ������Դ����
	// ����4��Ҫ��ȡ����������
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[0]), cudaRBOResources[0], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[1]), cudaRBOResources[1], 0, 0));
	CHECKCUDA(cudaGraphicsSubResourceGetMappedArray(&(cudaMappedArrays[2]), cudaRBOResources[2], 0, 0));

	// ��������
	cudaResourceDesc resourceDesc = resourceDescCudaArray();	// ��Դ��������������������Դ������Ĭ�ϵ�����
	cudaTextureDesc textureDesc = textureDescDefault2d();		// ��������������������������������Ĭ�ϵ�����
	resourceDesc.res.array.array = cudaMappedArrays[0];	// ��Դ�������е�������Դ��cudaMappedArrays��ֵ
	// ����1��&(cudaMappedTexture[0]) ��һ��ָ�� CUDA ��������ָ�룬���ڴ洢�����������������һ�����������������Ϊ�丳ֵ
	// ����2��&resourceDesc ��һ��ָ�� CUDA ��Դ��������ָ�룬������Ҫ��������� CUDA ����
	// ����3��&textureDesc ��һ��ָ�� CUDA ������������ָ�룬���������������
	// ����4��NULL ��һ����ѡ�� CUDA ����ָ�룬����ָ��������������
	cudaCreateTextureObject(&(cudaMappedTexture[0]), &resourceDesc, &textureDesc, NULL);	// ʹ�����úõ���Դ������������������������������󣬲��洢��cudaMappedTexture[0]��
	resourceDesc.res.array.array = cudaMappedArrays[1];
	cudaCreateTextureObject(&(cudaMappedTexture[1]), &resourceDesc, &textureDesc, NULL);
	resourceDesc.res.array.array = cudaMappedArrays[2];
	cudaCreateTextureObject(&(cudaMappedTexture[2]), &resourceDesc, &textureDesc, NULL);


	// ������洢���������ݽ����ĵ�ַ�У����ø�ֵ
	canonicalvertextexture = cudaMappedTexture[0];
	canonicalnormaltexture = cudaMappedTexture[1];
	colorTimeTexture = cudaMappedTexture[2];
}

void SparseSurfelFusion::GLFusionDepthSurfelFrameRenderBufferObjects::unmapFromCuda(cudaStream_t stream)
{
	CHECKCUDA(cudaGraphicsUnmapResources(3, cudaRBOResources, stream));
}
