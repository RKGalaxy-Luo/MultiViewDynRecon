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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

/** ʹ��cudaGraphicsMapResources��������Դӳ�䵽������ **/
	
	// ����1��Ҫӳ��� CUDA ͼ����Դ��������
	// ����2��ָ�� CUDA ͼ����Դ�����ָ�롣
	// ����3����ѡ����������ָ��Ҫ������ִ��ӳ������� CUDA ����Ĭ��ֵΪ 0����ʾʹ��Ĭ������
	CHECKCUDA(cudaGraphicsMapResources(5, cudaVBOResources, stream));	//����ӳ����Դ

	// ���buffer
	void* dptr;			// ���ڻ�ȡcuda��Դ�ĵ�ַ(�ظ�ʹ��)
	size_t bufferSize;	// ���ڻ�ȡcuda��Դbuffer�Ĵ�С
	const size_t validSurfelsNum = geometry.ValidSurfelsNum();	// ��Ч����Ԫ����
/** ��ʹ��cudaGraphicsMapResources��������Դӳ�䵽������֮��ʹ��cudaGraphicsResourceGetMappedPointer ��������ȡ����Դ���豸�ڴ��е�ָ�� **/
	//�ο��������Ŷ�
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[0]));//��cudaVBOResources[0]���豸�ϵĵ�ַ������dptr����С����buffersize
	geometry.CanonicalVertexConfidence = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);//�����ǰ�CanonicalVertexConfidence���Ӹ���cudaVBOResources[0]

	//�ο�����-�뾶
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[1]));
	geometry.CanonicalNormalRadius = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//ʵʱ�������Ŷ�
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[2]));
	geometry.LiveVertexConfidence = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//ʵʱ����-�뾶
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[3]));
	geometry.LiveNormalRadius = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//�۲�ʱ��
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[4]));
	geometry.ColorTime = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);
}

void SparseSurfelFusion::GLSurfelGeometryVBO::mapToCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

	/** ʹ��cudaGraphicsMapResources��������Դӳ�䵽������ **/
	CHECKCUDA(cudaGraphicsMapResources(5, cudaVBOResources, stream));		// ӳ��ͼ����Դ�Թ� CUDA ����
}

void SparseSurfelFusion::GLSurfelGeometryVBO::unmapFromCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

	/** ����ȡ��ӳ����ӳ��� CUDA ͼ����Դ��ʹ�䲻���뵱ǰ�� CUDA �����Ĺ��� **/
	CHECKCUDA(cudaGraphicsUnmapResources(5, cudaVBOResources, stream));	// ȡ���Ѿ�ӳ���cuda��������Դ
	// ����ִ�гɹ�����ӳ�����Դ���뵱ǰ�� CUDA �����ķ��룬���ٿ����� GPU �ϵĲ�����
	// ��������ȷ����Դ���������豸֮���һ���ԣ����ͷ������Դ���ڴ�
}

void SparseSurfelFusion::initializeGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// �󶨵���OpenGL��������

/*********************   ��surfel�ĸ��������һ������������(������ GPU �ϴ洢�͹������ݵ� OpenGL ����)   *********************/
	// ����1�����ɻ���������
	// ����2���������ַ�����������Ա㽫���ɵı�ʶ���洢�ڸñ�����
	glGenBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glGenBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glGenBuffers(1, &(surfelVBO.LiveVertexConfidence));
	glGenBuffers(1, &(surfelVBO.LiveNormalRadius));
	glGenBuffers(1, &(surfelVBO.ColorTime));

	//glGenBuffers(1, &(surfelVBO.From));

/*******   ��֮ǰ���ɵĻ���������󶨵��������Ի�����Ŀ�� GL_ARRAY_BUFFER �ϣ�������һ����С���ڴ棬���ڴ洢������������   *******/
	// ����1��GL_ARRAY_BUFFER �ǻ�����Ŀ�꣬��ʾ����������������ڴ洢������������
	// ����2��֮ǰ���ɵĻ���������ı�ʶ����ͨ������������ã����û����������� GL_ARRAY_BUFFER Ŀ����а�
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalVertexConfidence);
	// ����1��GL_ARRAY_BUFFER ��ʾ��ǰ�󶨵Ļ�����������һ���������Ի�����
	// ����2��4 * sizeof(float) * Constants::maxSurfelsNum ��ʾҪ������ڴ��С
	// ����3��NULL ��ʾ��ʱ���ṩ����
	// ����4��GL_DYNAMIC_DRAW ��ʾ������������󽫱�Ƶ���޸ĺ�ʹ��
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

/*******   ����Ҫ��CUDA��OpenGL֮��������ݽ���ʱ������ʹ��cudaGraphicsGLRegisterBuffer������OpenGL����������ע��ΪCUDAͼ����Դ   *******/
/*******************   �������Ϳ����� CUDA ��������ʹ�� CUDA API �����ʺͲ����û��������󣬶�������ʽ�ؽ������ݿ���   *******************/
	// ����1������ע���CUDAͼ����Դ���������OpenGL�л����������Ӧ��cuda��Դ�������ֱ�Ӷ����cuda��Դ���в�����
	// ����2��Ҫע��� OpenGL ����������ı�ʶ�����������ʶ����Ӧ��OpenGL��bufferע�ᵽcuda�����ġ�
	// ����3����ѡ����������ָ��ע���־����ʾע�� CUDA ͼ����Դʱ��ʹ���κα�־����ע�� CUDA ͼ����Դʱ��Ӧ���ض�����Ϊ���޸�Ĭ����Ϊ��
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[0]), surfelVBO.CanonicalVertexConfidence, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[1]), surfelVBO.CanonicalNormalRadius, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[2]), surfelVBO.LiveVertexConfidence, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[3]), surfelVBO.LiveNormalRadius, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[4]), surfelVBO.ColorTime, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::initializeGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// �󶨵���OpenGL��������

	/*********************   ��surfel�ĸ��������һ������������(������ GPU �ϴ洢�͹������ݵ� OpenGL ����)   *********************/
	// ����1�����ɻ���������
	// ����2���������ַ�����������Ա㽫���ɵı�ʶ���洢�ڸñ�����
	glGenBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glGenBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glGenBuffers(1, &(surfelVBO.ColorTime));

/*******   ��֮ǰ���ɵĻ���������󶨵��������Ի�����Ŀ�� GL_ARRAY_BUFFER �ϣ�������һ����С���ڴ棬���ڴ洢������������   *******/
	// ����1��GL_ARRAY_BUFFER �ǻ�����Ŀ�꣬��ʾ����������������ڴ洢������������
	// ����2��֮ǰ���ɵĻ���������ı�ʶ����ͨ������������ã����û����������� GL_ARRAY_BUFFER Ŀ����а�
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalVertexConfidence);
	// ����1��GL_ARRAY_BUFFER ��ʾ��ǰ�󶨵Ļ�����������һ���������Ի�����
	// ����2��4 * sizeof(float) * Constants::maxSurfelsNum ��ʾҪ������ڴ��С
	// ����3��NULL ��ʾ��ʱ���ṩ����
	// ����4��GL_DYNAMIC_DRAW ��ʾ������������󽫱�Ƶ���޸ĺ�ʹ��
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.CanonicalNormalRadius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, surfelVBO.ColorTime);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::maxSurfelsNum, NULL, GL_DYNAMIC_DRAW);


/*******   ����Ҫ��CUDA��OpenGL֮��������ݽ���ʱ������ʹ��cudaGraphicsGLRegisterBuffer������OpenGL����������ע��ΪCUDAͼ����Դ   *******/
/*******************   �������Ϳ����� CUDA ��������ʹ�� CUDA API �����ʺͲ����û��������󣬶�������ʽ�ؽ������ݿ���   *******************/
	// ����1������ע���CUDAͼ����Դ���������OpenGL�л����������Ӧ��cuda��Դ�������ֱ�Ӷ����cuda��Դ���в�����
	// ����2��Ҫע��� OpenGL ����������ı�ʶ�����������ʶ����Ӧ��OpenGL��bufferע�ᵽcuda�����ġ�
	// ����3����ѡ����������ָ��ע���־����ʾע�� CUDA ͼ����Դʱ��ʹ���κα�־����ע�� CUDA ͼ����Դʱ��Ӧ���ض�����Ϊ���޸�Ĭ����Ϊ��
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[0]), surfelVBO.CanonicalVertexConfidence, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[1]), surfelVBO.CanonicalNormalRadius, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGraphicsGLRegisterBuffer(&(surfelVBO.cudaVBOResources[2]), surfelVBO.ColorTime, cudaGraphicsRegisterFlagsNone));
	CHECKCUDA(cudaGetLastError());
}

void SparseSurfelFusion::releaseGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// �󶨵���OpenGL��������

	// ע��OpenGL������ע����cuda�������е���Դ
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[0]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[1]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[2]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[3]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[4]));

	// ɾ��֮ǰ���������VBO(���㻺����)
	glDeleteBuffers(1, &(surfelVBO.CanonicalVertexConfidence));
	glDeleteBuffers(1, &(surfelVBO.CanonicalNormalRadius));
	glDeleteBuffers(1, &(surfelVBO.LiveVertexConfidence));
	glDeleteBuffers(1, &(surfelVBO.LiveNormalRadius));
	glDeleteBuffers(1, &(surfelVBO.ColorTime));
}

void SparseSurfelFusion::releaseGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO)
{
	glfwMakeContextCurrent(surfelVBO.GLFW_Window);	// �󶨵���OpenGL��������

	// ע��OpenGL������ע����cuda�������е���Դ
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[0]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[1]));
	CHECKCUDA(cudaGraphicsUnregisterResource(surfelVBO.cudaVBOResources[2]));
	// ɾ��֮ǰ���������VBO(���㻺����)
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
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

	/** ʹ��cudaGraphicsMapResources��������Դӳ�䵽������ **/

	// ����1��Ҫӳ��� CUDA ͼ����Դ��������
	// ����2��ָ�� CUDA ͼ����Դ�����ָ�롣
	// ����3����ѡ����������ָ��Ҫ������ִ��ӳ������� CUDA ����Ĭ��ֵΪ 0����ʾʹ��Ĭ������
	CHECKCUDA(cudaGraphicsMapResources(3, cudaVBOResources, stream));	//����ӳ����Դ

	// ���buffer
	void* dptr;			// ���ڻ�ȡcuda��Դ�ĵ�ַ(�ظ�ʹ��)
	size_t bufferSize;	// ���ڻ�ȡcuda��Դbuffer�Ĵ�С
	const size_t validSurfelsNum = geometry.ValidSurfelsNum();	// ��Ч����Ԫ����
/** ��ʹ��cudaGraphicsMapResources��������Դӳ�䵽������֮��ʹ��cudaGraphicsResourceGetMappedPointer ��������ȡ����Դ���豸�ڴ��е�ָ�� **/
	//�ο��������Ŷ�
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[0]));//��cudaVBOResources[0]���豸�ϵĵ�ַ������dptr����С����buffersize
	geometry.CanonicalVertexConfidence = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);//�����ǰ�CanonicalVertexConfidence���Ӹ���cudaVBOResources[0]
	//�������ܽ���geometry���CanonicalVertexConfidence����cuda�϶�Ӧ��������
	//printf("map fusionDepthSurfelGepmetryVBO� %#x \n", geometry.CanonicalVertexConfidence.ArrayView().RawPtr());

	//�ο�����-�뾶
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[1]));
	geometry.CanonicalNormalRadius = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);

	//�۲�ʱ��
	CHECKCUDA(cudaGraphicsResourceGetMappedPointer(&dptr, &bufferSize, cudaVBOResources[2]));
	geometry.ColorTime = DeviceSliceBufferArray<float4>((float4*)dptr, bufferSize / sizeof(float4), validSurfelsNum);
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::mapToCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

	/** ʹ��cudaGraphicsMapResources��������Դӳ�䵽������ **/

	CHECKCUDA(cudaGraphicsMapResources(3, cudaVBOResources, stream));		// ӳ��ͼ����Դ�Թ� CUDA ����
}

void SparseSurfelFusion::GLfusionDepthSurfelVBO::unmapFromCuda(cudaStream_t stream)
{
	glfwMakeContextCurrent(GLFW_Window);	// �󶨵���OpenGL��������

	/** ����ȡ��ӳ����ӳ��� CUDA ͼ����Դ��ʹ�䲻���뵱ǰ�� CUDA �����Ĺ��� **/
	CHECKCUDA(cudaGraphicsUnmapResources(3, cudaVBOResources, stream));	// ȡ���Ѿ�ӳ���cuda��������Դ
	// ����ִ�гɹ�����ӳ�����Դ���뵱ǰ�� CUDA �����ķ��룬���ٿ����� GPU �ϵĲ�����
	// ��������ȷ����Դ���������豸֮���һ���ԣ����ͷ������Դ���ڴ�
}
