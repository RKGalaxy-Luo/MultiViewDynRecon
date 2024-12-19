/*****************************************************************//**
 * \file   GLSurfelGeometryVBO.h
 * \brief  ��������(����)��Ԫ����Ⱦ����
 * 
 * \author LUO
 * \date   February 21st 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <core/Geometry/SurfelGeometry.h>
#include <core/Geometry/FusionDepthGeometry.h>
#include <cuda_gl_interop.h>	// ��������ע�ᵽcuda
#include <base/Constants.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>

namespace SparseSurfelFusion {
	//����ÿ֡�����Ԫ�ںϵ�VBO
	struct GLfusionDepthSurfelVBO
	{
		// ���㻺������Ӧ��SurfelGeometry�ĳ�Ա
		GLuint CanonicalVertexConfidence;				// �ο���������
		GLuint CanonicalNormalRadius;					// �ο����߼��뾶
		GLuint ColorTime;								// �۲쵽��ʱ��

		cudaGraphicsResource_t cudaVBOResources[3];		// ��SurfelGeometryͼ����ص�CUDA��Դ

		GLFWwindow* GLFW_Window = NULL;

		// ����ֻ���ɳ���������
		/**
		 * \brief ��ʼ����ǰ�������.
		 *
		 */
		void initialize();

		/**
		 * \brief �ͷŵ�ǰ�������.
		 *
		 */
		void release();

		/**
		 * \brief ��SurfelGeometryӳ�䵽CUDA.
		 *
		 * \param geometry ��Ԫ������(��Ҫ��Ⱦ��)
		 * \param stream CUDA��ID
		 */
		void mapToCuda(FusionDepthGeometry& geometry, cudaStream_t stream = 0);
		/**
		 * \brief ��SurfelGeometryӳ�䵽CUDA(��һ������ CUDA ͼ����Դ������ CUDA ͼ��CUDA ����ȣ�ӳ�䵽��ǰ��CUDA��������)
		 *		����ִ�гɹ���ӳ�����Դ���뵱ǰ��CUDA�����Ĺ���������������GPU�Ͻ��ж�д����.
		 *
		 * \param stream CUDA��ID
		 */
		void mapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief ��������cuda�����ڸ����̣߳�Ϊ�˺���OpenGL��ͼ��pipeline����ִ�гɹ�����ӳ�����Դ���뵱ǰ�� CUDA �����ķ��룬
		 *		���ٿ����� GPU �ϵĲ�������������ȷ����Դ���������豸֮���һ���ԣ����ͷ������Դ���ڴ�.
		 *
		 * \param stream CUDA��ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

	};



	/**
	 * \brief һ���࣬����ά��һ��surfel������ʵ�������ж��㻺������Լ���cuda�Ϸ��ʵ���Դ��
	 *        �����ֻ������Ⱦ����ʹ�á�
	 */
	struct GLSurfelGeometryVBO
	{
		// ���㻺������Ӧ��SurfelGeometry�ĳ�Ա
		GLuint CanonicalVertexConfidence;				// �ο���������
		GLuint CanonicalNormalRadius;					// �ο����߼��뾶
		GLuint LiveVertexConfidence;					// ʵʱ��������
		GLuint LiveNormalRadius;						// ʵʱ���߼��뾶
		GLuint ColorTime;								// �۲쵽��ʱ��

		cudaGraphicsResource_t cudaVBOResources[5];		// ��SurfelGeometryͼ����ص�CUDA��Դ

		GLFWwindow* GLFW_Window = NULL;

		// ����ֻ���ɳ���������
		/**
		 * \brief ��ʼ����ǰ�������.
		 * 
		 */
		void initialize();

		/**
		 * \brief �ͷŵ�ǰ�������.
		 * 
		 */
		void release();

		/**
		 * \brief ��SurfelGeometryӳ�䵽CUDA.
		 * 
		 * \param geometry ��Ԫ������(��Ҫ��Ⱦ��)
		 * \param stream CUDA��ID
		 */
		void mapToCuda(SurfelGeometry& geometry, cudaStream_t stream = 0);
		/**
		 * \brief ��SurfelGeometryӳ�䵽CUDA(��һ������ CUDA ͼ����Դ������ CUDA ͼ��CUDA ����ȣ�ӳ�䵽��ǰ��CUDA��������)
		 *		����ִ�гɹ���ӳ�����Դ���뵱ǰ��CUDA�����Ĺ���������������GPU�Ͻ��ж�д����.
		 * 
		 * \param stream CUDA��ID
		 */
		void mapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief ��������cuda�����ڸ����̣߳�Ϊ�˺���OpenGL��ͼ��pipeline����ִ�гɹ�����ӳ�����Դ���뵱ǰ�� CUDA �����ķ��룬
		 *		���ٿ����� GPU �ϵĲ�������������ȷ����Դ���������豸֮���һ���ԣ����ͷ������Դ���ڴ�.
		 * 
		 * \param stream CUDA��ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

	};


	/**
	 * \brief ��SparseSurfelFusion�����ռ��ȫ�ֺ�������ʼ����Ⱦ��Ԫ��pipeline(������Ⱦ���̣�����д������).
	 * 
	 * \param ������Ҫ��Ⱦ����Ԫ�������Ե���.
	 */
	void initializeGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO);
	void initializeGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO);

	/**
	 * \brief ��SparseSurfelFusion�����ռ��ȫ�ֺ������ͷ���Ⱦ��Ԫ��pipeline(������Ⱦ���̣�����д������).
	 * 
	 * \param ������Ҫ�ͷŵ���Ԫ�������Ե���.
	 */
	void releaseGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO);
	void releaseGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO);

}


