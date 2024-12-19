/*****************************************************************//**
 * \file   GLRenderMaps.h
 * \brief  ��OpenGL����Դ��CUDA��Դ����ӳ�䡾����CUDAֱ�ӵ���OpenGL��Դ������Ⱦ������Ⱦ������Ԫ�������
 * 
 * \author LUO
 * \date   February 22nd 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>
#include <base/CommonTypes.h>
#include <base/Logging.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>

namespace SparseSurfelFusion {
	//����ÿ֡������ںϣ�fusionDepthSurfel����FBO��RBO  ��������������fusionmap�ĺ�����������ʹ��fusionmap��
	struct GLFusionDepthSurfelFrameRenderBufferObjects {
		// FBO �������������Ļ�������ִ����Ⱦ��������Щ������������Ϊ�������ȾĿ�깩��������Ⱦ����ʹ�á�
		// ͨ��ʹ�� FBO��������ʵ��һЩ�߼���Ⱦ����������Ⱦ��������Ӱӳ�䡢��Ļ����Ч�������ز�������ݵȡ�
		GLuint fusionDepthSurfelFBO; // ֡�������(FBO)��ʶ��

		// ��Щ����Ⱦ����������
		GLuint canonicalVertexMap;	// canonical�򶥵�ͼ
		GLuint canonicalNormalMap;	// canonical����ͼ
		GLuint colorTimeMap;	// Color-Timeͼ
		GLuint depthBuffer;		// ���Buffer �����֪����ɶ��


		// cuda���ʵĻ�����
		// RBO��Render Buffer Object���� OpenGL �е�һ�ֶ������ڴ洢��Ⱦ�����Ľ�����ݡ�
		// ���ṩ��һ�ָ�Ч�ķ�ʽ���洢��ȡ�ģ�����ɫ�����������ݣ���������������
		cudaGraphicsResource_t cudaRBOResources[3];		// cuda�д洢��OpenGL��RBO��Դ���
		cudaArray_t cudaMappedArrays[3];				// ��cudaArray_t����RBO��Դ
		cudaTextureObject_t cudaMappedTexture[3];		// �����յ�cudaArray_t��Դת����CUDA������Դ



		//ֻ������Ⱦ������
		/**
		 * \brief ��ʼ��֡�������(FBO)����Ⱦ�������(RBO)��������Ԫ���Ե�RBO�󶨵�FBO��.
		 *
		 * \param scaledWidth  ��Ҫ�������Ⱦ������(RBO)����Ŀ��
		 * \param scaledHeight ��Ҫ�������Ⱦ������(RBO)����ĸ߶�
		 */
		void initialize(int scaledWidth, int scaledHeight);

		/**
		 * \brief ���CUDA��OpenGL֮���ӳ�䣬�ͷ�֡�������(FBO)����Ⱦ�������(RBO)�ڴ�.
		 *
		 */
		void release();

		/**
		 * \brief OpenGL�е���Ⱦ�������(RBO)��Դת����CUDA������󣬱����ڳ�ʼ�������.
		 *
		 * \param liveVertexTexture live���еĶ�������
		 * \param liveNormalTexture live���еķ�������
		 * \param indexTexture ����ͼ����
		 * \param colorTimeTexture Color-Time����
		 * \param stream CUDA��ID
		 */
		void mapToCuda(
			cudaTextureObject_t& canonicalvertextexture,
			cudaTextureObject_t& canonicalnormaltexture,
			cudaTextureObject_t& colorTimeTexture,
			cudaStream_t stream = 0
		);
		/**
		 * \brief ȡ����ǰCUDA���У�OpenGL��CUDA֮�����Դӳ���ϵ.
		 *
		 * \param stream CUDA��ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);
	};

	/**
	 * \brief ����FusionMap��֡�������(FBO)����Ⱦ�������(RBO)������ṹ�崦���cuda��ӳ��
	 *         
	 */
	struct GLFusionMapsFrameRenderBufferObjects {
		// FBO �������������Ļ�������ִ����Ⱦ��������Щ������������Ϊ�������ȾĿ�깩��������Ⱦ����ʹ�á�
		// ͨ��ʹ�� FBO��������ʵ��һЩ�߼���Ⱦ����������Ⱦ��������Ӱӳ�䡢��Ļ����Ч�������ز�������ݵȡ�
		GLuint fusionMapFBO; // ֡�������(FBO)��ʶ��

		// ��Щ����Ⱦ����������
		GLuint liveVertexMap;	// Live�򶥵�ͼ
		GLuint liveNormalMap;	// Live����ͼ
		GLuint indexMap;		// Live������ͼ
		GLuint colorTimeMap;	// Color-Timeͼ
		GLuint depthBuffer;		// ���Buffer

		//GLuint flag;			// ��־λ

		// cuda���ʵĻ�����
		// RBO��Render Buffer Object���� OpenGL �е�һ�ֶ������ڴ洢��Ⱦ�����Ľ�����ݡ�
		// ���ṩ��һ�ָ�Ч�ķ�ʽ���洢��ȡ�ģ�����ɫ�����������ݣ���������������
		cudaGraphicsResource_t cudaRBOResources[4];		// cuda�д洢��OpenGL��RBO��Դ���
		cudaArray_t cudaMappedArrays[4];				// ��cudaArray_t����RBO��Դ
		cudaTextureObject_t cudaMappedTexture[4];		// �����յ�cudaArray_t��Դת����CUDA������Դ

		//ֻ������Ⱦ������
		/**
		 * \brief ��ʼ��֡�������(FBO)����Ⱦ�������(RBO)��������Ԫ���Ե�RBO�󶨵�FBO��.
		 * 
		 * \param scaledWidth  ��Ҫ�������Ⱦ������(RBO)����Ŀ��
		 * \param scaledHeight ��Ҫ�������Ⱦ������(RBO)����ĸ߶�
		 */
		void initialize(int scaledWidth, int scaledHeight);

		/**
		 * \brief ���CUDA��OpenGL֮���ӳ�䣬�ͷ�֡�������(FBO)����Ⱦ�������(RBO)�ڴ�.
		 * 
		 */
		void release();

		/**
		 * \brief OpenGL�е���Ⱦ�������(RBO)��Դת����CUDA������󣬱����ڳ�ʼ�������.
		 * 
		 * \param liveVertexTexture live���еĶ�������
		 * \param liveNormalTexture live���еķ�������
		 * \param indexTexture ����ͼ����
		 * \param colorTimeTexture Color-Time����
		 * \param stream CUDA��ID
		 */
		void mapToCuda(cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& colorTimeTexture, cudaStream_t stream = 0);
		/**
		 * \brief ȡ����ǰCUDA���У�OpenGL��CUDA֮�����Դӳ���ϵ.
		 * 
		 * \param stream CUDA��ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);
	};


	/**
	 * \brief ����SolverMap��֡�������(FBO)����Ⱦ�������(RBO)����OpenGL��Щ��������ӳ�䵽cuda��.
	 *
	 */
	struct GLSolverMapsFrameRenderBufferObjects {
		GLuint solverMapFBO;	// SolerMap��֡�������(FBO)

		// ��Щ����Ⱦ����������RBO
		GLuint canonicalVertexMap;		// canonical�򶥵�ͼ
		GLuint canonicalNormalMap;		// canonical����ͼ
		GLuint liveVertexMap;			// live�򶥵�ͼ
		GLuint liveNormalMap;			// live����ͼ
		GLuint indexMap;				// ����ͼ
		GLuint normalizedRGBMap;		// ��һ����RGBͼ
		GLuint depthBuffer;				// ��Ȼ���

		// cuda���ʵ���Դ
		cudaGraphicsResource_t  cudaRBOResources[6];		// OpenGLӳ�䵽CUDA����Դ
		cudaArray_t cudaMappedArrays[6];					// ��cudaArray_t����CUDA��Դ
		cudaTextureObject_t cudaMappedTexture[6];			// ��cudaArray_tת��������ͨ�����ô��δ��ݳ�ȥ

		
		/**
		 * \brief ��ʼ��֡�������(FBO)����Ⱦ�������(RBO)�����ʼ��ֻ������Ⱦ�з���.
		 * 
		 * \param width  ��Ⱦ�������Ŀ��
		 * \param height ��Ⱦ�������ĸ߶�
		 */
		void initialize(int width, int height);

		/**
		 * \brief �ͷ�֡�������(FBO)����Ⱦ�������(RBO).
		 * 
		 */
		void release();

		/**
		 * \brief ��OpenGL�е���Ⱦ�������(RBO)��Դת����CUDA������󣬱����ڳ�ʼ�������.
		 * 
		 * \param canonicalVertexTexture	�����������OpenGL�е�canonical���ж�������
		 * \param canonicalNormalTexture	�����������OpenGL�е�canonical���з�������
		 * \param liveVertexTexture			�����������OpenGL�е�live���ж�������
		 * \param liveNormalMapTexture		�����������OpenGL�е�live���ж�������
		 * \param indexTexture				�����������OpenGL�е�����ͼ����
		 * \param normalizedRGBTexture		�����������OpenGL�еĹ�һ��RGB����
		 * \param stream CUDA��ID
		 */
		void mapToCuda(cudaTextureObject_t& canonicalVertexTexture, cudaTextureObject_t& canonicalNormalTexture, cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalMapTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& normalizedRGBTexture, cudaStream_t stream = 0);
		
		/**
		 * \brief ȡ����ǰCUDA���У�OpenGL��CUDA֮�����Դӳ���ϵ.
		 * 
		 * \param stream CUDA��ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);
	};



	/**
	 * \brief float4�����������Ⱦͼ������������Ҫӳ�䵽cuda����������Ҫ���߱���.
	 */
	struct GLOfflineVisualizationFrameRenderBufferObjects {

		/**
		 * \brief ��Ⱦ���ͣ���������Live�� �� Canonical����Ⱦ��ʽ��Normal��Albedo��Phong.
		 */
		enum RenderType {
			LiveNormalMap,			// Live��Normal��Ⱦ����
			LiveAlbedoMap,			// Live��Albedo��Ⱦ����
			LivePhongMap,			// Live��Phong��Ⱦ����
			CanonicalNormalMap,		// Canonical��Normal��Ⱦ����
			CanonicalAlbedoMap,		// Canonical��Albedo��Ⱦ����
			CanonicalPhongMap		// Canonical��Phong��Ⱦ����
		};

		GLuint visualizationMapFBO;
		GLuint normalizedRGBARBO;		// һ��float4������Ԫ����[0,1]
		GLuint depthBuffer;

		unsigned int frameIndex = 0;	// ���ڼ�¼����ͼƬ��index

		/**
		 * \brief ��ʼ��.
		 * 
		 * \param width 
		 * \param height
		 */
		void initialize(int width, int height);
		/**
		 * \brief ����.
		 * 
		 */
		void release();

		/**
		 * \brief ��ͼƬ���浽����.
		 * 
		 * \param path �ļ�·��
		 */
		void save(const std::string& path);

		/**
		 * \brief ��ʾͼ��.
		 * 
		 * \param symbol
		 */
		void show(RenderType symbol);
	};
}
