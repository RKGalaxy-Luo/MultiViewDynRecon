/*****************************************************************//**
 * \file   GLRenderMaps.h
 * \brief  将OpenGL的资源与CUDA资源进行映射【方便CUDA直接调用OpenGL资源进行渲染】，渲染绘制面元，顶点等
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
	//用于每帧的深度融合（fusionDepthSurfel）的FBO和RBO  由于输入和输出和fusionmap的很像，所以主体使用fusionmap的
	struct GLFusionDepthSurfelFrameRenderBufferObjects {
		// FBO 允许您在离屏的缓冲区中执行渲染操作，这些缓冲区可以作为纹理或渲染目标供后续的渲染过程使用。
		// 通过使用 FBO，您可以实现一些高级渲染技术，如渲染到纹理、阴影映射、屏幕后处理效果、多重采样抗锯齿等。
		GLuint fusionDepthSurfelFBO; // 帧缓冲对象(FBO)标识符

		// 这些是渲染缓冲区对象
		GLuint canonicalVertexMap;	// canonical域顶点图
		GLuint canonicalNormalMap;	// canonical域法线图
		GLuint colorTimeMap;	// Color-Time图
		GLuint depthBuffer;		// 深度Buffer 这个不知道干啥的


		// cuda访问的缓冲区
		// RBO（Render Buffer Object）是 OpenGL 中的一种对象，用于存储渲染操作的结果数据。
		// 它提供了一种高效的方式来存储深度、模板和颜色缓冲区的数据，但不能用作纹理。
		cudaGraphicsResource_t cudaRBOResources[3];		// cuda中存储的OpenGL的RBO资源句柄
		cudaArray_t cudaMappedArrays[3];				// 用cudaArray_t接收RBO资源
		cudaTextureObject_t cudaMappedTexture[3];		// 将接收的cudaArray_t资源转化成CUDA纹理资源



		//只能由渲染器调用
		/**
		 * \brief 初始化帧缓冲对象(FBO)和渲染缓冲对象(RBO)，并将面元属性的RBO绑定到FBO上.
		 *
		 * \param scaledWidth  需要分配的渲染缓冲区(RBO)对象的宽度
		 * \param scaledHeight 需要分配的渲染缓冲区(RBO)对象的高度
		 */
		void initialize(int scaledWidth, int scaledHeight);

		/**
		 * \brief 解除CUDA与OpenGL之间的映射，释放帧缓冲对象(FBO)和渲染缓冲对象(RBO)内存.
		 *
		 */
		void release();

		/**
		 * \brief OpenGL中的渲染缓冲对象(RBO)资源转换成CUDA纹理对象，必须在初始化后调用.
		 *
		 * \param liveVertexTexture live域中的顶点纹理
		 * \param liveNormalTexture live域中的法线纹理
		 * \param indexTexture 索引图纹理
		 * \param colorTimeTexture Color-Time纹理
		 * \param stream CUDA流ID
		 */
		void mapToCuda(
			cudaTextureObject_t& canonicalvertextexture,
			cudaTextureObject_t& canonicalnormaltexture,
			cudaTextureObject_t& colorTimeTexture,
			cudaStream_t stream = 0
		);
		/**
		 * \brief 取消当前CUDA流中，OpenGL与CUDA之间的资源映射关系.
		 *
		 * \param stream CUDA流ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);
	};

	/**
	 * \brief 用于FusionMap的帧缓冲对象(FBO)和渲染缓冲对象(RBO)。这个结构体处理从cuda的映射
	 *         
	 */
	struct GLFusionMapsFrameRenderBufferObjects {
		// FBO 允许您在离屏的缓冲区中执行渲染操作，这些缓冲区可以作为纹理或渲染目标供后续的渲染过程使用。
		// 通过使用 FBO，您可以实现一些高级渲染技术，如渲染到纹理、阴影映射、屏幕后处理效果、多重采样抗锯齿等。
		GLuint fusionMapFBO; // 帧缓冲对象(FBO)标识符

		// 这些是渲染缓冲区对象
		GLuint liveVertexMap;	// Live域顶点图
		GLuint liveNormalMap;	// Live域法线图
		GLuint indexMap;		// Live域索引图
		GLuint colorTimeMap;	// Color-Time图
		GLuint depthBuffer;		// 深度Buffer

		//GLuint flag;			// 标志位

		// cuda访问的缓冲区
		// RBO（Render Buffer Object）是 OpenGL 中的一种对象，用于存储渲染操作的结果数据。
		// 它提供了一种高效的方式来存储深度、模板和颜色缓冲区的数据，但不能用作纹理。
		cudaGraphicsResource_t cudaRBOResources[4];		// cuda中存储的OpenGL的RBO资源句柄
		cudaArray_t cudaMappedArrays[4];				// 用cudaArray_t接收RBO资源
		cudaTextureObject_t cudaMappedTexture[4];		// 将接收的cudaArray_t资源转化成CUDA纹理资源

		//只能由渲染器调用
		/**
		 * \brief 初始化帧缓冲对象(FBO)和渲染缓冲对象(RBO)，并将面元属性的RBO绑定到FBO上.
		 * 
		 * \param scaledWidth  需要分配的渲染缓冲区(RBO)对象的宽度
		 * \param scaledHeight 需要分配的渲染缓冲区(RBO)对象的高度
		 */
		void initialize(int scaledWidth, int scaledHeight);

		/**
		 * \brief 解除CUDA与OpenGL之间的映射，释放帧缓冲对象(FBO)和渲染缓冲对象(RBO)内存.
		 * 
		 */
		void release();

		/**
		 * \brief OpenGL中的渲染缓冲对象(RBO)资源转换成CUDA纹理对象，必须在初始化后调用.
		 * 
		 * \param liveVertexTexture live域中的顶点纹理
		 * \param liveNormalTexture live域中的法线纹理
		 * \param indexTexture 索引图纹理
		 * \param colorTimeTexture Color-Time纹理
		 * \param stream CUDA流ID
		 */
		void mapToCuda(cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& colorTimeTexture, cudaStream_t stream = 0);
		/**
		 * \brief 取消当前CUDA流中，OpenGL与CUDA之间的资源映射关系.
		 * 
		 * \param stream CUDA流ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);
	};


	/**
	 * \brief 用于SolverMap的帧缓冲对象(FBO)和渲染缓冲对象(RBO)。将OpenGL这些个缓冲区映射到cuda上.
	 *
	 */
	struct GLSolverMapsFrameRenderBufferObjects {
		GLuint solverMapFBO;	// SolerMap的帧缓冲对象(FBO)

		// 这些是渲染缓冲区对象RBO
		GLuint canonicalVertexMap;		// canonical域顶点图
		GLuint canonicalNormalMap;		// canonical域法线图
		GLuint liveVertexMap;			// live域顶点图
		GLuint liveNormalMap;			// live域法线图
		GLuint indexMap;				// 索引图
		GLuint normalizedRGBMap;		// 归一化的RGB图
		GLuint depthBuffer;				// 深度缓存

		// cuda访问的资源
		cudaGraphicsResource_t  cudaRBOResources[6];		// OpenGL映射到CUDA的资源
		cudaArray_t cudaMappedArrays[6];					// 用cudaArray_t接收CUDA资源
		cudaTextureObject_t cudaMappedTexture[6];			// 将cudaArray_t转成纹理，并通过引用传参传递出去

		
		/**
		 * \brief 初始化帧缓冲对象(FBO)和渲染缓冲对象(RBO)，这初始化只能在渲染中访问.
		 * 
		 * \param width  渲染缓冲区的宽度
		 * \param height 渲染缓冲区的高度
		 */
		void initialize(int width, int height);

		/**
		 * \brief 释放帧缓冲对象(FBO)和渲染缓冲对象(RBO).
		 * 
		 */
		void release();

		/**
		 * \brief 将OpenGL中的渲染缓冲对象(RBO)资源转换成CUDA纹理对象，必须在初始化后调用.
		 * 
		 * \param canonicalVertexTexture	【输出参数】OpenGL中的canonical域中顶点纹理
		 * \param canonicalNormalTexture	【输出参数】OpenGL中的canonical域中法线纹理
		 * \param liveVertexTexture			【输出参数】OpenGL中的live域中顶点纹理
		 * \param liveNormalMapTexture		【输出参数】OpenGL中的live域中顶点纹理
		 * \param indexTexture				【输出参数】OpenGL中的索引图纹理
		 * \param normalizedRGBTexture		【输出参数】OpenGL中的归一化RGB纹理
		 * \param stream CUDA流ID
		 */
		void mapToCuda(cudaTextureObject_t& canonicalVertexTexture, cudaTextureObject_t& canonicalNormalTexture, cudaTextureObject_t& liveVertexTexture, cudaTextureObject_t& liveNormalMapTexture, cudaTextureObject_t& indexTexture, cudaTextureObject_t& normalizedRGBTexture, cudaStream_t stream = 0);
		
		/**
		 * \brief 取消当前CUDA流中，OpenGL与CUDA之间的资源映射关系.
		 * 
		 * \param stream CUDA流ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);
	};



	/**
	 * \brief float4输出的离线渲染图。缓冲区不需要映射到cuda，但可能需要离线保存.
	 */
	struct GLOfflineVisualizationFrameRenderBufferObjects {

		/**
		 * \brief 渲染类型：内容来自Live域 或 Canonical域，渲染方式是Normal、Albedo或Phong.
		 */
		enum RenderType {
			LiveNormalMap,			// Live域Normal渲染类型
			LiveAlbedoMap,			// Live域Albedo渲染类型
			LivePhongMap,			// Live域Phong渲染类型
			CanonicalNormalMap,		// Canonical域Normal渲染类型
			CanonicalAlbedoMap,		// Canonical域Albedo渲染类型
			CanonicalPhongMap		// Canonical域Phong渲染类型
		};

		GLuint visualizationMapFBO;
		GLuint normalizedRGBARBO;		// 一个float4纹理，其元素在[0,1]
		GLuint depthBuffer;

		unsigned int frameIndex = 0;	// 用于记录保存图片的index

		/**
		 * \brief 初始化.
		 * 
		 * \param width 
		 * \param height
		 */
		void initialize(int width, int height);
		/**
		 * \brief 析构.
		 * 
		 */
		void release();

		/**
		 * \brief 将图片保存到本地.
		 * 
		 * \param path 文件路径
		 */
		void save(const std::string& path);

		/**
		 * \brief 显示图像.
		 * 
		 * \param symbol
		 */
		void show(RenderType symbol);
	};
}
