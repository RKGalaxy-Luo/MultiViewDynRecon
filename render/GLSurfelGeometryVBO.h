/*****************************************************************//**
 * \file   GLSurfelGeometryVBO.h
 * \brief  用来处理(稠密)面元的渲染缓存
 * 
 * \author LUO
 * \date   February 21st 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <core/Geometry/SurfelGeometry.h>
#include <core/Geometry/FusionDepthGeometry.h>
#include <cuda_gl_interop.h>	// 将缓冲区注册到cuda
#include <base/Constants.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>

namespace SparseSurfelFusion {
	//用于每帧深度面元融合的VBO
	struct GLfusionDepthSurfelVBO
	{
		// 顶点缓存对象对应于SurfelGeometry的成员
		GLuint CanonicalVertexConfidence;				// 参考顶点配置
		GLuint CanonicalNormalRadius;					// 参考法线及半径
		GLuint ColorTime;								// 观察到的时间

		cudaGraphicsResource_t cudaVBOResources[3];		// 与SurfelGeometry图形相关的CUDA资源

		GLFWwindow* GLFW_Window = NULL;

		// 方法只能由呈现器访问
		/**
		 * \brief 初始化当前的这个类.
		 *
		 */
		void initialize();

		/**
		 * \brief 释放当前的这个类.
		 *
		 */
		void release();

		/**
		 * \brief 将SurfelGeometry映射到CUDA.
		 *
		 * \param geometry 面元几何体(需要渲染的)
		 * \param stream CUDA流ID
		 */
		void mapToCuda(FusionDepthGeometry& geometry, cudaStream_t stream = 0);
		/**
		 * \brief 将SurfelGeometry映射到CUDA(将一个或多个 CUDA 图形资源（例如 CUDA 图像、CUDA 表面等）映射到当前的CUDA上下文中)
		 *		函数执行成功后，映射的资源将与当前的CUDA上下文关联起来，可以在GPU上进行读写操作.
		 *
		 * \param stream CUDA流ID
		 */
		void mapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief 阻塞所有cuda调用在给定线程，为了后面OpenGL绘图的pipeline函数执行成功后，已映射的资源将与当前的 CUDA 上下文分离，
		 *		不再可用于 GPU 上的操作。这样可以确保资源在主机和设备之间的一致性，并释放相关资源的内存.
		 *
		 * \param stream CUDA流ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

	};



	/**
	 * \brief 一个类，用于维护一个surfel几何体实例的所有顶点缓冲对象，以及在cuda上访问的资源。
	 *        这个类只能在渲染类中使用。
	 */
	struct GLSurfelGeometryVBO
	{
		// 顶点缓存对象对应于SurfelGeometry的成员
		GLuint CanonicalVertexConfidence;				// 参考顶点配置
		GLuint CanonicalNormalRadius;					// 参考法线及半径
		GLuint LiveVertexConfidence;					// 实时顶点配置
		GLuint LiveNormalRadius;						// 实时法线及半径
		GLuint ColorTime;								// 观察到的时间

		cudaGraphicsResource_t cudaVBOResources[5];		// 与SurfelGeometry图形相关的CUDA资源

		GLFWwindow* GLFW_Window = NULL;

		// 方法只能由呈现器访问
		/**
		 * \brief 初始化当前的这个类.
		 * 
		 */
		void initialize();

		/**
		 * \brief 释放当前的这个类.
		 * 
		 */
		void release();

		/**
		 * \brief 将SurfelGeometry映射到CUDA.
		 * 
		 * \param geometry 面元几何体(需要渲染的)
		 * \param stream CUDA流ID
		 */
		void mapToCuda(SurfelGeometry& geometry, cudaStream_t stream = 0);
		/**
		 * \brief 将SurfelGeometry映射到CUDA(将一个或多个 CUDA 图形资源（例如 CUDA 图像、CUDA 表面等）映射到当前的CUDA上下文中)
		 *		函数执行成功后，映射的资源将与当前的CUDA上下文关联起来，可以在GPU上进行读写操作.
		 * 
		 * \param stream CUDA流ID
		 */
		void mapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief 阻塞所有cuda调用在给定线程，为了后面OpenGL绘图的pipeline函数执行成功后，已映射的资源将与当前的 CUDA 上下文分离，
		 *		不再可用于 GPU 上的操作。这样可以确保资源在主机和设备之间的一致性，并释放相关资源的内存.
		 * 
		 * \param stream CUDA流ID
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

	};


	/**
	 * \brief 【SparseSurfelFusion命名空间的全局函数】初始化渲染面元的pipeline(程序化渲染流程，这样写更清晰).
	 * 
	 * \param 传入需要渲染的面元几何属性的类.
	 */
	void initializeGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO);
	void initializeGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO);

	/**
	 * \brief 【SparseSurfelFusion命名空间的全局函数】释放渲染面元的pipeline(程序化渲染流程，这样写更清晰).
	 * 
	 * \param 传入需要释放的面元几何属性的类.
	 */
	void releaseGLSurfelGeometry(GLSurfelGeometryVBO& surfelVBO);
	void releaseGLfusionDepthSurfel(GLfusionDepthSurfelVBO& surfelVBO);

}


