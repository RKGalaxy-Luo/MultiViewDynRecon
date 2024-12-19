#pragma once
#include "Renderer.h"
#include <base/EncodeUtils.h>

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief 调整点坐标的核函数，将其缩放到第一象限，并解码depthsurfel的颜色.
		 *
		 * \param rawSurfels 融合后的稠密面元颜色
		 * \param center 中心点的坐标
		 * \param maxEdge 最长的边
		 * \param renderedSurfels 需要绘制的点
		 */
		__global__ void AdjustPointsCoordinateAndColorKernel(DeviceArrayView<DepthSurfel> rawSurfels, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedSurfels* renderedSurfels);
		
		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 *
		 * \param rawSurfels 融合后的稠密面元坐标
		 * \param AdjustMatrix 调整位置的位姿矩阵
		 * \param PointNum 点的数量
		 * \param renderedSurfels 渲染的点
		 */
		__global__ void AdjustModelPositionKernel(DeviceArrayView<DepthSurfel> rawSurfels, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedSurfels* renderedSurfels);

	}

	class DynamicallyRenderSurfels
	{
	public:

		enum RenderSurfelsType{
			Albedo,
			Phong,
			Normal
		};

		using Ptr = std::shared_ptr<DynamicallyRenderSurfels>;

		DynamicallyRenderSurfels(RenderSurfelsType renderType, Intrinsic intrinsic);

		~DynamicallyRenderSurfels();

	private:

		RenderSurfelsType RenderType;

		const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(180.0f, 0, 0);
		const float3 AdjustTranslation = make_float3(-0.75f, 0.0f, 0.35f);
		const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);

		const float3 Center = make_float3(-1.59550f, -0.74243f, -2.22310f);
		const float MaxEdge = 1.70285f;

		float radius = 4.5f;//摄像头绕的半径
		float camX = 0.0f;
		float camZ = 0.0f;

		glm::mat4 view = glm::mat4(1.0f);			// 确保初始化矩阵是单位矩阵
		glm::mat4 projection = glm::mat4(1.0f);		// 投影矩阵，选择是透视还是正射
		glm::mat4 model = glm::mat4(1.0f);			// 计算每个对象的模型矩阵，并在绘制之前将其传递给着色器

		Intrinsic rendererIntrinsic;
		int frameIndex;
		float2 confidenceTimeThreshold;
		GLFWwindow* SurfelsWindow;										// 显示窗口

		GLShaderProgram LiveSurfelsShader;								// 渲染融合点的程序
		GLuint LiveSurfelsVAO;											// 融合后点Shader的VAO
		GLuint LiveSurfelsVBO;											// 融合后点Shader的VBO
		GLuint LiveSurfelsIBO;											// 融合后点Shader的EBO/IBO
		cudaGraphicsResource_t cudaVBOResources;						// 注册缓冲区对象到CUDA

		// 绘制渲染窗口坐标系
		GLuint coordinateSystemVAO;											// 坐标系VAO
		GLuint coordinateSystemVBO;											// 坐标系轴点网格VBO
		GLShaderProgram coordinateShader;									// 坐标系的渲染程序

		DeviceBufferArray<Renderer::RenderedSurfels> RenderedSurfels;		// 将渲染的深度面元与计算的深度面元进行隔离

		/**
		 * \brief 初始化动态渲染器，该渲染器用来渲染融合后的点，同时分配渲染内存.
		 *
		 */
		void InitialDynamicRendererAndAllocateBuffer(RenderSurfelsType renderType);

		/**
		 * \brief 注册cuda资源.
		 *
		 */
		void RegisterCudaResources();

		/**
		 * \brief 初始化并加载坐标系.
		 *
		 */
		void InitialCoordinateSystem();

		/**
		 * \brief 调整点的坐标范围,解码颜色.
		 *
		 * \param SolvedPoints 融合后的稠密面元
		 * \param center 中心位置
		 * \param maxEdge 包围框最大边长
		 * \param stream cuda流
		 */
		void AdjustSurfelsCoordinateAndColor(DeviceArrayView<DepthSurfel> surfels, const float3 center, const float maxEdge, cudaStream_t stream = 0);

		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 *
		 * \param SolvedPoints 融合后的稠密面元坐标
		 * \param stream cuda流
		 */
		void AdjustModelPosition(DeviceArrayView<DepthSurfel> SolvedPoints, cudaStream_t stream = 0);

		/**
		 * \brief 将需要渲染的点映射到cuda资源上.
		 *
		 */
		void MapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief 将需要渲染的点映射到cuda资源上.
		 *
		 */
		void UnmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief 清空屏幕.
		 *
		 */
		void ClearWindow();

		/**
		 * \brief 绘制点.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void DrawSurfels(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief 绘制坐标系.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void DrawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \param 双缓冲并捕捉事件.
		 *
		 */
		void SwapBufferAndCatchEvent();

		/**
		 * \brief 渲染窗口的截图.
		 * 
		 */
		void ScreenShot(const unsigned int frameIdx);

	public:
		void DrawRenderedSurfels(const DeviceArrayView<DepthSurfel>& surfels, const unsigned int frameIdx, cudaStream_t stream = 0);
	};
}


