/*****************************************************************//**
 * \file   DynamicallyDrawPoints.h
 * \brief  动态绘制点
 * 
 * \author LUOJIAXUAN
 * \date   June 29th 2024
 *********************************************************************/
#pragma once
#include "Renderer.h"
namespace SparseSurfelFusion {

	namespace device {
		enum {
			CudaThreadsPerBlock = 128
		};
	
		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 * 
		 * \param SolvedPointsCoor 融合后的稠密面元坐标
		 * \param AdjustMatrix 调整位置的位姿矩阵
		 * \param PointNum 点的数量
		 * \param point 渲染的点
		 */
		__global__ void adjustModelPositionKernel(DeviceArrayView<float4> SolvedPointsCoor, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedPoints* point);

		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 *
		 * \param SolvedPointsCoor 融合后的稠密面元坐标
		 * \param AdjustMatrix 调整位置的位姿矩阵
		 * \param PointNum 点的数量
		 * \param point 渲染的点
		 */
		__global__ void adjustModelPositionKernel(DeviceArrayView<DepthSurfel> SolvedPoints, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedPoints* point);

		/**
		 * \brief 规约获得每个block最大和最小的Point3D.
		 *
		 * \param maxBlockData 当前block中的最大值
		 * \param minBlockData 当前block中的最小值
		 * \param points 稠密点云
		 * \param pointsCount 点云数量
		 */
		__global__ void reduceMaxMinKernel(float3* maxBlockData, float3* minBlockData, DeviceArrayView<Renderer::RenderedPoints> points, const unsigned int pointsCount);
		/**
		 * \brief 找规约后的最大和最小点.
		 *
		 * \param MaxPoint 输出最大点
		 * \param MinPoint 输出最小点
		 * \param maxArray 每个block最大的点Array
		 * \param minArray 每个block最小点点Array
		 * \param GridNum 规约时分配的网格数量
		 */
		__host__ void findMaxMinPoint(float3& MaxPoint, float3& MinPoint, float3* maxArray, float3* minArray, const unsigned int GridNum);

		/**
		 * \brief 调整点坐标的核函数，将其缩放到第一象限，并解码depthsurfel的颜色.
		 *
		 * \param SolvedPointsColor 融合后的稠密面元颜色
		 * \param center 中心点的坐标
		 * \param maxEdge 最长的边
		 * \param points 需要绘制的点
		 */
		__global__ void adjustPointsCoordinateAndColorKernel(DeviceArrayView<float4> SolvedPointsColor, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedPoints* points);

		/**
		 * \brief 调整点坐标的核函数，将其缩放到第一象限，并解码depthsurfel的颜色.
		 *
		 * \param SolvedPoints 融合后的稠密面元
		 * \param center 中心点的坐标
		 * \param maxEdge 最长的边
		 * \param points 需要绘制的点
		 */
		__global__ void adjustPointsCoordinateAndColorKernel(DeviceArrayView<DepthSurfel> SolvedPointsColor, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedPoints* points);
	}
	class DynamicallyDrawPoints
	{
	public:
		using Ptr = std::shared_ptr<DynamicallyDrawPoints>;

		/**
		 * \brief 初始化.
		 * 
		 */
		DynamicallyDrawPoints();

		/**
		 * \brief 析构.
		 * 
		 */
		~DynamicallyDrawPoints();

		/**
		 * \brief 绘制实时融合后的点.
		 *
		 * \param SolvedPointsCoor 求解的融合后的点位置
		 * \param SolvedPointsColor 求解的融合后的点颜色
		 * \param stream cuda流ID
		 */
		void DrawLiveFusedPoints(DeviceArrayView<float4> SolvedPointsCoor, DeviceArrayView<float4> SolvedPointsColor, cudaStream_t stream = 0);

		/**
		 * \brief 绘制实时融合后的点.
		 *
		 * \param SolvedPoints 求解后的带颜色的面元
		 * \param CheckPeriod 环绕检查的周期
		 * \param stream cuda流ID
		 */
		void DrawLiveFusedPoints(DeviceArrayView<DepthSurfel> SolvedPoints, bool CheckPeriod, cudaStream_t stream = 0);

	private:

		SynchronizeArray<float3> perBlockMaxPoint;	// 记录每个线程块的最大点
		SynchronizeArray<float3> perBlockMinPoint;	// 记录每个线程块的最小点

		GLFWwindow* LiveWindow;						// 显示窗口
		GLShaderProgram FusedLivePointsShader;		// 渲染融合点的程序
		GLuint FusedLivePointsVAO;					// 融合后点Shader的VAO
		GLuint FusedLivePointsVBO;					// 融合后点Shader的VBO
		GLuint FusedLivePointsIBO;					// 融合后点Shader的EBO/IBO

		// 绘制渲染窗口坐标系
		GLuint coordinateSystemVAO;					// 坐标系VAO
		GLuint coordinateSystemVBO;					// 坐标系轴点网格VBO
		GLShaderProgram coordinateShader;			// 坐标系的渲染程序
		cudaGraphicsResource_t cudaVBOResources;	// 注册缓冲区对象到CUDA

		const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(180.0f, 0, 0);
		const float3 AdjustTranslation = make_float3(-0.75f, 0.0f, 0.35f);
		//const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(150, 0, -90);
		//const float3 AdjustTranslation = make_float3(0.6f, 0.5f, 0.0f);
		const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);

		float radius = 4.0f;//摄像头绕的半径
		float camX = 0.0f;
		float camZ = 5.0f;

		glm::mat4 view = glm::mat4(1.0f);			// 确保初始化矩阵是单位矩阵
		glm::mat4 projection = glm::mat4(1.0f);		// 投影矩阵，选择是透视还是正射
		glm::mat4 model = glm::mat4(1.0f);			// 计算每个对象的模型矩阵，并在绘制之前将其传递给着色器

		DeviceBufferArray<Renderer::RenderedPoints> RenderedFusedPoints;	// 将渲染的深度面元与计算的深度面元进行隔离

		bool CheckSpecificFrame = true;

		/**
		 * \brief 初始化动态渲染器，该渲染器用来渲染融合后的点，同时分配渲染内存.
		 *
		 */
		void initialDynamicRendererAndAllocateBuffer();

		/**
		 * \brief 注册cuda资源.
		 *
		 */
		void registerCudaResources();

		/**
		 * \brief 初始化并加载坐标系.
		 *
		 */
		void initialCoordinateSystem();

		/**
		 * \brief 找到点云范围.
		 *
		 * \param points 传入点云
		 * \param MaxPoint 计算得到的包围盒最大point
		 * \param MinPoint 计算得到的包围盒最小point
		 * \param stream cuda流
		 */
		void getBoundingBox(DeviceArrayView<Renderer::RenderedPoints> points, float3& MaxPoint, float3& MinPoint, cudaStream_t stream = 0);

		/**
		 * \brief 调整点的坐标范围,解码颜色.
		 *
		 * \param SolvedPointsCoor 融合后的稠密面元坐标
		 * \param SolvedPointsColor 融合后的稠密面元颜色
		 * \param MxPoint 最大坐标
		 * \param MnPoint 最小坐标
		 * \param stream cuda流
		 */
		void adjustPointsCoordinateAndColor(DeviceArrayView<float4> SolvedPointsColor, const float3 MxPoint, const float3 MnPoint, cudaStream_t stream = 0);

		/**
		 * \brief 调整点的坐标范围,解码颜色.
		 *
		 * \param SolvedPoints 融合后的稠密面元
		 * \param MxPoint 最大坐标
		 * \param MnPoint 最小坐标
		 * \param stream cuda流
		 */
		void adjustPointsCoordinateAndColor(DeviceArrayView<DepthSurfel> SolvedPoints, const float3 MxPoint, const float3 MnPoint, cudaStream_t stream = 0);

		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 * 
		 * \param SolvedPointsCoor 融合后的稠密面元坐标
		 * \param stream cuda流
		 */
		void adjustModelPosition(DeviceArrayView<float4> SolvedPointsCoor, cudaStream_t stream = 0);

		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 *
		 * \param SolvedPoints 融合后的稠密面元坐标
		 * \param stream cuda流
		 */
		void adjustModelPosition(DeviceArrayView<DepthSurfel> SolvedPoints, cudaStream_t stream = 0);

		/**
		 * \brief 将需要渲染的点映射到cuda资源上.
		 * 
		 */
		void mapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief 将需要渲染的点映射到cuda资源上.
		 *
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief 清空屏幕.
		 * 
		 */
		void clearWindow();

		/**
		 * \brief 绘制点.
		 * 
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawLivePoints(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief 绘制坐标系.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \param 双缓冲并捕捉事件.
		 * 
		 */
		void swapBufferAndCatchEvent();

		// 监控输入,并进行控制的函数实现
		bool CheckPressESC(GLFWwindow* window) {
			//glfwGetKey函数，它需要一个窗口以及一个按键作为输入。这个函数将会返回这个按键是否正在被按下。
			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) return true;
			else return false;
			//这里我们检查用户是否按下了返回键(Esc)
		}
	};
}


