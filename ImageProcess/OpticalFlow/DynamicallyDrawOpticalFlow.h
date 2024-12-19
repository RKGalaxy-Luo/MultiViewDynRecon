#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>//OpenGL矩阵运算库
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>	// CUDA与OpenGL显存共享
#include <base/Logging.h>
#include <base/GlobalConfigs.h>
#include <base/CommonTypes.h>
#include <render/GLShaderProgram.h>
#include <render/Renderer.h>
#include <math/MatUtils.h>

namespace SparseSurfelFusion {

	namespace device {


		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 *
		 * \param SolvedPointsCoor 融合后的稠密面元坐标
		 * \param AdjustMatrix 调整位置的位姿矩阵
		 * \param PointNum 点的数量
		 * \param point 渲染的点
		 */
		__global__ void adjustModelPositionKernel(const mat34 AdjustMatrix, const unsigned int PointNum, float3* point, ColorVertex* vertex);

		/**
		 * \brief 调整点坐标的核函数.
		 *
		 * \param PointNum 点的数量
		 * \param points 需要绘制的点
		 * \param point 渲染的点
		 */
		__global__ void adjustPointsCoordinate(const unsigned int pointsCount, float3* points, ColorVertex* vertex);
	}

	class DynamicallyDrawOpticalFlow
	{
	public:
		DynamicallyDrawOpticalFlow();

		~DynamicallyDrawOpticalFlow();

		using Ptr = std::shared_ptr<DynamicallyDrawOpticalFlow>;

	private:
		GLFWwindow* OpticalFlowWindow = NULL;		// 光流窗口
		GLShaderProgram OpticalFlowShader;			// 渲染融合点的程序
		GLuint OpticalFlowVAO;						// 光流的Shader的VAO
		GLuint OpticalFlowVBO;						// 光流的Shader的VBO

		GLShaderProgram ColorVertexShader;			// 渲染融合点的程序
		GLuint ColorVertexVAO;						// 光流的Shader的VAO
		GLuint ColorVertexVBO;						// 光流的Shader的VBO

		cudaGraphicsResource_t cudaVBOResources[2];	// 注册缓冲区对象到CUDA

		GLuint coordinateSystemVAO;					// 坐标系VAO
		GLuint coordinateSystemVBO;					// 坐标系轴点网格VBO
		GLShaderProgram coordinateShader;			// 坐标系的渲染程序

		glm::mat4 view = glm::mat4(1.0f);			// 确保初始化矩阵是单位矩阵
		glm::mat4 projection = glm::mat4(1.0f);		// 投影矩阵，选择是透视还是正射
		glm::mat4 model = glm::mat4(1.0f);			// 计算每个对象的模型矩阵，并在绘制之前将其传递给着色器

		const unsigned int ImageWidth = 640;
		const unsigned int ImageHeight = 400;
		const unsigned int ImageSize = ImageWidth * ImageHeight;

		const unsigned int WindowHeight = 1080.0f * 0.5f;
		const unsigned int WindowWidth = 1920.0f * 0.5f;

		unsigned int ValidNum = 0;

		const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(180, 0, -45);
		const float3 AdjustTranslation = make_float3(0, 0, 0);
		const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);

		//const float3 AdjustRotation = make_float3(0, 0, 0);
		//const float3 AdjustTranslation = make_float3(0, 0, 0);
		//const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);

		/**
		 * \brief 注册光流的cuda资源.
		 *
		 */
		void registerFlowCudaResources();

		/**
		 * \brief 注册带颜色顶点的cuda资源.
		 *
		 */
		void registerVertexCudaResources();

		/**
		 * \brief 初始化并加载坐标系.
		 *
		 */
		void initialCoordinateSystem();

		/**
		 * \brief 分配运行所需显存.
		 *
		 */
		void allocateBuffer();

		/**
		 * \brief 调整点的坐标范围.
		 *
		 * \param points 融合后的稠密面元坐标
		 * \param vertex RGB点
		 * \param stream cuda流
		 */
		void adjustPointsCoordinate(float3* points, ColorVertex* vertex, cudaStream_t stream = 0);

		/**
		 * \brief 调整模型的位置，使其正对摄像头.
		 *
		 * \param points 融合后的稠密面元坐标
		 * \param vertex RGB点
		 * \param stream cuda流
		 */
		void adjustModelPosition(float3* points, ColorVertex* vertex, cudaStream_t stream = 0);
		/**
		 * \brief 将光流映射到与cuda绑定的OpenGL资源.
		 *
		 * \param validFlow 有效的光流
		 * \param validFlowNum 有效光流的数量
		 * \param stream cuda流
		 */
		void OpticalFlowMapToCuda(float3* validFlow, cudaStream_t stream = 0);

		/**
		 * \brief 将带颜色的点拷贝到OpenGL资源.
		 *
		 * \param validVertex 有效的顶点
		 * \param validVertexNum 有效顶点数量
		 * \param stream cuda流
		 */
		void LivePointsMapToCuda(ColorVertex* validVertex, cudaStream_t stream = 0);

		/**
		 * \brief 取消映射.
		 *
		 */
		void UnmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief 绘制光流.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawOpticalFlow(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief 绘制光流.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawColorVertex(glm::mat4& view, glm::mat4& projection, glm::mat4& model);
		/**
		 * \brief 绘制坐标系.
		 *
		 * \param view 传入视角矩阵
		 * \param projection 传入投影矩阵
		 * \param model 传入模型矩阵
		 */
		void drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief 清空屏幕.
		 *
		 */
		void clearWindow();

		/**
		 * \param 双缓冲并捕捉事件.
		 *
		 */
		void swapBufferAndCatchEvent();

	public:
		/**
		 * \brief 绘制3D光流.
		 *
		 * \param validFlow 有效的光流
		 * \param validFlowNum 有效光流的数量
		 * \param stream cuda流
		 */
		void imshow(float3* validFlow, ColorVertex* validColorVertex, const unsigned int validFlowNum, cudaStream_t stream);
	};

}