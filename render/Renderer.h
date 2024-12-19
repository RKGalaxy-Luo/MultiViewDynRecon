/*****************************************************************//**
 * \file   Renderer.h
 * \brief  渲染器对象，主要是显示并渲染，需要注意的是，渲染是以某一个相机(0号相机)空间作为展示，所有非0号相机需全部转换到0号相机空间中
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>			// <GLFW/glfw3.h>会自动包含很多老版本，如果<glad/glad.h>在前面，那么就使用glad对应最新版本的OpenGL，这也是为什么<glad/glad.h>必须在<GLFW/glfw3.h>之前
//#define GLFW_INCLUDE_NONE		// 显示的禁用 <GLFW/glfw3.h> 自动包含开发环境的功能，使用这个功能之后就不会再从开发环境中包含，即<GLFW/glfw3.h>也可以在<glad/glad.h>之前
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>//OpenGL矩阵运算库
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>	// CUDA与OpenGL显存共享
#include <tuple>				// tuple是一个固定大小的不同类型值的集合，是泛化的std::pair
#include <vector>
#include <Eigen/Eigen>
#include <memory>

#include <base/Camera.h>
#include <base/CameraObservation.h>
#include <base/CommonTypes.h>
#include <base/CommonUtils.h>
#include <base/EncodeUtils.h>
#include <base/ConfigParser.h>
#include <base/Constants.h>
#include <base/Logging.h>

#include "GLSurfelGeometryVBO.h"
#include "GLSurfelGeometryVAO.h"
#include "GLShaderProgram.h"
#include "GLRenderMaps.h"

static glm::vec3 box[68] = {	
	// x轴								x轴颜色
	{ 0.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 0.8f,   0.1f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
	{ 0.8f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.2f,   0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.3f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.2f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
	{ 1.3f,   0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
											     	     
	{ 0.0f,   0.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.1f,   0.8f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ -0.1,   0.8f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.05f,  1.3f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ -0.05f, 1.3f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
	{ 0.0f,   1.1f,   0.0f },			{0.0f,   1.0f,   0.0f},

	{ 0.0f,   0.0f,   0.0f },			{0.0f,   0.0f,   1.0f},
	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
	{ 0.1f,   0.0f,   0.8f },			{0.0f,   0.0f,   1.0f},
	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
	{ -0.1f,  0.0f,   0.8f },			{0.0f,   0.0f,   1.0f},
	{ -0.05f, 0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
	{ 0.05f,  0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
	{ 0.05f,  0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
	{ -0.05f, 0.0f,   1.1f },			{0.0f,   0.0f,   1.0f},
	{ -0.05f, 0.0f,   1.1f },			{0.0f,   0.0f,   1.0f},
	{ 0.05f,  0.0f,   1.1f },			{0.0f,   0.0f,   1.0f}
};



namespace SparseSurfelFusion {

	namespace device {
		struct SolverMapFilteringInterface {
			cudaTextureObject_t warpVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t warpNormalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t warpIndexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t warpColorViewTimeMap[MAX_CAMERA_COUNT];


			cudaTextureObject_t observedVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t observedNormalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t observedForeground[MAX_CAMERA_COUNT];

			cudaSurfaceObject_t filteredIndexMap[MAX_CAMERA_COUNT];

			mat34 Live2Camera[MAX_CAMERA_COUNT];
		};


		struct FusionMapFilteringInterface {
			cudaTextureObject_t warpVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t warpNormalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t warpIndexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t warpColorViewTimeMap[MAX_CAMERA_COUNT];

			cudaTextureObject_t observedVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t observedNormalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t observedForeground[MAX_CAMERA_COUNT];

			cudaSurfaceObject_t filteredIndexMap[MAX_CAMERA_COUNT];

			mat34 World2Camera[MAX_CAMERA_COUNT];
			mat34 initialCameraSE3Inverse[MAX_CAMERA_COUNT];
		};

	}

	/**
	 * \brief 这是一个渲染器的类型，主要负责渲染及可视化.
	 */
	class Renderer
	{
	private:
		int imageWidth = 0;										// 裁剪后图像的宽
		int imageHeight = 0;									// 裁剪后图像的高

		int fusionImageWidth = 0;								// 融合图像的宽
		int fusionImageHeight = 0;								// 融合图像的高
		const unsigned int FusionMapScale;						// 上采用的尺度

		unsigned int deviceCount = 0;							// 实际接入相机个数

		float4 rendererIntrinsic;								// 渲染的相机内参(0号相机的内参：以某一相机的相机空间作为渲染的空间做展示)
		float4 widthHeightMaxMinDepth;							// 宽、高及最大、最小深度
		float4 ACameraRendererIntrinsic[MAX_CAMERA_COUNT];		// 每个相机的内参
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];				// 相机初始位姿

	public:
		using Ptr = std::shared_ptr<Renderer>;
		explicit Renderer(int imageRows, int imageCols, ConfigParser::Ptr config);
		~Renderer();
		GLFWwindow* GLFW_Window = nullptr;			//不透明窗口对象

		NO_COPY_ASSIGN_MOVE(Renderer);
/******************************************************** 显示对象 ********************************************************/
	/* GLFW窗口相关变量和函数  GLFW是一个专门针对OpenGL的C语言库,它提供了一些渲染物体所需的最低限度的接口
	 * 一般采用 glad 获取一堆 glxxx 函数的函数指针。用 GLFW 管理操作系统的窗口管理器给到的 framebuffer，然后OpenGL 在上面画画
	 */
	private:
		/**
		 * \brief 初始化GLFW.
		 * 
		 */
		void initGLFW();

		/**
		 * \brief 初始化过滤SolverMap的IndexMap和FusionMap的IndexMap透射面元的FilteredIndexMap.
		 * 
		 */
		void initFilteredSolverMapAndFusionMap();

		/**
		 * \brief 释放FilteredMap.
		 * 
		 */
		void releaseFilteredSolverMapAndFusionMap();

	public:

		/**
		 * \brief 渲染点的数据.
		 */
		struct RenderedPoints {
			float3 coordinate;						// 渲染点的坐标
			float3 rgb;								// 渲染点的颜色
		};

		/**
		 * \brief 渲染点的数据.
		 */
		struct RenderedSurfels {
			float4 vertexConfidence;						// 渲染点的坐标,置信度
			float4 normalRadius;							// 渲染点的法线,半径
			float4 rgbTime;									// 渲染的RGB,视角,初始观测帧和当前帧
		};

		/**
		 * \brief 传回GLFW窗口指针.
		 *
		 * \return GLFW窗口指针.
		 */
		GLFWwindow* getWindowGLFW(unsigned int WindowsIndex) { 
			if (GLFW_Window != nullptr) return GLFW_Window;
			else LOGGING(FATAL) << "窗口指针为nullptr";
		}
/******************************************************** 显示对象 ********************************************************/

/******************************************************** 默认清空的值 ********************************************************/
	private:
		// 小型类管理清除(默认)的值
		struct GLClearValues {
			GLfloat vertexMapClear[4];
			GLfloat normalMapClear[4];
			GLfloat colorTimeClear[4];
			GLfloat solverRGBAClear[4];
			GLfloat visualizeRGBAClear[4];
			GLfloat zBufferClear;
			GLuint indexMapClear;

			/**
			 * \brief 构造函数，声明即赋值.
			 * 
			 */
			inline void initialize() {
				memset(vertexMapClear, 0, sizeof(GLfloat) * 4);
				memset(normalMapClear, 0, sizeof(GLfloat) * 4);
				memset(colorTimeClear, 0, sizeof(GLfloat) * 4);
				memset(solverRGBAClear, 0, sizeof(GLfloat) * 4);
				visualizeRGBAClear[0] = 1.0f;
				visualizeRGBAClear[1] = 1.0f;
				visualizeRGBAClear[2] = 1.0f;
				visualizeRGBAClear[3] = 1.0f;
				zBufferClear = 1;
				indexMapClear = 0xFFFFFFFF;
			}
		};

		GLClearValues clearValues;
		/**
		 * \brief 初始化默认(clear)值.
		 * 
		 */
		void initClearValues();
/******************************************************** VBO顶点缓冲对象管理 ********************************************************/

	private:
		GLSurfelGeometryVBO surfelGeometryVBOs[MAX_CAMERA_COUNT][2];	// 双缓冲
		/**
		 * \brief 初始化顶点缓存对象.
		 * hsg加了surfelGeometryVBOIndexMap的初始化  加了fusiondepthVBO的  双视角  初始化 
		 */
		void initVertexBufferObjects();
		/**
		 * \brief 释放顶点缓存对象.
		 * hsg加了surfelGeometryVBOIndexMap的释放    加了fusiondepthVBO的  双视角  释放 
		 */
		void freeVertexBufferObjects();

	public:
		/**
		 * \brief 将SurfelGeometry映射到CUDA，以便渲染显示，同时开辟SurfelGeometry中属性的内存.
		 * 
		 * \param idx 用来更新渲染的标志索引(双缓冲方式显示)
		 * \param cameraID 相机的id，如果为非法id，默认所有surfelGeometryVBOs全部进行操作
		 * \param geometry 需要渲染的面元几何对象
		 * \param stream CUDA流ID
		 */
		void MapSurfelGeometryToCUDA(int idx, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], int cameraID = 0xFFFF, cudaStream_t stream = 0);
		/**
		 * \brief 将(所有)资源映射到CUDA，以便渲染显示.
		 *
		 *
		 * \param idx 用来更新渲染的标志索引(双缓冲方式显示)
		 * \param cameraID 相机的id，如果为非法id，默认所有surfelGeometryVBOs全部进行操作
		 * \param stream CUDA流ID
		 */
		void MapSurfelGeometryToCUDA(int idx, int cameraID = 0xFFFF, cudaStream_t stream = 0);
		/**
		 * \brief 已映射的资源将与当前的 CUDA 上下文分离，不再可用于 GPU 上的操作.
		 * 
		 * 
		 * \param idx 用来更新渲染的标志索引(双缓冲方式显示)
		 * \param cameraID 相机的id，如果为非法id，默认所有surfelGeometryVBOs全部进行操作
		 * \param stream CUDA流ID
		 */
		void UnmapSurfelGeometryFromCUDA(int idx, int cameraID = 0xFFFF, cudaStream_t stream = 0);

/******************************************************** VBO顶点缓冲对象管理 ********************************************************/

/******************************************************** VAO顶点数组对象管理 ********************************************************/
	/*
	 * 用于渲染的vao必须在vbo初始化之后初始化 
	 */
	private:
		// 实时显示的VAO，对VAO进行处理，对应双缓冲方案
		GLuint fusionMapVAO[MAX_CAMERA_COUNT][2];
		GLuint solverMapVAO[MAX_CAMERA_COUNT][2];
		GLuint fusionDepthSurfelVAO[MAX_CAMERA_COUNT];

		// 这里的vao用于Canonical域和Live域面元的离线可视化
		GLuint canonicalGeometryVAO[2]; // 用于Canonical域面元的离线可视化的VAO
		GLuint liveGeometryVAO[2];		// 用于Live域面元的离线可视化的VAO

		/**
		 * \brief 初始化所需的所有VAO.
		 * 
		 */
		void initMapRenderVAO();

/******************************************************** VAO顶点数组对象管理 ********************************************************/


/******************************************************** OpenGL与CUDA之间映射管理 ********************************************************/
	private:
		// 在线处理所需的帧缓冲区/渲染缓冲区
		GLFusionMapsFrameRenderBufferObjects fusionMapBuffers[MAX_CAMERA_COUNT];		// Live域(融合到Live域)的Buffer
		GLSolverMapsFrameRenderBufferObjects solverMapBuffers[MAX_CAMERA_COUNT];		// Live域 + Canonical域的Buffer
	
		////用于每帧融合
		//GLFusionDepthSurfelFrameRenderBufferObjects fusionDepthSurfelBuffers[MAX_CAMERA_COUNT];

		GLOfflineVisualizationFrameRenderBufferObjects visualizationDrawBuffers;	// 用于离线可视化的帧Buffer/渲染Buffer
		/**
		 * \brief 初始化帧Buffer/渲染Buffer.
		 * hsg加了
		 */
		void initFrameRenderBuffers();
		/**
		 * \brief 释放帧Buffer/渲染Buffer.
		 * hsg加了
		 */
		void freeFrameRenderBuffers();
/******************************************************** OpenGL与CUDA之间映射管理 ********************************************************/
	
/******************************************************** OpenGL着色器 ********************************************************/
	private:
		GLShaderProgram fusionMapShader;	// Live域(融合到Live域)的着色器 
		GLShaderProgram solverMapShader;	// Live域 + Canonical域的着色器

		GLShaderProgram solverMapShaderIndexMap;
		GLShaderProgram fusionMapShaderIndexMap;

		/**
		 * \brief 初始化“处理”着色器.
		 * 
		 */
		void initProcessingShaders();
		/**
		 * \brief 初始化“可视化”着色器.
		 *
		 */
		void initVisualizationShaders();

		/**
		 * \brief 初始化所有着色器.
		 *
		 */
		void initShaders();

		// 用于可视化的着色器集合
		struct {
			GLShaderProgram normalMap;
			GLShaderProgram phongMap;
			GLShaderProgram albedoMap;
			GLShaderProgram coordinateSystem;
		} visualizationShaders;

		// 绘制渲染窗口坐标系
		GLuint coordinateSystemVAO;					// 坐标系VAO
		GLuint coordinateSystemVBO;					// 坐标系轴点网格VBO


		void drawVisualizationMap(GLShaderProgram& shader, GLuint geometry_vao, unsigned num_vertex, int current_time, const Matrix4f& world2camera, bool with_recent_observation);

	public:
		void SetFrameIndex(const unsigned int frameIdx) { visualizationDrawBuffers.frameIndex = frameIdx; }
		//画实时图像
		void ShowLiveNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowLiveAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowLivePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowReferenceNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowReferenceAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowReferencePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);

/******************************************************** OpenGL着色器 ********************************************************/


/******************************************************** SolverMap ********************************************************/
	public:
		struct SolverMaps {
			cudaTextureObject_t reference_vertex_map;	// 当前视角的Canonical域(将0号视角下的Canonical域点通过位姿反矩阵得到)
			cudaTextureObject_t reference_normal_map;	// 当前视角的Canonical域(将0号视角下的Canonical域点通过位姿反矩阵得到)
			cudaTextureObject_t warp_vertex_map;		// 当前视角的Live域: Canonical域用非刚性SE3 Warp得到
			cudaTextureObject_t warp_normal_map;		// 当前视角的Live域: Canonical域用非刚性SE3 Warp得到
			cudaTextureObject_t index_map;				// 当前视角的IndexMap(Canonical域中反变换到当前视角，能出现的点)
			cudaTextureObject_t normalized_rgb_map;		// 当前视角下的Rgb图
		};
		void MapSolverMapsToCuda(SolverMaps* maps, cudaStream_t stream = 0);
		void UnmapSolverMapsFromCuda(cudaStream_t stream = 0);

		void DrawSolverMapsWithRecentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f* world2camera);
		void DrawSolverMapsConfidentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f* world2camera);
		
		void FilterSolverMapsIndexMap(SolverMaps* maps, CameraObservation& observation, mat34* Live2Camera, float filterRatio = 0.2f, cudaStream_t stream = 0);
		bool ShouldDoRefresh() { return shouldRefresh; }
	private:
		CudaTextureSurface solverMapFilteredIndexMap[MAX_CAMERA_COUNT];				// 滤波后的IndexMap，除去透射点
		device::SolverMapFilteringInterface solverMapFilteringInterface;
		unsigned int singleViewTotalSurfelsHost[MAX_CAMERA_COUNT];	
		unsigned int singleViewFilteredSurfelsHost[MAX_CAMERA_COUNT];

		unsigned int totalValidData[MAX_CAMERA_COUNT];
		unsigned int outlierData[MAX_CAMERA_COUNT];

		DeviceArray<unsigned int> singleViewTotalSurfels;	   // 记录当前视角下，shader处理得到的有效面元的数量
		DeviceArray<unsigned int> singleViewFilteredSurfels;   // 记录当前视角下被筛掉的有效面元数量
		bool shouldRefresh = false;
		void initSolverMapsFilteredIndexMap();
		void releaseSolverMapsFilteredIndexMap();
		/**
		 * \brief 检查是否需要进入刷新帧，当solverMap中Live域与观察帧匹配度很低的时候，直接进入刷新帧.
		 * 
		 * \param ratio 滤除的比例大于这个阈值即刷新
		 * \return 是否需要刷新
		 */
		bool CheckFilteredSolverMapsSurfelRatio(float ratio = 0.3f);
		//The workforce method for solver maps drawing
		void drawZeroViewSolverMapsIndexMap(const unsigned int num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera0, bool with_recent_observation);
		void drawOtherViewsSolverMapsIndexMap(const unsigned int CameraID, const unsigned int num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera1, bool with_recent_observation);
/******************************************************** SolverMap ********************************************************/

/******************************************************** FusionMap ********************************************************/

	public:
		struct FusionMaps {
			cudaTextureObject_t warp_vertex_map;//没有world2camera后的
			cudaTextureObject_t warp_normal_map;//没有world2camera后的
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		};
		void MapFusionMapsToCuda(FusionMaps* maps, cudaStream_t stream = 0);
		void UnmapFusionMapsFromCuda(cudaStream_t stream = 0);

		void DrawFusionMap(const unsigned int num_vertex, int vao_idx, const mat34* world2camera);
		void DrawZeroViewFusionMap(const unsigned int num_vertex, int vao_idx, const Matrix4f& world2camera);
		void DrawOtherViewsFusionMaps(const unsigned int CameraID, const unsigned int num_vertex, int vao_idx, const Matrix4f& world2camera);

		void FilterFusionMapsIndexMap(FusionMaps* maps, CameraObservation& observation, mat34* Live2Camera, cudaStream_t stream = 0);


	private:
		CudaTextureSurface fusionMapFilteredIndexMap[MAX_CAMERA_COUNT];	// 滤波后的IndexMap，除去透射点
		device::FusionMapFilteringInterface fusionMapFilteringInterface;
		void initFusionMapsFilteredIndexMap();
		void releaseFusionMapsFilteredIndexMap();

/******************************************************** FusionMap ********************************************************/
	};
}


