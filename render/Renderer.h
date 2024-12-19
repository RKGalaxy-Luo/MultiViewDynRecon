/*****************************************************************//**
 * \file   Renderer.h
 * \brief  ��Ⱦ��������Ҫ����ʾ����Ⱦ����Ҫע����ǣ���Ⱦ����ĳһ�����(0�����)�ռ���Ϊչʾ�����з�0�������ȫ��ת����0������ռ���
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>			// <GLFW/glfw3.h>���Զ������ܶ��ϰ汾�����<glad/glad.h>��ǰ�棬��ô��ʹ��glad��Ӧ���°汾��OpenGL����Ҳ��Ϊʲô<glad/glad.h>������<GLFW/glfw3.h>֮ǰ
//#define GLFW_INCLUDE_NONE		// ��ʾ�Ľ��� <GLFW/glfw3.h> �Զ��������������Ĺ��ܣ�ʹ���������֮��Ͳ����ٴӿ��������а�������<GLFW/glfw3.h>Ҳ������<glad/glad.h>֮ǰ
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>//OpenGL���������
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>	// CUDA��OpenGL�Դ湲��
#include <tuple>				// tuple��һ���̶���С�Ĳ�ͬ����ֵ�ļ��ϣ��Ƿ�����std::pair
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
	// x��								x����ɫ
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
	 * \brief ����һ����Ⱦ�������ͣ���Ҫ������Ⱦ�����ӻ�.
	 */
	class Renderer
	{
	private:
		int imageWidth = 0;										// �ü���ͼ��Ŀ�
		int imageHeight = 0;									// �ü���ͼ��ĸ�

		int fusionImageWidth = 0;								// �ں�ͼ��Ŀ�
		int fusionImageHeight = 0;								// �ں�ͼ��ĸ�
		const unsigned int FusionMapScale;						// �ϲ��õĳ߶�

		unsigned int deviceCount = 0;							// ʵ�ʽ����������

		float4 rendererIntrinsic;								// ��Ⱦ������ڲ�(0��������ڲΣ���ĳһ���������ռ���Ϊ��Ⱦ�Ŀռ���չʾ)
		float4 widthHeightMaxMinDepth;							// ���߼������С���
		float4 ACameraRendererIntrinsic[MAX_CAMERA_COUNT];		// ÿ��������ڲ�
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];				// �����ʼλ��

	public:
		using Ptr = std::shared_ptr<Renderer>;
		explicit Renderer(int imageRows, int imageCols, ConfigParser::Ptr config);
		~Renderer();
		GLFWwindow* GLFW_Window = nullptr;			//��͸�����ڶ���

		NO_COPY_ASSIGN_MOVE(Renderer);
/******************************************************** ��ʾ���� ********************************************************/
	/* GLFW������ر����ͺ���  GLFW��һ��ר�����OpenGL��C���Կ�,���ṩ��һЩ��Ⱦ�������������޶ȵĽӿ�
	 * һ����� glad ��ȡһ�� glxxx �����ĺ���ָ�롣�� GLFW �������ϵͳ�Ĵ��ڹ����������� framebuffer��Ȼ��OpenGL �����滭��
	 */
	private:
		/**
		 * \brief ��ʼ��GLFW.
		 * 
		 */
		void initGLFW();

		/**
		 * \brief ��ʼ������SolverMap��IndexMap��FusionMap��IndexMap͸����Ԫ��FilteredIndexMap.
		 * 
		 */
		void initFilteredSolverMapAndFusionMap();

		/**
		 * \brief �ͷ�FilteredMap.
		 * 
		 */
		void releaseFilteredSolverMapAndFusionMap();

	public:

		/**
		 * \brief ��Ⱦ�������.
		 */
		struct RenderedPoints {
			float3 coordinate;						// ��Ⱦ�������
			float3 rgb;								// ��Ⱦ�����ɫ
		};

		/**
		 * \brief ��Ⱦ�������.
		 */
		struct RenderedSurfels {
			float4 vertexConfidence;						// ��Ⱦ�������,���Ŷ�
			float4 normalRadius;							// ��Ⱦ��ķ���,�뾶
			float4 rgbTime;									// ��Ⱦ��RGB,�ӽ�,��ʼ�۲�֡�͵�ǰ֡
		};

		/**
		 * \brief ����GLFW����ָ��.
		 *
		 * \return GLFW����ָ��.
		 */
		GLFWwindow* getWindowGLFW(unsigned int WindowsIndex) { 
			if (GLFW_Window != nullptr) return GLFW_Window;
			else LOGGING(FATAL) << "����ָ��Ϊnullptr";
		}
/******************************************************** ��ʾ���� ********************************************************/

/******************************************************** Ĭ����յ�ֵ ********************************************************/
	private:
		// С����������(Ĭ��)��ֵ
		struct GLClearValues {
			GLfloat vertexMapClear[4];
			GLfloat normalMapClear[4];
			GLfloat colorTimeClear[4];
			GLfloat solverRGBAClear[4];
			GLfloat visualizeRGBAClear[4];
			GLfloat zBufferClear;
			GLuint indexMapClear;

			/**
			 * \brief ���캯������������ֵ.
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
		 * \brief ��ʼ��Ĭ��(clear)ֵ.
		 * 
		 */
		void initClearValues();
/******************************************************** VBO���㻺�������� ********************************************************/

	private:
		GLSurfelGeometryVBO surfelGeometryVBOs[MAX_CAMERA_COUNT][2];	// ˫����
		/**
		 * \brief ��ʼ�����㻺�����.
		 * hsg����surfelGeometryVBOIndexMap�ĳ�ʼ��  ����fusiondepthVBO��  ˫�ӽ�  ��ʼ�� 
		 */
		void initVertexBufferObjects();
		/**
		 * \brief �ͷŶ��㻺�����.
		 * hsg����surfelGeometryVBOIndexMap���ͷ�    ����fusiondepthVBO��  ˫�ӽ�  �ͷ� 
		 */
		void freeVertexBufferObjects();

	public:
		/**
		 * \brief ��SurfelGeometryӳ�䵽CUDA���Ա���Ⱦ��ʾ��ͬʱ����SurfelGeometry�����Ե��ڴ�.
		 * 
		 * \param idx ����������Ⱦ�ı�־����(˫���巽ʽ��ʾ)
		 * \param cameraID �����id�����Ϊ�Ƿ�id��Ĭ������surfelGeometryVBOsȫ�����в���
		 * \param geometry ��Ҫ��Ⱦ����Ԫ���ζ���
		 * \param stream CUDA��ID
		 */
		void MapSurfelGeometryToCUDA(int idx, SurfelGeometry::Ptr geometry[MAX_CAMERA_COUNT][2], int cameraID = 0xFFFF, cudaStream_t stream = 0);
		/**
		 * \brief ��(����)��Դӳ�䵽CUDA���Ա���Ⱦ��ʾ.
		 *
		 *
		 * \param idx ����������Ⱦ�ı�־����(˫���巽ʽ��ʾ)
		 * \param cameraID �����id�����Ϊ�Ƿ�id��Ĭ������surfelGeometryVBOsȫ�����в���
		 * \param stream CUDA��ID
		 */
		void MapSurfelGeometryToCUDA(int idx, int cameraID = 0xFFFF, cudaStream_t stream = 0);
		/**
		 * \brief ��ӳ�����Դ���뵱ǰ�� CUDA �����ķ��룬���ٿ����� GPU �ϵĲ���.
		 * 
		 * 
		 * \param idx ����������Ⱦ�ı�־����(˫���巽ʽ��ʾ)
		 * \param cameraID �����id�����Ϊ�Ƿ�id��Ĭ������surfelGeometryVBOsȫ�����в���
		 * \param stream CUDA��ID
		 */
		void UnmapSurfelGeometryFromCUDA(int idx, int cameraID = 0xFFFF, cudaStream_t stream = 0);

/******************************************************** VBO���㻺�������� ********************************************************/

/******************************************************** VAO�������������� ********************************************************/
	/*
	 * ������Ⱦ��vao������vbo��ʼ��֮���ʼ�� 
	 */
	private:
		// ʵʱ��ʾ��VAO����VAO���д�����Ӧ˫���巽��
		GLuint fusionMapVAO[MAX_CAMERA_COUNT][2];
		GLuint solverMapVAO[MAX_CAMERA_COUNT][2];
		GLuint fusionDepthSurfelVAO[MAX_CAMERA_COUNT];

		// �����vao����Canonical���Live����Ԫ�����߿��ӻ�
		GLuint canonicalGeometryVAO[2]; // ����Canonical����Ԫ�����߿��ӻ���VAO
		GLuint liveGeometryVAO[2];		// ����Live����Ԫ�����߿��ӻ���VAO

		/**
		 * \brief ��ʼ�����������VAO.
		 * 
		 */
		void initMapRenderVAO();

/******************************************************** VAO�������������� ********************************************************/


/******************************************************** OpenGL��CUDA֮��ӳ����� ********************************************************/
	private:
		// ���ߴ��������֡������/��Ⱦ������
		GLFusionMapsFrameRenderBufferObjects fusionMapBuffers[MAX_CAMERA_COUNT];		// Live��(�ںϵ�Live��)��Buffer
		GLSolverMapsFrameRenderBufferObjects solverMapBuffers[MAX_CAMERA_COUNT];		// Live�� + Canonical���Buffer
	
		////����ÿ֡�ں�
		//GLFusionDepthSurfelFrameRenderBufferObjects fusionDepthSurfelBuffers[MAX_CAMERA_COUNT];

		GLOfflineVisualizationFrameRenderBufferObjects visualizationDrawBuffers;	// �������߿��ӻ���֡Buffer/��ȾBuffer
		/**
		 * \brief ��ʼ��֡Buffer/��ȾBuffer.
		 * hsg����
		 */
		void initFrameRenderBuffers();
		/**
		 * \brief �ͷ�֡Buffer/��ȾBuffer.
		 * hsg����
		 */
		void freeFrameRenderBuffers();
/******************************************************** OpenGL��CUDA֮��ӳ����� ********************************************************/
	
/******************************************************** OpenGL��ɫ�� ********************************************************/
	private:
		GLShaderProgram fusionMapShader;	// Live��(�ںϵ�Live��)����ɫ�� 
		GLShaderProgram solverMapShader;	// Live�� + Canonical�����ɫ��

		GLShaderProgram solverMapShaderIndexMap;
		GLShaderProgram fusionMapShaderIndexMap;

		/**
		 * \brief ��ʼ����������ɫ��.
		 * 
		 */
		void initProcessingShaders();
		/**
		 * \brief ��ʼ�������ӻ�����ɫ��.
		 *
		 */
		void initVisualizationShaders();

		/**
		 * \brief ��ʼ��������ɫ��.
		 *
		 */
		void initShaders();

		// ���ڿ��ӻ�����ɫ������
		struct {
			GLShaderProgram normalMap;
			GLShaderProgram phongMap;
			GLShaderProgram albedoMap;
			GLShaderProgram coordinateSystem;
		} visualizationShaders;

		// ������Ⱦ��������ϵ
		GLuint coordinateSystemVAO;					// ����ϵVAO
		GLuint coordinateSystemVBO;					// ����ϵ�������VBO


		void drawVisualizationMap(GLShaderProgram& shader, GLuint geometry_vao, unsigned num_vertex, int current_time, const Matrix4f& world2camera, bool with_recent_observation);

	public:
		void SetFrameIndex(const unsigned int frameIdx) { visualizationDrawBuffers.frameIndex = frameIdx; }
		//��ʵʱͼ��
		void ShowLiveNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowLiveAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowLivePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowReferenceNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowReferenceAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);
		void ShowReferencePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent = true);

/******************************************************** OpenGL��ɫ�� ********************************************************/


/******************************************************** SolverMap ********************************************************/
	public:
		struct SolverMaps {
			cudaTextureObject_t reference_vertex_map;	// ��ǰ�ӽǵ�Canonical��(��0���ӽ��µ�Canonical���ͨ��λ�˷�����õ�)
			cudaTextureObject_t reference_normal_map;	// ��ǰ�ӽǵ�Canonical��(��0���ӽ��µ�Canonical���ͨ��λ�˷�����õ�)
			cudaTextureObject_t warp_vertex_map;		// ��ǰ�ӽǵ�Live��: Canonical���÷Ǹ���SE3 Warp�õ�
			cudaTextureObject_t warp_normal_map;		// ��ǰ�ӽǵ�Live��: Canonical���÷Ǹ���SE3 Warp�õ�
			cudaTextureObject_t index_map;				// ��ǰ�ӽǵ�IndexMap(Canonical���з��任����ǰ�ӽǣ��ܳ��ֵĵ�)
			cudaTextureObject_t normalized_rgb_map;		// ��ǰ�ӽ��µ�Rgbͼ
		};
		void MapSolverMapsToCuda(SolverMaps* maps, cudaStream_t stream = 0);
		void UnmapSolverMapsFromCuda(cudaStream_t stream = 0);

		void DrawSolverMapsWithRecentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f* world2camera);
		void DrawSolverMapsConfidentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f* world2camera);
		
		void FilterSolverMapsIndexMap(SolverMaps* maps, CameraObservation& observation, mat34* Live2Camera, float filterRatio = 0.2f, cudaStream_t stream = 0);
		bool ShouldDoRefresh() { return shouldRefresh; }
	private:
		CudaTextureSurface solverMapFilteredIndexMap[MAX_CAMERA_COUNT];				// �˲����IndexMap����ȥ͸���
		device::SolverMapFilteringInterface solverMapFilteringInterface;
		unsigned int singleViewTotalSurfelsHost[MAX_CAMERA_COUNT];	
		unsigned int singleViewFilteredSurfelsHost[MAX_CAMERA_COUNT];

		unsigned int totalValidData[MAX_CAMERA_COUNT];
		unsigned int outlierData[MAX_CAMERA_COUNT];

		DeviceArray<unsigned int> singleViewTotalSurfels;	   // ��¼��ǰ�ӽ��£�shader����õ�����Ч��Ԫ������
		DeviceArray<unsigned int> singleViewFilteredSurfels;   // ��¼��ǰ�ӽ��±�ɸ������Ч��Ԫ����
		bool shouldRefresh = false;
		void initSolverMapsFilteredIndexMap();
		void releaseSolverMapsFilteredIndexMap();
		/**
		 * \brief ����Ƿ���Ҫ����ˢ��֡����solverMap��Live����۲�֡ƥ��Ⱥܵ͵�ʱ��ֱ�ӽ���ˢ��֡.
		 * 
		 * \param ratio �˳��ı������������ֵ��ˢ��
		 * \return �Ƿ���Ҫˢ��
		 */
		bool CheckFilteredSolverMapsSurfelRatio(float ratio = 0.3f);
		//The workforce method for solver maps drawing
		void drawZeroViewSolverMapsIndexMap(const unsigned int num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera0, bool with_recent_observation);
		void drawOtherViewsSolverMapsIndexMap(const unsigned int CameraID, const unsigned int num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera1, bool with_recent_observation);
/******************************************************** SolverMap ********************************************************/

/******************************************************** FusionMap ********************************************************/

	public:
		struct FusionMaps {
			cudaTextureObject_t warp_vertex_map;//û��world2camera���
			cudaTextureObject_t warp_normal_map;//û��world2camera���
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
		CudaTextureSurface fusionMapFilteredIndexMap[MAX_CAMERA_COUNT];	// �˲����IndexMap����ȥ͸���
		device::FusionMapFilteringInterface fusionMapFilteringInterface;
		void initFusionMapsFilteredIndexMap();
		void releaseFusionMapsFilteredIndexMap();

/******************************************************** FusionMap ********************************************************/
	};
}


