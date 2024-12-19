#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>//OpenGL���������
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>	// CUDA��OpenGL�Դ湲��
#include <base/Logging.h>
#include <base/GlobalConfigs.h>
#include <base/CommonTypes.h>
#include <render/GLShaderProgram.h>
#include <render/Renderer.h>
#include <math/MatUtils.h>

namespace SparseSurfelFusion {

	namespace device {


		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 *
		 * \param SolvedPointsCoor �ںϺ�ĳ�����Ԫ����
		 * \param AdjustMatrix ����λ�õ�λ�˾���
		 * \param PointNum �������
		 * \param point ��Ⱦ�ĵ�
		 */
		__global__ void adjustModelPositionKernel(const mat34 AdjustMatrix, const unsigned int PointNum, float3* point, ColorVertex* vertex);

		/**
		 * \brief ����������ĺ˺���.
		 *
		 * \param PointNum �������
		 * \param points ��Ҫ���Ƶĵ�
		 * \param point ��Ⱦ�ĵ�
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
		GLFWwindow* OpticalFlowWindow = NULL;		// ��������
		GLShaderProgram OpticalFlowShader;			// ��Ⱦ�ںϵ�ĳ���
		GLuint OpticalFlowVAO;						// ������Shader��VAO
		GLuint OpticalFlowVBO;						// ������Shader��VBO

		GLShaderProgram ColorVertexShader;			// ��Ⱦ�ںϵ�ĳ���
		GLuint ColorVertexVAO;						// ������Shader��VAO
		GLuint ColorVertexVBO;						// ������Shader��VBO

		cudaGraphicsResource_t cudaVBOResources[2];	// ע�Ỻ��������CUDA

		GLuint coordinateSystemVAO;					// ����ϵVAO
		GLuint coordinateSystemVBO;					// ����ϵ�������VBO
		GLShaderProgram coordinateShader;			// ����ϵ����Ⱦ����

		glm::mat4 view = glm::mat4(1.0f);			// ȷ����ʼ�������ǵ�λ����
		glm::mat4 projection = glm::mat4(1.0f);		// ͶӰ����ѡ����͸�ӻ�������
		glm::mat4 model = glm::mat4(1.0f);			// ����ÿ�������ģ�;��󣬲��ڻ���֮ǰ���䴫�ݸ���ɫ��

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
		 * \brief ע�������cuda��Դ.
		 *
		 */
		void registerFlowCudaResources();

		/**
		 * \brief ע�����ɫ�����cuda��Դ.
		 *
		 */
		void registerVertexCudaResources();

		/**
		 * \brief ��ʼ������������ϵ.
		 *
		 */
		void initialCoordinateSystem();

		/**
		 * \brief �������������Դ�.
		 *
		 */
		void allocateBuffer();

		/**
		 * \brief ����������귶Χ.
		 *
		 * \param points �ںϺ�ĳ�����Ԫ����
		 * \param vertex RGB��
		 * \param stream cuda��
		 */
		void adjustPointsCoordinate(float3* points, ColorVertex* vertex, cudaStream_t stream = 0);

		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 *
		 * \param points �ںϺ�ĳ�����Ԫ����
		 * \param vertex RGB��
		 * \param stream cuda��
		 */
		void adjustModelPosition(float3* points, ColorVertex* vertex, cudaStream_t stream = 0);
		/**
		 * \brief ������ӳ�䵽��cuda�󶨵�OpenGL��Դ.
		 *
		 * \param validFlow ��Ч�Ĺ���
		 * \param validFlowNum ��Ч����������
		 * \param stream cuda��
		 */
		void OpticalFlowMapToCuda(float3* validFlow, cudaStream_t stream = 0);

		/**
		 * \brief ������ɫ�ĵ㿽����OpenGL��Դ.
		 *
		 * \param validVertex ��Ч�Ķ���
		 * \param validVertexNum ��Ч��������
		 * \param stream cuda��
		 */
		void LivePointsMapToCuda(ColorVertex* validVertex, cudaStream_t stream = 0);

		/**
		 * \brief ȡ��ӳ��.
		 *
		 */
		void UnmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief ���ƹ���.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void drawOpticalFlow(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief ���ƹ���.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void drawColorVertex(glm::mat4& view, glm::mat4& projection, glm::mat4& model);
		/**
		 * \brief ��������ϵ.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief �����Ļ.
		 *
		 */
		void clearWindow();

		/**
		 * \param ˫���岢��׽�¼�.
		 *
		 */
		void swapBufferAndCatchEvent();

	public:
		/**
		 * \brief ����3D����.
		 *
		 * \param validFlow ��Ч�Ĺ���
		 * \param validFlowNum ��Ч����������
		 * \param stream cuda��
		 */
		void imshow(float3* validFlow, ColorVertex* validColorVertex, const unsigned int validFlowNum, cudaStream_t stream);
	};

}