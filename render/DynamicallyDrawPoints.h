/*****************************************************************//**
 * \file   DynamicallyDrawPoints.h
 * \brief  ��̬���Ƶ�
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
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 * 
		 * \param SolvedPointsCoor �ںϺ�ĳ�����Ԫ����
		 * \param AdjustMatrix ����λ�õ�λ�˾���
		 * \param PointNum �������
		 * \param point ��Ⱦ�ĵ�
		 */
		__global__ void adjustModelPositionKernel(DeviceArrayView<float4> SolvedPointsCoor, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedPoints* point);

		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 *
		 * \param SolvedPointsCoor �ںϺ�ĳ�����Ԫ����
		 * \param AdjustMatrix ����λ�õ�λ�˾���
		 * \param PointNum �������
		 * \param point ��Ⱦ�ĵ�
		 */
		__global__ void adjustModelPositionKernel(DeviceArrayView<DepthSurfel> SolvedPoints, const mat34 AdjustMatrix, const unsigned int PointNum, Renderer::RenderedPoints* point);

		/**
		 * \brief ��Լ���ÿ��block������С��Point3D.
		 *
		 * \param maxBlockData ��ǰblock�е����ֵ
		 * \param minBlockData ��ǰblock�е���Сֵ
		 * \param points ���ܵ���
		 * \param pointsCount ��������
		 */
		__global__ void reduceMaxMinKernel(float3* maxBlockData, float3* minBlockData, DeviceArrayView<Renderer::RenderedPoints> points, const unsigned int pointsCount);
		/**
		 * \brief �ҹ�Լ���������С��.
		 *
		 * \param MaxPoint �������
		 * \param MinPoint �����С��
		 * \param maxArray ÿ��block���ĵ�Array
		 * \param minArray ÿ��block��С���Array
		 * \param GridNum ��Լʱ�������������
		 */
		__host__ void findMaxMinPoint(float3& MaxPoint, float3& MinPoint, float3* maxArray, float3* minArray, const unsigned int GridNum);

		/**
		 * \brief ����������ĺ˺������������ŵ���һ���ޣ�������depthsurfel����ɫ.
		 *
		 * \param SolvedPointsColor �ںϺ�ĳ�����Ԫ��ɫ
		 * \param center ���ĵ������
		 * \param maxEdge ��ı�
		 * \param points ��Ҫ���Ƶĵ�
		 */
		__global__ void adjustPointsCoordinateAndColorKernel(DeviceArrayView<float4> SolvedPointsColor, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedPoints* points);

		/**
		 * \brief ����������ĺ˺������������ŵ���һ���ޣ�������depthsurfel����ɫ.
		 *
		 * \param SolvedPoints �ںϺ�ĳ�����Ԫ
		 * \param center ���ĵ������
		 * \param maxEdge ��ı�
		 * \param points ��Ҫ���Ƶĵ�
		 */
		__global__ void adjustPointsCoordinateAndColorKernel(DeviceArrayView<DepthSurfel> SolvedPointsColor, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedPoints* points);
	}
	class DynamicallyDrawPoints
	{
	public:
		using Ptr = std::shared_ptr<DynamicallyDrawPoints>;

		/**
		 * \brief ��ʼ��.
		 * 
		 */
		DynamicallyDrawPoints();

		/**
		 * \brief ����.
		 * 
		 */
		~DynamicallyDrawPoints();

		/**
		 * \brief ����ʵʱ�ںϺ�ĵ�.
		 *
		 * \param SolvedPointsCoor �����ںϺ�ĵ�λ��
		 * \param SolvedPointsColor �����ںϺ�ĵ���ɫ
		 * \param stream cuda��ID
		 */
		void DrawLiveFusedPoints(DeviceArrayView<float4> SolvedPointsCoor, DeviceArrayView<float4> SolvedPointsColor, cudaStream_t stream = 0);

		/**
		 * \brief ����ʵʱ�ںϺ�ĵ�.
		 *
		 * \param SolvedPoints ����Ĵ���ɫ����Ԫ
		 * \param CheckPeriod ���Ƽ�������
		 * \param stream cuda��ID
		 */
		void DrawLiveFusedPoints(DeviceArrayView<DepthSurfel> SolvedPoints, bool CheckPeriod, cudaStream_t stream = 0);

	private:

		SynchronizeArray<float3> perBlockMaxPoint;	// ��¼ÿ���߳̿������
		SynchronizeArray<float3> perBlockMinPoint;	// ��¼ÿ���߳̿����С��

		GLFWwindow* LiveWindow;						// ��ʾ����
		GLShaderProgram FusedLivePointsShader;		// ��Ⱦ�ںϵ�ĳ���
		GLuint FusedLivePointsVAO;					// �ںϺ��Shader��VAO
		GLuint FusedLivePointsVBO;					// �ںϺ��Shader��VBO
		GLuint FusedLivePointsIBO;					// �ںϺ��Shader��EBO/IBO

		// ������Ⱦ��������ϵ
		GLuint coordinateSystemVAO;					// ����ϵVAO
		GLuint coordinateSystemVBO;					// ����ϵ�������VBO
		GLShaderProgram coordinateShader;			// ����ϵ����Ⱦ����
		cudaGraphicsResource_t cudaVBOResources;	// ע�Ỻ��������CUDA

		const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(180.0f, 0, 0);
		const float3 AdjustTranslation = make_float3(-0.75f, 0.0f, 0.35f);
		//const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(150, 0, -90);
		//const float3 AdjustTranslation = make_float3(0.6f, 0.5f, 0.0f);
		const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);

		float radius = 4.0f;//����ͷ�Ƶİ뾶
		float camX = 0.0f;
		float camZ = 5.0f;

		glm::mat4 view = glm::mat4(1.0f);			// ȷ����ʼ�������ǵ�λ����
		glm::mat4 projection = glm::mat4(1.0f);		// ͶӰ����ѡ����͸�ӻ�������
		glm::mat4 model = glm::mat4(1.0f);			// ����ÿ�������ģ�;��󣬲��ڻ���֮ǰ���䴫�ݸ���ɫ��

		DeviceBufferArray<Renderer::RenderedPoints> RenderedFusedPoints;	// ����Ⱦ�������Ԫ�����������Ԫ���и���

		bool CheckSpecificFrame = true;

		/**
		 * \brief ��ʼ����̬��Ⱦ��������Ⱦ��������Ⱦ�ںϺ�ĵ㣬ͬʱ������Ⱦ�ڴ�.
		 *
		 */
		void initialDynamicRendererAndAllocateBuffer();

		/**
		 * \brief ע��cuda��Դ.
		 *
		 */
		void registerCudaResources();

		/**
		 * \brief ��ʼ������������ϵ.
		 *
		 */
		void initialCoordinateSystem();

		/**
		 * \brief �ҵ����Ʒ�Χ.
		 *
		 * \param points �������
		 * \param MaxPoint ����õ��İ�Χ�����point
		 * \param MinPoint ����õ��İ�Χ����Сpoint
		 * \param stream cuda��
		 */
		void getBoundingBox(DeviceArrayView<Renderer::RenderedPoints> points, float3& MaxPoint, float3& MinPoint, cudaStream_t stream = 0);

		/**
		 * \brief ����������귶Χ,������ɫ.
		 *
		 * \param SolvedPointsCoor �ںϺ�ĳ�����Ԫ����
		 * \param SolvedPointsColor �ںϺ�ĳ�����Ԫ��ɫ
		 * \param MxPoint �������
		 * \param MnPoint ��С����
		 * \param stream cuda��
		 */
		void adjustPointsCoordinateAndColor(DeviceArrayView<float4> SolvedPointsColor, const float3 MxPoint, const float3 MnPoint, cudaStream_t stream = 0);

		/**
		 * \brief ����������귶Χ,������ɫ.
		 *
		 * \param SolvedPoints �ںϺ�ĳ�����Ԫ
		 * \param MxPoint �������
		 * \param MnPoint ��С����
		 * \param stream cuda��
		 */
		void adjustPointsCoordinateAndColor(DeviceArrayView<DepthSurfel> SolvedPoints, const float3 MxPoint, const float3 MnPoint, cudaStream_t stream = 0);

		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 * 
		 * \param SolvedPointsCoor �ںϺ�ĳ�����Ԫ����
		 * \param stream cuda��
		 */
		void adjustModelPosition(DeviceArrayView<float4> SolvedPointsCoor, cudaStream_t stream = 0);

		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 *
		 * \param SolvedPoints �ںϺ�ĳ�����Ԫ����
		 * \param stream cuda��
		 */
		void adjustModelPosition(DeviceArrayView<DepthSurfel> SolvedPoints, cudaStream_t stream = 0);

		/**
		 * \brief ����Ҫ��Ⱦ�ĵ�ӳ�䵽cuda��Դ��.
		 * 
		 */
		void mapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief ����Ҫ��Ⱦ�ĵ�ӳ�䵽cuda��Դ��.
		 *
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief �����Ļ.
		 * 
		 */
		void clearWindow();

		/**
		 * \brief ���Ƶ�.
		 * 
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void drawLivePoints(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief ��������ϵ.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void drawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \param ˫���岢��׽�¼�.
		 * 
		 */
		void swapBufferAndCatchEvent();

		// �������,�����п��Ƶĺ���ʵ��
		bool CheckPressESC(GLFWwindow* window) {
			//glfwGetKey����������Ҫһ�������Լ�һ��������Ϊ���롣����������᷵����������Ƿ����ڱ����¡�
			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) return true;
			else return false;
			//�������Ǽ���û��Ƿ����˷��ؼ�(Esc)
		}
	};
}


