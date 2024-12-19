#pragma once
#include "Renderer.h"
#include <base/EncodeUtils.h>

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief ����������ĺ˺������������ŵ���һ���ޣ�������depthsurfel����ɫ.
		 *
		 * \param rawSurfels �ںϺ�ĳ�����Ԫ��ɫ
		 * \param center ���ĵ������
		 * \param maxEdge ��ı�
		 * \param renderedSurfels ��Ҫ���Ƶĵ�
		 */
		__global__ void AdjustPointsCoordinateAndColorKernel(DeviceArrayView<DepthSurfel> rawSurfels, const float3 center, const float maxEdge, const unsigned int pointsCount, Renderer::RenderedSurfels* renderedSurfels);
		
		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 *
		 * \param rawSurfels �ںϺ�ĳ�����Ԫ����
		 * \param AdjustMatrix ����λ�õ�λ�˾���
		 * \param PointNum �������
		 * \param renderedSurfels ��Ⱦ�ĵ�
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

		float radius = 4.5f;//����ͷ�Ƶİ뾶
		float camX = 0.0f;
		float camZ = 0.0f;

		glm::mat4 view = glm::mat4(1.0f);			// ȷ����ʼ�������ǵ�λ����
		glm::mat4 projection = glm::mat4(1.0f);		// ͶӰ����ѡ����͸�ӻ�������
		glm::mat4 model = glm::mat4(1.0f);			// ����ÿ�������ģ�;��󣬲��ڻ���֮ǰ���䴫�ݸ���ɫ��

		Intrinsic rendererIntrinsic;
		int frameIndex;
		float2 confidenceTimeThreshold;
		GLFWwindow* SurfelsWindow;										// ��ʾ����

		GLShaderProgram LiveSurfelsShader;								// ��Ⱦ�ںϵ�ĳ���
		GLuint LiveSurfelsVAO;											// �ںϺ��Shader��VAO
		GLuint LiveSurfelsVBO;											// �ںϺ��Shader��VBO
		GLuint LiveSurfelsIBO;											// �ںϺ��Shader��EBO/IBO
		cudaGraphicsResource_t cudaVBOResources;						// ע�Ỻ��������CUDA

		// ������Ⱦ��������ϵ
		GLuint coordinateSystemVAO;											// ����ϵVAO
		GLuint coordinateSystemVBO;											// ����ϵ�������VBO
		GLShaderProgram coordinateShader;									// ����ϵ����Ⱦ����

		DeviceBufferArray<Renderer::RenderedSurfels> RenderedSurfels;		// ����Ⱦ�������Ԫ�����������Ԫ���и���

		/**
		 * \brief ��ʼ����̬��Ⱦ��������Ⱦ��������Ⱦ�ںϺ�ĵ㣬ͬʱ������Ⱦ�ڴ�.
		 *
		 */
		void InitialDynamicRendererAndAllocateBuffer(RenderSurfelsType renderType);

		/**
		 * \brief ע��cuda��Դ.
		 *
		 */
		void RegisterCudaResources();

		/**
		 * \brief ��ʼ������������ϵ.
		 *
		 */
		void InitialCoordinateSystem();

		/**
		 * \brief ����������귶Χ,������ɫ.
		 *
		 * \param SolvedPoints �ںϺ�ĳ�����Ԫ
		 * \param center ����λ��
		 * \param maxEdge ��Χ�����߳�
		 * \param stream cuda��
		 */
		void AdjustSurfelsCoordinateAndColor(DeviceArrayView<DepthSurfel> surfels, const float3 center, const float maxEdge, cudaStream_t stream = 0);

		/**
		 * \brief ����ģ�͵�λ�ã�ʹ����������ͷ.
		 *
		 * \param SolvedPoints �ںϺ�ĳ�����Ԫ����
		 * \param stream cuda��
		 */
		void AdjustModelPosition(DeviceArrayView<DepthSurfel> SolvedPoints, cudaStream_t stream = 0);

		/**
		 * \brief ����Ҫ��Ⱦ�ĵ�ӳ�䵽cuda��Դ��.
		 *
		 */
		void MapToCuda(cudaStream_t stream = 0);

		/**
		 * \brief ����Ҫ��Ⱦ�ĵ�ӳ�䵽cuda��Դ��.
		 *
		 */
		void UnmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief �����Ļ.
		 *
		 */
		void ClearWindow();

		/**
		 * \brief ���Ƶ�.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void DrawSurfels(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \brief ��������ϵ.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void DrawCoordinateSystem(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

		/**
		 * \param ˫���岢��׽�¼�.
		 *
		 */
		void SwapBufferAndCatchEvent();

		/**
		 * \brief ��Ⱦ���ڵĽ�ͼ.
		 * 
		 */
		void ScreenShot(const unsigned int frameIdx);

	public:
		void DrawRenderedSurfels(const DeviceArrayView<DepthSurfel>& surfels, const unsigned int frameIdx, cudaStream_t stream = 0);
	};
}


