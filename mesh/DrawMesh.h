/*****************************************************************//**
 * \file   DrawMesh.h
 * \brief  OpenGL������Ⱦ����
 * 
 * \author LUOJIAXUAN
 * \date   June 5th 2024
 *********************************************************************/
#pragma once
#include <glad/glad.h>			// <GLFW/glfw3.h>���Զ������ܶ��ϰ汾�����<glad/glad.h>��ǰ�棬��ô��ʹ��glad��Ӧ���°汾��OpenGL����Ҳ��Ϊʲô<glad/glad.h>������<GLFW/glfw3.h>֮ǰ
//#define GLFW_INCLUDE_NONE		// ��ʾ�Ľ��� <GLFW/glfw3.h> �Զ��������������Ĺ��ܣ�ʹ���������֮��Ͳ����ٴӿ��������а�������<GLFW/glfw3.h>Ҳ������<glad/glad.h>֮ǰ
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>//OpenGL���������
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Geometry.h"

#include <chrono>
#include <render/GLShaderProgram.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <render/Renderer.h>
#include <mesh/MeshConfigs.h>


//static glm::vec3 box[68] = {
//	// x��								x����ɫ
//	{ 0.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
//	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
//	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
//	{ 0.8f,   0.1f,   0.0f },			{1.0f,   0.0f,   0.0f},
//	{ 1.0f,   0.0f,   0.0f },			{1.0f,   0.0f,   0.0f},
//	{ 0.8f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
//	{ 1.2f,   0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
//	{ 1.3f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
//	{ 1.2f,  -0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
//	{ 1.3f,   0.1f,   0.0f },		    {1.0f,   0.0f,   0.0f},
//
//	{ 0.0f,   0.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.1f,   0.8f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.0f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ -0.1,   0.8f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.05f,  1.3f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ -0.05f, 1.3f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.2f,   0.0f },			{0.0f,   1.0f,   0.0f},
//	{ 0.0f,   1.1f,   0.0f },			{0.0f,   1.0f,   0.0f},
//
//	{ 0.0f,   0.0f,   0.0f },			{0.0f,   0.0f,   1.0f},
//	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
//	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
//	{ 0.1f,   0.0f,   0.8f },			{0.0f,   0.0f,   1.0f},
//	{ 0.0f,   0.0f,   1.0f },			{0.0f,   0.0f,   1.0f},
//	{ -0.1f,  0.0f,   0.8f },			{0.0f,   0.0f,   1.0f},
//	{ -0.05f, 0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
//	{ 0.05f,  0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
//	{ 0.05f,  0.0f,   1.2f },			{0.0f,   0.0f,   1.0f},
//	{ -0.05f, 0.0f,   1.1f },			{0.0f,   0.0f,   1.0f},
//	{ -0.05f, 0.0f,   1.1f },			{0.0f,   0.0f,   1.0f},
//	{ 0.05f,  0.0f,   1.1f },			{0.0f,   0.0f,   1.0f}
//};



namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief GPU�������KNN.
		 */
		struct KnnHeapDevice {
			float4& distance;
			uint4& index;

			// ���캯��ֻ�Ǹ���ָ�룬����޸���
			__host__ __device__ KnnHeapDevice(float4& dist, uint4& node_idx) : distance(dist), index(node_idx) {}

			// ���º���
			__host__ __device__ __forceinline__ void update(unsigned int idx, float dist);
		};

		/**
		 * \brief ���������붥�������4��������.
		 */
		__device__ __forceinline__ void bruteForceSearch4KNN(const float3& vertex, DeviceArrayView<OrientedPoint3D<float>> samplePoint, const unsigned int samplePointsCount, float4& distance, uint4& sampleIndex);

		/**
		 * \brief ������һ��.
		 */
		__device__ float3 VectorNormalize(const float3& normal);

		/**
		 * \brief �������.
		 */
		__device__ float3 CrossProduct(const float3& Vector_OA, const float3& Vector_OB);

		/**
		 * \brief ���������ߺ˺���.
		 */
		__global__ void CalculateMeshNormalsKernel(DeviceArrayView<Point3D<float>> verticesArray, DeviceArrayView<TriangleIndex> indicesArray, const unsigned int meshCount, Point3D<float>* normalsArray);

		/**
		 * \brief ���㶥���ڽӶ��ٸ�������.
		 */
		__global__ void CountConnectedTriangleNumKernel(DeviceArrayView<TriangleIndex> indicesArray, const unsigned int meshCount, unsigned int* ConnectedTriangleNum);

		/**
		 * \brief �����ڽ������η�������.
		 */
		__global__ void VerticesNormalsSumKernel(const Point3D<float>* meshNormals, DeviceArrayView<TriangleIndex> indicesArray, const unsigned int meshCount, Point3D<float>* VerticesNormalsSum);

		/**
		 * \brief ͨ��ƽ���ڽ�����Mesh�ķ��ߣ����㵱ǰ����ķ���.
		 */
		__global__ void CalculateVerticesAverageNormals(const unsigned int* ConnectedTriangleNum, const Point3D<float>* VerticesNormalsSum, const unsigned int verticesCount, Point3D<float>* VerticesAverageNormals);

		/**
		 * \brief ����ģ�͵�λ��.
		 */
		__global__ void AdjustModelPositionKernel(DeviceArrayView<Point3D<float>> originVertices, const mat34 AdjustModelSE3, const unsigned int verticesNum, Point3D<float>* adjustedVertices);

		/**
		 * \brief ���ݶ�������Ĳ������ھӼ��㶥�����ɫ.
		 */
		__global__ void CalculateVerticesAverageColors(DeviceArrayView<Point3D<float>> meshVertices, DeviceArrayView<OrientedPoint3D<float>> samplePoints, const unsigned int verticesCount, const unsigned int samplePointsCount, Point3D<float>* VerticesAverageColors);
	}
	class DrawMesh
	{
	public:
		DrawMesh();
		~DrawMesh();
		
		using Ptr = std::shared_ptr<DrawMesh>;

		/**
		 * \brief ���ò���.
		 *
		 * \param meshVertices ���񶥵�
		 * \param meshTriangleIndices ������Ԫ����
		 * \param samplePoints ��������RGB������
		 */
		void setInput(DeviceArrayView<Point3D<float>> meshVertices, DeviceArrayView<TriangleIndex> meshTriangleIndices, DeviceArrayView<OrientedPoint3D<float>> samplePoints);

		/**
		 * \brief �����ؽ�����������.
		 * 
		 * \param MeshVertices ���񶥵�
		 * \param MeshTriangleIndices ����index
		 * \param stream cuda��
		 */
		void CalculateMeshNormals(DeviceArrayView<Point3D<float>> MeshVertices, DeviceArrayView<TriangleIndex> MeshTriangleIndices, cudaStream_t stream = 0);

		/**
		 * \brief ��������Ķ�����ɫ��ͨ��Ѱ�������(KNN)�����㣬��������ɫ��Ȩƽ��.
		 *
		 * \param sampleDensePoints �����ĳ��ܵ�
		 * \param meshVertices ���񶥵�
		 * \param stream cuda��
		 */
		void CalculateMeshVerticesColor(DeviceArrayView<OrientedPoint3D<float>> sampleDensePoints, DeviceArrayView<Point3D<float>> meshVertices, cudaStream_t stream = 0);

		/**
		 * \brief ������Ⱦ������.
		 * 
		 * \param stream
		 */
		void DrawRenderedMesh(cudaStream_t stream = 0);

	private:

		DeviceBufferArray<Point3D<float>> VerticesAverageNormals;	// ��һ���Ķ���ƽ��������
		DeviceBufferArray<Point3D<float>> VerticesAverageColors;	// ������ɫ
		DeviceBufferArray<Point3D<float>> MeshVertices;				// ���񶥵�
		DeviceBufferArray<TriangleIndex> MeshTriangleIndices;		// ����������Ԫ����

		const unsigned int WindowWidth = 1920 * 0.9;
		const unsigned int WindowHeight = 1080 * 0.9;

		GLFWwindow* window;					// ����ָ��
		GLShaderProgram meshShader;			// ������Ⱦ
		GLShaderProgram coordinateShader;	// ����ϵ��Ⱦ

		// ���Ƶ���
		GLuint GeometryVAO;						// �������ɵ������VAO
		GLuint GeometryVBO;						// �������ɵ������VBO
		GLuint GeometryIBO;						// �������ɵ������EBO/IBO
		cudaGraphicsResource_t cudaVBOResources;// ע�Ỻ��������CUDA
		cudaGraphicsResource_t cudaIBOResources;// ע��IBO����CUDA
		// ������Ⱦ��������ϵ
		GLuint coordinateSystemVAO;			// ����ϵVAO
		GLuint coordinateSystemVBO;			// ����ϵ�������VBO

		unsigned int TranglesCount = 0;		// ����ʵʱ���������
		unsigned int VerticesCount = 0;		// ����������
		unsigned int DensePointsCount = 0;	// ���ܵ������

		//const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(180, 0, -75);
		//const float3 AdjustTranslation = make_float3(0, 0.5f, 1.0f);
		//const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);
		const float3 AdjustRotation = (EIGEN_PI / 180.0f) * make_float3(180.0f, 0, 0);
		const float3 AdjustTranslation = make_float3(-0.75f, 0.5f, 0.35f);
		const mat34 AdjustModelSE3 = mat34(AdjustRotation, AdjustTranslation);
		// �����任
		glm::mat4 view = glm::mat4(1.0f);		// ȷ����ʼ�������ǵ�λ����
		glm::mat4 projection = glm::mat4(1.0f);	// ͶӰ����ѡ����͸�ӻ�������
		glm::mat4 model = glm::mat4(1.0f);		// ����ÿ�������ģ�;��󣬲��ڻ���֮ǰ���䴫�ݸ���ɫ��
		
		/**
		 * \brief ��ʼ������������ϵ.
		 *
		 */
		void initialCoordinateSystem();

		/**
		 * \brief ע��cuda��Դ.
		 * 
		 */
		void registerCudaResources();

		/**
		 * \brief ��������Դӳ�䵽cuda.
		 * 
		 * \param MeshVertices ���񶥵�
		 * \param MeshTriangleIndices ����������Ԫ����
		 * \param stream cuda��
		 */
		void mapToCuda(DeviceArrayView<Point3D<float>> MeshVertices, DeviceArrayView<TriangleIndex> MeshTriangleIndices, cudaStream_t stream = 0);

		/**
		 * \brief ����Ҫ��Ⱦ�ĵ�ӳ�䵽cuda��Դ��.
		 *
		 */
		void unmapFromCuda(cudaStream_t stream = 0);

		/**
		 * \brief ��մ���.
		 * 
		 */
		void clearWindow();

		/**
		 * \brief ��������.
		 *
		 * \param view �����ӽǾ���
		 * \param projection ����ͶӰ����
		 * \param model ����ģ�;���
		 */
		void drawMesh(glm::mat4& view, glm::mat4& projection, glm::mat4& model);

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


		//�������,�����п��Ƶĺ���ʵ��
		bool CheckPressESC(GLFWwindow* window) {
			//glfwGetKey����������Ҫһ�������Լ�һ��������Ϊ���롣����������᷵����������Ƿ����ڱ����¡�
			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) return true;
			else return false;
			//�������Ǽ���û��Ƿ����˷��ؼ�(Esc)
		}
	};

}

