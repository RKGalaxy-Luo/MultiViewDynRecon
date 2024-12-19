/*****************************************************************//**
 * \file   Camera.h
 * \brief  ������������Լ���Ӧ��һЩ����
 * 
 * \author LUO
 * \date   January 15th 2024
 *********************************************************************/
#pragma once
#include "CommonTypes.h"
#include <math/MatUtils.h>
#include <math/VectorUtils.h>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <base/Logging.h>

namespace SparseSurfelFusion {

	const static float CameraRotatingAnglePerFrame = 5.0f;

	// Ĭ���������
	const static float YAW = -90.0f;
	const static float PITCH = 0.0f;
	const static float SPEED = 2.5f;
	const static float SENSITIVITY = 0.1f;
	const static float ZOOM = 45.0f;

	class Camera {
	private:

		unsigned int deviceCount = 0;			// �������

		//��ǰ��������ⲿ����
		mat34 world2camera[MAX_CAMERA_COUNT];	//��������ϵת���������ϵ    �����ŵ�Ӧ���ǵ�0֡live����뵱ǰ֡��ȵ�ľ����
		mat34 camera2world[MAX_CAMERA_COUNT];	//�������ϵת����������ϵ

		//��ʼ�ⲿ����
		Eigen::Matrix4f init_world2camera;		//��ʼ����������ϵת���������ϵ����
		Eigen::Matrix4f init_camera2world;		//��ʼ���������ϵת����������ϵ����
	public:

		using Ptr = std::shared_ptr<Camera>;

		/**
		 * \brief ���캯��������һ��Camera�࣬����world2camera��camera2world��init_camera2world��init_world2camera���󶼸�ֵΪ��λ����
		 *		ͬʱ��OpenGL����ӽǽ��г�ʼ��.
		 *
		 */
		Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);
		// ���б���ֵ�Ĺ��캯��
		Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

		/**
		 * \brief ��Eigen���е�ŷʽ�任(λ�˾���)������Eigen::Isometry3f����Camera��Ĳ�����ֵ.
		 * 
		 * \param init_camera2world Eigen���е�ŷʽ�任(λ�˾���)
		 */
		Camera(const Eigen::Isometry3f& init_camera2world_Isometry3f);

		~Camera() = default;

		void SetDevicesCount(const unsigned int devCount) { deviceCount = devCount; }

		//Ψһ�����޸ĵĽ��棬���漰world2camera��camera2world
		/**
		 * \brief ͨ���������������ϵ���������ϵ��λ�˾���
		 *		  ��Camera���е�λ��ת��world2camera��camera2world����ֵ.
		 * 
		 * \param world2camera_mat34
		 */
		void SetWorld2Camera(const mat34 & world2camera_mat34,int k);

		/**
		 * \brief �����������ϵת���������ϵ�е�λ�˱任����.
		 * 
		 * \return ��������ϵת���������ϵ�е�λ�˱任����.
		 */
		const SparseSurfelFusion::mat34& GetWorld2Camera(int k) const;
		/**
		 * \brief ����������ϵ����������ϵ�е�λ�˱任����.
		 * 
		 * \return �������ϵ����������ϵ�е�λ�˱任����
		 */
		const SparseSurfelFusion::mat34& GetCamera2World(int k) const;

		/**
		 * \brief ��world2camera�����mat34ת��Eigen::Matrix4f���ͣ�������Ⱦ.
		 * 
		 * \return Eigen::Matrix4f���͵�world2camera����
		 */
		Eigen::Matrix4f GetWorld2CameraEigen(int k) const;

		/**
		 * \brief ���World2Camera����mat34����.
		 * 
		 * \return World2Camera����mat34����
		 */
		mat34* GetWorld2CameraMat34Array();

		/**
		 * \brief ��camera2world�����mat34ת��Eigen::Matrix4f���ͣ�������Ⱦ.
		 * 
		 * \return Eigen::Matrix4f���͵�camera2world����
		 */
		Eigen::Matrix4f GetCamera2WorldEigen(int k ) const;

		/**
		 * \brief ���Eigen::Matrix4f���͵ĳ�ʼ��world2camera�ľ���������Ⱦ.
		 * 
		 * \return Eigen::Matrix4f���͵�init_world2camera����.
		 */
		const Eigen::Matrix4f& GetInitWorld2CameraEigen() const;

		/**
		 * \brief ���Eigen::Matrix4f���͵ĳ�ʼ��camera2world�ľ���������Ⱦ.
		 * 
		 * \return Eigen::Matrix4f���͵�init_camera2world����.
		 */
		const Eigen::Matrix4f& GetInitCamera2WorldEigen() const;

	public:

		enum class CameraMovement {
			FORWARD,
			BACKWARD,
			LEFT,
			RIGHT,
			AXISROTATION_X,
			AXISROTATION_Y,
			AXISROTATION_Z
		};



		// �������
		glm::vec3 Position;	// �������λ��
		glm::vec3 Front;
		glm::vec3 Up;
		glm::vec3 Right;
		glm::vec3 WorldUp;
		// ŷ����
		float Yaw;
		float Pitch;
		// camera options
		float MovementSpeed;
		float MouseSensitivity;
		float Zoom;


		// ����ʹ��ŷ���Ǻ�LookAt����������ͼ����
		glm::mat4 GetViewMatrix();
		// ������κ����Ƽ��̵�����ϵͳ���յ����롣������������enum��ʽ���������(�Ӵ���ϵͳ�г������)
		void ProcessKeyboard(CameraMovement direction, float deltaTime);
		// ������������ϵͳ���յ����롣����x��y�����ϵ�ƫ��ֵ��
		void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);
		// ������������¼����յ����롣ֻ��Ҫ�ڴ�ֱ����������
		void ProcessMouseScroll(float yoffset);

	private:
		// �������(���µ�)ŷ���Ǽ���ǰʸ��
		void updateCameraVectors()
		{
			// �����µ�Front����
			glm::vec3 front;
			front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
			front.y = sin(glm::radians(Pitch));
			front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
			Front = glm::normalize(front);
			// ͬ�����¼���Right��Up����
			Right = glm::normalize(glm::cross(Front, WorldUp));  // ���������й�һ������Ϊ���ϻ����¿��Ĵ���Խ�࣬���ǵĳ��Ⱦ�Խ�ӽ� 0����ᵼ���ƶ��ٶȱ�����
			Up = glm::normalize(glm::cross(Right, Front));
		}


	};
}
