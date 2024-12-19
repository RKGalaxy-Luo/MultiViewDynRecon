/*****************************************************************//**
 * \file   Camera.h
 * \brief  声明摄像机类以及对应的一些方法
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

	// 默认相机参数
	const static float YAW = -90.0f;
	const static float PITCH = 0.0f;
	const static float SPEED = 2.5f;
	const static float SENSITIVITY = 0.1f;
	const static float ZOOM = 45.0f;

	class Camera {
	private:

		unsigned int deviceCount = 0;			// 相机数量

		//当前摄像机的外部参数
		mat34 world2camera[MAX_CAMERA_COUNT];	//世界坐标系转到相机坐标系    这里存放的应该是第0帧live点对齐当前帧深度点的矩阵吧
		mat34 camera2world[MAX_CAMERA_COUNT];	//相机坐标系转到世界坐标系

		//初始外部参数
		Eigen::Matrix4f init_world2camera;		//初始化世界坐标系转到相机坐标系矩阵
		Eigen::Matrix4f init_camera2world;		//初始化相机坐标系转到世界坐标系矩阵
	public:

		using Ptr = std::shared_ptr<Camera>;

		/**
		 * \brief 构造函数，构造一个Camera类，并将world2camera，camera2world，init_camera2world，init_world2camera矩阵都赋值为单位矩阵
		 *		同时给OpenGL相机视角进行初始化.
		 *
		 */
		Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);
		// 带有标量值的构造函数
		Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

		/**
		 * \brief 用Eigen库中的欧式变换(位姿矩阵)的类型Eigen::Isometry3f，给Camera类的参数赋值.
		 * 
		 * \param init_camera2world Eigen库中的欧式变换(位姿矩阵)
		 */
		Camera(const Eigen::Isometry3f& init_camera2world_Isometry3f);

		~Camera() = default;

		void SetDevicesCount(const unsigned int devCount) { deviceCount = devCount; }

		//唯一允许修改的界面，将涉及world2camera和camera2world
		/**
		 * \brief 通过传入的世界坐标系到相机坐标系的位姿矩阵，
		 *		  给Camera类中的位姿转换world2camera和camera2world矩阵赋值.
		 * 
		 * \param world2camera_mat34
		 */
		void SetWorld2Camera(const mat34 & world2camera_mat34,int k);

		/**
		 * \brief 获得世界坐标系转到相机坐标系中的位姿变换矩阵.
		 * 
		 * \return 世界坐标系转到相机坐标系中的位姿变换矩阵.
		 */
		const SparseSurfelFusion::mat34& GetWorld2Camera(int k) const;
		/**
		 * \brief 获得相机坐标系到世界坐标系中的位姿变换矩阵.
		 * 
		 * \return 相机坐标系到世界坐标系中的位姿变换矩阵
		 */
		const SparseSurfelFusion::mat34& GetCamera2World(int k) const;

		/**
		 * \brief 将world2camera矩阵从mat34转到Eigen::Matrix4f类型，用于渲染.
		 * 
		 * \return Eigen::Matrix4f类型的world2camera矩阵
		 */
		Eigen::Matrix4f GetWorld2CameraEigen(int k) const;

		/**
		 * \brief 获得World2Camera矩阵mat34数组.
		 * 
		 * \return World2Camera矩阵mat34数组
		 */
		mat34* GetWorld2CameraMat34Array();

		/**
		 * \brief 将camera2world矩阵从mat34转到Eigen::Matrix4f类型，用于渲染.
		 * 
		 * \return Eigen::Matrix4f类型的camera2world矩阵
		 */
		Eigen::Matrix4f GetCamera2WorldEigen(int k ) const;

		/**
		 * \brief 获得Eigen::Matrix4f类型的初始化world2camera的矩阵，用于渲染.
		 * 
		 * \return Eigen::Matrix4f类型的init_world2camera矩阵.
		 */
		const Eigen::Matrix4f& GetInitWorld2CameraEigen() const;

		/**
		 * \brief 获得Eigen::Matrix4f类型的初始化camera2world的矩阵，用于渲染.
		 * 
		 * \return Eigen::Matrix4f类型的init_camera2world矩阵.
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



		// 相机属性
		glm::vec3 Position;	// 摄像机的位置
		glm::vec3 Front;
		glm::vec3 Up;
		glm::vec3 Right;
		glm::vec3 WorldUp;
		// 欧拉角
		float Yaw;
		float Pitch;
		// camera options
		float MovementSpeed;
		float MouseSensitivity;
		float Zoom;


		// 返回使用欧拉角和LookAt矩阵计算的视图矩阵
		glm::mat4 GetViewMatrix();
		// 处理从任何类似键盘的输入系统接收的输入。接受相机定义的enum形式的输入参数(从窗口系统中抽象出来)
		void ProcessKeyboard(CameraMovement direction, float deltaTime);
		// 处理从鼠标输入系统接收的输入。期望x和y方向上的偏移值。
		void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);
		// 处理从鼠标滚轮事件接收的输入。只需要在垂直轮轴上输入
		void ProcessMouseScroll(float yoffset);

	private:
		// 从相机的(更新的)欧拉角计算前矢量
		void updateCameraVectors()
		{
			// 计算新的Front向量
			glm::vec3 front;
			front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
			front.y = sin(glm::radians(Pitch));
			front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
			Front = glm::normalize(front);
			// 同样重新计算Right和Up向量
			Right = glm::normalize(glm::cross(Front, WorldUp));  // 对向量进行归一化，因为向上或向下看的次数越多，它们的长度就越接近 0，这会导致移动速度变慢。
			Up = glm::normalize(glm::cross(Right, Front));
		}


	};
}
