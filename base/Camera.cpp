/*****************************************************************//**
 * \file   Camera.cpp
 * \brief  算法中的摄像头类，主要是涉及世界坐标系转相机坐标系
 * 
 * \author LUO
 * \date   January 12th 2024
 *********************************************************************/
#include "Camera.h"

SparseSurfelFusion::Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, 0.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    /************************************ 存储非刚性对齐特征点的持续的位姿变换 ************************************/
   //hsg
    for (int k = 0; k < MAX_CAMERA_COUNT; k++) {
    world2camera[k] = mat34::identity();
    camera2world[k] = mat34::identity();
    }
   //hsg
    init_camera2world.setIdentity();
    init_world2camera.setIdentity();
    /************************************ 记录OpenGL相机的观察角度变换 ************************************/
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

SparseSurfelFusion::Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    /************************************ 存储非刚性对齐特征点的持续的位姿变换 ************************************/
    
    //hsg 
    for (int k = 0; k < MAX_CAMERA_COUNT; k++) {
        world2camera[k] = mat34::identity();
        camera2world[k] = mat34::identity();
    }

    init_camera2world.setIdentity();
    init_world2camera.setIdentity();
    /************************************ 记录OpenGL相机的观察角度变换 ************************************/
    Position = glm::vec3(posX, posY, posZ);
    WorldUp = glm::vec3(upX, upY, upZ);
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

SparseSurfelFusion::Camera::Camera(const Eigen::Isometry3f& init_camera2world_Isometry3f)
{
	init_camera2world = init_camera2world_Isometry3f.matrix();
	init_world2camera = init_camera2world_Isometry3f.inverse().matrix();
    
    for (int k = 0; k < MAX_CAMERA_COUNT; k++) {
        world2camera[k] = mat34(init_camera2world_Isometry3f.inverse());
        camera2world[k] = mat34(init_camera2world_Isometry3f);
    }

}


void SparseSurfelFusion::Camera::SetWorld2Camera(const mat34& world2camera_mat34, int k)
{
    if (k >= deviceCount) {
        LOGGING(FATAL) << "输入的相机编号太大了！！！";
    }
    else {
        world2camera[k] = world2camera_mat34;
	    camera2world[k] = world2camera_mat34.inverse();
    }
	
}

const SparseSurfelFusion::mat34& SparseSurfelFusion::Camera::GetWorld2Camera(int k) const
{
	return world2camera[k];
}

const SparseSurfelFusion::mat34& SparseSurfelFusion::Camera::GetCamera2World(int k) const
{
	return camera2world[k];
}

Eigen::Matrix4f SparseSurfelFusion::Camera::GetWorld2CameraEigen(int k) const
{
	return toEigen(world2camera[k]);
}

SparseSurfelFusion::mat34* SparseSurfelFusion::Camera::GetWorld2CameraMat34Array()
{
    return world2camera;
}

Eigen::Matrix4f SparseSurfelFusion::Camera::GetCamera2WorldEigen(int k) const
{
	return toEigen(camera2world[k]);
}

const Eigen::Matrix4f& SparseSurfelFusion::Camera::GetInitWorld2CameraEigen() const
{
	return init_world2camera;
}


const Eigen::Matrix4f& SparseSurfelFusion::Camera::GetInitCamera2WorldEigen() const
{
	return init_camera2world;
}



glm::mat4 SparseSurfelFusion::Camera::GetViewMatrix()
{
	//return glm::lookAt(Position, Position + Front, Up);
    return glm::lookAt(Position, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

}

void SparseSurfelFusion::Camera::ProcessKeyboard(CameraMovement direction, float deltaTime)
{
    float velocity = MovementSpeed * deltaTime; //速度
    switch (direction)
    {
    case CameraMovement::FORWARD: {
        Position += Front * velocity;
        break;
    }

    case CameraMovement::BACKWARD: {
        Position -= Front * velocity;
        break;
    }

    case CameraMovement::LEFT: {
        Position -= Right * velocity;
        break;
    }

    case CameraMovement::RIGHT: {
        Position += Right * velocity;
        break;
    }

    case CameraMovement::AXISROTATION_X: {
        glm::mat4 rotationMatrix = glm::mat4(1.0f);
        rotationMatrix = glm::rotate(rotationMatrix, glm::radians(CameraRotatingAnglePerFrame), glm::vec3(1.0f, 0.0f, 0.0f));//绕向量(1,1,1)旋转
        glm::vec4 nextPosition(Position, 1.0f);
        nextPosition = nextPosition * rotationMatrix;
        Position.x = nextPosition.x;
        Position.y = nextPosition.y;
        Position.z = nextPosition.z;
        break;
    }
    case CameraMovement::AXISROTATION_Y: {
        glm::mat4 rotationMatrix = glm::mat4(1.0f);
        rotationMatrix = glm::rotate(rotationMatrix, glm::radians(CameraRotatingAnglePerFrame), glm::vec3(0.0f, 1.0f, 0.0f));//绕向量(1,1,1)旋转
        glm::vec4 nextPosition(Position, 1.0f);
        nextPosition = nextPosition * rotationMatrix;
        Position.x = nextPosition.x;
        Position.y = nextPosition.y;
        Position.z = nextPosition.z;
        break;
    }
    case CameraMovement::AXISROTATION_Z: {
        glm::mat4 rotationMatrix = glm::mat4(1.0f);
        rotationMatrix = glm::rotate(rotationMatrix, glm::radians(CameraRotatingAnglePerFrame), glm::vec3(0.0f, 0.0f, 1.0f));//绕向量(1,1,1)旋转
        glm::vec4 nextPosition(Position, 1.0f);
        nextPosition = nextPosition * rotationMatrix;
        Position.x = nextPosition.x;
        Position.y = nextPosition.y;
        Position.z = nextPosition.z;
        break;
    }
    default:
        break;
    }
}

void SparseSurfelFusion::Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
{
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    //当pitch超出边界, 屏幕不会翻转
    if (constrainPitch)
    {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    // 使用更新的欧拉角更新前、右、上矢量
    updateCameraVectors();
}

void SparseSurfelFusion::Camera::ProcessMouseScroll(float yoffset)
{
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f)    Zoom = 1.0f;
    if (Zoom > 45.0f)   Zoom = 45.0f;
}

