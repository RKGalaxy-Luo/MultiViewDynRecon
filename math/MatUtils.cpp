/*****************************************************************//**
 * \file   MatUtils.cpp
 * \brief  一些基本矩阵构造函数及相应方法的实现，以及求顶点矩阵法线和半径的PCL算法
 * 
 * \author LUO
 * \date   January 13th 2024
 *********************************************************************/
#include "MatUtils.h"
#include "DualQuaternion/Quaternion.h"

SparseSurfelFusion::mat34::mat34(const float3& twist_rot, const float3& twist_trans)
{
	// 旋转部分
	if (fabsf_sum(twist_rot) < 1e-4f) {// 旋转太小了就近似成没有旋转
		rot.set_identity();
	}
	else { // 将绕轴旋转的角度分量转化成矩阵的形式
		float angle = SparseSurfelFusion::norm(twist_rot);
		float3 axis = (1.0f / angle) * twist_rot;

		float c = cosf(angle);
		float s = sinf(angle);
		float t = 1.0f - c;

		rot.m00() = t * axis.x * axis.x + c;
		rot.m01() = t * axis.x * axis.y - axis.z * s;
		rot.m02() = t * axis.x * axis.z + axis.y * s;

		rot.m10() = t * axis.x * axis.y + axis.z * s;
		rot.m11() = t * axis.y * axis.y + c;
		rot.m12() = t * axis.y * axis.z - axis.x * s;

		rot.m20() = t * axis.x * axis.z - axis.y * s;
		rot.m21() = t * axis.y * axis.z + axis.x * s;
		rot.m22() = t * axis.z * axis.z + c;
	}

	//平移部分
	trans = twist_trans;
}


SparseSurfelFusion::mat34::mat34(const Eigen::Isometry3f& se3)
{
	rot = se3.linear().matrix();
	Eigen::Vector3f translation = se3.translation();
	trans = fromEigen(translation);
}

SparseSurfelFusion::mat34::mat34(const Eigen::Matrix4f& matrix4f)
{
	rot = matrix4f.block<3, 3>(0, 0);
	Eigen::Vector3f eigen_trans = matrix4f.block<3, 1>(0, 3);
	trans = fromEigen(eigen_trans);
}

Eigen::Matrix3f SparseSurfelFusion::toEigen(const mat33& rhs)
{
	Matrix3f lhs;
	lhs(0, 0) = rhs.m00();
	lhs(0, 1) = rhs.m01();
	lhs(0, 2) = rhs.m02();
	lhs(1, 0) = rhs.m10();
	lhs(1, 1) = rhs.m11();
	lhs(1, 2) = rhs.m12();
	lhs(2, 0) = rhs.m20();
	lhs(2, 1) = rhs.m21();
	lhs(2, 2) = rhs.m22();
	return lhs;
}

Eigen::Matrix4f SparseSurfelFusion::toEigen(const mat34& rhs)
{
	Matrix4f lhs;
	lhs.setIdentity();
	//旋转部分
	lhs(0, 0) = rhs.rot.m00();
	lhs(0, 1) = rhs.rot.m01();
	lhs(0, 2) = rhs.rot.m02();
	lhs(1, 0) = rhs.rot.m10();
	lhs(1, 1) = rhs.rot.m11();
	lhs(1, 2) = rhs.rot.m12();
	lhs(2, 0) = rhs.rot.m20();
	lhs(2, 1) = rhs.rot.m21();
	lhs(2, 2) = rhs.rot.m22();
	//平移部分
	lhs.block<3, 1>(0, 3) = toEigen(rhs.trans);
	return lhs;
}

Eigen::Vector3f SparseSurfelFusion::toEigen(const float3& rhs)
{
	Vector3f lhs;
	lhs(0) = rhs.x;
	lhs(1) = rhs.y;
	lhs(2) = rhs.z;
	return lhs;
}

Eigen::Vector4f SparseSurfelFusion::toEigen(const float4& rhs)
{
	Vector4f lhs;
	lhs(0) = rhs.x;
	lhs(1) = rhs.y;
	lhs(2) = rhs.z;
	lhs(3) = rhs.w;
	return lhs;
}

float3 SparseSurfelFusion::fromEigen(const Vector3f& rhs)
{
	float3 lhs;
	lhs.x = rhs(0);
	lhs.y = rhs(1);
	lhs.z = rhs(2);
	return lhs;
}

float4 SparseSurfelFusion::fromEigen(const Vector4f& rhs)
{
	float4 lhs;
	lhs.x = rhs(0);
	lhs.y = rhs(1);
	lhs.z = rhs(2);
	lhs.w = rhs(3);
	return lhs;
}

void SparseSurfelFusion::fromEigen(const Isometry3f& se3, Quaternion& rotation, float3& translation)
{
	mat33 rot(se3.linear().matrix());
	rotation = Quaternion(rot);
	Vector3f trans_eigen = se3.translation();
	translation = fromEigen(trans_eigen);
}

