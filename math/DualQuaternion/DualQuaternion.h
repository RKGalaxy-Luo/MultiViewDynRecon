/*****************************************************************//**
 * \file   DualQuaternion.h
 * \brief  对偶数以及对偶四元数的定义与基本计算，线性插值蒙皮(Skinning)
 * 
 * \author LUO
 * \date   January 2024
 *********************************************************************/
#pragma once
#include "Quaternion.h"
#include <math/VectorUtils.h>

namespace SparseSurfelFusion {
	//对偶数:q0 + q1*ε
	struct DualNumber {
		float q0; //对偶数实部
		float q1; //对偶数对偶系数
		/**
		 * \brief 对偶数构造函数.
		 * 
		 * \return void
		 */
		__host__ __device__ DualNumber() : q0(0), q1(0) {}
		/**
		 * \brief 构造对偶数，_q0实部，_q1对偶系数.
		 * 
		 * \param _q0 对偶数实部
		 * \param _q1 对偶数对偶系数
		 * \return 
		 */
		__host__ __device__ DualNumber(float _q0, float _q1) : q0(_q0), q1(_q1) {}
		/**
		 * \brief 重载“+”：对偶数相加.
		 * 
		 * \param _dn 相加的对偶数
		 * \return 对偶数相加的结果
		 */
		__host__ __device__ DualNumber operator+(const DualNumber& _dn) const
		{
			return{ q0 + _dn.q0, q1 + _dn.q1 };
		}
		/**
		 * \brief 重载“+=”：对偶数自身加另一个对偶数，并赋值给自己.
		 * 
		 * \param _dn 相加的对偶数
		 * \return 对偶数自身加上另一个对偶数_dn的结果
		 */
		__host__ __device__ DualNumber& operator+=(const DualNumber& _dn)
		{
			*this = *this + _dn;
			return *this;
		}
		/**
		 * \brief 重载“*”：对偶数的乘法.
		 * 
		 * \param _dn 对偶数
		 * \return 对偶数相乘的结果
		 */
		__host__ __device__ DualNumber operator*(const DualNumber& _dn) const
		{
			return{ q0 * _dn.q0, q0 * _dn.q1 + q1 * _dn.q0 };
		}
		/**
		 * \brief 重载“*=”：对偶数自身乘另一对偶数，并赋值给自己.
		 * 
		 * \param _dn 另一个对偶乘数
		 * \return 返回相乘后的对偶数
		 */
		__host__ __device__ DualNumber& operator*=(const DualNumber& _dn)
		{
			*this = *this * _dn;
			return *this;
		}
		/**
		 * \brief 获得对偶数的倒数.
		 * 
		 * \return 返回对偶数的倒数
		 */
		__host__ __device__ DualNumber reciprocal() const
		{
			return{ 1.0f / q0, -q1 / (q0 * q0) };
		}

		/**
		 * \brief 获得对偶数开根号的结果.
		 * 
		 * \return 返回对偶数开根号的值
		 */
		__host__ __device__ DualNumber sqrt() const
		{
			return{ sqrtf(q0), q1 / (2 * sqrtf(q0)) };
		}
	};

	// 前向声明：对偶四元数
	struct DualQuaternion;
	__host__ __device__ DualQuaternion operator*(const DualNumber& _dn, const DualQuaternion& _dq);

	// 对偶四元数：q0 + q1*ε  (其中q0和q1都是四元数)
	struct DualQuaternion {

		Quaternion q0; // 对偶四元数中，四元数构成的实部
		Quaternion q1; // 对偶四元数中，四元数构成的对偶系数
		/**
		 * \brief 空构造函数，声明对偶四元数.
		 * 
		 * \return void
		 */
		__host__ __device__ DualQuaternion() {}
		/**
		 * \brief 传入两个四元数构造对偶四元数.
		 * 
		 * \param _q0 实部部分的四元数
		 * \param _q1 对偶部分的四元数
		 * \return void
		 */
		__host__ __device__ DualQuaternion(const Quaternion& _q0, const Quaternion& _q1) : q0(_q0), q1(_q1) {}
		/**
		 * \brief 将mat34类型的位姿矩阵转化成对偶四元数.
		 * 
		 * \param T 传入mat34位姿矩阵
		 * \return void
		 */
		__host__ __device__ DualQuaternion(const mat34& T)
		{
			mat33 r = T.rot;
			float3 t = T.trans;
			DualQuaternion rot_part(Quaternion(r), Quaternion(0, 0, 0, 0));
			DualQuaternion vec_part(Quaternion(1, 0, 0, 0), Quaternion(0, 0.5f * t.x, 0.5f * t.y, 0.5f * t.z));
			*this = vec_part * rot_part;
		}

		//对四元数重定义“+”
		/**
		 * \brief 重载“+”：对偶四元数相加.
		 * 
		 * \param _dq 对偶四元数加数
		 * \return 返回相加后的值
		 */
		__host__ __device__ DualQuaternion operator+(const DualQuaternion& _dq) const
		{
			Quaternion quat0(q0 + _dq.q0);
			Quaternion quat1(q1 + _dq.q1);
			return{ quat0, quat1 };
		}

		/**
		 * \brief 对偶四元数相乘.
		 * 
		 * \param _dq 对偶四元数的右乘数
		 * \return 对偶四元数相乘的结果
		 */
		__host__ __device__ DualQuaternion operator*(const DualQuaternion& _dq) const
		{
			Quaternion quat0(q0 * _dq.q0);
			Quaternion quat1(q1 * _dq.q0 + q0 * _dq.q1);
			return{ quat0, quat1 };
		}

		/**
		 * \brief 重载“*”：位姿放缩.
		 * 
		 * \param t 插值比例，t∈[0, 1]
		 * \return 位姿放缩的结果
		 */
		__host__ __device__ __forceinline__ DualQuaternion operator*(const float& ratio) const
		{
			float t = ratio;
			if (t > 1.0f) t = 1.0f;
			else if (t < 1e-10f) t = 1e-10f;

			Quaternion q0Scaled = q0.pow(t);								// 缩放旋转部分
			Quaternion q1Scaled = ((q1 * q0.conjugate()) * t) * q0Scaled;	// 缩放位移部分

			DualQuaternion dq = DualQuaternion(q0Scaled, q1Scaled);

			return dq;
		}

		/**
		 * \brief 对偶四元数自加_dq.
		 * 
		 * \param _dq 对偶四元数的加数
		 * \return 当前对偶四元数的自加后的结果
		 */
		__host__ __device__ DualQuaternion& operator+=(const DualQuaternion& _dq)
		{
			*this = *this + _dq;
			return *this;
		}

		/**
		 * \brief 对偶四元数自乘.
		 * 
		 * \param _dq 对偶四元数乘数
		 * \return 自乘的结果
		 */
		__host__ __device__ DualQuaternion& operator*=(const DualQuaternion& _dq)
		{
			*this = *this * _dq;
			return *this;
		}
		/**
		 * \brief 对偶四元数与四元数相乘.
		 * 
		 * \param _dn 相乘的对偶数
		 * \return 
		 */
		__host__ __device__ DualQuaternion operator*(const DualNumber& _dn) const
		{
			return _dn * *this;
		}

		/**
		 * \brief 对偶四元数自乘对偶数.
		 * 
		 * \param _dn 对偶数乘数
		 * \return 对偶四元数自乘一个对偶数的结果
		 */
		__host__ __device__ DualQuaternion& operator*=(const DualNumber& _dn)
		{
			*this = *this * _dn;
			return *this;
		}

		/**
		 * \brief 类型转化，将对偶四元数实部和对偶部分的四元数，四元数实部提取出来，组成新的对偶数.
		 * 
		 * \return void
		 */
		__host__ __device__ operator DualNumber() const
		{
			return DualNumber(q0.w(), q1.w());
		}
		
		/**
		 * \brief 对偶四元数的共轭.
		 * 
		 * \return 返回对偶四元数的共轭
		 */
		__host__ __device__ DualQuaternion conjugate() const
		{
			return{ q0.conjugate(), q1.conjugate() };
		}

		/**
		 * \brief 对偶四元数的2-范数(qq*).
		 * 
		 * \return 获得对偶数(取对偶四元数，四元数的实部，组成对偶数)
		 */
		__host__ __device__ DualNumber squared_norm() const
		{
			return (*this) * (this->conjugate());
		}
		/**
		 * \brief 对偶四元数的范数.
		 * 
		 * \return 获得对偶数(取对偶四元数，四元数的实部，组成对偶数)
		 */
		__host__ __device__ DualNumber norm() const
		{
			float a0 = q0.norm();
			float a1 = q0.dot(q1) / q0.norm();
			return{ a0, a1 };
		}

		/**
		 * \brief 对偶四元数的逆.
		 * 
		 * \return 
		 */
		__host__ __device__ DualQuaternion inverse() const
		{
			return this->conjugate() * this->squared_norm().reciprocal();
		}

		/**
		 * \brief 将当前对偶四元数归一化.
		 * 
		 * \return void
		 */
		__host__ __device__ void normalize()
		{
			const float inv_norm = q0.norm_inversed();
			q0 = inv_norm * q0;
			q1 = inv_norm * q1;
			q1 = q1 - dot(q0, q1) * q0;
		}
		__host__ __device__ void normalize_indirect()
		{
			*this = *this * this->norm().reciprocal();
		}

		/**
		 * \brief 将当前对偶四元数归一化的结果拷贝给新对偶四元数.
		 * 
		 * \return 新的已归一化的对偶四元数
		 */
		__host__ __device__ DualQuaternion normalized() const {
			DualQuaternion dq = *this;
			dq.normalize();
			return dq;
		}

		/**
		 * \brief 实现DualQuaternion类型转化到mat34类型，获得当前对偶四元数对应的位姿变换矩阵mat34.
		 * 
		 * \return 
		 */
		__host__ __device__ operator mat34() const
		{
			mat33 r;
			float3 t;
			DualQuaternion quat_normalized = this->normalized();
			r = quat_normalized.q0.matrix();
			Quaternion vec_part = 2.0f * quat_normalized.q1 * quat_normalized.q0.conjugate();
			t = vec_part.vec();

			return mat34(r, t);
		}

		/**
		 * \brief 获得当前对偶四元数对应的位姿矩阵mat34.(这可能会改变值)
		 * 
		 * \return 当前对偶四元数对应的位姿矩阵mat34.
		 */
		__host__ __device__ mat34 se3_matrix() {
			this->normalize();
			const mat33 rotate = this->q0.rotation_matrix(false);
			const Quaternion trans_part = 2.0f * q1 * q0.conjugate();
			const float3 translate = make_float3(trans_part.x(), trans_part.y(), trans_part.z());
			return mat34(rotate, translate);
		}

		/**
		 * \brief 将当前对偶四元数清0，当四元数用于平均值时使用此方法，
		 *		  调用这个之后，不要使用normalized()去获得当前已清零的对偶四元数归一化值.
		 * 
		 * \return 
		 */
		__host__ __device__ void set_zero() {
			q0.x() = q0.y() = q0.z() = q0.w() = 0.f;
			q1.x() = q1.y() = q1.z() = q1.w() = 0.f;
		}

		/**
		 * \brief 设置单位对偶四元数(不会产生位姿变换的单位对偶四元数).
		 * 
		 * \return 
		 */
		__host__ __device__ void set_identity() {
			q0.w() = 1.0f;
			q0.x() = q0.y() = q0.z() = 0.f;
			q1.x() = q1.y() = q1.z() = q1.w() = 0.f;
		}


	};

	/**
	 * \brief 对偶数与对偶四元数相乘.
	 * 
	 * \param _dn 对偶数
	 * \param _dq 对偶四元数
	 * \return 相乘后得到的对偶四元数
	 */
	__host__ __device__ __forceinline__ DualQuaternion operator*(const DualNumber& _dn, const DualQuaternion& _dq)
	{
		const Quaternion quat0 = _dn.q0 * _dq.q0;
		const Quaternion quat1 = _dn.q0 * _dq.q1 + _dn.q1 * _dq.q0;
		return{ quat0, quat1 };
	}

	/**
	 * \brief 求对偶四元数的平均值，确定当前点的位姿变化，邻居稀疏点影响当前点的位姿.
	 * 
	 * \param warp_field 稀疏顶点对应的对偶四元数数组
	 * \param knn 当前点对应的邻居稀疏点
	 * \param weight 每个邻居节点对我这个点的影响权重
	 * \return 经过邻居稀疏点的影响之后，我这个点的对偶四元数(位姿变换情况)
	 */
	__host__ __device__ __forceinline__ DualQuaternion averageDualQuaternion(
		const DualQuaternion* warp_field,
		const ushort4& knn,
		const float4& weight
	) {
		DualQuaternion dq_average;
		dq_average.set_zero();
		dq_average += DualNumber(weight.x, 0) * warp_field[knn.x];
		dq_average += DualNumber(weight.y, 0) * warp_field[knn.y];
		dq_average += DualNumber(weight.z, 0) * warp_field[knn.z];
		dq_average += DualNumber(weight.w, 0) * warp_field[knn.w];
		return dq_average;
	}
	/**
	 * \brief 求对偶四元数的平均值，确定当前点的位姿变化，邻居稀疏点影响当前点的位姿.
	 *
	 * \param warp_field 稀疏顶点对应的对偶四元数数组
	 * \param knn 当前点对应的邻居稀疏点
	 * \param weight 每个邻居节点对我这个点的影响权重
	 * \return 经过邻居稀疏点的影响之后，我这个点的对偶四元数(位姿变换情况)
	 */
	__host__ __device__ __forceinline__ DualQuaternion averageDualQuaternion(
		const DualQuaternion* warp_field,
		const int4& knn,
		const float4& weight
	) {
		DualQuaternion dq_average;
		dq_average.set_zero();
		dq_average += DualNumber(weight.x, 0) * warp_field[knn.x];
		dq_average += DualNumber(weight.y, 0) * warp_field[knn.y];
		dq_average += DualNumber(weight.z, 0) * warp_field[knn.z];
		dq_average += DualNumber(weight.w, 0) * warp_field[knn.w];
		return dq_average;
	}

	/**
	 * \brief 将绕x,y,z轴旋转的旋转向量，以及沿x,y,z轴平移的平移向量，转化成方便运算的对偶四元数类型.
	 * 
	 * \param twist_rot 绕x,y,z轴旋转的旋转向量
	 * \param twist_trans 沿x,y,z轴平移的平移向量
	 * \param dq 包含上述位姿变换的对偶四元数类型
	 * \return void
	 */
	__host__ __device__ __forceinline__ void apply_twist(
		const float3& twist_rot,
		const float3& twist_trans,
		DualQuaternion& dq
	) {
		mat34 SE3;
		if (fabsf_sum(twist_rot) < 1e-4f) {
			SE3.rot.set_identity();
		}
		else {
			float angle = SparseSurfelFusion::norm(twist_rot);
			float3 axis = (1.0f / angle) * twist_rot;

			float c = cosf(angle);
			float s = sinf(angle);
			float t = 1.0f - c;

			SE3.rot.m00() = t * axis.x * axis.x + c;
			SE3.rot.m01() = t * axis.x * axis.y - axis.z * s;
			SE3.rot.m02() = t * axis.x * axis.z + axis.y * s;

			SE3.rot.m10() = t * axis.x * axis.y + axis.z * s;
			SE3.rot.m11() = t * axis.y * axis.y + c;
			SE3.rot.m12() = t * axis.y * axis.z - axis.x * s;

			SE3.rot.m20() = t * axis.x * axis.z - axis.y * s;
			SE3.rot.m21() = t * axis.y * axis.z + axis.x * s;
			SE3.rot.m22() = t * axis.z * axis.z + c;
		}

		SE3.trans = twist_trans;
		dq = SE3 * dq;
	}

	__host__ __device__ __forceinline__ DualQuaternion ScrewInterpolation(const DualQuaternion& dq, const float& ratio)
	{
		float t = ratio;
		if (t > 1.0f) t = 1.0f;
		else if (t < 1e-10f) t = 1e-10f;

		Quaternion q0Scaled = dq.q0.pow(t);								// 缩放旋转部分
		Quaternion q1Scaled = ((dq.q1 * dq.q0.conjugate()) * t) * q0Scaled;	// 缩放位移部分

		DualQuaternion InterpolationDq = DualQuaternion(q0Scaled, q1Scaled);

		return InterpolationDq;
	}
}
