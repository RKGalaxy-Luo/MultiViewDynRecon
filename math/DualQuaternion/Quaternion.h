/*****************************************************************//**
 * \file   Quaternion.h
 * \brief  四元数的相关声明、计算以及对应符号的重载
 * 
 * \author LUO
 * \date   January 13th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <math/MatUtils.h>		//矩阵运算工具包
#include <math/VectorUtils.h>	//向量运算工具包

namespace SparseSurfelFusion {

	struct Quaternion {
		float4 q0; //四元数：w + x * i + y * j + z * k;
		/**
		 * \brief 空构造函数，声明四元数
		 * 
		 * \return 无返回
		 */
		__host__ __device__ Quaternion() {}
		/**
		 * \brief 构造四元数
		 * 
		 * \param _w 四元数的实部
		 * \param _x 四元数的虚部i的系数
		 * \param _y 四元数的虚部j的系数
		 * \param _z 四元数的虚部k的系数
		 * \return 构造一个四元数
		 */
		__host__ __device__ Quaternion(float _w, float _x, float _y, float _z) : q0(make_float4(_x, _y, _z, _w)) {}
		/**
		 * \brief 构造四元数
		 * 
		 * \param _q 传入float4类型的四元数
		 * \return 构造一个四元数
		 */
		__host__ __device__ Quaternion(const float4& _q) : q0(_q) {}
		/**
		 * \brief 用mat33旋转矩阵构造四元数，将旋转矩阵转化为四元数表示
		 * 
		 * \param _rot 输入旋转矩阵
		 * \return 构造四元数
		 */
		__host__ __device__ Quaternion(const mat33 _rot)
		{
			float tr = _rot.m00() + _rot.m11() + _rot.m22();
			if (tr > 0) {
				float s = sqrtf(tr + 1.0f) * 2;
				q0.w = s * 0.25f;
				q0.x = (_rot.m21() - _rot.m12()) / s;
				q0.y = (_rot.m02() - _rot.m20()) / s;
				q0.z = (_rot.m10() - _rot.m01()) / s;
			}
			else if ((_rot.m00() > _rot.m11()) && (_rot.m00() > _rot.m22())) {
				float s = sqrtf(1.0f + _rot.m00() - _rot.m11() - _rot.m22()) * 2;
				q0.w = (_rot.m21() - _rot.m12()) / s;
				q0.x = 0.25f * s;
				q0.y = (_rot.m01() + _rot.m10()) / s;
				q0.z = (_rot.m02() + _rot.m20()) / s;
			}
			else if (_rot.m11() > _rot.m22()) {
				float s = sqrtf(1.0f + _rot.m11() - _rot.m00() - _rot.m22()) * 2;
				q0.w = (_rot.m02() - _rot.m20()) / s;
				q0.x = (_rot.m01() + _rot.m10()) / s;
				q0.y = 0.25f * s;
				q0.z = (_rot.m12() + _rot.m21()) / s;
			}
			else {
				float s = sqrtf(1.0f + _rot.m22() - _rot.m00() - _rot.m11()) * 2;
				q0.w = (_rot.m10() - _rot.m01()) / s;
				q0.x = (_rot.m02() + _rot.m20()) / s;
				q0.y = (_rot.m12() + _rot.m21()) / s;
				q0.z = 0.25f * s;
			}
		}

		/**
		 * \brief 获得四元数x虚部的系数.
		 * 
		 * \return 返回四元数x虚部的系数
		 */
		__host__ __device__ float& x() { return q0.x; }
		/**
		 * \brief 获得四元数y虚部的系数.
		 * 
		 * \return 返回四元数y虚部的系数
		 */
		__host__ __device__ float& y() { return q0.y; }
		/**
		 * \brief 获得四元数z虚部的系数.
		 * 
		 * \return 返回四元数z虚部的系数
		 */
		__host__ __device__ float& z() { return q0.z; }
		/**
		 * \brief 获得四元数实部w.
		 * 
		 * \return 返回四元数实部w
		 */
		__host__ __device__ float& w() { return q0.w; }

		/**
		 * \brief 获得四元数x虚部系数.
		 * 
		 * \return 返回四元数x虚部系数
		 */
		__host__ __device__ const float& x() const { return q0.x; }
		/**
		 * \brief 获得四元数y虚部系数.
		 * 
		 * \return 返回四元数y虚部系数
		 */
		__host__ __device__ const float& y() const { return q0.y; }

		/**
		 * \brief 获得四元数z虚部系数.
		 * 
		 * \return 返回四元数z虚部系数
		 */
		__host__ __device__ const float& z() const { return q0.z; }
		/**
		 * \brief 获得四元数实部w.
		 * 
		 * \return 返回四元数z实部w
		 */
		__host__ __device__ const float& w() const { return q0.w; }
		
		/**
		 * \brief 获得四元数的共轭四元数.
		 * 
		 * \return 返回共轭四元数
		 */
		__host__ __device__ Quaternion conjugate() const { return Quaternion(q0.w, -q0.x, -q0.y, -q0.z); }
		/**
		 * \brief 获得当前四元数的模的平方.
		 * 
		 * \return 返回四元数的模的平方
		 */
		__host__ __device__ float square_norm() const { return q0.w * q0.w + q0.x * q0.x + q0.y * q0.y + q0.z * q0.z; }
		/**
		 * \brief 获得当前四元数的模.
		 * 
		 * \return 返回四元数的模
		 */
		__host__ __device__ float norm() const { return sqrtf(square_norm()); }
		/**
		 * \brief 获得当前四元数的模的倒数.
		 * 
		 * \return 返回四元数的模的倒数
		 */
		__host__ __device__ float norm_inversed() const { return SparseSurfelFusion::norm_inversed(q0); }
		/**
		 * \brief 四元数点乘.
		 * 
		 * \param _quat 与之点乘的四元数
		 * \return 四元数点乘后的结果
		 */
		__host__ __device__ float dot(const Quaternion& _quat) const { return q0.w * _quat.w() + q0.x * _quat.x() + q0.y * _quat.y() + q0.z * _quat.z(); }
		/**
		 * \brief 将当前四元数进行归一化.
		 * 
		 * \return void
		 */
		__host__ __device__ void normalize() { SparseSurfelFusion::normalize(q0); }
		/**
		 * \brief 获得一个新的归一化的四元数(非当前四元数)，即当前四元数归一化后的拷贝.
		 * 
		 * \return 拷贝的归一化四元数
		 */
		__host__ __device__ Quaternion normalized() const 
		{ 
			Quaternion q(*this); 
			q.normalize(); 
			return q; 
		}
		/**
		 * \brief 将当前四元数转化成旋转矩阵mat33类型.
		 * 
		 * \return 由四元数转化成的旋转矩阵
		 */
		__host__ __device__ mat33 matrix() const
		{
			//在转化成se3矩阵之前先对四元数归一化
			Quaternion q(*this);
			q.normalize();

			mat33 rot;
			rot.m00() = 1 - 2 * q.y() * q.y() - 2 * q.z() * q.z();
			rot.m01() = 2 * q.x() * q.y() - 2 * q.z() * q.w();
			rot.m02() = 2 * q.x() * q.z() + 2 * q.y() * q.w();
			rot.m10() = 2 * q.x() * q.y() + 2 * q.z() * q.w();
			rot.m11() = 1 - 2 * q.x() * q.x() - 2 * q.z() * q.z();
			rot.m12() = 2 * q.y() * q.z() - 2 * q.x() * q.w();
			rot.m20() = 2 * q.x() * q.z() - 2 * q.y() * q.w();
			rot.m21() = 2 * q.y() * q.z() + 2 * q.x() * q.w();
			rot.m22() = 1 - 2 * q.x() * q.x() - 2 * q.y() * q.y();
			return rot;
		}
		/**
		 * \brief 将当前四元数转化成旋转矩阵mat33类型.
		 * 
		 * \param normalize 是否对四元数进行归一化
		 * \return 返回旋转矩阵
		 */
		__host__ __device__ mat33 rotation_matrix(bool normalize) {
			if (normalize) this->normalize();
			mat33 rot;
			rot.m00() = 1 - 2 * y() * y() - 2 * z() * z();
			rot.m01() = 2 * x() * y() - 2 * z() * w();
			rot.m02() = 2 * x() * z() + 2 * y() * w();
			rot.m10() = 2 * x() * y() + 2 * z() * w();
			rot.m11() = 1 - 2 * x() * x() - 2 * z() * z();
			rot.m12() = 2 * y() * z() - 2 * x() * w();
			rot.m20() = 2 * x() * z() - 2 * y() * w();
			rot.m21() = 2 * y() * z() + 2 * x() * w();
			rot.m22() = 1 - 2 * x() * x() - 2 * y() * y();
			return rot;
		}

		/**
		 * \brief 将四元数x,y,z虚部的三个系数组成向量float3类型.
		 * 
		 * \return 四元数虚部系数组成的向量
		 */
		__host__ __device__ float3 vec() const { return make_float3(q0.x, q0.y, q0.z); }

		/**
		 * \brief 四元数t次幂(tθ).
		 * 
		 * \param t 幂指数
		 * \return 四元数t次幂
		 */
		__host__ __device__ Quaternion pow(float t) const {
			// 计算四元数的模长
			float norm = sqrtf(q0.x * q0.x + q0.y * q0.y + q0.z * q0.z);
			float theta = atan2f(norm, q0.w); // 计算旋转角度
			float newTheta = theta * t;
			float sinNewTheta = sinf(newTheta);

			// 防止除以零
			if (fabs(norm) < 1e-6f) {
				return Quaternion(cosf(newTheta), 0.0f, 0.0f, 0.0f);
			}

			float scale = sinNewTheta / norm;
			return Quaternion(cosf(newTheta), q0.x * scale, q0.y * scale, q0.z * scale);
		}

	};
	
	/**
	 * \brief 四元数相加.
	 * 
	 * \param _left 四元数
	 * \param _right 四元数
	 * \return 四元数相加后的结果
	 */
	__host__ __device__ __forceinline__ Quaternion operator+(const Quaternion& _left, const Quaternion& _right)
	{
		return{ _left.w() + _right.w(), _left.x() + _right.x(), _left.y() + _right.y(), _left.z() + _right.z() };
	}
	/**
	 * \brief 四元数相减.
	 * 
	 * \param _left 被减四元数
	 * \param _right 减四元数
	 * \return 四元数相减的结果
	 */
	__host__ __device__ __forceinline__ Quaternion operator-(const Quaternion& _left, const Quaternion& _right) {
		return{ _left.w() - _right.w(), _left.x() - _right.x(), _left.y() - _right.y(), _left.z() - _right.z() };
	}
	/**
	 * \brief 常数乘以四元数.
	 * 
	 * \param _scalar 常数
	 * \param _quat 四元数
	 * \return 常数乘以四元数的结果
	 */
	__host__ __device__ __forceinline__ Quaternion operator*(float _scalar, const Quaternion& _quat)
	{
		return{ _scalar * _quat.w(), _scalar * _quat.x(), _scalar * _quat.y(), _scalar * _quat.z() };
	}
	/**
	 * \brief 四元数乘以常数.
	 * 
	 * \param _quat 四元数
	 * \param _scalar 常数
	 * \return 四元数乘以常数的结果
	 */
	__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion& _quat, float _scalar)
	{
		return _scalar * _quat;
	}
	/**
	 * \brief 四元数相乘.
	 * 
	 * \param _q0 四元数左乘数
	 * \param _q1 四元数右乘数
	 * \return 四元数相乘结果
	 */
	__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion& _q0, const Quaternion& _q1)
	{
		Quaternion q;
		q.w() = _q0.w() * _q1.w() - _q0.x() * _q1.x() - _q0.y() * _q1.y() - _q0.z() * _q1.z();
		q.x() = _q0.w() * _q1.x() + _q0.x() * _q1.w() + _q0.y() * _q1.z() - _q0.z() * _q1.y();
		q.y() = _q0.w() * _q1.y() - _q0.x() * _q1.z() + _q0.y() * _q1.w() + _q0.z() * _q1.x();
		q.z() = _q0.w() * _q1.z() + _q0.x() * _q1.y() - _q0.y() * _q1.x() + _q0.z() * _q1.w();

		return q;
	}

	/**
	 * \brief 四元数乘以常数.
	 *
	 * \param _quat 四元数
	 * \param _scalar 常数
	 * \return 四元数乘以常数的结果
	 */
	__host__ __device__ __forceinline__ Quaternion operator/(const Quaternion& _quat, float _scalar)
	{
		if (_scalar > 1e-10f) {
			return { _quat.w() / _scalar, _quat.x() / _scalar, _quat.y() / _scalar, _quat.z() / _scalar };
		}
		else {
			return{ _quat.w(),  _quat.x(),  _quat.y(), _quat.z() };
		}
	}

	/**
	 * \brief 四元数点乘.
	 * 
	 * \param q0 四元数左乘数
	 * \param q1 四元数右乘数
	 * \return 两个四元数点乘的结果
	 */
	__host__ __device__ __forceinline__ float dot(const Quaternion& q0, const Quaternion& q1) {
		return dot(q0.q0, q1.q0);
	}
}
