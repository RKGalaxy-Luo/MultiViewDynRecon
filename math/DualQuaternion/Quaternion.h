/*****************************************************************//**
 * \file   Quaternion.h
 * \brief  ��Ԫ������������������Լ���Ӧ���ŵ�����
 * 
 * \author LUO
 * \date   January 13th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <math/MatUtils.h>		//�������㹤�߰�
#include <math/VectorUtils.h>	//�������㹤�߰�

namespace SparseSurfelFusion {

	struct Quaternion {
		float4 q0; //��Ԫ����w + x * i + y * j + z * k;
		/**
		 * \brief �չ��캯����������Ԫ��
		 * 
		 * \return �޷���
		 */
		__host__ __device__ Quaternion() {}
		/**
		 * \brief ������Ԫ��
		 * 
		 * \param _w ��Ԫ����ʵ��
		 * \param _x ��Ԫ�����鲿i��ϵ��
		 * \param _y ��Ԫ�����鲿j��ϵ��
		 * \param _z ��Ԫ�����鲿k��ϵ��
		 * \return ����һ����Ԫ��
		 */
		__host__ __device__ Quaternion(float _w, float _x, float _y, float _z) : q0(make_float4(_x, _y, _z, _w)) {}
		/**
		 * \brief ������Ԫ��
		 * 
		 * \param _q ����float4���͵���Ԫ��
		 * \return ����һ����Ԫ��
		 */
		__host__ __device__ Quaternion(const float4& _q) : q0(_q) {}
		/**
		 * \brief ��mat33��ת��������Ԫ��������ת����ת��Ϊ��Ԫ����ʾ
		 * 
		 * \param _rot ������ת����
		 * \return ������Ԫ��
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
		 * \brief �����Ԫ��x�鲿��ϵ��.
		 * 
		 * \return ������Ԫ��x�鲿��ϵ��
		 */
		__host__ __device__ float& x() { return q0.x; }
		/**
		 * \brief �����Ԫ��y�鲿��ϵ��.
		 * 
		 * \return ������Ԫ��y�鲿��ϵ��
		 */
		__host__ __device__ float& y() { return q0.y; }
		/**
		 * \brief �����Ԫ��z�鲿��ϵ��.
		 * 
		 * \return ������Ԫ��z�鲿��ϵ��
		 */
		__host__ __device__ float& z() { return q0.z; }
		/**
		 * \brief �����Ԫ��ʵ��w.
		 * 
		 * \return ������Ԫ��ʵ��w
		 */
		__host__ __device__ float& w() { return q0.w; }

		/**
		 * \brief �����Ԫ��x�鲿ϵ��.
		 * 
		 * \return ������Ԫ��x�鲿ϵ��
		 */
		__host__ __device__ const float& x() const { return q0.x; }
		/**
		 * \brief �����Ԫ��y�鲿ϵ��.
		 * 
		 * \return ������Ԫ��y�鲿ϵ��
		 */
		__host__ __device__ const float& y() const { return q0.y; }

		/**
		 * \brief �����Ԫ��z�鲿ϵ��.
		 * 
		 * \return ������Ԫ��z�鲿ϵ��
		 */
		__host__ __device__ const float& z() const { return q0.z; }
		/**
		 * \brief �����Ԫ��ʵ��w.
		 * 
		 * \return ������Ԫ��zʵ��w
		 */
		__host__ __device__ const float& w() const { return q0.w; }
		
		/**
		 * \brief �����Ԫ���Ĺ�����Ԫ��.
		 * 
		 * \return ���ع�����Ԫ��
		 */
		__host__ __device__ Quaternion conjugate() const { return Quaternion(q0.w, -q0.x, -q0.y, -q0.z); }
		/**
		 * \brief ��õ�ǰ��Ԫ����ģ��ƽ��.
		 * 
		 * \return ������Ԫ����ģ��ƽ��
		 */
		__host__ __device__ float square_norm() const { return q0.w * q0.w + q0.x * q0.x + q0.y * q0.y + q0.z * q0.z; }
		/**
		 * \brief ��õ�ǰ��Ԫ����ģ.
		 * 
		 * \return ������Ԫ����ģ
		 */
		__host__ __device__ float norm() const { return sqrtf(square_norm()); }
		/**
		 * \brief ��õ�ǰ��Ԫ����ģ�ĵ���.
		 * 
		 * \return ������Ԫ����ģ�ĵ���
		 */
		__host__ __device__ float norm_inversed() const { return SparseSurfelFusion::norm_inversed(q0); }
		/**
		 * \brief ��Ԫ�����.
		 * 
		 * \param _quat ��֮��˵���Ԫ��
		 * \return ��Ԫ����˺�Ľ��
		 */
		__host__ __device__ float dot(const Quaternion& _quat) const { return q0.w * _quat.w() + q0.x * _quat.x() + q0.y * _quat.y() + q0.z * _quat.z(); }
		/**
		 * \brief ����ǰ��Ԫ�����й�һ��.
		 * 
		 * \return void
		 */
		__host__ __device__ void normalize() { SparseSurfelFusion::normalize(q0); }
		/**
		 * \brief ���һ���µĹ�һ������Ԫ��(�ǵ�ǰ��Ԫ��)������ǰ��Ԫ����һ����Ŀ���.
		 * 
		 * \return �����Ĺ�һ����Ԫ��
		 */
		__host__ __device__ Quaternion normalized() const 
		{ 
			Quaternion q(*this); 
			q.normalize(); 
			return q; 
		}
		/**
		 * \brief ����ǰ��Ԫ��ת������ת����mat33����.
		 * 
		 * \return ����Ԫ��ת���ɵ���ת����
		 */
		__host__ __device__ mat33 matrix() const
		{
			//��ת����se3����֮ǰ�ȶ���Ԫ����һ��
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
		 * \brief ����ǰ��Ԫ��ת������ת����mat33����.
		 * 
		 * \param normalize �Ƿ����Ԫ�����й�һ��
		 * \return ������ת����
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
		 * \brief ����Ԫ��x,y,z�鲿������ϵ���������float3����.
		 * 
		 * \return ��Ԫ���鲿ϵ����ɵ�����
		 */
		__host__ __device__ float3 vec() const { return make_float3(q0.x, q0.y, q0.z); }

		/**
		 * \brief ��Ԫ��t����(t��).
		 * 
		 * \param t ��ָ��
		 * \return ��Ԫ��t����
		 */
		__host__ __device__ Quaternion pow(float t) const {
			// ������Ԫ����ģ��
			float norm = sqrtf(q0.x * q0.x + q0.y * q0.y + q0.z * q0.z);
			float theta = atan2f(norm, q0.w); // ������ת�Ƕ�
			float newTheta = theta * t;
			float sinNewTheta = sinf(newTheta);

			// ��ֹ������
			if (fabs(norm) < 1e-6f) {
				return Quaternion(cosf(newTheta), 0.0f, 0.0f, 0.0f);
			}

			float scale = sinNewTheta / norm;
			return Quaternion(cosf(newTheta), q0.x * scale, q0.y * scale, q0.z * scale);
		}

	};
	
	/**
	 * \brief ��Ԫ�����.
	 * 
	 * \param _left ��Ԫ��
	 * \param _right ��Ԫ��
	 * \return ��Ԫ����Ӻ�Ľ��
	 */
	__host__ __device__ __forceinline__ Quaternion operator+(const Quaternion& _left, const Quaternion& _right)
	{
		return{ _left.w() + _right.w(), _left.x() + _right.x(), _left.y() + _right.y(), _left.z() + _right.z() };
	}
	/**
	 * \brief ��Ԫ�����.
	 * 
	 * \param _left ������Ԫ��
	 * \param _right ����Ԫ��
	 * \return ��Ԫ������Ľ��
	 */
	__host__ __device__ __forceinline__ Quaternion operator-(const Quaternion& _left, const Quaternion& _right) {
		return{ _left.w() - _right.w(), _left.x() - _right.x(), _left.y() - _right.y(), _left.z() - _right.z() };
	}
	/**
	 * \brief ����������Ԫ��.
	 * 
	 * \param _scalar ����
	 * \param _quat ��Ԫ��
	 * \return ����������Ԫ���Ľ��
	 */
	__host__ __device__ __forceinline__ Quaternion operator*(float _scalar, const Quaternion& _quat)
	{
		return{ _scalar * _quat.w(), _scalar * _quat.x(), _scalar * _quat.y(), _scalar * _quat.z() };
	}
	/**
	 * \brief ��Ԫ�����Գ���.
	 * 
	 * \param _quat ��Ԫ��
	 * \param _scalar ����
	 * \return ��Ԫ�����Գ����Ľ��
	 */
	__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion& _quat, float _scalar)
	{
		return _scalar * _quat;
	}
	/**
	 * \brief ��Ԫ�����.
	 * 
	 * \param _q0 ��Ԫ�������
	 * \param _q1 ��Ԫ���ҳ���
	 * \return ��Ԫ����˽��
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
	 * \brief ��Ԫ�����Գ���.
	 *
	 * \param _quat ��Ԫ��
	 * \param _scalar ����
	 * \return ��Ԫ�����Գ����Ľ��
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
	 * \brief ��Ԫ�����.
	 * 
	 * \param q0 ��Ԫ�������
	 * \param q1 ��Ԫ���ҳ���
	 * \return ������Ԫ����˵Ľ��
	 */
	__host__ __device__ __forceinline__ float dot(const Quaternion& q0, const Quaternion& q1) {
		return dot(q0.q0, q1.q0);
	}
}
