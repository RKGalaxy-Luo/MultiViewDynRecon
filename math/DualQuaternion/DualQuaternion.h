/*****************************************************************//**
 * \file   DualQuaternion.h
 * \brief  ��ż���Լ���ż��Ԫ���Ķ�����������㣬���Բ�ֵ��Ƥ(Skinning)
 * 
 * \author LUO
 * \date   January 2024
 *********************************************************************/
#pragma once
#include "Quaternion.h"
#include <math/VectorUtils.h>

namespace SparseSurfelFusion {
	//��ż��:q0 + q1*��
	struct DualNumber {
		float q0; //��ż��ʵ��
		float q1; //��ż����żϵ��
		/**
		 * \brief ��ż�����캯��.
		 * 
		 * \return void
		 */
		__host__ __device__ DualNumber() : q0(0), q1(0) {}
		/**
		 * \brief �����ż����_q0ʵ����_q1��żϵ��.
		 * 
		 * \param _q0 ��ż��ʵ��
		 * \param _q1 ��ż����żϵ��
		 * \return 
		 */
		__host__ __device__ DualNumber(float _q0, float _q1) : q0(_q0), q1(_q1) {}
		/**
		 * \brief ���ء�+������ż�����.
		 * 
		 * \param _dn ��ӵĶ�ż��
		 * \return ��ż����ӵĽ��
		 */
		__host__ __device__ DualNumber operator+(const DualNumber& _dn) const
		{
			return{ q0 + _dn.q0, q1 + _dn.q1 };
		}
		/**
		 * \brief ���ء�+=������ż���������һ����ż��������ֵ���Լ�.
		 * 
		 * \param _dn ��ӵĶ�ż��
		 * \return ��ż�����������һ����ż��_dn�Ľ��
		 */
		__host__ __device__ DualNumber& operator+=(const DualNumber& _dn)
		{
			*this = *this + _dn;
			return *this;
		}
		/**
		 * \brief ���ء�*������ż���ĳ˷�.
		 * 
		 * \param _dn ��ż��
		 * \return ��ż����˵Ľ��
		 */
		__host__ __device__ DualNumber operator*(const DualNumber& _dn) const
		{
			return{ q0 * _dn.q0, q0 * _dn.q1 + q1 * _dn.q0 };
		}
		/**
		 * \brief ���ء�*=������ż���������һ��ż��������ֵ���Լ�.
		 * 
		 * \param _dn ��һ����ż����
		 * \return ������˺�Ķ�ż��
		 */
		__host__ __device__ DualNumber& operator*=(const DualNumber& _dn)
		{
			*this = *this * _dn;
			return *this;
		}
		/**
		 * \brief ��ö�ż���ĵ���.
		 * 
		 * \return ���ض�ż���ĵ���
		 */
		__host__ __device__ DualNumber reciprocal() const
		{
			return{ 1.0f / q0, -q1 / (q0 * q0) };
		}

		/**
		 * \brief ��ö�ż�������ŵĽ��.
		 * 
		 * \return ���ض�ż�������ŵ�ֵ
		 */
		__host__ __device__ DualNumber sqrt() const
		{
			return{ sqrtf(q0), q1 / (2 * sqrtf(q0)) };
		}
	};

	// ǰ����������ż��Ԫ��
	struct DualQuaternion;
	__host__ __device__ DualQuaternion operator*(const DualNumber& _dn, const DualQuaternion& _dq);

	// ��ż��Ԫ����q0 + q1*��  (����q0��q1������Ԫ��)
	struct DualQuaternion {

		Quaternion q0; // ��ż��Ԫ���У���Ԫ�����ɵ�ʵ��
		Quaternion q1; // ��ż��Ԫ���У���Ԫ�����ɵĶ�żϵ��
		/**
		 * \brief �չ��캯����������ż��Ԫ��.
		 * 
		 * \return void
		 */
		__host__ __device__ DualQuaternion() {}
		/**
		 * \brief ����������Ԫ�������ż��Ԫ��.
		 * 
		 * \param _q0 ʵ�����ֵ���Ԫ��
		 * \param _q1 ��ż���ֵ���Ԫ��
		 * \return void
		 */
		__host__ __device__ DualQuaternion(const Quaternion& _q0, const Quaternion& _q1) : q0(_q0), q1(_q1) {}
		/**
		 * \brief ��mat34���͵�λ�˾���ת���ɶ�ż��Ԫ��.
		 * 
		 * \param T ����mat34λ�˾���
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

		//����Ԫ���ض��塰+��
		/**
		 * \brief ���ء�+������ż��Ԫ�����.
		 * 
		 * \param _dq ��ż��Ԫ������
		 * \return ������Ӻ��ֵ
		 */
		__host__ __device__ DualQuaternion operator+(const DualQuaternion& _dq) const
		{
			Quaternion quat0(q0 + _dq.q0);
			Quaternion quat1(q1 + _dq.q1);
			return{ quat0, quat1 };
		}

		/**
		 * \brief ��ż��Ԫ�����.
		 * 
		 * \param _dq ��ż��Ԫ�����ҳ���
		 * \return ��ż��Ԫ����˵Ľ��
		 */
		__host__ __device__ DualQuaternion operator*(const DualQuaternion& _dq) const
		{
			Quaternion quat0(q0 * _dq.q0);
			Quaternion quat1(q1 * _dq.q0 + q0 * _dq.q1);
			return{ quat0, quat1 };
		}

		/**
		 * \brief ���ء�*����λ�˷���.
		 * 
		 * \param t ��ֵ������t��[0, 1]
		 * \return λ�˷����Ľ��
		 */
		__host__ __device__ __forceinline__ DualQuaternion operator*(const float& ratio) const
		{
			float t = ratio;
			if (t > 1.0f) t = 1.0f;
			else if (t < 1e-10f) t = 1e-10f;

			Quaternion q0Scaled = q0.pow(t);								// ������ת����
			Quaternion q1Scaled = ((q1 * q0.conjugate()) * t) * q0Scaled;	// ����λ�Ʋ���

			DualQuaternion dq = DualQuaternion(q0Scaled, q1Scaled);

			return dq;
		}

		/**
		 * \brief ��ż��Ԫ���Լ�_dq.
		 * 
		 * \param _dq ��ż��Ԫ���ļ���
		 * \return ��ǰ��ż��Ԫ�����ԼӺ�Ľ��
		 */
		__host__ __device__ DualQuaternion& operator+=(const DualQuaternion& _dq)
		{
			*this = *this + _dq;
			return *this;
		}

		/**
		 * \brief ��ż��Ԫ���Գ�.
		 * 
		 * \param _dq ��ż��Ԫ������
		 * \return �Գ˵Ľ��
		 */
		__host__ __device__ DualQuaternion& operator*=(const DualQuaternion& _dq)
		{
			*this = *this * _dq;
			return *this;
		}
		/**
		 * \brief ��ż��Ԫ������Ԫ�����.
		 * 
		 * \param _dn ��˵Ķ�ż��
		 * \return 
		 */
		__host__ __device__ DualQuaternion operator*(const DualNumber& _dn) const
		{
			return _dn * *this;
		}

		/**
		 * \brief ��ż��Ԫ���Գ˶�ż��.
		 * 
		 * \param _dn ��ż������
		 * \return ��ż��Ԫ���Գ�һ����ż���Ľ��
		 */
		__host__ __device__ DualQuaternion& operator*=(const DualNumber& _dn)
		{
			*this = *this * _dn;
			return *this;
		}

		/**
		 * \brief ����ת��������ż��Ԫ��ʵ���Ͷ�ż���ֵ���Ԫ������Ԫ��ʵ����ȡ����������µĶ�ż��.
		 * 
		 * \return void
		 */
		__host__ __device__ operator DualNumber() const
		{
			return DualNumber(q0.w(), q1.w());
		}
		
		/**
		 * \brief ��ż��Ԫ���Ĺ���.
		 * 
		 * \return ���ض�ż��Ԫ���Ĺ���
		 */
		__host__ __device__ DualQuaternion conjugate() const
		{
			return{ q0.conjugate(), q1.conjugate() };
		}

		/**
		 * \brief ��ż��Ԫ����2-����(qq*).
		 * 
		 * \return ��ö�ż��(ȡ��ż��Ԫ������Ԫ����ʵ������ɶ�ż��)
		 */
		__host__ __device__ DualNumber squared_norm() const
		{
			return (*this) * (this->conjugate());
		}
		/**
		 * \brief ��ż��Ԫ���ķ���.
		 * 
		 * \return ��ö�ż��(ȡ��ż��Ԫ������Ԫ����ʵ������ɶ�ż��)
		 */
		__host__ __device__ DualNumber norm() const
		{
			float a0 = q0.norm();
			float a1 = q0.dot(q1) / q0.norm();
			return{ a0, a1 };
		}

		/**
		 * \brief ��ż��Ԫ������.
		 * 
		 * \return 
		 */
		__host__ __device__ DualQuaternion inverse() const
		{
			return this->conjugate() * this->squared_norm().reciprocal();
		}

		/**
		 * \brief ����ǰ��ż��Ԫ����һ��.
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
		 * \brief ����ǰ��ż��Ԫ����һ���Ľ���������¶�ż��Ԫ��.
		 * 
		 * \return �µ��ѹ�һ���Ķ�ż��Ԫ��
		 */
		__host__ __device__ DualQuaternion normalized() const {
			DualQuaternion dq = *this;
			dq.normalize();
			return dq;
		}

		/**
		 * \brief ʵ��DualQuaternion����ת����mat34���ͣ���õ�ǰ��ż��Ԫ����Ӧ��λ�˱任����mat34.
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
		 * \brief ��õ�ǰ��ż��Ԫ����Ӧ��λ�˾���mat34.(����ܻ�ı�ֵ)
		 * 
		 * \return ��ǰ��ż��Ԫ����Ӧ��λ�˾���mat34.
		 */
		__host__ __device__ mat34 se3_matrix() {
			this->normalize();
			const mat33 rotate = this->q0.rotation_matrix(false);
			const Quaternion trans_part = 2.0f * q1 * q0.conjugate();
			const float3 translate = make_float3(trans_part.x(), trans_part.y(), trans_part.z());
			return mat34(rotate, translate);
		}

		/**
		 * \brief ����ǰ��ż��Ԫ����0������Ԫ������ƽ��ֵʱʹ�ô˷�����
		 *		  �������֮�󣬲�Ҫʹ��normalized()ȥ��õ�ǰ������Ķ�ż��Ԫ����һ��ֵ.
		 * 
		 * \return 
		 */
		__host__ __device__ void set_zero() {
			q0.x() = q0.y() = q0.z() = q0.w() = 0.f;
			q1.x() = q1.y() = q1.z() = q1.w() = 0.f;
		}

		/**
		 * \brief ���õ�λ��ż��Ԫ��(�������λ�˱任�ĵ�λ��ż��Ԫ��).
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
	 * \brief ��ż�����ż��Ԫ�����.
	 * 
	 * \param _dn ��ż��
	 * \param _dq ��ż��Ԫ��
	 * \return ��˺�õ��Ķ�ż��Ԫ��
	 */
	__host__ __device__ __forceinline__ DualQuaternion operator*(const DualNumber& _dn, const DualQuaternion& _dq)
	{
		const Quaternion quat0 = _dn.q0 * _dq.q0;
		const Quaternion quat1 = _dn.q0 * _dq.q1 + _dn.q1 * _dq.q0;
		return{ quat0, quat1 };
	}

	/**
	 * \brief ���ż��Ԫ����ƽ��ֵ��ȷ����ǰ���λ�˱仯���ھ�ϡ���Ӱ�쵱ǰ���λ��.
	 * 
	 * \param warp_field ϡ�趥���Ӧ�Ķ�ż��Ԫ������
	 * \param knn ��ǰ���Ӧ���ھ�ϡ���
	 * \param weight ÿ���ھӽڵ����������Ӱ��Ȩ��
	 * \return �����ھ�ϡ����Ӱ��֮���������Ķ�ż��Ԫ��(λ�˱任���)
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
	 * \brief ���ż��Ԫ����ƽ��ֵ��ȷ����ǰ���λ�˱仯���ھ�ϡ���Ӱ�쵱ǰ���λ��.
	 *
	 * \param warp_field ϡ�趥���Ӧ�Ķ�ż��Ԫ������
	 * \param knn ��ǰ���Ӧ���ھ�ϡ���
	 * \param weight ÿ���ھӽڵ����������Ӱ��Ȩ��
	 * \return �����ھ�ϡ����Ӱ��֮���������Ķ�ż��Ԫ��(λ�˱任���)
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
	 * \brief ����x,y,z����ת����ת�������Լ���x,y,z��ƽ�Ƶ�ƽ��������ת���ɷ�������Ķ�ż��Ԫ������.
	 * 
	 * \param twist_rot ��x,y,z����ת����ת����
	 * \param twist_trans ��x,y,z��ƽ�Ƶ�ƽ������
	 * \param dq ��������λ�˱任�Ķ�ż��Ԫ������
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

		Quaternion q0Scaled = dq.q0.pow(t);								// ������ת����
		Quaternion q1Scaled = ((dq.q1 * dq.q0.conjugate()) * t) * q0Scaled;	// ����λ�Ʋ���

		DualQuaternion InterpolationDq = DualQuaternion(q0Scaled, q1Scaled);

		return InterpolationDq;
	}
}
