/*****************************************************************//**
 * \file   MatUtils.h
 * \brief  �������㹤�߰�����Ҫ���һЩ�����������⣬�Լ�3��4��λ�˱任����SE(3)���ļ�������
 *
 * \author LUO
 * \date   January 12th 2024
 *********************************************************************/
#pragma once
#include <vector_functions.h>
#include <base/CommonTypes.h>
#include <base/CommonUtils.h>
#include "VectorUtils.h"
#include "NumericLimits.h"

namespace SparseSurfelFusion {
	/**
	 * ��������3����������ɵ�3��3�ľ���.
	 */
	struct mat33 {
		float3 cols[3]; //�����������3����������ɵ�
		/**
		 * \brief �չ��캯��������һ���յ�3��3����
		 * \return
		 */
		__host__ __device__ mat33() {}
		/**
		 * \brief mat33���캯��������3�����������3��3�ľ���
		 * \param _a0 ����������_a0
		 * \param _a1 ����������_a1
		 * \param _a2 ����������_a2
		 * \return ���캯�����޷���ֵ
		 */
		__host__ __device__ mat33(const float3 _a0, const float3 _a1, const float3 _a2) { cols[0] = _a0; cols[1] = _a1; cols[2] = _a2; }
		/**
		 * \brief �������鹹��3��3����
		 * \param _data �������飺index = 0 - 2(������1)��index = 3 - 5(������2)��index = 6 - 8(������3)
		 * \return ���캯�����޷���ֵ
		 */
		__host__ __device__ mat33(const float* _data)
		{
			/*_data����������9��floatԪ�أ����ﲻ��鷶Χ*/
			cols[0] = make_float3(_data[0], _data[1], _data[2]);
			cols[1] = make_float3(_data[3], _data[4], _data[5]);
			cols[2] = make_float3(_data[6], _data[7], _data[8]);
		}
		/**
		 * \brief ��Eigen::Matrix3fת��mat33����
		 * \param matrix3f ����Eigen::Matrix3f���;���
		 * \return
		 */
		__host__ mat33(const Eigen::Matrix3f matrix3f)
		{
			cols[0] = make_float3(matrix3f(0, 0), matrix3f(1, 0), matrix3f(2, 0));
			cols[1] = make_float3(matrix3f(0, 1), matrix3f(1, 1), matrix3f(2, 1));
			cols[2] = make_float3(matrix3f(0, 2), matrix3f(1, 2), matrix3f(2, 2));
		}
		/**
		 * \brief ���ء�=������Eigen::Matrix3f��ֵ����mat33����
		 * \param matrix3f ����Eigen::Matrix3f���͵ľ���
		 * \return ��ֵ��mat33
		 */
		__host__ mat33 operator=(const Eigen::Matrix3f matrix3f)
		{
			cols[0] = make_float3(matrix3f(0, 0), matrix3f(1, 0), matrix3f(2, 0));
			cols[1] = make_float3(matrix3f(0, 1), matrix3f(1, 1), matrix3f(2, 1));
			cols[2] = make_float3(matrix3f(0, 2), matrix3f(1, 2), matrix3f(2, 2));
			return *this;
		}
		/**
		 * \brief ���ؾ���Ԫ��(0, 0)
		 * \return Ԫ��(0, 0)��ֵ const
		 */
		__host__ __device__ const float& m00() const { return cols[0].x; }
		/**
		 * \brief ���ؾ���Ԫ��(1, 0)
		 * \return Ԫ��(1, 0)��ֵ const
		 */
		__host__ __device__ const float& m10() const { return cols[0].y; }
		/**
		 * \brief ���ؾ���Ԫ��(2, 0)
		 * \return Ԫ��(2, 0)��ֵ const
		 */
		__host__ __device__ const float& m20() const { return cols[0].z; }
		/**
		 * \brief ���ؾ���Ԫ��(0, 1)
		 * \return Ԫ��(0, 1)��ֵ const
		 */
		__host__ __device__ const float& m01() const { return cols[1].x; }
		/**
		 * \brief ���ؾ���Ԫ��(1, 1)
		 * \return Ԫ��(1, 1)��ֵ const
		 */
		__host__ __device__ const float& m11() const { return cols[1].y; }
		/**
		 * \brief ���ؾ���Ԫ��(2, 1)
		 * \return Ԫ��(2, 1)��ֵ const
		 */
		__host__ __device__ const float& m21() const { return cols[1].z; }
		/**
		 * \brief ���ؾ���Ԫ��(0, 2)
		 * \return Ԫ��(0, 2)��ֵ const
		 */
		__host__ __device__ const float& m02() const { return cols[2].x; }
		/**
		 * \brief ���ؾ���Ԫ��(1, 2)
		 * \return Ԫ��(1, 2)��ֵ const
		 */
		__host__ __device__ const float& m12() const { return cols[2].y; }
		/**
		 * \brief ���ؾ���Ԫ��(2, 2)
		 * \return Ԫ��(2, 2)��ֵ const
		 */
		__host__ __device__ const float& m22() const { return cols[2].z; }

		/**
		 * \brief ���ؾ���Ԫ��(0, 0)
		 * \return ����Ԫ��(0, 0)��ֵ
		 */
		__host__ __device__ float& m00() { return cols[0].x; }
		/**
		 * \brief ���ؾ���Ԫ��(1, 0)
		 * \return ����Ԫ��(1, 0)��ֵ
		 */
		__host__ __device__ float& m10() { return cols[0].y; }
		/**
		 * \brief ���ؾ���Ԫ��(2, 0)
		 * \return ����Ԫ��(2, 0)��ֵ
		 */
		__host__ __device__ float& m20() { return cols[0].z; }
		/**
		 * \brief ���ؾ���Ԫ��(0, 1)
		 * \return ����Ԫ��(0, 1)��ֵ
		 */
		__host__ __device__ float& m01() { return cols[1].x; }
		/**
		 * \brief ���ؾ���Ԫ��(1, 1)
		 * \return ����Ԫ��(1, 1)��ֵ
		 */
		__host__ __device__ float& m11() { return cols[1].y; }
		/**
		 * \brief ���ؾ���Ԫ��(2, 1)
		 * \return ����Ԫ��(2, 1)��ֵ
		 */
		__host__ __device__ float& m21() { return cols[1].z; }
		/**
		 * \brief ���ؾ���Ԫ��(0, 2)
		 * \return ����Ԫ��(0, 2)��ֵ
		 */
		__host__ __device__ float& m02() { return cols[2].x; }
		/**
		 * \brief ���ؾ���Ԫ��(2, 2)
		 * \return ����Ԫ��(2, 2)��ֵ
		 */
		__host__ __device__ float& m12() { return cols[2].y; }
		/**
		 * \brief ���ؾ���Ԫ��(1, 2)
		 * \return ����Ԫ��(1, 2)��ֵ
		 */
		__host__ __device__ float& m22() { return cols[2].z; }

		/**
		 * \brief ����ת��
		 * \return ����ת�õľ���
		 */
		__host__ __device__ mat33 transpose() const
		{
			float3 row0 = make_float3(cols[0].x, cols[1].x, cols[2].x);
			float3 row1 = make_float3(cols[0].y, cols[1].y, cols[2].y);
			float3 row2 = make_float3(cols[0].z, cols[1].z, cols[2].z);
			return mat33(row0, row1, row2);
		}

		/**
		 * \brief ���ء�*����3��3�������(���)
		 * \param _mat ��˵ľ���
		 * \return 3��3������˵Ľ��
		 */
		__host__ __device__ mat33 operator* (const mat33& _mat) const
		{
			mat33 mat;
			mat.m00() = m00() * _mat.m00() + m01() * _mat.m10() + m02() * _mat.m20();
			mat.m01() = m00() * _mat.m01() + m01() * _mat.m11() + m02() * _mat.m21();
			mat.m02() = m00() * _mat.m02() + m01() * _mat.m12() + m02() * _mat.m22();
			mat.m10() = m10() * _mat.m00() + m11() * _mat.m10() + m12() * _mat.m20();
			mat.m11() = m10() * _mat.m01() + m11() * _mat.m11() + m12() * _mat.m21();
			mat.m12() = m10() * _mat.m02() + m11() * _mat.m12() + m12() * _mat.m22();
			mat.m20() = m20() * _mat.m00() + m21() * _mat.m10() + m22() * _mat.m20();
			mat.m21() = m20() * _mat.m01() + m21() * _mat.m11() + m22() * _mat.m21();
			mat.m22() = m20() * _mat.m02() + m21() * _mat.m12() + m22() * _mat.m22();
			return mat;
		}

		/**
		 * \brief ���ء�+����3��3�������
		 * \param _mat ��ӵľ���
		 * \return 3��3������ӵĽ��
		 */
		__host__ __device__ mat33 operator+ (const mat33& _mat) const
		{
			mat33 mat_sum;
			mat_sum.m00() = m00() + _mat.m00();
			mat_sum.m01() = m01() + _mat.m01();
			mat_sum.m02() = m02() + _mat.m02();

			mat_sum.m10() = m10() + _mat.m10();
			mat_sum.m11() = m11() + _mat.m11();
			mat_sum.m12() = m12() + _mat.m12();

			mat_sum.m20() = m20() + _mat.m20();
			mat_sum.m21() = m21() + _mat.m21();
			mat_sum.m22() = m22() + _mat.m22();

			return mat_sum;
		}
		/**
		 * \brief ���ء�-����3��3�������
		 * \param _mat ����ľ���
		 * \return 3��3��������Ľ��
		 */
		__host__ __device__ mat33 operator- (const mat33& _mat) const
		{
			mat33 mat_diff;
			mat_diff.m00() = m00() - _mat.m00();
			mat_diff.m01() = m01() - _mat.m01();
			mat_diff.m02() = m02() - _mat.m02();

			mat_diff.m10() = m10() - _mat.m10();
			mat_diff.m11() = m11() - _mat.m11();
			mat_diff.m12() = m12() - _mat.m12();

			mat_diff.m20() = m20() - _mat.m20();
			mat_diff.m21() = m21() - _mat.m21();
			mat_diff.m22() = m22() - _mat.m22();

			return mat_diff;
		}
		/**
		 * \brief ����ȡ��
		 * \return ȡ�������ֵ
		 */
		__host__ __device__ mat33 operator-() const
		{
			mat33 mat_neg;
			mat_neg.m00() = -m00();
			mat_neg.m01() = -m01();
			mat_neg.m02() = -m02();

			mat_neg.m10() = -m10();
			mat_neg.m11() = -m11();
			mat_neg.m12() = -m12();

			mat_neg.m20() = -m20();
			mat_neg.m21() = -m21();
			mat_neg.m22() = -m22();

			return mat_neg;
		}

		/**
		 * \brief ����ÿ��Ԫ�س���һ������
		 * \return ���ŵľ���
		 */
		__host__ __device__ mat33 operator/(const float Factor) const
		{
			mat33 mat_neg;
			mat_neg.m00() = m00() / Factor;
			mat_neg.m01() = m01() / Factor;
			mat_neg.m02() = m02() / Factor;

			mat_neg.m10() = m10() / Factor;
			mat_neg.m11() = m11() / Factor;
			mat_neg.m12() = m12() / Factor;

			mat_neg.m20() = m20() / Factor;
			mat_neg.m21() = m21() / Factor;
			mat_neg.m22() = m22() / Factor;

			return mat_neg;
		}

		/**
		 * \brief ����ÿ��Ԫ�س���һ������
		 * \return ���ŵľ���
		 */
		__host__ __device__ mat33 operator*(const float Factor) const
		{
			mat33 mat_neg;
			mat_neg.m00() = m00() * Factor;
			mat_neg.m01() = m01() * Factor;
			mat_neg.m02() = m02() * Factor;
								  
			mat_neg.m10() = m10() * Factor;
			mat_neg.m11() = m11() * Factor;
			mat_neg.m12() = m12() * Factor;
								  
			mat_neg.m20() = m20() * Factor;
			mat_neg.m21() = m21() * Factor;
			mat_neg.m22() = m22() * Factor;

			return mat_neg;
		}

		/**
		 * \brief ���ء�*=���������������_mat���(���)
		 * \param _mat ��˵ľ���
		 * \return ���������_mat��˵Ľ��
		 */
		__host__ __device__ mat33& operator*= (const mat33& _mat)
		{
			*this = *this * _mat;
			return *this;
		}
		/**
		 * \brief ���ء�*������3��3������3��1�������(���)
		 * \param _vec 3��1�������
		 * \return 3��3������3��1������˵õ���3��1����
		 */
		__host__ __device__ float3 operator* (const float3& _vec) const
		{
			const float x = m00() * _vec.x + m01() * _vec.y + m02() * _vec.z;
			const float y = m10() * _vec.x + m11() * _vec.y + m12() * _vec.z;
			const float z = m20() * _vec.x + m21() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		//Just ignore the vec.w elements
		/**
		 * \brief ���ء�*������3��3������4��1�����е�x,y,z�������(����vec.w����)(���)
		 * \param _vec 4��1����
		 * \return 3��3������4��1�����е�x,y,z������˵õ���3��1����
		 */
		__host__ __device__ float3 operator* (const float4& _vec) const
		{
			const float x = m00() * _vec.x + m01() * _vec.y + m02() * _vec.z;
			const float y = m10() * _vec.x + m11() * _vec.y + m12() * _vec.z;
			const float z = m20() * _vec.x + m21() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		/**
		 * \brief �Ƚ�3��3������3��1�������(���)
		 * \param _vec 3��1�������
		 * \return 3��3������3��1������˵Ľ��
		 */
		__host__ __device__ float3 dot(const float3& _vec) const
		{
			const float x = m00() * _vec.x + m01() * _vec.y + m02() * _vec.z;
			const float y = m10() * _vec.x + m11() * _vec.y + m12() * _vec.z;
			const float z = m20() * _vec.x + m21() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		/**
		 * \brief �Ƚ�3��3����ת�ã�����3��1�������(���)
		 * \param _vec 3��1�������
		 * \return 3��3ת�þ�����3��1������˵Ľ��
		 */
		__host__ __device__ float3 transpose_dot(const float3& _vec) const
		{
			const float x = m00() * _vec.x + m10() * _vec.y + m20() * _vec.z;
			const float y = m01() * _vec.x + m11() * _vec.y + m21() * _vec.z;
			const float z = m02() * _vec.x + m12() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}
		/**
		 * \brief �Ƚ�3��3����ת�ã�����4��1������x,y,z�������(���)
		 * \param _vec 3��1�������
		 * \return 3��3ת�þ�����3��1������˵Ľ��
		 */
		__host__ __device__ float3 transpose_dot(const float4& _vec) const
		{
			const float x = m00() * _vec.x + m10() * _vec.y + m20() * _vec.z;
			const float y = m01() * _vec.x + m11() * _vec.y + m21() * _vec.z;
			const float z = m02() * _vec.x + m12() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		/**
		 * \brief ���������ó�Ϊ��λ����
		 * \return void
		 */
		__host__ __device__ void set_identity()
		{
			cols[0] = make_float3(1, 0, 0);
			cols[1] = make_float3(0, 1, 0);
			cols[2] = make_float3(0, 0, 1);
		}
		/**
		 * \brief ���һ����λ����
		 * \return ����һ����λ����
		 */
		__host__ __device__ static mat33 identity()
		{
			mat33 idmat;
			idmat.set_identity();
			return idmat;
		}
	};

	//3��4����λ�˾���SE(3)
	struct mat34 {
		mat33 rot;		//��ת����
		float3 trans;	//ƽ�ƾ���
		/**
		 * \brief �յĹ��캯��������mat34����
		 * \return 
		 */
		__host__ __device__ mat34() {}
		/**
		 * \brief ����mat34��������ת����_rot��ƽ�ƾ���_trans����mat34
		 * \param _rot ��ת����
		 * \param _trans ƽ�ƾ���
		 * \return 
		 */
		__host__ __device__ mat34(const mat33& _rot, const float3& _trans) : rot(_rot), trans(_trans) {}
		/**
		 * \brief ����x,y,z����ת�ĽǶȺ���x,y,z��ƽ�Ƶĳ���ת����λ�˾���mat34
		 * \param twist_rot ��x,y,z����ת�ĽǶ�
		 * \param twist_trans ����x,y,z��ƽ�Ƶĳ���
		 * \return ���캯���޷���ֵ������mat34λ�˾���
		 */
		__host__ __device__ mat34(const float3& twist_rot, const float3& twist_trans);

		/**
		 * \brief ��õ�λλ�˾���
		 * \return ���ص�λλ�˾��󣬼�û���κ���ת��ƽ��
		 */
		__host__ __device__ static mat34 identity()
		{
			return mat34(mat33::identity(), make_float3(0, 0, 0));
		}
		/**
		 * \brief ��Eigen::Isometry3f����ת����mat34����
		 * \param se3
		 * \return 
		 */
		__host__ mat34(const Eigen::Isometry3f& se3);
		/**
		 * \brief ��Eigen::Matrix4f
		 * \param matrix4f
		 * \return 
		 */
		__host__ mat34(const Eigen::Matrix4f& matrix4f);
		/**
		 * \brief ���ء�*����ͨ��*�ű�ʾ����λ�������
		 * \param _right_se3 ��Ҫ���е�λ�˱任
		 * \return �任���λ�˾���
		 */
		__host__ __device__ mat34 operator* (const mat34 _right_se3) const
		{
			mat34 se3;
			se3.rot = rot * _right_se3.rot;
			se3.trans = (rot * _right_se3.trans) + trans;
			return se3;
		}

		/**
		 * \brief ���Լ�����λ�˱任
		 * \param _right_se3 ��Ҫ���е�λ�˱任
		 * \return �任���λ�˾���
		 */
		__host__ __device__ mat34& operator*= (const mat34& _right_se3)
		{
			*this = *this * _right_se3;
			return *this;
		}

		/**
		 * \brief ͨ��������ķ����Լ�λ�ã��������������λ�˱任����.
		 *
		 * \param preVertex ��һ֡����λ��
		 * \param currVertex ��ǰ֡����λ��
		 * \param preNormal ��һ֡����
		 * \param currNormal ��ǰ֡����
		 * \return λ�˱任����
		 */
		__host__ __device__ __forceinline__ mat34 ComputeVertexSE3(const float4& preVertex, const float4& currVertex, const float4& preNormal, const float4& currNormal) {
			float dot = dotxyz(preNormal, currNormal);
			if (dot < NUMERIC_LIMITS_MIN_NORMAL_DOT) {		// ����
				rot = -mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else if (dot > NUMERIC_LIMITS_MAX_NORMAL_DOT) {	// ͬ��
				rot = mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else {
				float3 k = cross_xyz(preNormal, currNormal);	// ��ת��
				k = normalized(k);								// ��ת��������һ��
				float theta = acosf(dot);						// ��ת�Ƕ�
				// �޵������ת��ʽ
				// K����ת��k�ķ��Գƾ���
				mat33 K = mat33(make_float3(0.0f, k.z, -k.y), make_float3(-k.z, 0.0f, k.x), make_float3(k.y, -k.x, 0.0f));
				mat33 part_0 = mat33::identity();
				mat33 part_1 = K * sinf(theta);
				mat33 part_2 = K * K * (1.0f - dot);
				rot = part_0 + part_1 + part_2;
				trans = currVertex - rot * preVertex;
			}
		}

		/**
		 * \brief ͨ��������ķ����Լ�λ�ã��������������λ�˱任����.
		 *
		 * \param preVertex ��һ֡����λ��
		 * \param currVertex ��ǰ֡����λ��
		 * \param preNormal ��һ֡����
		 * \param currNormal ��ǰ֡����
		 * \return λ�˱任����
		 */
		__host__ __device__ __forceinline__ mat34 ComputeVertexSE3(const float3& preVertex, const float3& currVertex, const float3& preNormal, const float3& currNormal) {
			float dot = dotxyz(preNormal, currNormal);
			if (dot < NUMERIC_LIMITS_MIN_NORMAL_DOT) {		// ����
				rot = -mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else if (dot > NUMERIC_LIMITS_MAX_NORMAL_DOT) {	// ͬ��
				rot = mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else {
				float3 k = cross_xyz(preNormal, currNormal);	// ��ת��
				k = normalized(k);								// ��ת��������һ��
				float theta = acosf(dot);						// ��ת�Ƕ�
				// �޵������ת��ʽ
				// K����ת��k�ķ��Գƾ���
				mat33 K = mat33(make_float3(0.0f, k.z, -k.y), make_float3(-k.z, 0.0f, k.x), make_float3(k.y, -k.x, 0.0f));
				mat33 part_0 = mat33::identity();
				mat33 part_1 = K * sinf(theta);
				mat33 part_2 = K * K * (1.0f - dot);
				rot = part_0 + part_1 + part_2;
				trans = currVertex - rot * preVertex;
			}
		}

		/**
		 * \brief ͨ��������ķ����Լ�λ�ã��������������λ�˱任����.
		 *
		 * \param preVertex ��һ֡����λ��
		 * \param currVertex ��ǰ֡����λ��
		 * \param preNormal ��һ֡����
		 * \param currNormal ��ǰ֡����
		 * \return λ�˱任����
		 */
		__host__ __device__ __forceinline__ static mat34 ComputeSurfelsSE3(const float3& preVertex, const float3& currVertex, const float3& preNormal, const float3& currNormal) {
			mat34 se3;
			float dot = dotxyz(preNormal, currNormal);
			if (dot < NUMERIC_LIMITS_MIN_NORMAL_DOT) {		// ����
				se3.rot = -mat33::identity();
				se3.trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else if (dot > NUMERIC_LIMITS_MAX_NORMAL_DOT) {	// ͬ��
				se3.rot = mat33::identity();
				se3.trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else {
				float3 k = cross_xyz(preNormal, currNormal);	// ��ת��
				k = normalized(k);								// ��ת��������һ��
				float theta = acosf(dot);						// ��ת�Ƕ�
				// �޵������ת��ʽ
				// K����ת��k�ķ��Գƾ���
				mat33 K = mat33(make_float3(0.0f, k.z, -k.y), make_float3(-k.z, 0.0f, k.x), make_float3(k.y, -k.x, 0.0f));
				mat33 part_0 = mat33::identity();
				mat33 part_1 = K * sinf(theta);
				mat33 part_2 = K * K * (1.0f - dot);
				se3.rot = part_0 + part_1 + part_2;
				se3.trans = currVertex - se3.rot * preVertex;
			}
			return se3;
		}

		/**
		 * \brief ����ǰ�������õ�λ�任���󣬼��Ȳ�ƽ��Ҳ����ת�ľ���
		 * \return void
		 */
		__host__ __device__ __forceinline__ void set_identity() {
			rot.set_identity();
			trans.x = trans.y = trans.z = 0.0f;
		}

		/**
		 * \brief ��λ�˷��仯����se(3)�����ӵ�ǰλ�˱任����һλ�˵�λ�˾���
		 * \return ����λ�˾������(λ�˷��仯����)
		 */
		__host__ __device__ __forceinline__ mat34 inverse() const {
			mat34 inversed;
			inversed.rot = rot.transpose();
			inversed.trans = -(inversed.rot * trans);
			return inversed;
		}
		/**
		 * \brief ����ǰ�ĵ�����任����һ״̬�ĵ����괦��vec_last = R^-1 �� (vec_now - trans) ������ת����R��R^T = R^-1
		 * \param vec ���뵱ǰ������
		 * \return ������һ״̬�ĵ������float3
		 */
		__host__ __device__ __forceinline__ float3 apply_inversed_se3(const float3& vec) const {
			return rot.transpose_dot(vec - trans);
		}

		/**
		 * \brief ��4ά������x,y,z������Ϊ������꣬�任����һ״̬�ĵ����꣺vec_last = R^-1 �� (vec_now - trans) ������ת����R��R^T = R^-1
		 * \param vec ����4ά����
		 * \return ������һת״̬�ĵ������float3
		 */
		__host__ __device__ __forceinline__ float3 apply_inversed_se3(const float4& vec) const {
			return rot.transpose_dot(make_float3(vec.x - trans.x, vec.y - trans.y, vec.z - trans.z));
		}

		/**
		 * \brief ��mat34ת����ת�ǶȺ�ƽ�ƾ���.
		 * 
		 * \param rot ��ת�Ƕ�(��λ:����)
		 * \param trans ƽ�ƾ���(��λ:��)
		 */
		__host__ __device__ __forceinline__ void de_twist_se3(float3& rotAngle, float3& translation) {
			float trace = rot.m00() + rot.m11() + rot.m22();
			float angle = acosf((trace - 1.0f) / 2.0f);
			if (fabsf(angle) < 1e-4f) {
				rotAngle = make_float3(0.0f, 0.0f, 0.0f);
			}
			else {
				float sinAngle_2 = sinf(angle) * 2.0f;
				float3 axis = make_float3((rot.m21() - rot.m12()) / sinAngle_2, (rot.m02() - rot.m20()) / sinAngle_2, (rot.m10() - rot.m01()) / sinAngle_2);
				rotAngle = make_float3(axis.x * angle, axis.y * angle, axis.z * angle);
			}
			translation = trans;
		}
	};

	/**
	 * \brief ��mat33����ת����Eigen::Matrix3f����
	 * \param rhs ����mat33����
	 * \return ����Eigen::Matrix3f��������
	 */
	Eigen::Matrix3f toEigen(const mat33& rhs);

	/**
	 * \brief ��mat34����ת����Eigen::Matrix4f����
	 * \param rhs ����mat34����
	 * \return ����Eigen::Matrix4f��������
	 */
	Eigen::Matrix4f toEigen(const mat34& rhs);

	/**
	 * \brief ��CUDA����float3ת����Eigen::Vector3f����
	 * \param rhs ����float3����
	 * \return ����Eigen::Vector3f��������
	 */
	Eigen::Vector3f toEigen(const float3& rhs);

	/**
	 * \brief ��CUDA����float4ת����Eigen::Vector4f����
	 * \param rhs ����float4����
	 * \return ����Eigen::Vector4f��������
	 */
	Eigen::Vector4f toEigen(const float4& rhs);

	/**
	 * \brief ��Eigen::Vector3f����ת����CUDA����float3
	 * \param rhs ����Eigen::Vector3f��������
	 * \return ����float3����
	 */
	float3 fromEigen(const Vector3f& rhs);

	/**
	 * \brief ��Eigen::Vector4f����ת����CUDA����float4
	 * \param rhs ����Eigen::Vector4f��������
	 * \return ����float4����
	 */
	float4 fromEigen(const Vector4f& rhs);

	struct Quaternion; //��ǰ������cpp��Ҫ��Quaternion�ṹ���������ʵ��

	/**
	 * \brief ��Isometry3f���͵�λ�˱任����se3��ת����Quaternion���͵���ת��Ԫ��rotation �� float3���͵�ƽ������translation.
	 * 
	 * \param se3 λ�˱任����(Isometry3f����)
	 * \param rotation ��ʾ��ת����Ԫ��(Quaternion����)
	 * \param translation ��ʾƽ�Ƶ�����(float3����)
	 */
	void fromEigen(const Isometry3f& se3, Quaternion& rotation, float3& translation);


/******************************************** ����PCL���������㶥�㷨�ߵ����� ********************************************/
	struct eigen33 {
	private:
		template<int Rows>
		struct MiniMat
		{
			float3 data[Rows];
			__host__ __device__ __forceinline__ float3& operator[](int i) { return data[i]; }
			__host__ __device__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
		};
		typedef MiniMat<3> Mat33;

	public:
		// ���㷽�̣�x^2 - b x + c = 0�ĸ�, ����ʵ��
		__host__ __device__ static void compute_root2(const float b, const float c, float3& roots);

		//���㷽�̣�x^3 - c2*x^2 + c1*x - c0 = 0�ĸ�
		__host__ __device__ static void compute_root3(const float c0, const float c1, const float c2, float3& roots);

		// ���캯��
		__host__ __device__ eigen33(float* psd33) : psd_matrix33(psd33) {}

		// ������������
		__host__ __device__ static float3 unit_orthogonal(const float3& src);
		// ������������
		__host__ __device__ __forceinline__ void compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals);
		// ������������
		__host__ __device__ __forceinline__ void compute(float3& eigen_vec);
		// ������������
		__host__ __device__ __forceinline__ void compute(float3& eigen_vec, float& eigen_value);

	private:
		// ��psd����(����������)����������������СΪ6
		float* psd_matrix33;

		// ����Ԫ��
		__host__ __device__  __forceinline__ float m00() const { return psd_matrix33[0]; }
		__host__ __device__  __forceinline__ float m01() const { return psd_matrix33[1]; }
		__host__ __device__  __forceinline__ float m02() const { return psd_matrix33[2]; }
		__host__ __device__  __forceinline__ float m10() const { return psd_matrix33[1]; }
		__host__ __device__  __forceinline__ float m11() const { return psd_matrix33[3]; }
		__host__ __device__  __forceinline__ float m12() const { return psd_matrix33[4]; }
		__host__ __device__  __forceinline__ float m20() const { return psd_matrix33[2]; }
		__host__ __device__  __forceinline__ float m21() const { return psd_matrix33[4]; }
		__host__ __device__  __forceinline__ float m22() const { return psd_matrix33[5]; }

		__host__ __device__  __forceinline__ float3 row0() const { return make_float3(m00(), m01(), m02()); }
		__host__ __device__  __forceinline__ float3 row1() const { return make_float3(m10(), m11(), m12()); }
		__host__ __device__  __forceinline__ float3 row2() const { return make_float3(m20(), m21(), m22()); }

		__host__ __device__ __forceinline__ static bool isMuchSmallerThan(float x, float y) {
			const float prec_sqr = numeric_limits<float>::epsilon() * numeric_limits<float>::epsilon();
			return x * x <= prec_sqr * y * y;
		}

		//The inverse sqrt function
#if defined (__CUDACC__)
		__host__ __device__ __forceinline__ static float inv_sqrt(float x) {
			return rsqrtf(x);
		}
#else
		__host__ __device__ __forceinline__ static float inv_sqrt(float x) {
			return 1.0f / sqrtf(x);
		}
#endif
	};
}

/******************************************** ����PCL��ʵ�ּ��㶥�㷨�ߵ����� ********************************************/

__host__  __device__ __forceinline__ void SparseSurfelFusion::eigen33::compute_root2(const float b, const float c, float3& roots)
{
	roots.x = 0.0f; // ���� compute_root3
	float d = b * b - 4.f * c;
	if (d < 0.f)	// û�������ĸ�!!!!�ⲻӦ�÷���!
		d = 0.f;

	float sd = sqrtf(d);

	roots.z = 0.5f * (b + sd);
	roots.y = 0.5f * (b - sd);
}

__host__  __device__ __forceinline__ void SparseSurfelFusion::eigen33::compute_root3(const float c0, const float c1, const float c2, float3& roots)
{
	if (fabsf(c0) < numeric_limits<float>::epsilon()) {
		compute_root2(c2, c1, roots);
	}
	else {
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		// �������ڶԷ��̵ĸ����з�����Աո���ʽ��ⷽ�̵Ĳ�����
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;
		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));
		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		// ͨ��������ʽ�ĸ�����������ֵ
		float rho = sqrtf(-a_over_3);
		float theta = atan2f(sqrtf(-q), half_b) * s_inv3;

		// Using intrinsic here
		float cos_theta, sin_theta;
		cos_theta = cosf(theta);
		sin_theta = sinf(theta);
		// �����
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		// �������ǵ�ֵ�Ը���������
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z) {
			swap(roots.y, roots.z);
			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}

		// �Գ����붨���������ֵ����Ϊ��!��Ϊ0
		if (roots.x <= 0.0f)
			compute_root2(c2, c1, roots);
	}
}

__host__  __device__ __forceinline__ float3 SparseSurfelFusion::eigen33::unit_orthogonal(const float3& src)
{
	float3 perp;
	// ����*this��һ����̫�ӽ���*this���ߵ������Ĳ�ˡ�


	// ����x��y���ӽ���0�����ǿ��Լ򵥵�ȡ(-y, x, 0)�������׼����

	if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
	{
		float invnm = inv_sqrt(src.x * src.x + src.y * src.y);
		perp.x = -src.y * invnm;
		perp.y = src.x * invnm;
		perp.z = 0.0f;
	}

	// ���x��y���ӽ���0����ô��������ͽӽ���z�ᣬ��������x�᲻���ߣ�����ȡ����(1,0,0)�Ĳ�˲���׼��.
	else
	{
		float invnm = inv_sqrt(src.z * src.z + src.y * src.y);
		perp.x = 0.0f;
		perp.y = -src.z * invnm;
		perp.z = src.y * invnm;
	}

	return perp;
}

__host__  __device__ __forceinline__ void SparseSurfelFusion::eigen33::compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
{
	// ���ž���ʹ����Ԫ����[-1,1]�С�ֻ�е�����һ��������Ŀ�Ĵ�С����1ʱ����Ӧ������.
	float max01 = fmaxf(fabsf(psd_matrix33[0]), fabsf(psd_matrix33[1]));
	float max23 = fmaxf(fabsf(psd_matrix33[2]), fabsf(psd_matrix33[3]));
	float max45 = fmaxf(fabsf(psd_matrix33[4]), fabsf(psd_matrix33[5]));
	float m0123 = fmaxf(max01, max23);
	float scale = fmaxf(max45, m0123);

	if (scale <= numeric_limits<float>::min_positive())
		scale = 1.f;

	psd_matrix33[0] /= scale;
	psd_matrix33[1] /= scale;
	psd_matrix33[2] /= scale;
	psd_matrix33[3] /= scale;
	psd_matrix33[4] /= scale;
	psd_matrix33[5] /= scale;

	// ����������x^3 - c2*x^2 + c1*x - c0 = 0.  
	// ����ֵ��������̵ĸ�������֤��ʵֵ����Ϊ�����ǶԳƵ�.
	float c0 = m00() * m11() * m22()
		+ 2.f * m01() * m02() * m12()
		- m00() * m12() * m12()
		- m11() * m02() * m02()
		- m22() * m01() * m01();
	float c1 = m00() * m11() -
		m01() * m01() +
		m00() * m22() -
		m02() * m02() +
		m11() * m22() -
		m12() * m12();
	float c2 = m00() + m11() + m22();

	compute_root3(c0, c1, c2, evals);

	if (evals.z - evals.x <= numeric_limits<float>::epsilon())
	{
		evecs[0] = make_float3(1.f, 0.f, 0.f);
		evecs[1] = make_float3(0.f, 1.f, 0.f);
		evecs[2] = make_float3(0.f, 0.f, 1.f);
	}
	else if (evals.y - evals.x <= numeric_limits<float>::epsilon())
	{
		// ��һ�͵ڶ����       
		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		float len1 = dot(vec_tmp[0], vec_tmp[0]);
		float len2 = dot(vec_tmp[1], vec_tmp[1]);
		float len3 = dot(vec_tmp[2], vec_tmp[2]);

		if (len1 >= len2 && len1 >= len3)
		{
			evecs[2] = vec_tmp[0] * inv_sqrt(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			evecs[2] = vec_tmp[1] * inv_sqrt(len2);
		}
		else
		{
			evecs[2] = vec_tmp[2] * inv_sqrt(len3);
		}

		evecs[1] = unit_orthogonal(evecs[2]);
		evecs[0] = cross(evecs[1], evecs[2]);
	}
	else if (evals.z - evals.y <= numeric_limits<float>::epsilon())
	{
		// �ڶ��͵������                              
		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		float len1 = dot(vec_tmp[0], vec_tmp[0]);
		float len2 = dot(vec_tmp[1], vec_tmp[1]);
		float len3 = dot(vec_tmp[2], vec_tmp[2]);

		if (len1 >= len2 && len1 >= len3)
		{
			evecs[0] = vec_tmp[0] * inv_sqrt(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			evecs[0] = vec_tmp[1] * inv_sqrt(len2);
		}
		else
		{
			evecs[0] = vec_tmp[2] * inv_sqrt(len3);
		}

		evecs[1] = unit_orthogonal(evecs[0]);
		evecs[2] = cross(evecs[0], evecs[1]);
	}
	else
	{

		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		float len1 = dot(vec_tmp[0], vec_tmp[0]);
		float len2 = dot(vec_tmp[1], vec_tmp[1]);
		float len3 = dot(vec_tmp[2], vec_tmp[2]);

		float mmax[3];

		unsigned int min_el = 2;
		unsigned int max_el = 2;
		if (len1 >= len2 && len1 >= len3)
		{
			mmax[2] = len1;
			evecs[2] = vec_tmp[0] * inv_sqrt(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[2] = len2;
			evecs[2] = vec_tmp[1] * inv_sqrt(len2);
		}
		else
		{
			mmax[2] = len3;
			evecs[2] = vec_tmp[2] * inv_sqrt(len3);
		}

		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		len1 = dot(vec_tmp[0], vec_tmp[0]);
		len2 = dot(vec_tmp[1], vec_tmp[1]);
		len3 = dot(vec_tmp[2], vec_tmp[2]);

		if (len1 >= len2 && len1 >= len3)
		{
			mmax[1] = len1;
			evecs[1] = vec_tmp[0] * inv_sqrt(len1);
			min_el = len1 <= mmax[min_el] ? 1 : min_el;
			max_el = len1 > mmax[max_el] ? 1 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[1] = len2;
			evecs[1] = vec_tmp[1] * inv_sqrt(len2);
			min_el = len2 <= mmax[min_el] ? 1 : min_el;
			max_el = len2 > mmax[max_el] ? 1 : max_el;
		}
		else
		{
			mmax[1] = len3;
			evecs[1] = vec_tmp[2] * inv_sqrt(len3);
			min_el = len3 <= mmax[min_el] ? 1 : min_el;
			max_el = len3 > mmax[max_el] ? 1 : max_el;
		}

		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		len1 = dot(vec_tmp[0], vec_tmp[0]);
		len2 = dot(vec_tmp[1], vec_tmp[1]);
		len3 = dot(vec_tmp[2], vec_tmp[2]);


		if (len1 >= len2 && len1 >= len3)
		{
			mmax[0] = len1;
			evecs[0] = vec_tmp[0] * inv_sqrt(len1);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3 > mmax[max_el] ? 0 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[0] = len2;
			evecs[0] = vec_tmp[1] * inv_sqrt(len2);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3 > mmax[max_el] ? 0 : max_el;
		}
		else
		{
			mmax[0] = len3;
			evecs[0] = vec_tmp[2] * inv_sqrt(len3);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3 > mmax[max_el] ? 0 : max_el;
		}

		unsigned mid_el = 3 - min_el - max_el;
		evecs[min_el] = normalized(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
		evecs[mid_el] = normalized(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));
	}
	// �������Ż�ԭʼ��С.
	evals = evals * scale;
}

__host__  __device__ __forceinline__ void SparseSurfelFusion::eigen33::compute(float3& eigen_vec)
{
	Mat33 tmp, vec_tmp, evecs;
	float3 evals;
	compute(tmp, vec_tmp, evecs, evals);
	eigen_vec = evecs[0];
}

__host__  __device__ __forceinline__ void SparseSurfelFusion::eigen33::compute(float3& eigen_vec, float& eigen_value)
{
	Mat33 tmp, vec_tmp, evecs;
	float3 evals;
	compute(tmp, vec_tmp, evecs, evals);
	eigen_vec = evecs[0];
	eigen_value = evals.x;
}
