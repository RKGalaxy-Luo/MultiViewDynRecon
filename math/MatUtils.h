/*****************************************************************//**
 * \file   MatUtils.h
 * \brief  矩阵运算工具包：主要解决一些矩阵运算问题，以及3×4的位姿变换矩阵【SE(3)】的计算问题
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
	 * 矩阵是由3个列向量组成的3×3的矩阵.
	 */
	struct mat33 {
		float3 cols[3]; //这个矩阵是由3个列向量组成的
		/**
		 * \brief 空构造函数，声明一个空的3×3矩阵
		 * \return
		 */
		__host__ __device__ mat33() {}
		/**
		 * \brief mat33构造函数：传入3个列向量组成3×3的矩阵
		 * \param _a0 输入列向量_a0
		 * \param _a1 输入列向量_a1
		 * \param _a2 输入列向量_a2
		 * \return 构造函数，无返回值
		 */
		__host__ __device__ mat33(const float3 _a0, const float3 _a1, const float3 _a2) { cols[0] = _a0; cols[1] = _a1; cols[2] = _a2; }
		/**
		 * \brief 根据数组构造3×3矩阵
		 * \param _data 传入数组：index = 0 - 2(列向量1)；index = 3 - 5(列向量2)；index = 6 - 8(列向量3)
		 * \return 构造函数，无返回值
		 */
		__host__ __device__ mat33(const float* _data)
		{
			/*_data必须至少有9个float元素，这里不检查范围*/
			cols[0] = make_float3(_data[0], _data[1], _data[2]);
			cols[1] = make_float3(_data[3], _data[4], _data[5]);
			cols[2] = make_float3(_data[6], _data[7], _data[8]);
		}
		/**
		 * \brief 将Eigen::Matrix3f转成mat33类型
		 * \param matrix3f 输入Eigen::Matrix3f类型矩阵
		 * \return
		 */
		__host__ mat33(const Eigen::Matrix3f matrix3f)
		{
			cols[0] = make_float3(matrix3f(0, 0), matrix3f(1, 0), matrix3f(2, 0));
			cols[1] = make_float3(matrix3f(0, 1), matrix3f(1, 1), matrix3f(2, 1));
			cols[2] = make_float3(matrix3f(0, 2), matrix3f(1, 2), matrix3f(2, 2));
		}
		/**
		 * \brief 重载“=”：将Eigen::Matrix3f的值赋给mat33类型
		 * \param matrix3f 输入Eigen::Matrix3f类型的矩阵
		 * \return 赋值的mat33
		 */
		__host__ mat33 operator=(const Eigen::Matrix3f matrix3f)
		{
			cols[0] = make_float3(matrix3f(0, 0), matrix3f(1, 0), matrix3f(2, 0));
			cols[1] = make_float3(matrix3f(0, 1), matrix3f(1, 1), matrix3f(2, 1));
			cols[2] = make_float3(matrix3f(0, 2), matrix3f(1, 2), matrix3f(2, 2));
			return *this;
		}
		/**
		 * \brief 返回矩阵元素(0, 0)
		 * \return 元素(0, 0)的值 const
		 */
		__host__ __device__ const float& m00() const { return cols[0].x; }
		/**
		 * \brief 返回矩阵元素(1, 0)
		 * \return 元素(1, 0)的值 const
		 */
		__host__ __device__ const float& m10() const { return cols[0].y; }
		/**
		 * \brief 返回矩阵元素(2, 0)
		 * \return 元素(2, 0)的值 const
		 */
		__host__ __device__ const float& m20() const { return cols[0].z; }
		/**
		 * \brief 返回矩阵元素(0, 1)
		 * \return 元素(0, 1)的值 const
		 */
		__host__ __device__ const float& m01() const { return cols[1].x; }
		/**
		 * \brief 返回矩阵元素(1, 1)
		 * \return 元素(1, 1)的值 const
		 */
		__host__ __device__ const float& m11() const { return cols[1].y; }
		/**
		 * \brief 返回矩阵元素(2, 1)
		 * \return 元素(2, 1)的值 const
		 */
		__host__ __device__ const float& m21() const { return cols[1].z; }
		/**
		 * \brief 返回矩阵元素(0, 2)
		 * \return 元素(0, 2)的值 const
		 */
		__host__ __device__ const float& m02() const { return cols[2].x; }
		/**
		 * \brief 返回矩阵元素(1, 2)
		 * \return 元素(1, 2)的值 const
		 */
		__host__ __device__ const float& m12() const { return cols[2].y; }
		/**
		 * \brief 返回矩阵元素(2, 2)
		 * \return 元素(2, 2)的值 const
		 */
		__host__ __device__ const float& m22() const { return cols[2].z; }

		/**
		 * \brief 返回矩阵元素(0, 0)
		 * \return 矩阵元素(0, 0)的值
		 */
		__host__ __device__ float& m00() { return cols[0].x; }
		/**
		 * \brief 返回矩阵元素(1, 0)
		 * \return 矩阵元素(1, 0)的值
		 */
		__host__ __device__ float& m10() { return cols[0].y; }
		/**
		 * \brief 返回矩阵元素(2, 0)
		 * \return 矩阵元素(2, 0)的值
		 */
		__host__ __device__ float& m20() { return cols[0].z; }
		/**
		 * \brief 返回矩阵元素(0, 1)
		 * \return 矩阵元素(0, 1)的值
		 */
		__host__ __device__ float& m01() { return cols[1].x; }
		/**
		 * \brief 返回矩阵元素(1, 1)
		 * \return 矩阵元素(1, 1)的值
		 */
		__host__ __device__ float& m11() { return cols[1].y; }
		/**
		 * \brief 返回矩阵元素(2, 1)
		 * \return 矩阵元素(2, 1)的值
		 */
		__host__ __device__ float& m21() { return cols[1].z; }
		/**
		 * \brief 返回矩阵元素(0, 2)
		 * \return 矩阵元素(0, 2)的值
		 */
		__host__ __device__ float& m02() { return cols[2].x; }
		/**
		 * \brief 返回矩阵元素(2, 2)
		 * \return 矩阵元素(2, 2)的值
		 */
		__host__ __device__ float& m12() { return cols[2].y; }
		/**
		 * \brief 返回矩阵元素(1, 2)
		 * \return 矩阵元素(1, 2)的值
		 */
		__host__ __device__ float& m22() { return cols[2].z; }

		/**
		 * \brief 矩阵转置
		 * \return 返回转置的矩阵
		 */
		__host__ __device__ mat33 transpose() const
		{
			float3 row0 = make_float3(cols[0].x, cols[1].x, cols[2].x);
			float3 row1 = make_float3(cols[0].y, cols[1].y, cols[2].y);
			float3 row2 = make_float3(cols[0].z, cols[1].z, cols[2].z);
			return mat33(row0, row1, row2);
		}

		/**
		 * \brief 重载“*”：3×3矩阵相乘(点乘)
		 * \param _mat 相乘的矩阵
		 * \return 3×3矩阵相乘的结果
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
		 * \brief 重载“+”：3×3矩阵相加
		 * \param _mat 相加的矩阵
		 * \return 3×3矩阵相加的结果
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
		 * \brief 重载“-”：3×3矩阵相减
		 * \param _mat 相减的矩阵
		 * \return 3×3矩阵相减的结果
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
		 * \brief 矩阵取反
		 * \return 取反矩阵的值
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
		 * \brief 矩阵每个元素除以一个常数
		 * \return 缩放的矩阵
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
		 * \brief 矩阵每个元素除以一个常数
		 * \return 缩放的矩阵
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
		 * \brief 重载“*=”：将自身矩阵与_mat相乘(点乘)
		 * \param _mat 相乘的矩阵
		 * \return 自身矩阵与_mat相乘的结果
		 */
		__host__ __device__ mat33& operator*= (const mat33& _mat)
		{
			*this = *this * _mat;
			return *this;
		}
		/**
		 * \brief 重载“*”：将3×3矩阵与3×1向量相乘(点乘)
		 * \param _vec 3×1向量相乘
		 * \return 3×3矩阵与3×1向量相乘得到的3×1向量
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
		 * \brief 重载“*”：将3×3矩阵与4×1向量中的x,y,z分量相乘(忽略vec.w分量)(点乘)
		 * \param _vec 4×1向量
		 * \return 3×3矩阵与4×1向量中的x,y,z分量相乘得到的3×1向量
		 */
		__host__ __device__ float3 operator* (const float4& _vec) const
		{
			const float x = m00() * _vec.x + m01() * _vec.y + m02() * _vec.z;
			const float y = m10() * _vec.x + m11() * _vec.y + m12() * _vec.z;
			const float z = m20() * _vec.x + m21() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		/**
		 * \brief 先将3×3矩阵与3×1向量相乘(点乘)
		 * \param _vec 3×1向量相乘
		 * \return 3×3矩阵与3×1向量相乘的结果
		 */
		__host__ __device__ float3 dot(const float3& _vec) const
		{
			const float x = m00() * _vec.x + m01() * _vec.y + m02() * _vec.z;
			const float y = m10() * _vec.x + m11() * _vec.y + m12() * _vec.z;
			const float z = m20() * _vec.x + m21() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		/**
		 * \brief 先将3×3矩阵转置，在与3×1向量相乘(点乘)
		 * \param _vec 3×1向量相乘
		 * \return 3×3转置矩阵与3×1向量相乘的结果
		 */
		__host__ __device__ float3 transpose_dot(const float3& _vec) const
		{
			const float x = m00() * _vec.x + m10() * _vec.y + m20() * _vec.z;
			const float y = m01() * _vec.x + m11() * _vec.y + m21() * _vec.z;
			const float z = m02() * _vec.x + m12() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}
		/**
		 * \brief 先将3×3矩阵转置，在与4×1向量的x,y,z分量相乘(点乘)
		 * \param _vec 3×1向量相乘
		 * \return 3×3转置矩阵与3×1向量相乘的结果
		 */
		__host__ __device__ float3 transpose_dot(const float4& _vec) const
		{
			const float x = m00() * _vec.x + m10() * _vec.y + m20() * _vec.z;
			const float y = m01() * _vec.x + m11() * _vec.y + m21() * _vec.z;
			const float z = m02() * _vec.x + m12() * _vec.y + m22() * _vec.z;
			return make_float3(x, y, z);
		}

		/**
		 * \brief 将矩阵设置成为单位矩阵
		 * \return void
		 */
		__host__ __device__ void set_identity()
		{
			cols[0] = make_float3(1, 0, 0);
			cols[1] = make_float3(0, 1, 0);
			cols[2] = make_float3(0, 0, 1);
		}
		/**
		 * \brief 获得一个单位矩阵
		 * \return 返回一个单位矩阵
		 */
		__host__ __device__ static mat33 identity()
		{
			mat33 idmat;
			idmat.set_identity();
			return idmat;
		}
	};

	//3×4矩阵，位姿矩阵SE(3)
	struct mat34 {
		mat33 rot;		//旋转矩阵
		float3 trans;	//平移矩阵
		/**
		 * \brief 空的构造函数，声明mat34类型
		 * \return 
		 */
		__host__ __device__ mat34() {}
		/**
		 * \brief 构造mat34矩阵，用旋转矩阵_rot和平移矩阵_trans构造mat34
		 * \param _rot 旋转矩阵
		 * \param _trans 平移矩阵
		 * \return 
		 */
		__host__ __device__ mat34(const mat33& _rot, const float3& _trans) : rot(_rot), trans(_trans) {}
		/**
		 * \brief 将绕x,y,z轴旋转的角度和沿x,y,z轴平移的长度转化成位姿矩阵mat34
		 * \param twist_rot 绕x,y,z轴旋转的角度
		 * \param twist_trans 沿着x,y,z轴平移的长度
		 * \return 构造函数无返回值，构造mat34位姿矩阵
		 */
		__host__ __device__ mat34(const float3& twist_rot, const float3& twist_trans);

		/**
		 * \brief 获得单位位姿矩阵
		 * \return 返回单位位姿矩阵，即没有任何旋转和平移
		 */
		__host__ __device__ static mat34 identity()
		{
			return mat34(mat33::identity(), make_float3(0, 0, 0));
		}
		/**
		 * \brief 将Eigen::Isometry3f对象转化成mat34类型
		 * \param se3
		 * \return 
		 */
		__host__ mat34(const Eigen::Isometry3f& se3);
		/**
		 * \brief 将Eigen::Matrix4f
		 * \param matrix4f
		 * \return 
		 */
		__host__ mat34(const Eigen::Matrix4f& matrix4f);
		/**
		 * \brief 重载“*”：通过*号表示两个位姿相叠加
		 * \param _right_se3 需要进行的位姿变换
		 * \return 变换后的位姿矩阵
		 */
		__host__ __device__ mat34 operator* (const mat34 _right_se3) const
		{
			mat34 se3;
			se3.rot = rot * _right_se3.rot;
			se3.trans = (rot * _right_se3.trans) + trans;
			return se3;
		}

		/**
		 * \brief 对自己进行位姿变换
		 * \param _right_se3 需要进行的位姿变换
		 * \return 变换后的位姿矩阵
		 */
		__host__ __device__ mat34& operator*= (const mat34& _right_se3)
		{
			*this = *this * _right_se3;
			return *this;
		}

		/**
		 * \brief 通过两个点的法线以及位置，计算这两个点的位姿变换矩阵.
		 *
		 * \param preVertex 上一帧顶点位置
		 * \param currVertex 当前帧顶点位置
		 * \param preNormal 上一帧法线
		 * \param currNormal 当前帧法线
		 * \return 位姿变换矩阵
		 */
		__host__ __device__ __forceinline__ mat34 ComputeVertexSE3(const float4& preVertex, const float4& currVertex, const float4& preNormal, const float4& currNormal) {
			float dot = dotxyz(preNormal, currNormal);
			if (dot < NUMERIC_LIMITS_MIN_NORMAL_DOT) {		// 反向
				rot = -mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else if (dot > NUMERIC_LIMITS_MAX_NORMAL_DOT) {	// 同向
				rot = mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else {
				float3 k = cross_xyz(preNormal, currNormal);	// 旋转轴
				k = normalized(k);								// 旋转轴向量归一化
				float theta = acosf(dot);						// 旋转角度
				// 罗德里格旋转公式
				// K是旋转轴k的反对称矩阵
				mat33 K = mat33(make_float3(0.0f, k.z, -k.y), make_float3(-k.z, 0.0f, k.x), make_float3(k.y, -k.x, 0.0f));
				mat33 part_0 = mat33::identity();
				mat33 part_1 = K * sinf(theta);
				mat33 part_2 = K * K * (1.0f - dot);
				rot = part_0 + part_1 + part_2;
				trans = currVertex - rot * preVertex;
			}
		}

		/**
		 * \brief 通过两个点的法线以及位置，计算这两个点的位姿变换矩阵.
		 *
		 * \param preVertex 上一帧顶点位置
		 * \param currVertex 当前帧顶点位置
		 * \param preNormal 上一帧法线
		 * \param currNormal 当前帧法线
		 * \return 位姿变换矩阵
		 */
		__host__ __device__ __forceinline__ mat34 ComputeVertexSE3(const float3& preVertex, const float3& currVertex, const float3& preNormal, const float3& currNormal) {
			float dot = dotxyz(preNormal, currNormal);
			if (dot < NUMERIC_LIMITS_MIN_NORMAL_DOT) {		// 反向
				rot = -mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else if (dot > NUMERIC_LIMITS_MAX_NORMAL_DOT) {	// 同向
				rot = mat33::identity();
				trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else {
				float3 k = cross_xyz(preNormal, currNormal);	// 旋转轴
				k = normalized(k);								// 旋转轴向量归一化
				float theta = acosf(dot);						// 旋转角度
				// 罗德里格旋转公式
				// K是旋转轴k的反对称矩阵
				mat33 K = mat33(make_float3(0.0f, k.z, -k.y), make_float3(-k.z, 0.0f, k.x), make_float3(k.y, -k.x, 0.0f));
				mat33 part_0 = mat33::identity();
				mat33 part_1 = K * sinf(theta);
				mat33 part_2 = K * K * (1.0f - dot);
				rot = part_0 + part_1 + part_2;
				trans = currVertex - rot * preVertex;
			}
		}

		/**
		 * \brief 通过两个点的法线以及位置，计算这两个点的位姿变换矩阵.
		 *
		 * \param preVertex 上一帧顶点位置
		 * \param currVertex 当前帧顶点位置
		 * \param preNormal 上一帧法线
		 * \param currNormal 当前帧法线
		 * \return 位姿变换矩阵
		 */
		__host__ __device__ __forceinline__ static mat34 ComputeSurfelsSE3(const float3& preVertex, const float3& currVertex, const float3& preNormal, const float3& currNormal) {
			mat34 se3;
			float dot = dotxyz(preNormal, currNormal);
			if (dot < NUMERIC_LIMITS_MIN_NORMAL_DOT) {		// 反向
				se3.rot = -mat33::identity();
				se3.trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else if (dot > NUMERIC_LIMITS_MAX_NORMAL_DOT) {	// 同向
				se3.rot = mat33::identity();
				se3.trans = make_float3(currVertex.x - preVertex.x, currVertex.y - preVertex.y, currVertex.z - preVertex.z);
			}
			else {
				float3 k = cross_xyz(preNormal, currNormal);	// 旋转轴
				k = normalized(k);								// 旋转轴向量归一化
				float theta = acosf(dot);						// 旋转角度
				// 罗德里格旋转公式
				// K是旋转轴k的反对称矩阵
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
		 * \brief 将当前矩阵设置单位变换矩阵，即既不平移也不旋转的矩阵
		 * \return void
		 */
		__host__ __device__ __forceinline__ void set_identity() {
			rot.set_identity();
			trans.x = trans.y = trans.z = 0.0f;
		}

		/**
		 * \brief 求位姿反变化矩阵se(3)，即从当前位姿变换到上一位姿的位姿矩阵
		 * \return 返回位姿矩阵的逆(位姿反变化矩阵)
		 */
		__host__ __device__ __forceinline__ mat34 inverse() const {
			mat34 inversed;
			inversed.rot = rot.transpose();
			inversed.trans = -(inversed.rot * trans);
			return inversed;
		}
		/**
		 * \brief 将当前的点坐标变换到上一状态的点坐标处：vec_last = R^-1 · (vec_now - trans) 其中旋转矩阵R有R^T = R^-1
		 * \param vec 输入当前点坐标
		 * \return 返回上一状态的点的坐标float3
		 */
		__host__ __device__ __forceinline__ float3 apply_inversed_se3(const float3& vec) const {
			return rot.transpose_dot(vec - trans);
		}

		/**
		 * \brief 将4维向量的x,y,z分量作为点的坐标，变换到上一状态的点坐标：vec_last = R^-1 · (vec_now - trans) 其中旋转矩阵R有R^T = R^-1
		 * \param vec 输入4维向量
		 * \return 返回上一转状态的点的坐标float3
		 */
		__host__ __device__ __forceinline__ float3 apply_inversed_se3(const float4& vec) const {
			return rot.transpose_dot(make_float3(vec.x - trans.x, vec.y - trans.y, vec.z - trans.z));
		}

		/**
		 * \brief 将mat34转换旋转角度和平移距离.
		 * 
		 * \param rot 旋转角度(单位:弧度)
		 * \param trans 平移距离(单位:米)
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
	 * \brief 将mat33类型转化成Eigen::Matrix3f类型
	 * \param rhs 输入mat33类型
	 * \return 返回Eigen::Matrix3f类型数据
	 */
	Eigen::Matrix3f toEigen(const mat33& rhs);

	/**
	 * \brief 将mat34类型转化成Eigen::Matrix4f类型
	 * \param rhs 输入mat34类型
	 * \return 返回Eigen::Matrix4f类型数据
	 */
	Eigen::Matrix4f toEigen(const mat34& rhs);

	/**
	 * \brief 将CUDA类型float3转化成Eigen::Vector3f类型
	 * \param rhs 输入float3类型
	 * \return 返回Eigen::Vector3f类型数据
	 */
	Eigen::Vector3f toEigen(const float3& rhs);

	/**
	 * \brief 将CUDA类型float4转化成Eigen::Vector4f类型
	 * \param rhs 输入float4类型
	 * \return 返回Eigen::Vector4f类型数据
	 */
	Eigen::Vector4f toEigen(const float4& rhs);

	/**
	 * \brief 将Eigen::Vector3f类型转化成CUDA类型float3
	 * \param rhs 输入Eigen::Vector3f类型数据
	 * \return 返回float3类型
	 */
	float3 fromEigen(const Vector3f& rhs);

	/**
	 * \brief 将Eigen::Vector4f类型转化成CUDA类型float4
	 * \param rhs 输入Eigen::Vector4f类型数据
	 * \return 返回float4类型
	 */
	float4 fromEigen(const Vector4f& rhs);

	struct Quaternion; //向前声明，cpp中要有Quaternion结构体的声明和实现

	/**
	 * \brief 将Isometry3f类型的位姿变换矩阵se3，转化成Quaternion类型的旋转四元数rotation 和 float3类型的平移向量translation.
	 * 
	 * \param se3 位姿变换矩阵(Isometry3f类型)
	 * \param rotation 表示旋转的四元数(Quaternion类型)
	 * \param translation 表示平移的向量(float3类型)
	 */
	void fromEigen(const Isometry3f& se3, Quaternion& rotation, float3& translation);


/******************************************** 根据PCL库声明计算顶点法线的内容 ********************************************/
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
		// 计算方程：x^2 - b x + c = 0的根, 假设实根
		__host__ __device__ static void compute_root2(const float b, const float c, float3& roots);

		//计算方程：x^3 - c2*x^2 + c1*x - c0 = 0的根
		__host__ __device__ static void compute_root3(const float c0, const float c1, const float c2, float3& roots);

		// 构造函数
		__host__ __device__ eigen33(float* psd33) : psd_matrix33(psd33) {}

		// 计算特征向量
		__host__ __device__ static float3 unit_orthogonal(const float3& src);
		// 计算特征向量
		__host__ __device__ __forceinline__ void compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals);
		// 计算特征向量
		__host__ __device__ __forceinline__ void compute(float3& eigen_vec);
		// 计算特征向量
		__host__ __device__ __forceinline__ void compute(float3& eigen_vec, float& eigen_value);

	private:
		// 用psd矩阵(半正定矩阵)计算特征向量，大小为6
		float* psd_matrix33;

		// 访问元素
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

/******************************************** 根据PCL库实现计算顶点法线的内容 ********************************************/

__host__  __device__ __forceinline__ void SparseSurfelFusion::eigen33::compute_root2(const float b, const float c, float3& roots)
{
	roots.x = 0.0f; // 用于 compute_root3
	float d = b * b - 4.f * c;
	if (d < 0.f)	// 没有真正的根!!!!这不应该发生!
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
		// 构造用于对方程的根进行分类和以闭根形式求解方程的参数。
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;
		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));
		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		// 通过求解多项式的根来计算特征值
		float rho = sqrtf(-a_over_3);
		float theta = atan2f(sqrtf(-q), half_b) * s_inv3;

		// Using intrinsic here
		float cos_theta, sin_theta;
		cos_theta = cosf(theta);
		sin_theta = sinf(theta);
		// 计算根
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		// 根据它们的值对根进行排序
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z) {
			swap(roots.y, roots.z);
			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}

		// 对称正半定矩阵的特征值不能为负!设为0
		if (roots.x <= 0.0f)
			compute_root2(c2, c1, roots);
	}
}

__host__  __device__ __forceinline__ float3 SparseSurfelFusion::eigen33::unit_orthogonal(const float3& src)
{
	float3 perp;
	// 计算*this与一个不太接近于*this共线的向量的叉乘。


	// 除非x和y都接近于0，我们可以简单地取(-y, x, 0)并将其标准化。

	if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
	{
		float invnm = inv_sqrt(src.x * src.x + src.y * src.y);
		perp.x = -src.y * invnm;
		perp.y = src.x * invnm;
		perp.z = 0.0f;
	}

	// 如果x和y都接近于0，那么这个向量就接近于z轴，所以它与x轴不共线，我们取它与(1,0,0)的叉乘并标准化.
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
	// 缩放矩阵，使它的元素在[-1,1]中。只有当至少一个矩阵条目的大小大于1时，才应用缩放.
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

	// 特征方程是x^3 - c2*x^2 + c1*x - c0 = 0.  
	// 特征值是这个方程的根，都保证是实值，因为矩阵是对称的.
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
		// 第一和第二相等       
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
		// 第二和第三相等                              
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
	// 重新缩放回原始大小.
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
