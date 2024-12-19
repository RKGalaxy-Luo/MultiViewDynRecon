/*****************************************************************//**
 * \file   NumericLimits.h
 * \brief  主要记录数值极限
 * 
 * \author LUO
 * \date   January 30th 2024
 *********************************************************************/
#pragma once
#include "VectorUtils.h"

#define NUMERIC_LIMITS_MIN_NORMAL_DOT -1.0f + 1e-10f		// 小于这个值默认为法线反向
#define NUMERIC_LIMITS_MAX_NORMAL_DOT 1.0f - 1e-10f			// 大于这个值默认为法线同相


namespace SparseSurfelFusion {

	/* 用于数值范围的结构体
	*/
	template <typename T> struct numeric_limits;


	/* 对于float类型，主机CPU和设备代码GPU可能不同
	*/
	template <> struct numeric_limits<float> {
#if defined (__CUDACC__)
		__device__ __forceinline__ static float quiet_nan() {
			return __int_as_float(0x7fffffff);
		}
#else
		__host__ __device__ __forceinline__ static float quiet_nan() {
			int value = 0x7fffffff;
			return (*((float*)&(value)));
		}
#endif

		__host__ __device__ __forceinline__ static float epsilon() {
			return 1.192092896e-07f;
		}

		__host__ __device__ __forceinline__ static float min_positive() {
			return 1.175494351e-38f;
		}

		__host__ __device__ __forceinline__ static float max() {
			return 3.402823466e+38f;
		}
	};

	/* 对于16位无符号短类型
	*/
	template <> struct numeric_limits<signed short> {
		__host__ __device__ __forceinline__ static signed short min() {
			return -32768;
		}

		__host__ __device__ __forceinline__ static signed short max() {
			return 32767;
		}
	};

	/* 对于16位无符号短类型
	*/
	template <> struct numeric_limits<unsigned short> {
		__host__ __device__ __forceinline__ static unsigned short min() {
			return 0;
		}

		__host__ __device__ __forceinline__ static unsigned short max() {
			return 0xffff;
		}
	};
}
