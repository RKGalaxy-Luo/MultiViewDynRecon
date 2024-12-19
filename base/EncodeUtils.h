/*****************************************************************//**
 * \file   EncodeUtils.h
 * \brief  编码的方式，包括RGBA颜色编码，体素Voxel位置编码
 * 
 * \author LUO
 * \date   January 31st 2024
 *********************************************************************/
#pragma once

#include <base/CommonTypes.h>
#include <cuda_texture_types.h>
#include <exception>
#include <cuda.h>

namespace SparseSurfelFusion
{

	//The transfer between uint and float		unsigned int和float之间的转换
#if defined(__CUDA_ARCH__) //GPU处理
	__device__ __forceinline__ float uint_as_float(unsigned i_value) {
		return __uint_as_float(i_value);
	}

	__device__ __forceinline__ unsigned float_as_uint(float f_value) {
		return __float_as_uint(f_value);
	}
#else	//CPU处理
	//The transfer from into to float			int转float
	union uint2float_union {
		unsigned i_value;
		float f_value;
	};

	//方法
	__host__ __forceinline__ float uint_as_float(unsigned i_value) {
		uint2float_union u;
		u.i_value = i_value;
		return u.f_value;
	}
	__host__ __forceinline__ unsigned float_as_uint(float f_value) {
		uint2float_union u;
		u.f_value = f_value;
		return u.i_value;
	}
#endif
	/*	函数使用三个无符号字符（8 位每个）表示 RGB 颜色，将它们合并成一个 24 或者 32 位的无符号整数。
		每8位表示一个颜色通道
		通过这种方式，将四个颜色通道合并到一个整数中，形成了一个 32 位的 RGBA 编码。
		这种编码方式通常用于在计算机图形学、图像处理和 GPU 编程等领域中表示颜色信息。
		编码成一个整数可以在传递颜色信息时减少参数传递的开销，有时可以提高程序的性能。
		同样需要注意，这种编码方式也有其限制，无法表示超过 8 位的颜色精度
	*/


	/**
	* \brief Encode uint8_t as float32_t or uint32_t		将uint8_t编码为float32_t或uint32_t
	*/
	__host__ __device__ __forceinline__ unsigned uint_encode_rgba(
		const unsigned char r,
		const unsigned char g,
		const unsigned char b,
		const unsigned char a
	) {
		const unsigned encoded = ((a << 24) + (r << 16) + (g << 8) + b);
		return encoded;
	}

	__host__ __device__ __forceinline__ unsigned uint_encode_rgb(
		const unsigned char r,
		const unsigned char g,
		const unsigned char b
	) {
		const unsigned encoded = ((r << 16) + (g << 8) + b);
		return encoded;
	}

	__host__ __device__ __forceinline__
		unsigned uint_encode_rgb(const uchar3 rgb) {
		const unsigned encoded = ((rgb.x << 16) + (rgb.y << 8) + rgb.z);
		return encoded;
	}

	__host__ __device__ __forceinline__ float float_encode_rgba(
		const unsigned char r,
		const unsigned char g,
		const unsigned char b,
		const unsigned char a
	) {
		return uint_as_float(uint_encode_rgba(r, g, b, a));
	}

	__host__ __device__ __forceinline__
		float float_encode_rgb(const uchar3 rgb) {

		return uint_as_float(uint_encode_rgb(rgb));
		//return (float)uint_encode_rgb(rgb);
	}

	/**
	* \brief Dncode uint8_t as float32_t or uint32_t	将uint8_t编码为float32_t或uint32_t
	*/
	__host__ __device__ __forceinline__ void uint_decode_rgba(
		const unsigned encoded,
		unsigned char& r,
		unsigned char& g,
		unsigned char& b,
		unsigned char& a
	) {
		a = ((encoded & 0xff000000) >> 24);
		r = ((encoded & 0x00ff0000) >> 16);
		g = ((encoded & 0x0000ff00) >> 8);
		b = ((encoded & 0x000000ff) /*0*/);
	}

	__host__ __device__ __forceinline__ void uint_decode_rgb(
		const unsigned encoded,
		unsigned char& r,
		unsigned char& g,
		unsigned char& b
	) {
		r = ((encoded & 0x00ff0000) >> 16);
		g = ((encoded & 0x0000ff00) >> 8);
		b = ((encoded & 0x000000ff) /*0*/);
	}

	__host__ __device__ __forceinline__ void uint_decode_rgb(
		const unsigned encoded,
		uchar3& rgb
	) {
		uint_decode_rgb(encoded, rgb.x, rgb.y, rgb.z);
	}

	__host__ __device__ __forceinline__ void float_decode_rgba(
		const float encoded,
		unsigned char& r,
		unsigned char& g,
		unsigned char& b,
		unsigned char& a
	) {
		const unsigned int unsigned_encoded = float_as_uint(encoded);
		uint_decode_rgba(unsigned_encoded, r, g, b, a);
	}

	__host__ __device__ __forceinline__
		void float_decode_rgb(const float encoded, uchar3& rgb)
	{
		const unsigned int unsigned_encoded = float_as_uint(encoded);
		uint_decode_rgb(unsigned_encoded, rgb);
	}



	//Assume x, y, z are in (-512, 512)		
	/**
	 * \brief 根据三维坐标编码体素，编码体素：假设x, y, z在(-512,512)中，1000个体素单位，最大可以展示NODE_RADIUS * 1000 m的范围.
	 * 
	 * \param x x轴坐标
	 * \param y y轴坐标
	 * \param z z轴坐标
	 * \return 编码后的体素
	 */
	__host__ __device__ __forceinline__ int encodeVoxel(const int x, const int y, const int z) {
		//int 类型 只能存x, y, z在(-512,512)的体素 int是(-2^31,2^31)
		return (x + 512) + (y + 512) * 1024 + (z + 512) * 1024 * 1024;
	}

	/**
	 * \brief 解码体素.
	 * 
	 * \param encoded 传入key值
	 * \param x x轴坐标
	 * \param y	y轴坐标
	 * \param z	z轴坐标
	 * \return 
	 */
	__host__ __device__ __forceinline__	 void decodeVoxel(const int encoded, int& x, int& y, int& z) {
		z = encoded / (1024 * 1024);
		x = encoded % 1024;
		y = (encoded - z * 1024 * 1024) / 1024;
		x -= 512;
		y -= 512;
		z -= 512;
	}
	//hsg 非刚性SE3求解时需要的节点对编码
	//To encode and decode the pair into int, this number shall
	//be larger than any of the y (which is the node index)
	const int large_number = MAX_NODE_COUNT;

	//A row major encoding of the (row, col) pair
	__host__ __device__ __forceinline__ unsigned encode_nodepair(unsigned short x, unsigned short y) {
		return x * large_number + y;
	}
	__host__ __device__ __forceinline__ void decode_nodepair(const unsigned encoded, unsigned& x, unsigned& y) {
		x = encoded / large_number;
		y = encoded % large_number;
	}

	__host__ __device__ __forceinline__ unsigned short encoded_row(const unsigned encoded) {
		return encoded / large_number;
	}

	__host__ __device__ __forceinline__ unsigned short encoded_col(const unsigned encoded) {
		return encoded % large_number;
	}
}