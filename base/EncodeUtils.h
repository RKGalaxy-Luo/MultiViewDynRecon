/*****************************************************************//**
 * \file   EncodeUtils.h
 * \brief  ����ķ�ʽ������RGBA��ɫ���룬����Voxelλ�ñ���
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

	//The transfer between uint and float		unsigned int��float֮���ת��
#if defined(__CUDA_ARCH__) //GPU����
	__device__ __forceinline__ float uint_as_float(unsigned i_value) {
		return __uint_as_float(i_value);
	}

	__device__ __forceinline__ unsigned float_as_uint(float f_value) {
		return __float_as_uint(f_value);
	}
#else	//CPU����
	//The transfer from into to float			intתfloat
	union uint2float_union {
		unsigned i_value;
		float f_value;
	};

	//����
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
	/*	����ʹ�������޷����ַ���8 λÿ������ʾ RGB ��ɫ�������Ǻϲ���һ�� 24 ���� 32 λ���޷���������
		ÿ8λ��ʾһ����ɫͨ��
		ͨ�����ַ�ʽ�����ĸ���ɫͨ���ϲ���һ�������У��γ���һ�� 32 λ�� RGBA ���롣
		���ֱ��뷽ʽͨ�������ڼ����ͼ��ѧ��ͼ����� GPU ��̵������б�ʾ��ɫ��Ϣ��
		�����һ�����������ڴ�����ɫ��Ϣʱ���ٲ������ݵĿ�������ʱ������߳�������ܡ�
		ͬ����Ҫע�⣬���ֱ��뷽ʽҲ�������ƣ��޷���ʾ���� 8 λ����ɫ����
	*/


	/**
	* \brief Encode uint8_t as float32_t or uint32_t		��uint8_t����Ϊfloat32_t��uint32_t
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
	* \brief Dncode uint8_t as float32_t or uint32_t	��uint8_t����Ϊfloat32_t��uint32_t
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
	 * \brief ������ά����������أ��������أ�����x, y, z��(-512,512)�У�1000�����ص�λ��������չʾNODE_RADIUS * 1000 m�ķ�Χ.
	 * 
	 * \param x x������
	 * \param y y������
	 * \param z z������
	 * \return ����������
	 */
	__host__ __device__ __forceinline__ int encodeVoxel(const int x, const int y, const int z) {
		//int ���� ֻ�ܴ�x, y, z��(-512,512)������ int��(-2^31,2^31)
		return (x + 512) + (y + 512) * 1024 + (z + 512) * 1024 * 1024;
	}

	/**
	 * \brief ��������.
	 * 
	 * \param encoded ����keyֵ
	 * \param x x������
	 * \param y	y������
	 * \param z	z������
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
	//hsg �Ǹ���SE3���ʱ��Ҫ�Ľڵ�Ա���
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