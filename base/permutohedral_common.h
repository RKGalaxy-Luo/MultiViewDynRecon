#pragma once
//
// Created by wei on 2/25/18.
//

//hsg

#pragma once

#include "CommonTypes.h"


namespace SparseSurfelFusion {

	template<int FeatureDim = 5>
	__host__ __device__ __forceinline__
		float permutohedral_scale_factor(const int index);

	//һ��С�ṹ�壬�����������������Ϊ��
	template<int FeatureDim = 5>
	struct LatticeCoordKey {
		//ֻά����һ��FeatureDimԪ��.
		//�ܺ͵���0.
		short key[FeatureDim];

		//�������ɢ��
		__host__ __device__ __forceinline__ unsigned hash() const;


		/**
		 * \brief ���ıȽ���
		 * \param rhs 
		 * \return С�ڷ���1, ���ڷ���-1, ��ȷ���0
		 */
		__host__ __device__ __forceinline__ char less_than(const LatticeCoordKey<FeatureDim>& rhs) const;

		//������Ϊ��ptr
		__host__ __device__ __forceinline__ void set_null();
		__host__ __device__ __forceinline__ bool is_null() const;

		//Operator ����
		__host__ __device__ __forceinline__ bool operator==(const LatticeCoordKey<FeatureDim>& rhs) const
		{
			bool equal = true;
			for (auto i = 0; i < FeatureDim; i++) {
				if (key[i] != rhs.key[i]) equal = false;
			}
			return equal;
		}
	};



	/**
	 * \brief ���㾧�����Χ�Ƹ������ľ�����Ȩ��.
	 * \param FeatureDim����ά��
	 * \param feature������������FeatureDim�Ĵ�С��ʾ
	 * \param lattice_coord_keys�����Ը����ĸ������������Ĵ�СΪFeatureDim + 1.
	 * \param barycentricȨ��ֵ����FeatureDim + 2�Ĵ�С�У�����һ��FeatureDim + 1Ԫ��ƥ��lattice_coord_keys��Ȩ��.
	 */
	template<int FeatureDim = 5>
	__host__ __device__ __forceinline__
		void permutohedral_lattice(
			const float* feature,
			LatticeCoordKey<FeatureDim>* lattice_coord_keys,
			float* barycentric
		);
}


//ʵ���ļ�
template<int FeatureDim>
float SparseSurfelFusion::permutohedral_scale_factor(const int index) {
	return (FeatureDim + 1) * sqrtf((1.0f / 6.0f) / ((index + 1) * (index + 2)));
}

template<int FeatureDim>
unsigned SparseSurfelFusion::LatticeCoordKey<FeatureDim>::hash() const
{
	unsigned hash_value = 0;
	for (auto i = 0; i < FeatureDim; i++) {
		hash_value += key[i];
		//hash_value *= 1664525;
		hash_value *= 1500007;
	}
	return hash_value;
}

template<int FeatureDim>
char SparseSurfelFusion::LatticeCoordKey<FeatureDim>::less_than(const LatticeCoordKey<FeatureDim>& rhs) const
{
	char is_less_than = 0;
	for (auto i = 0; i < FeatureDim; i++) {
		if (key[i] < rhs.key[i]) {
			is_less_than = 1;
			break;
		}
		else if (key[i] > rhs.key[i]) {
			is_less_than = -1;
			break;
		}
		//Else, continue
	}
	return is_less_than;
}


template<int FeatureDim>
void SparseSurfelFusion::LatticeCoordKey<FeatureDim>::set_null()
{
	for (auto i = 0; i < FeatureDim; i++) {
		key[i] = 1 << 14;
	}
}

template<int FeatureDim>
bool SparseSurfelFusion::LatticeCoordKey<FeatureDim>::is_null() const {
	bool null_key = true;
	for (auto i = 0; i < FeatureDim; i++) {
		if (key[i] != (1 << 14)) null_key = false;
	}
	return null_key;
}


template<int FeatureDim>
void SparseSurfelFusion::permutohedral_lattice(
	const float* feature,
	LatticeCoordKey<FeatureDim>* lattice_coord_keys,
	float* barycentric
) {
	float elevated[FeatureDim + 1];
	elevated[FeatureDim] = -FeatureDim * (feature[FeatureDim - 1]) * permutohedral_scale_factor<FeatureDim>(FeatureDim - 1);
	for (int i = FeatureDim - 1; i > 0; i--) {
		elevated[i] = (elevated[i + 1] -
			i * (feature[i - 1]) * permutohedral_scale_factor<FeatureDim>(i - 1) +
			(i + 2) * (feature[i]) * permutohedral_scale_factor<FeatureDim>(i));
	}
	elevated[0] = elevated[1] + 2 * (feature[0]) * permutohedral_scale_factor<FeatureDim>(0);

	short greedy[FeatureDim + 1];
	signed short sum = 0;
	for (int i = 0; i <= FeatureDim; i++) {
		float v = elevated[i] * (1.0f / (FeatureDim + 1));
		float up = ceilf(v) * (FeatureDim + 1);
		float down = floorf(v) * (FeatureDim + 1);
		if (up - elevated[i] < elevated[i] - down) {
			greedy[i] = (signed short)up;
		}
		else {
			greedy[i] = (signed short)down;
		}
		sum += greedy[i];
	}
	sum /= FeatureDim + 1;

	//��΢�ֽ��������ҳ���������κ�������֮�������
	short rank[FeatureDim + 1];
	for (int i = 0; i <= FeatureDim; i++) {
		rank[i] = 0;
		for (int j = 0; j <= FeatureDim; j++) {
			if (elevated[i] - greedy[i] < elevated[j] - greedy[j] ||
				(elevated[i] - greedy[i] == elevated[j] - greedy[j]
					&& i > j)) {
				rank[i]++;
			}
		}
	}

	//��̫����Ҫ��΢����С��ȡ����
	if (sum > 0) {
		for (int i = 0; i <= FeatureDim; i++) {
			if (rank[i] >= FeatureDim + 1 - sum) {
				greedy[i] -= FeatureDim + 1;
				rank[i] += sum - (FeatureDim + 1);
			}
			else {
				rank[i] += sum;
			}
		}
	}
	else if (sum < 0) { //��̫С����Ҫ�Ѳ�ֵ���������
		for (int i = 0; i <= FeatureDim; i++) {
			if (rank[i] < -sum) {
				greedy[i] += FeatureDim + 1;
				rank[i] += (FeatureDim + 1) + sum;
			}
			else {
				rank[i] += sum;
			}
		}
	}


	//�ѱ����������
	for (int i = 0; i <= FeatureDim + 1; i++) {
		barycentric[i] = 0;
	}

	for (int i = 0; i <= FeatureDim; i++) {
		float delta = (elevated[i] - greedy[i]) * (1.0f / (FeatureDim + 1));
		barycentric[FeatureDim - rank[i]] += delta;
		barycentric[FeatureDim + 1 - rank[i]] -= delta;
	}
	barycentric[0] += 1.0f + barycentric[FeatureDim + 1];

	//����Կ�׺����ǵ�����
	for (auto color = 0; color <= FeatureDim; color++) {
		//��ʽ�ؼ��㾧����λ��(�������һ������֮����������ꡪ�����Ƕ���ģ���Ϊ���Ǻ�Ϊ��)
		short* key = lattice_coord_keys[color].key;
		for (int i = 0; i < FeatureDim; i++) {
			key[i] = greedy[i] + color;
			if (rank[i] > FeatureDim - color) key[i] -= (FeatureDim + 1);
		}
	}
}