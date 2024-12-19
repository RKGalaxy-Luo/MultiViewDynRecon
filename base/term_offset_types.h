#pragma once

#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <vector_types.h>

#define ENERGY_TERM_NUM 5		// 项的数量

namespace SparseSurfelFusion {


	enum class TermType {
		DenseImage = 0,		// 稠密点
		Smooth = 1,			// 平滑项
		Foreground = 2,		// 前景项
		Feature = 3,		// 特征项
		CrossViewCorr = 4,	// 跨镜项
		Invalid = 5			// 无效值
	};

	struct TermTypeOffset {
		unsigned offset_value[ENERGY_TERM_NUM];

		//The accessed interface
		__host__ __device__ __forceinline__ const unsigned& operator[](const int idx) const {
			return offset_value[idx];
		}

		//The size of terms
		__host__ __device__ __forceinline__ unsigned TermSize() const { return offset_value[ENERGY_TERM_NUM - 1]; }
		__host__ __forceinline__ unsigned ScalarTermSize() const {
			return DenseImageTermSize() + ForegroundTermSize() + 3 * (SmoothTermSize() + FeatureTermSize() + CrossViewTermSize());
		}

		//The different type of terms
		__host__ __device__ __forceinline__ unsigned DenseImageTermSize() const { return offset_value[0]; }
		__host__ __device__ __forceinline__ unsigned SmoothTermSize() const { return offset_value[1] - offset_value[0]; }
		__host__ __device__ __forceinline__ unsigned ForegroundTermSize() const { return offset_value[2] - offset_value[1]; }
		__host__ __device__ __forceinline__ unsigned FeatureTermSize() const { return offset_value[3] - offset_value[2]; }
		__host__ __device__ __forceinline__ unsigned CrossViewTermSize() const { return offset_value[4] - offset_value[3]; }

	};


	inline void size2offset(
		TermTypeOffset& offset,
		DeviceArrayView<ushort4> dense_depth_knn,
		DeviceArrayView<ushort2> node_graph,
		//These costs might be empty
		DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
		DeviceArrayView<ushort4> sparse_feature_knn = DeviceArrayView<ushort4>(), 
		DeviceArrayView<ushort4> cross_corr_knn = DeviceArrayView<ushort4>()
	) {
		unsigned prefix_sum = 0;
		prefix_sum += dense_depth_knn.Size();
		offset.offset_value[0] = prefix_sum;
		prefix_sum += node_graph.Size();
		offset.offset_value[1] = prefix_sum;
		prefix_sum += foreground_mask_knn.Size();
		offset.offset_value[2] = prefix_sum;
		prefix_sum += sparse_feature_knn.Size();
		offset.offset_value[3] = prefix_sum;
		prefix_sum += cross_corr_knn.Size();
		offset.offset_value[4] = prefix_sum;
	}

	__host__ __device__ __forceinline__
		void query_typed_index(unsigned term_idx, const TermTypeOffset& offset, TermType& type, unsigned& typed_idx)
	{
		if (term_idx < offset[0]) {
			type = TermType::DenseImage;
			typed_idx = term_idx - 0;
			return;
		}
		if (term_idx >= offset[0] && term_idx < offset[1]) {
			type = TermType::Smooth;
			typed_idx = term_idx - offset[0];
			return;
		}
		if (term_idx >= offset[1] && term_idx < offset[2]) {
			type = TermType::Foreground;
			typed_idx = term_idx - offset[1];
			return;
		}
		if (term_idx >= offset[2] && term_idx < offset[3]) {
			type = TermType::Feature;
			typed_idx = term_idx - offset[2];
			return;
		}

		if (term_idx >= offset[3] && term_idx < offset[4]) {
			type = TermType::CrossViewCorr;
			typed_idx = term_idx - offset[3];
			return;
		}

		//Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
	}

	__host__ __device__ __forceinline__
		void query_typed_index(unsigned term_idx, const TermTypeOffset& offset, TermType& type, unsigned& typed_idx, unsigned& scalar_term_idx)
	{
		unsigned scalar_offset = 0;
		if (term_idx < offset[0]) {
			type = TermType::DenseImage;
			typed_idx = term_idx - 0;
			scalar_term_idx = scalar_offset + 1 * term_idx;
			return;
		}

		scalar_offset += 1 * offset.DenseImageTermSize();
		if (term_idx >= offset[0] && term_idx < offset[1]) {
			type = TermType::Smooth;
			typed_idx = term_idx - offset[0];
			scalar_term_idx = scalar_offset + 3 * typed_idx;
			return;
		}

		scalar_offset += 3 * offset.SmoothTermSize();
		if (term_idx >= offset[1] && term_idx < offset[2]) {
			type = TermType::Foreground;
			typed_idx = term_idx - offset[1];
			scalar_term_idx = scalar_offset + 1 * typed_idx;
			return;
		}

		scalar_offset += 1 * offset.ForegroundTermSize();
		if (term_idx >= offset[2] && term_idx < offset[3]) {
			type = TermType::Feature;
			typed_idx = term_idx - offset[2];
			scalar_term_idx = scalar_offset + 3 * typed_idx;
			return;
		}

		scalar_offset += 3 * offset.FeatureTermSize();
		if (term_idx >= offset[3] && term_idx < offset[4]) {
			type = TermType::CrossViewCorr;
			typed_idx = term_idx - offset[3];
			scalar_term_idx = scalar_offset + 3 * typed_idx;
			return;
		}
		//Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
		scalar_term_idx = 0xFFFFFFFF;
	}


	__host__ __device__ __forceinline__
		void query_nodepair_index(unsigned term_idx, const TermTypeOffset& offset, TermType& type, unsigned& typed_idx, unsigned& nodepair_idx)
	{
		unsigned pair_offset = 0;
		//如果属于dense_image_knn
		if (term_idx < offset[0]) {
			type = TermType::DenseImage;
			//这个typed_idx是在自己原来项里的偏移
			typed_idx = term_idx - 0;
			nodepair_idx = pair_offset + 6 * term_idx;
			return;
		}

		pair_offset += 6 * offset.DenseImageTermSize();
		if (term_idx >= offset[0] && term_idx < offset[1]) {
			type = TermType::Smooth;
			typed_idx = term_idx - offset[0];
			nodepair_idx = pair_offset + 1 * typed_idx;
			return;
		}

		pair_offset += 1 * offset.SmoothTermSize();
		if (term_idx >= offset[1] && term_idx < offset[2]) {
			type = TermType::Foreground;
			typed_idx = term_idx - offset[1];
			nodepair_idx = pair_offset + 6 * typed_idx;
			return;
		}

		pair_offset += 6 * offset.ForegroundTermSize();
		if (term_idx >= offset[2] && term_idx < offset[3]) {
			type = TermType::Feature;
			typed_idx = term_idx - offset[2];
			nodepair_idx = pair_offset + 6 * typed_idx;
			return;
		}

		pair_offset += 6 * offset.FeatureTermSize();
		if (term_idx >= offset[3] && term_idx < offset[4]) {
			type = TermType::CrossViewCorr;
			typed_idx = term_idx - offset[3];
			nodepair_idx = pair_offset + 6 * typed_idx;
			return;
		}

		//Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
		nodepair_idx = 0xFFFFFFFF;
	}
}
