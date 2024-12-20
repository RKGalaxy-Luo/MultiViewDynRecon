//
// Created by wei on 4/21/18.
//

#pragma once
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include "ApplySpMVBase.h"
#include <base/CommonUtils.h>

namespace SparseSurfelFusion {

	template<int BlockDim>
	class ApplySpMVBinBlockCSR : public ApplySpMVBase<BlockDim> {
	public:
		using Ptr = std::shared_ptr<ApplySpMVBinBlockCSR>;
		DEFAULT_CONSTRUCT_DESTRUCT(ApplySpMVBinBlockCSR);

		//The interface for matrix size
		size_t MatrixSize() const override { return matrix_size_; }

		//The interface for spmv
		void ApplySpMV(DeviceArrayView<float> x, DeviceArrayHandle<float> spmv_x, cudaStream_t stream = 0) override;
		void ApplySpMVTextured(cudaTextureObject_t x, DeviceArrayHandle<float> spmv_x, cudaStream_t stream = 0) override;

		//The interface for init residual
		void InitResidual(
			DeviceArrayView<float> x_init,
			DeviceArrayView<float> b,
			DeviceArrayHandle<float> residual,
			cudaStream_t stream = 0
		) override;
		void InitResidualTextured(
			cudaTextureObject_t x_init,
			DeviceArrayView<float> b,
			DeviceArrayHandle<float> residual,
			cudaStream_t stream = 0
		) override;

		//Set the input
		void SetInputs(const float* A_data, const int* A_rowptr, const int* A_colptr, size_t mat_size) {
			this->A_data_ = A_data;
			this->A_rowptr_ = A_rowptr;
			this->A_colptr_ = A_colptr;
			this->matrix_size_ = mat_size;
		}

	private:
		//The matrix elements and size
		const float* A_data_;
		const int* A_rowptr_;
		const int* A_colptr_;
		size_t matrix_size_;
	};

}


#if defined(__CUDACC__)
#include "ApplySpMVBinBlockCSR.cuh"
#endif