//
// Created by wei on 4/21/18.
//

#pragma once
#include <base/DeviceReadWrite/DeviceArrayHandle.h>
#include <base/CommonUtils.h>
#include "sanity_check.h"
#include <memory>

namespace SparseSurfelFusion {

	template<int BlockDim>
	class ApplySpMVBase {
	public:
		using Ptr = std::shared_ptr<ApplySpMVBase>;
		ApplySpMVBase() = default;
		virtual ~ApplySpMVBase() = default;
		NO_COPY_ASSIGN(ApplySpMVBase);
		DEFAULT_MOVE(ApplySpMVBase);

		//The matrix size for this apply spmv
		virtual size_t MatrixSize() const = 0;

		//The application interface
		virtual void ApplySpMV(DeviceArrayView<float> x, DeviceArrayHandle<float> spmv_x, cudaStream_t stream = 0) {
			LOGGING(FATAL) << "The sparse matrix vector produce is not implemented";
		}
		virtual void ApplySpMVTextured(
			cudaTextureObject_t x,
			DeviceArrayHandle<float> spmv_x,
			cudaStream_t stream = 0
		) {
			LOGGING(FATAL) << "The textured sparse matrix-vector product is not implemented";
		}

		//residual <- b - Ax
		virtual void InitResidual(
			DeviceArrayView<float> x_init,
			DeviceArrayView<float> b,
			DeviceArrayHandle<float> residual,
			cudaStream_t stream = 0
		) {
			LOGGING(FATAL) << "The init resiudal computation is not implemented";
		}
		virtual void InitResidualTextured(
			cudaTextureObject_t x_init,
			DeviceArrayView<float> b,
			DeviceArrayHandle<float> residual,
			cudaStream_t stream = 0
		) {
			LOGGING(FATAL) << "The init resiudal computation is not implemented";
		}

		//The debug method
		static void CompareApplySpMV(typename ApplySpMVBase<BlockDim>::Ptr applier_0, typename ApplySpMVBase<BlockDim>::Ptr applier_1);
	};
}


template<int BlockDim>
void SparseSurfelFusion::ApplySpMVBase<BlockDim>::CompareApplySpMV(typename ApplySpMVBase<BlockDim>::Ptr applier_0, typename ApplySpMVBase<BlockDim>::Ptr applier_1) {
	FUNCTION_CHECK(applier_0->MatrixSize() == applier_1->MatrixSize());

	//Prepare the data
	std::vector<float> x_h;
	x_h.resize(applier_0->MatrixSize());
	fillRandomVector(x_h);

	//Upload to device
	DeviceArray<float> x_dev, spmv_0, spmv_1;
	x_dev.upload(x_h);
	spmv_0.create(applier_0->MatrixSize());
	spmv_1.create(applier_1->MatrixSize());

	//Apply it
	applier_0->ApplySpMV(DeviceArrayView<float>(x_dev), DeviceArrayHandle<float>(spmv_0));
	applier_1->ApplySpMV(DeviceArrayView<float>(x_dev), DeviceArrayHandle<float>(spmv_1));

	//Download and check
	std::vector<float> spmv_0_h, spmv_1_h;
	spmv_0.download(spmv_0_h);
	spmv_1.download(spmv_1_h);
	LOGGING(INFO) << "The relative error between two spmv is " << maxRelativeError(spmv_0_h, spmv_1_h, 1e-3, true);
}