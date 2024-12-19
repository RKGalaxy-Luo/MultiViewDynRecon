#pragma once
#include <base/CommonUtils.h>
#include "DenseGaussian.h"
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <memory>


namespace SparseSurfelFusion {
	template<int BlockDim>
	class BlockDiagonalPreconditionerInverse {
	private:
		//The buffer for the inverse of diagonal blocks
		DeviceBufferArray<float> m_inv_diag_blks;

		//The input to the preconditioner
		DeviceArrayView<float> m_diagonal_blks;
		size_t m_matrix_size;

	public:
		using Ptr = std::shared_ptr<BlockDiagonalPreconditionerInverse>;
		BlockDiagonalPreconditionerInverse() {
			m_matrix_size = 0;
		}
		BlockDiagonalPreconditionerInverse(size_t max_matrix_size) {
			AllocateBuffer(max_matrix_size);
			m_matrix_size = 0;
		}
		~BlockDiagonalPreconditionerInverse() {
			if (m_inv_diag_blks.Capacity() > 0) m_inv_diag_blks.ReleaseBuffer();
		}
		NO_COPY_ASSIGN(BlockDiagonalPreconditionerInverse);
		DEFAULT_MOVE(BlockDiagonalPreconditionerInverse);

		//Allocate and release the buffer
		void AllocateBuffer(size_t max_matrix_size) {
			const auto max_blk_num = divUp(max_matrix_size, BlockDim);
			m_inv_diag_blks.AllocateBuffer(max_blk_num * BlockDim * BlockDim);
		}
		void ReleaseBuffer() {
			m_inv_diag_blks.ReleaseBuffer();
		}

		//The input interface
		void SetInput(DeviceArrayView<float> diagonal_blks) {
			//Simple sanity check
			FUNCTION_CHECK(diagonal_blks.Size() % (BlockDim * BlockDim) == 0);

			m_diagonal_blks = diagonal_blks;
			m_matrix_size = diagonal_blks.Size() / BlockDim;
			m_inv_diag_blks.ResizeArrayOrException(diagonal_blks.Size());
		}
		void SetInput(DeviceArray<float> diagonal_blks) {
			DeviceArrayView<float> diagonal_blks_view(diagonal_blks);
			SetInput(diagonal_blks_view);
		}

		//The processing and access interface
		void PerformDiagonalInverse(cudaStream_t stream = 0);
		DeviceArrayView<float> InversedDiagonalBlocks() const { return m_inv_diag_blks.ArrayView(); }
	};
}

//#if defined(__CUDACC__)
//#include "BlockDiagonalPreconditionerInverse.cu"
//#endif