#include <base/DeviceAPI/device_intrinsics.h>
#include <base/SolverMethodConstants.h>
#include <base/term_offset_types.h>
#include <base/EncodeUtils.h>
#include "NodePair2TermsIndex.h"
#include <device_launch_parameters.h>
#include <math_functions.h>

namespace SparseSurfelFusion {
	namespace device {

		/**
		 * \brief ����nodepair�����е�RowOffset.
		 * 
		 * \param compacted_Iij_key ����ѹ����ĶԳƽڵ�ԣ�(node_i, node_j)��(node_j, node_i)
		 * \param rowoffset_array
		 * \return 
		 */
		__global__ void computeRowOffsetKernel(
			const DeviceArrayView<unsigned> compacted_Iij_key,
			DeviceArrayHandle<unsigned> rowoffset_array
		) {
			const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= compacted_Iij_key.Size()) return;
			if (idx == 0) {	// CRS�е�RowOffset��ע��rowoffset_array.Size() = nodeNum + 1
				rowoffset_array[0] = 0;	// ��0�е�0����������
				rowoffset_array[rowoffset_array.Size() - 1] = compacted_Iij_key.Size();	// ���һ�����ݴ�����з�0��������
			}
			else {
				/**
				 * ����˵������x����ں��桿compacted_Iij_key:(node_1, node_3),(node_1, node_5),(node_1, node_6),(node_3, node_1),(node_5, node_1),(node_6, node_1)
				 * idx = 1: key_prev = (node_1, node_3)     key_this: (node_1, node_5)     
				 * idx = 2: key_prev = (node_1, node_5)     key_this: (node_1, node_6)     
				 * idx = 3: key_prev = (node_1, node_6)     key_this: (node_3, node_1)     
				 * idx = 4: key_prev = (node_3, node_1)     key_this: (node_5, node_1)     
				 * idx = 5: key_prev = (node_5, node_1)     key_this: (node_6, node_1)     
				 * idx = 6: ����һ��if���жϵ�
				 */
				const unsigned int key_prev = compacted_Iij_key[idx - 1];	// �ڵ�Ե�key
				const unsigned int key_this = compacted_Iij_key[idx];		// ��һ���ڵ�Ե�key
				const unsigned short row_prev = encoded_row(key_prev);		// ����(node_i, node_j)��key���õ�i��ֵ
				const unsigned short row_this = encoded_row(key_this);		// ����(node_i, node_j)��key���õ�i��ֵ
				if (row_this != row_prev) {									// �������ͬһ��i
					rowoffset_array[row_this] = idx;						// ��¼�����λ��
				}
				// ϡ�����Ĺ��ɣ�ÿ���ڵ��ܷ���һ������飬node_i��Ӧ�����Ĵ�Сȡ����(�ǶԳ�)nodePair���ж��ٸ�node_i����������node_1�ܷ��䵽3��3�Ŀ�
			}
		}

		// Kernel for computing the length of each row (both diag and non-diagonal terms)
		// ����ÿ�г��ȵĺ˺����������Խ���ͷǶԽ��
		__global__ void computeRowBlockLengthKernel(
			const unsigned* rowoffset_array,
			const unsigned int nodeNum,
			DeviceArrayHandle<unsigned> blk_rowlength
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= nodeNum) return;
			// Note that the diagonal term is included ע�⣬��������˶Խ�����
			blk_rowlength[idx] = 1 + rowoffset_array[idx + 1] - rowoffset_array[idx];
			//if (idx <= 12) printf("NodeBlockLength[%d] = %d\n", idx, blk_rowlength[idx]);
		}

		__global__ void computeBinLengthKernel(
			const DeviceArrayView<unsigned> rowblk_length,
			DeviceArrayHandle<unsigned> valid_bin_length,
			const unsigned int nodesNum,
			const unsigned int binsNum,
			unsigned* valid_nonzeros_rowscan
		) {
			__shared__ unsigned partial_sum[32];

			//The idx is in [0, 1024)
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int warp_idx = idx >> 5;		// �߳���warp������
			const unsigned int lane_idx = idx & 31;		// �߳����߳���warp�е�����

			unsigned int bin_length = 0;			// �����bin����Ԫ�صĳ��ȡ�bin�ĳ���ȡ����bin�нڵ�����Ӧ��������󳤶ȣ�����󳤶� * 6����Ԫ�صĳ���
			if (idx < binsNum) {	// idx�ǵڼ���bin��idx������س���nodeNum��С��Ĭ��idx�ǿ���ȫ���ǵ�
				// 32 * idx is the real-matrix begin row
				// so does 32 * idx + 31 is the ending row
				// For a matrix row, its corresponding blk-row is matrix_row / 6
				const unsigned blkrow_begin = bin_size * idx / 6;			// ���bin����ʼ�ڵ������
				unsigned blkrow_end = (bin_size * idx + bin_size - 1) / 6;	// ���bin���һ���ڵ������������ϡ������С֮��ʼ�ձ���ֵΪ��ϡ������СNodeNum
				blkrow_end = umin(blkrow_end, nodesNum - 1);				// ����bin�����һ���ڵ����������blkrow_end > rowblk_length.Size()-1��˵��bin�ܴ��data�����Ѿ�������(�Գ�)NotePair����������������ϡ�����Ĵ�С
				unsigned max_length = 0;									// ���bin�����нڵ����Ŀ�ĳ���(ÿ���ڵ��ھ����ж���һ����)
				// �������bin�д洢�����нڵ㣬�ҵ����bin�нڵ�node_i(��ʵ�ϣ�һ��blkrow_idx��Ӧһ���ڵ�)������ĳ���
				for (unsigned blkrow_idx = blkrow_begin; blkrow_idx <= blkrow_end; blkrow_idx++) {
					// blkrow_end������node��Χ����Ȼ�ḳֵΪnodeNum��������blkrow_begin > blkrow_end������ִ�����ѭ����max_lengthֱ��Ϊ0
					max_length = umax(max_length, rowblk_length[blkrow_idx]);
				}

				//From block length to actual element length �ӽڵ��ĳ���ת�����Ԫ�س���
				bin_length = 6 * max_length;		// ���binLength������ĳ�����ڵ��ĳ���(��λ:elem)
				valid_bin_length[idx] = bin_length;	// �����bin�е�����ĳ�����Ϊbin����Ч����
			}

			bin_length = warp_scan(bin_length);	// ǰ׺������߳�����ÿ��lane
			if (lane_idx == 31) {
				partial_sum[warp_idx] = bin_length;	// ������߳���������binLength�ӵ�һ��ע�ⲻͬ�̶߳�Ӧ��binLength��ʵ��ͬ��ֻ��˵ͬһ��bin(һ���߳�һ��bin)�нڵ�blockLength��ǿ��Ĭ��Ϊ���blockLength����
			}
			__syncthreads();	// �˴�ͬ������block��ÿһ���̶߳���ɣ��Ѿ���partial_sum[32]�еĴ���32���߳�����binLength�ĺ�

			if (warp_idx == 0) {					// �������߳������ۺ͵�binLength�ӵ�һ��
				const unsigned int partial_scan = partial_sum[lane_idx];	// lane_idx �� [0, 31]
				partial_sum[lane_idx] = warp_scan(partial_scan);			// ���߳������й�Լ�ۺ�
			}
			__syncthreads();	// �˴�ͬ������block��ÿһ���̶߳���ɣ��Ѿ���partial_sum[32]�����Լ�õ��ĺͣ�(31, 1) = [31+..+0, 30+..+0,...,0]

			if (idx < binsNum) {	// �������Ч��bin(��binLength��Ϊ0)
				const unsigned offset = (warp_idx == 0 ? 0 : partial_sum[warp_idx - 1]);	// ��ǰ�߳������̵߳�ƫ��(ƫ������Ԫ��Ϊ��λ)
				valid_nonzeros_rowscan[idx + 1] = 32 * (bin_length + offset);				// ����̶߳�Ӧ��bin��ƫ��ֵ --> (����߳����߳����е�ƫ�� + ����߳�����block���е�ƫ�� = ����߳���block�е�ƫ��)
				//const unsigned int binOffset_elem = bin_length + offset;
				//if (idx < 5) printf("BinOffset[%d] = %d\n", idx, binOffset_elem);
			}

			//The first elements
			if (idx == 0) valid_nonzeros_rowscan[0] = 0;									// 0��ʱ��ƫ��Ϊ0
		}

		// �����idx��Ԫ�ص�λ��
		__global__ void computeBinBlockedCSRRowPtrKernel(
			const unsigned* valid_nonzeros_rowscan,
			DeviceArrayHandle<int> csr_rowptr
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= csr_rowptr.Size()) return;	// ���csr_rowptr�Ƕ�ԭʼ��binLength���˸�32������չ
			// idx��Χ[0, (numBins + 1) * 32) -> [0, MatrixSize + 32)
			const int bin_row_idx = idx / 32;		// ���Ԫ���ڵڼ���bin��	��Χ[0, numBins + 1)
			const int bin_row_offset = idx % 32;	// ��bin���е�ƫ�ƣ���Χ[0, 31]
			csr_rowptr[idx] = bin_row_offset + valid_nonzeros_rowscan[bin_row_idx];	// ��bin���е�ƫ��(�е�λ��) + ���bin��ʼλ�������������е�ƫ��(��λ:elem)
		}


		//The column index for bin-block csr format
		__global__ void computeBinBlockedCSRColPtrKernel(
			const unsigned matrix_size,
			const int* csr_rowptr,
			const unsigned* compacted_nodepair,	// �Գ�����õ�nodePair��Key
			const unsigned* blkrow_offset,		// ÿ���ڵ��RowOffset
			int* csr_colptr
		) {
			const unsigned int row_idx = threadIdx.x + blockDim.x * blockIdx.x;	// binsNum * 32
			if (row_idx >= matrix_size) return;
			/**
			 * row_idx��Χ[0, binsNum * 32) = [0, MatrixSize)
			 * binsNum = (MatrixSize / 32) = (nodesNum * 6 / 32)
			 * row_idx��Χ[0, nodesNum * 6)
			 * blkrow_idx��Χ[0, nodeNum)
			 * data_offset: ��bin���е�ƫ��(�е�λ��) + ���bin��ʼλ�������������е�ƫ��(��λ:elem)
			 * lane_idx: ������[0, 31]����ֵ, lane_idx��row_idx�й�ϵ��lane_idxʼ�ձ�ʾΪ�����е���һ��
			 */
			//From now, the query on rowptr should be safe
			const unsigned int blkrow_idx = row_idx / 6;	// ���Ԫ�������ĸ��ڵ�idx
			const int data_offset = csr_rowptr[row_idx];	// ֱ�Ӷ�λ���Ԫ�ص�offset
			const unsigned int lane_idx = threadIdx.x & 31;	// lane_idx��row_idx�й�ϵ��lane_idxʼ�ձ�ʾΪ�����е���һ��

			/**
			 * data_offset - lane_idx, ���ҵ���bin����ʼλ���ǵڼ���Ԫ��(��λ: elem)
			 * (data_offset - lane_idx) / 6, ���ҵ���bin����ʼλ���ǵڼ����ڵ��(��λ: node)
			 * column_idx_offset = (data_offset - lane_idx) / 6 + lane_idx, �ҵ���ǰ�����ǵڼ����ڵ��(��λ: node)
			 * 6 * blkrow_idx, �����Ԫ�������Ľڵ���ٴ���չΪ��Ԫ��Ϊ��λ�Ŀ飬���ֵ����������ֵ
			 * csr_colptr[column_idx_offset] = 6 * blkrow_idx ������ڵ�����ֵ(��λ: elem)����Ϊ����ڵ���colPtr
			 */
			//For the diagonal terms (��ԶԽ����ϵ�Ԫ��)
			unsigned int column_idx_offset = (data_offset - lane_idx) / 6 + lane_idx;	// ��ǰ�����ǵڼ����ڵ�(��λ: node), column_idx_offset������������
			csr_colptr[column_idx_offset] = 6 * blkrow_idx;								// ����ڵ�Ԫ��(node_i, node_i)�ھ����е�λ��
			column_idx_offset += bin_size;												// һ��bin��32��С��+32��ת����һ��

			//For the non-diagonal terms (��ԷǶԽ����ϵ�Ԫ��)
			unsigned int Iij_begin = blkrow_offset[blkrow_idx];			 // ��ýڵ��node_i��startƫ��idx
			const unsigned int Iij_end = blkrow_offset[blkrow_idx + 1];	 // ��ýڵ��node_i��endƫ��idx
			//[Iij_begin, Iij_end]֮������node_i��ԵĲ�ͬ��node_j1��node_j2..
			for (; Iij_begin < Iij_end; Iij_begin++, column_idx_offset += bin_size) {	// ����m_half_nodepair_keys��Сֱ�Ӳ���������ѭ��
				const unsigned int Iij_key = compacted_nodepair[Iij_begin];				// ��öѳɽڵ�Ե�key
				const auto blkcol_idx = encoded_col(Iij_key);							// ���node_j��index��ʵ����node_j��index���Ǹýڵ��(node_i, node_j)��CSR��ʽ�ĸýڵ��ھ����е����±�
				csr_colptr[column_idx_offset] = 6 * blkcol_idx;							// �ڵ���д�СΪ6
			}
		}

	} 
} 


void SparseSurfelFusion::NodePair2TermsIndex::computeBlockRowLength(cudaStream_t stream) {
	m_blkrow_offset_array.ResizeArrayOrException(m_num_nodes + 1);	

	// �����ĶԳƽڵ������
	DeviceArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	dim3 offset_blk(128);
	dim3 offset_grid(divUp(compacted_nodepair.Size(), offset_blk.x));
	device::computeRowOffsetKernel << <offset_grid, offset_blk, 0, stream >> > (
		compacted_nodepair,
		m_blkrow_offset_array.ArraySlice()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif

	//Compute the row length
	m_blkrow_length_array.ResizeArrayOrException(m_num_nodes);
	dim3 length_blk(64);
	dim3 length_grid(divUp(m_num_nodes, length_blk.x));
	device::computeRowBlockLengthKernel << <length_grid, length_blk, 0, stream >> > (
		m_blkrow_offset_array.ArrayView(),
		m_num_nodes,
		m_blkrow_length_array.ArraySlice()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif

	//Debug sanity check
	//blockRowOffsetSanityCheck();
	//blockRowLengthSanityCheck();
}


void SparseSurfelFusion::NodePair2TermsIndex::computeBinLength(cudaStream_t stream) {
	// У������Ĵ�С
	const unsigned int matrix_size = m_num_nodes * 6;
	const int num_bins = divUp(matrix_size, bin_size);

	m_binlength_array.ResizeArrayOrException(num_bins);
	m_binnonzeros_prefixsum.ResizeArrayOrException(num_bins + 1);
	device::computeBinLengthKernel << <1, 1024, 0, stream >> > (
		m_blkrow_length_array.ArrayView(),
		m_binlength_array.ArraySlice(),
		m_num_nodes,
		num_bins,
		m_binnonzeros_prefixsum.ArraySlice()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif

	//The sanity check method
	//binLengthNonzerosSanityCheck();
}


void SparseSurfelFusion::NodePair2TermsIndex::computeBinBlockCSRRowPtr(cudaStream_t stream) {
	// Compute the row pointer in bin-blocked csr format ����bin-block CSR ��ʽ�µ�row pointer
	m_binblocked_csr_rowptr.ResizeArrayOrException(32 * m_binnonzeros_prefixsum.ArraySize());

	// m_binblocked_csr_rowptr��С (numBins + 1) * 32
	dim3 rowptr_blk(128);
	dim3 rowptr_grid(divUp(m_binblocked_csr_rowptr.ArraySize(), rowptr_blk.x));
	device::computeBinBlockedCSRRowPtrKernel << <rowptr_grid, rowptr_blk, 0, stream >> > (
		m_binnonzeros_prefixsum.Ptr(),
		m_binblocked_csr_rowptr.ArraySlice()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif

	//Sanity check method
	//binBlockCSRRowPtrSanityCheck();
}

void SparseSurfelFusion::NodePair2TermsIndex::nullifyBinBlockCSRColumePtr(cudaStream_t stream) {
	//Compute the size to nullify ������Ҫ����Чֵ�����ݴ�С(�Գ�ѹ�����nodepair�������ǶԽ���Ԫ�ء� + �ڵ��������Խ���Ԫ�ء�)
	const unsigned long long total_blk_size = m_symmetric_kv_sorter.valid_sorted_key.size() + m_num_nodes;
	// 7 * total_blk_size�������Ƿ�ֹ6������ÿ��block��6��Ԫ�ء� �� m_binblocked_csr_colptr.BufferSize() = 6 * Constants::kMaxNumNodePairs����Сֵ��������Buffer���¸�ֵ���
	const unsigned long long nullify_size = std::min(7 * total_blk_size, m_binblocked_csr_colptr.BufferSize());

	//Do it
	CHECKCUDA(cudaMemsetAsync(m_binblocked_csr_colptr.Ptr(), 0xFF, sizeof(int) * m_binblocked_csr_colptr.BufferSize(), stream));
}


void SparseSurfelFusion::NodePair2TermsIndex::computeBinBlockCSRColumnPtr(cudaStream_t stream) {
	//The compacted full nodepair array
	DeviceArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	const unsigned int matrix_size = 6 * m_num_nodes;	// �����С
	const unsigned int num_bins = m_binlength_array.ArraySize();
	//Do not need to query the size of colptr?
	dim3 colptr_blk(128);
	dim3 colptr_grid(divUp(32 * num_bins, colptr_blk.x));
	device::computeBinBlockedCSRColPtrKernel << <colptr_grid, colptr_blk, 0, stream >> > (
		matrix_size,
		m_binblocked_csr_rowptr.Ptr(),
		compacted_nodepair.RawPtr(),
		m_blkrow_offset_array.Ptr(),
		m_binblocked_csr_colptr.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	CHECKCUDA(cudaStreamSynchronize(stream));
	CHECKCUDA(cudaGetLastError());
#endif

	//Debug method
	//binBlockCSRColumnPtrSanityCheck();
}