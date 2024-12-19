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
		 * \brief 计算nodepair数组中的RowOffset.
		 * 
		 * \param compacted_Iij_key 排序并压缩后的对称节点对，(node_i, node_j)和(node_j, node_i)
		 * \param rowoffset_array
		 * \return 
		 */
		__global__ void computeRowOffsetKernel(
			const DeviceArrayView<unsigned> compacted_Iij_key,
			DeviceArrayHandle<unsigned> rowoffset_array
		) {
			const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= compacted_Iij_key.Size()) return;
			if (idx == 0) {	// CRS中的RowOffset，注意rowoffset_array.Size() = nodeNum + 1
				rowoffset_array[0] = 0;	// 第0行第0个就有数据
				rowoffset_array[rowoffset_array.Size() - 1] = compacted_Iij_key.Size();	// 最后一个数据存矩阵中非0数的数量
			}
			else {
				/**
				 * 比如说【升序x大的在后面】compacted_Iij_key:(node_1, node_3),(node_1, node_5),(node_1, node_6),(node_3, node_1),(node_5, node_1),(node_6, node_1)
				 * idx = 1: key_prev = (node_1, node_3)     key_this: (node_1, node_5)     
				 * idx = 2: key_prev = (node_1, node_5)     key_this: (node_1, node_6)     
				 * idx = 3: key_prev = (node_1, node_6)     key_this: (node_3, node_1)     
				 * idx = 4: key_prev = (node_3, node_1)     key_this: (node_5, node_1)     
				 * idx = 5: key_prev = (node_5, node_1)     key_this: (node_6, node_1)     
				 * idx = 6: 在上一个if中判断的
				 */
				const unsigned int key_prev = compacted_Iij_key[idx - 1];	// 节点对的key
				const unsigned int key_this = compacted_Iij_key[idx];		// 下一个节点对的key
				const unsigned short row_prev = encoded_row(key_prev);		// 解码(node_i, node_j)的key，得到i的值
				const unsigned short row_this = encoded_row(key_this);		// 解码(node_i, node_j)的key，得到i的值
				if (row_this != row_prev) {									// 如果不是同一个i
					rowoffset_array[row_this] = idx;						// 记录下这个位置
				}
				// 稀疏矩阵的构成：每个节点能分配一个矩阵块，node_i对应矩阵块的大小取决于(非对称)nodePair中有多少个node_i，上述例子node_1能分配到3×3的块
			}
		}

		// Kernel for computing the length of each row (both diag and non-diagonal terms)
		// 计算每行长度的核函数（包括对角项和非对角项）
		__global__ void computeRowBlockLengthKernel(
			const unsigned* rowoffset_array,
			const unsigned int nodeNum,
			DeviceArrayHandle<unsigned> blk_rowlength
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= nodeNum) return;
			// Note that the diagonal term is included 注意，这里包含了对角线项
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
			const unsigned int warp_idx = idx >> 5;		// 线程束warp的索引
			const unsigned int lane_idx = idx & 31;		// 线程在线程束warp中的索引

			unsigned int bin_length = 0;			// 【这个bin包含元素的长度】bin的长度取决于bin中节点所对应矩阵块的最大长度，将最大长度 * 6即是元素的长度
			if (idx < binsNum) {	// idx是第几个bin，idx到后面必超出nodeNum大小，默认idx是可以全覆盖的
				// 32 * idx is the real-matrix begin row
				// so does 32 * idx + 31 is the ending row
				// For a matrix row, its corresponding blk-row is matrix_row / 6
				const unsigned blkrow_begin = bin_size * idx / 6;			// 这个bin的起始节点的索引
				unsigned blkrow_end = (bin_size * idx + bin_size - 1) / 6;	// 这个bin最后一个节点的索引，超出稀疏矩阵大小之后，始终被赋值为块稀疏矩阵大小NodeNum
				blkrow_end = umin(blkrow_end, nodesNum - 1);				// 纠正bin的最后一个节点索引，如果blkrow_end > rowblk_length.Size()-1，说明bin能存的data数量已经超出了(对称)NotePair的数量，即超出了稀疏矩阵的大小
				unsigned max_length = 0;									// 这个bin中所有节点最大的块的长度(每个节点在矩阵中都有一个块)
				// 遍历这个bin中存储的所有节点，找到这个bin中节点node_i(事实上，一个blkrow_idx对应一个节点)的最大块的长度
				for (unsigned blkrow_idx = blkrow_begin; blkrow_idx <= blkrow_end; blkrow_idx++) {
					// blkrow_end超出了node范围后自然会赋值为nodeNum，到后面blkrow_begin > blkrow_end都不会执行这个循环，max_length直接为0
					max_length = umax(max_length, rowblk_length[blkrow_idx]);
				}

				//From block length to actual element length 从节点块的长度转向矩阵元素长度
				bin_length = 6 * max_length;		// 这个binLength是其中某个最大节点块的长度(单位:elem)
				valid_bin_length[idx] = bin_length;	// 将这个bin中的最大块的长度作为bin的有效长度
			}

			bin_length = warp_scan(bin_length);	// 前缀和这个线程束中每个lane
			if (lane_idx == 31) {
				partial_sum[warp_idx] = bin_length;	// 将这个线程束的所有binLength加到一起，注意不同线程对应的binLength其实不同，只是说同一个bin(一个线程一个bin)中节点blockLength被强制默认为最大blockLength而已
			}
			__syncthreads();	// 此处同步，将block中每一个线程都完成，已经将partial_sum[32]中的存了32个线程束中binLength的和

			if (warp_idx == 0) {					// 将所有线程束中累和的binLength加到一起
				const unsigned int partial_scan = partial_sum[lane_idx];	// lane_idx ∈ [0, 31]
				partial_sum[lane_idx] = warp_scan(partial_scan);			// 对线程束进行规约累和
			}
			__syncthreads();	// 此处同步，将block中每一个线程都完成，已经将partial_sum[32]存入规约得到的和，(31, 1) = [31+..+0, 30+..+0,...,0]

			if (idx < binsNum) {	// 如果是有效的bin(即binLength不为0)
				const unsigned offset = (warp_idx == 0 ? 0 : partial_sum[warp_idx - 1]);	// 当前线程束中线程的偏移(偏移量以元素为单位)
				valid_nonzeros_rowscan[idx + 1] = 32 * (bin_length + offset);				// 这个线程对应的bin的偏移值 --> (这个线程在线程束中的偏移 + 这个线程束在block的中的偏移 = 这个线程在block中的偏移)
				//const unsigned int binOffset_elem = bin_length + offset;
				//if (idx < 5) printf("BinOffset[%d] = %d\n", idx, binOffset_elem);
			}

			//The first elements
			if (idx == 0) valid_nonzeros_rowscan[0] = 0;									// 0的时候偏移为0
		}

		// 计算第idx个元素的位置
		__global__ void computeBinBlockedCSRRowPtrKernel(
			const unsigned* valid_nonzeros_rowscan,
			DeviceArrayHandle<int> csr_rowptr
		) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= csr_rowptr.Size()) return;	// 这个csr_rowptr是对原始的binLength有了个32倍的拓展
			// idx范围[0, (numBins + 1) * 32) -> [0, MatrixSize + 32)
			const int bin_row_idx = idx / 32;		// 这个元素在第几个bin中	范围[0, numBins + 1)
			const int bin_row_offset = idx % 32;	// 在bin条中的偏移，范围[0, 31]
			csr_rowptr[idx] = bin_row_offset + valid_nonzeros_rowscan[bin_row_idx];	// 在bin条中的偏移(有单位？) + 这个bin起始位置在整个矩阵中的偏移(单位:elem)
		}


		//The column index for bin-block csr format
		__global__ void computeBinBlockedCSRColPtrKernel(
			const unsigned matrix_size,
			const int* csr_rowptr,
			const unsigned* compacted_nodepair,	// 对称排序好的nodePair的Key
			const unsigned* blkrow_offset,		// 每个节点的RowOffset
			int* csr_colptr
		) {
			const unsigned int row_idx = threadIdx.x + blockDim.x * blockIdx.x;	// binsNum * 32
			if (row_idx >= matrix_size) return;
			/**
			 * row_idx范围[0, binsNum * 32) = [0, MatrixSize)
			 * binsNum = (MatrixSize / 32) = (nodesNum * 6 / 32)
			 * row_idx范围[0, nodesNum * 6)
			 * blkrow_idx范围[0, nodeNum)
			 * data_offset: 在bin条中的偏移(有单位？) + 这个bin起始位置在整个矩阵中的偏移(单位:elem)
			 * lane_idx: 不仅是[0, 31]生成值, lane_idx与row_idx有关系，lane_idx始终表示为网格中的哪一列
			 */
			//From now, the query on rowptr should be safe
			const unsigned int blkrow_idx = row_idx / 6;	// 这个元素属于哪个节点idx
			const int data_offset = csr_rowptr[row_idx];	// 直接定位这个元素的offset
			const unsigned int lane_idx = threadIdx.x & 31;	// lane_idx与row_idx有关系，lane_idx始终表示为网格中的哪一列

			/**
			 * data_offset - lane_idx, 将找到该bin的起始位置是第几个元素(单位: elem)
			 * (data_offset - lane_idx) / 6, 将找到该bin的起始位置是第几个节点块(单位: node)
			 * column_idx_offset = (data_offset - lane_idx) / 6 + lane_idx, 找到当前网格是第几个节点块(单位: node)
			 * 6 * blkrow_idx, 将这个元素所属的节点块再次拓展为以元素为单位的块，这个值是这个块的首值
			 * csr_colptr[column_idx_offset] = 6 * blkrow_idx 将这个节点块的首值(单位: elem)，作为这个节点块的colPtr
			 */
			//For the diagonal terms (针对对角线上的元素)
			unsigned int column_idx_offset = (data_offset - lane_idx) / 6 + lane_idx;	// 当前网格是第几个节点(单位: node), column_idx_offset并不是连续的
			csr_colptr[column_idx_offset] = 6 * blkrow_idx;								// 这个节点元素(node_i, node_i)在矩阵中的位置
			column_idx_offset += bin_size;												// 一个bin是32大小，+32跳转到下一列

			//For the non-diagonal terms (针对非对角线上的元素)
			unsigned int Iij_begin = blkrow_offset[blkrow_idx];			 // 获得节点块node_i的start偏移idx
			const unsigned int Iij_end = blkrow_offset[blkrow_idx + 1];	 // 获得节点块node_i的end偏移idx
			//[Iij_begin, Iij_end]之间是与node_i配对的不同的node_j1，node_j2..
			for (; Iij_begin < Iij_end; Iij_begin++, column_idx_offset += bin_size) {	// 超出m_half_nodepair_keys大小直接不会进入这个循环
				const unsigned int Iij_key = compacted_nodepair[Iij_begin];				// 获得堆成节点对的key
				const auto blkcol_idx = encoded_col(Iij_key);							// 解出node_j的index，实际上node_j的index就是该节点对(node_i, node_j)的CSR格式的该节点在矩阵中的列下标
				csr_colptr[column_idx_offset] = 6 * blkcol_idx;							// 节点块有大小为6
			}
		}

	} 
} 


void SparseSurfelFusion::NodePair2TermsIndex::computeBlockRowLength(cudaStream_t stream) {
	m_blkrow_offset_array.ResizeArrayOrException(m_num_nodes + 1);	

	// 排序后的对称节点对数组
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
	// 校正矩阵的大小
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
	// Compute the row pointer in bin-blocked csr format 计算bin-block CSR 形式下的row pointer
	m_binblocked_csr_rowptr.ResizeArrayOrException(32 * m_binnonzeros_prefixsum.ArraySize());

	// m_binblocked_csr_rowptr大小 (numBins + 1) * 32
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
	//Compute the size to nullify 计算需要赋无效值的数据大小(对称压缩后的nodepair数量【非对角线元素】 + 节点数量【对角线元素】)
	const unsigned long long total_blk_size = m_symmetric_kv_sorter.valid_sorted_key.size() + m_num_nodes;
	// 7 * total_blk_size【可能是防止6不够？每个block有6个元素】 和 m_binblocked_csr_colptr.BufferSize() = 6 * Constants::kMaxNumNodePairs的最小值，不超过Buffer是怕赋值溢出
	const unsigned long long nullify_size = std::min(7 * total_blk_size, m_binblocked_csr_colptr.BufferSize());

	//Do it
	CHECKCUDA(cudaMemsetAsync(m_binblocked_csr_colptr.Ptr(), 0xFF, sizeof(int) * m_binblocked_csr_colptr.BufferSize(), stream));
}


void SparseSurfelFusion::NodePair2TermsIndex::computeBinBlockCSRColumnPtr(cudaStream_t stream) {
	//The compacted full nodepair array
	DeviceArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	const unsigned int matrix_size = 6 * m_num_nodes;	// 矩阵大小
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