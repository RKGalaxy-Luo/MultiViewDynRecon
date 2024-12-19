#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <core/AlgorithmTypes.h>
#include <base/term_offset_types.h>
#include <memory>

namespace SparseSurfelFusion {

	class NodePair2TermsIndex {
	public:
		using Ptr = std::shared_ptr<NodePair2TermsIndex>;
		NodePair2TermsIndex();
		~NodePair2TermsIndex() = default;
		NO_COPY_ASSIGN(NodePair2TermsIndex);

		//Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		void setColRow(unsigned col, unsigned row);//要和构造函数在一起，在allocatebuffer前

		//The input for index
		void SetInputs(
			unsigned num_nodes,
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph,
			//These costs might be empty
			DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> sparse_feature_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> cross_corr_knn = DeviceArrayView<ushort4>()
		);

		//The operation interface
		void BuildHalfIndex(cudaStream_t stream = 0);
		void QueryValidNodePairSize(cudaStream_t stream = 0); //Will block the stream
		unsigned NumTerms() const;
		unsigned NumKeyValuePairs() const;

		//Build the symmetric and row index
		void BuildSymmetricAndRowBlocksIndex(cudaStream_t stream = 0);

		//The access interface
		struct NodePair2TermMap {
			DeviceArrayView<unsigned> encoded_nodepair;
			DeviceArrayView<uint2> nodepair_term_range;
			DeviceArrayView<unsigned> nodepair_term_index;
			TermTypeOffset term_offset;
			//For bin-block csr
			DeviceArrayView<unsigned> blkrow_offset;
			DeviceArrayView<int> binblock_csr_rowptr;
			const int* binblock_csr_colptr;
		};
		NodePair2TermMap GetNodePair2TermMap() const;


		/* Fill the key and value given the terms
		 */
	private:

		int end_bit = 32;		// 将点对编码值所排序，相当于是行优先的排序，对键的前24位进行排序，节点数量[0, 4095]，编码的时候是x * 4096 + y，编码范围[0 * 4096 + 0, 4095 * 4096 + 4095] = [0, 2^24 -1]，因此编码最多涉及24位的排序比较

		//The input map from terms to nodes, the input might be empty for dense_density, foreground mask and sparse feature
		struct {
			DeviceArrayView<ushort4> dense_image_knn;		// Each depth scalar term has 4 nearest neighbour
			DeviceArrayView<ushort2> node_graph;			   
			//DeviceArrayView<ushort4> density_map_knn;		// Each density scalar term has 4 nearest neighbour
			DeviceArrayView<ushort4> foreground_mask_knn;	// The same as density term
			DeviceArrayView<ushort4> sparse_feature_knn;	// Each 4 nodes correspond to 3 scalar cost
			DeviceArrayView<ushort4> cross_corr_knn;		// Each 4 nodes correspond to 3 scalar cost
		} m_term2node;

		//The term offset of term2node map
		TermTypeOffset m_term_offset;
		unsigned int m_num_nodes;							// 现有NodeGragh中节点的数量
		unsigned int imageClipCol;
		unsigned int imageClipRow;

		/* The key-value buffer for indexing
		 */
	private:
		DeviceBufferArray<unsigned> m_nodepair_keys;	// 对节点对编码
		DeviceBufferArray<unsigned> m_term_idx_values;	// 这个节点对所在的位置
	public:
		void buildTermKeyValue(cudaStream_t stream = 0);


		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned int, unsigned int> m_nodepair2term_sorter;	// 对节点对排序
		DeviceBufferArray<unsigned int> m_segment_label;					// 记录排序好与之前不重复的点标记为1
		PrefixSum m_segment_label_prefixsum;								// 累和，可以得到当前idx不同node的数量

		//The compacted half key and values
		DeviceBufferArray<unsigned int> m_half_nodepair_keys;				// 节点对的编码(node_i, node_j),其中node_i的idx < node_j的idx
		DeviceBufferArray<unsigned int> m_half_nodepair2term_offset;		// 当前这个key在Array中的offset，因为肯定是有重复的
	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);


		/* Fill the other part of the matrix
		 */
	private:
		DeviceBufferArray<unsigned> m_compacted_nodepair_keys;				// 这里记录的是完整的节点对(node_i, node_j)和(node_j, node_i)，但不包含重复的
		DeviceBufferArray<uint2> m_nodepair_term_range;						// 这里记录某节点对的start_idx和end_idx，因为节点对是有重复的，所以需要在(排序好的)m_nodepair_keys和(排序好的)m_term_idx_values的数组中标记位置
		KeyValueSort<unsigned, uint2> m_symmetric_kv_sorter;				// 里面记录着压缩并排序后的对称节点对的key(node_i, node_j)以及这个节点对在m_half_nodepair_keys中的位置范围(有重复所以有start_idx和end_idx)
	public:
		// 计算成对节点对[(node_i, node_j)和(node_j, node_i)]的排序且压缩后的数组，并可以通过m_nodepair_term_range找到这个节点对在原数组中的位置
		void buildSymmetricCompactedIndex(cudaStream_t stream = 0);


		/* Compute the offset and length of each BLOCKED row
		 */
	private:
		DeviceBufferArray<unsigned> m_blkrow_offset_array;		// CSR稀疏矩阵的RowOffsets:row offsets 中包含有 rows+1 个值，其中 rows 指矩阵的总行数。最后一个数指的是矩阵中非零数的个数。前 rows 个数指每一行第一个非零数在values中的偏移量
		DeviceBufferArray<unsigned> m_blkrow_length_array;		// 每个节点矩阵块的长度，有k个节点与该节点相关联(不包含对称出来的)，就有这个块的长度就为k+1
		void blockRowOffsetSanityCheck();
		void blockRowLengthSanityCheck();
	public:
		// 1、计算每一个节点对应CSR格式中的RowOffset.
		// 2、计算每个节点矩阵块的长度，有k个节点与该节点相关联(不包含对称出来的)，就有这个块的长度就为k+1
		void computeBlockRowLength(cudaStream_t stream = 0);


		/* Compute the map from block row to the elements in this row block
		 */
	private:
		DeviceBufferArray<unsigned> m_binlength_array;			// 这个bin的长度(bin中节点对应块长度的最大值 * 6)
		DeviceBufferArray<unsigned> m_binnonzeros_prefixsum;	// 这个bin在整个flatten的矩阵元素(一共6 * nodeNum个)中的偏移
		DeviceBufferArray<int> m_binblocked_csr_rowptr;			// 这个元素在整个稀疏矩阵中行的首指针(因为是以bin中最大的节点块长度作为bin的长度，所以不连续很正常)
		void binLengthNonzerosSanityCheck();
		void binBlockCSRRowPtrSanityCheck();
	public:
		// 1、计算每一个bin元素的长度(bin中节点对应块长度的最大值 * 6)
		// 2、计算这个bin在整体flatten的矩阵元素中的偏移量
		void computeBinLength(cudaStream_t stream = 0);
		void computeBinBlockCSRRowPtr(cudaStream_t stream = 0);


		/* Compute the column ptr for bin block csr matrix
		 */
	private:
		DeviceBufferArray<int> m_binblocked_csr_colptr;
		void binBlockCSRColumnPtrSanityCheck();
	public:
		// 将m_binblocked_csr_colptr初始化为0xFF无效值
		void nullifyBinBlockCSRColumePtr(cudaStream_t stream = 0);

		void computeBinBlockCSRColumnPtr(cudaStream_t stream = 0);



		/* Perform sanity check for nodepair2term
		 */
	public:
		void CheckHalfIndex();
		void CompactedIndexSanityCheck();

		//Check the size and distribution of the size of index
		void IndexStatistics();

		//Check whether the smooth term contains nearly all index
		//that can be exploited to implement more efficient indexing
		//Required download data and should not be used in real-time code
		void CheckSmoothTermIndexCompleteness();
	private:
		static void check4NNTermIndex(int typed_term_idx,
			const std::vector<ushort4>& knn_vec,
			unsigned encoded_nodepair);
		static void checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort2>& node_graph, unsigned encoded_nodepair);
	};

}
