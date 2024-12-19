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
		void setColRow(unsigned col, unsigned row);//Ҫ�͹��캯����һ����allocatebufferǰ

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

		int end_bit = 32;		// ����Ա���ֵ�������൱���������ȵ����򣬶Լ���ǰ24λ�������򣬽ڵ�����[0, 4095]�������ʱ����x * 4096 + y�����뷶Χ[0 * 4096 + 0, 4095 * 4096 + 4095] = [0, 2^24 -1]����˱�������漰24λ������Ƚ�

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
		unsigned int m_num_nodes;							// ����NodeGragh�нڵ������
		unsigned int imageClipCol;
		unsigned int imageClipRow;

		/* The key-value buffer for indexing
		 */
	private:
		DeviceBufferArray<unsigned> m_nodepair_keys;	// �Խڵ�Ա���
		DeviceBufferArray<unsigned> m_term_idx_values;	// ����ڵ�����ڵ�λ��
	public:
		void buildTermKeyValue(cudaStream_t stream = 0);


		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned int, unsigned int> m_nodepair2term_sorter;	// �Խڵ������
		DeviceBufferArray<unsigned int> m_segment_label;					// ��¼�������֮ǰ���ظ��ĵ���Ϊ1
		PrefixSum m_segment_label_prefixsum;								// �ۺͣ����Եõ���ǰidx��ͬnode������

		//The compacted half key and values
		DeviceBufferArray<unsigned int> m_half_nodepair_keys;				// �ڵ�Եı���(node_i, node_j),����node_i��idx < node_j��idx
		DeviceBufferArray<unsigned int> m_half_nodepair2term_offset;		// ��ǰ���key��Array�е�offset����Ϊ�϶������ظ���
	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);


		/* Fill the other part of the matrix
		 */
	private:
		DeviceBufferArray<unsigned> m_compacted_nodepair_keys;				// �����¼���������Ľڵ��(node_i, node_j)��(node_j, node_i)�����������ظ���
		DeviceBufferArray<uint2> m_nodepair_term_range;						// �����¼ĳ�ڵ�Ե�start_idx��end_idx����Ϊ�ڵ�������ظ��ģ�������Ҫ��(����õ�)m_nodepair_keys��(����õ�)m_term_idx_values�������б��λ��
		KeyValueSort<unsigned, uint2> m_symmetric_kv_sorter;				// �����¼��ѹ���������ĶԳƽڵ�Ե�key(node_i, node_j)�Լ�����ڵ����m_half_nodepair_keys�е�λ�÷�Χ(���ظ�������start_idx��end_idx)
	public:
		// ����ɶԽڵ��[(node_i, node_j)��(node_j, node_i)]��������ѹ��������飬������ͨ��m_nodepair_term_range�ҵ�����ڵ����ԭ�����е�λ��
		void buildSymmetricCompactedIndex(cudaStream_t stream = 0);


		/* Compute the offset and length of each BLOCKED row
		 */
	private:
		DeviceBufferArray<unsigned> m_blkrow_offset_array;		// CSRϡ������RowOffsets:row offsets �а����� rows+1 ��ֵ������ rows ָ����������������һ����ָ���Ǿ����з������ĸ�����ǰ rows ����ָÿһ�е�һ����������values�е�ƫ����
		DeviceBufferArray<unsigned> m_blkrow_length_array;		// ÿ���ڵ�����ĳ��ȣ���k���ڵ���ýڵ������(�������ԳƳ�����)�����������ĳ��Ⱦ�Ϊk+1
		void blockRowOffsetSanityCheck();
		void blockRowLengthSanityCheck();
	public:
		// 1������ÿһ���ڵ��ӦCSR��ʽ�е�RowOffset.
		// 2������ÿ���ڵ�����ĳ��ȣ���k���ڵ���ýڵ������(�������ԳƳ�����)�����������ĳ��Ⱦ�Ϊk+1
		void computeBlockRowLength(cudaStream_t stream = 0);


		/* Compute the map from block row to the elements in this row block
		 */
	private:
		DeviceBufferArray<unsigned> m_binlength_array;			// ���bin�ĳ���(bin�нڵ��Ӧ�鳤�ȵ����ֵ * 6)
		DeviceBufferArray<unsigned> m_binnonzeros_prefixsum;	// ���bin������flatten�ľ���Ԫ��(һ��6 * nodeNum��)�е�ƫ��
		DeviceBufferArray<int> m_binblocked_csr_rowptr;			// ���Ԫ��������ϡ��������е���ָ��(��Ϊ����bin�����Ľڵ�鳤����Ϊbin�ĳ��ȣ����Բ�����������)
		void binLengthNonzerosSanityCheck();
		void binBlockCSRRowPtrSanityCheck();
	public:
		// 1������ÿһ��binԪ�صĳ���(bin�нڵ��Ӧ�鳤�ȵ����ֵ * 6)
		// 2���������bin������flatten�ľ���Ԫ���е�ƫ����
		void computeBinLength(cudaStream_t stream = 0);
		void computeBinBlockCSRRowPtr(cudaStream_t stream = 0);


		/* Compute the column ptr for bin block csr matrix
		 */
	private:
		DeviceBufferArray<int> m_binblocked_csr_colptr;
		void binBlockCSRColumnPtrSanityCheck();
	public:
		// ��m_binblocked_csr_colptr��ʼ��Ϊ0xFF��Чֵ
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
