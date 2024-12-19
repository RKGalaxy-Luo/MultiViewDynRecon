#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <core/AlgorithmTypes.h>
#include <base/term_offset_types.h>
#include <memory>

namespace SparseSurfelFusion {

	class Node2TermsIndex {
	private:
		//The input map from terms to nodes, the input might be empty for dense_density, foreground mask and sparse feature
		struct {
			DeviceArrayView<ushort4> dense_image_knn;		// Each depth scalar term has 4 nearest neighbour
			DeviceArrayView<ushort2> node_graph;			   
			DeviceArrayView<ushort4> foreground_mask_knn;	// The same as density term
			DeviceArrayView<ushort4> sparse_feature_knn;	// Each 4 nodes correspond to 3 scalar cost
			DeviceArrayView<ushort4> cross_corr_knn;		// �羵ƥ���Knn����feature_knn����
		} m_term2node;

		//The number of nodes
		unsigned m_num_nodes;
		unsigned imageClipCol;
		unsigned imageClipRow;

		//The term offset of term2node map
		TermTypeOffset m_term_offset;
	public:
		//Accessed by pointer, default construct/destruct
		using Ptr = std::shared_ptr<Node2TermsIndex>;
		Node2TermsIndex();
		~Node2TermsIndex() = default;
		NO_COPY_ASSIGN_MOVE(Node2TermsIndex);

		//Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		void setColRow(unsigned col, unsigned row);//Ҫ�͹��캯����һ����allocatebufferǰ

		//The input
		void SetInputs(
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph, unsigned num_nodes,
			//These costs might be empty
			DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> sparse_feature_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> cross_corr_knn = DeviceArrayView<ushort4>()
		);

		//The main interface
		void BuildIndex(cudaStream_t stream = 0);
		unsigned int NumTerms() const;
		unsigned int NumKeyValuePairs() const;


		/* Fill the key and value given the terms
		 */
	private:
		DeviceBufferArray<unsigned short> m_node_keys;		// ��������density + foreground + correspondence��knn �� nodeGraph��knn
		DeviceBufferArray<unsigned> m_term_idx_values;		// ��¼���node����nodeArrayterm�еĵڼ���
	public:
		void buildTermKeyValue(cudaStream_t stream = 0);



		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned short, unsigned> m_node2term_sorter;	// ����������term��knn��4��idxΪkey��term��term�����е�idxΪvalue
		DeviceBufferArray<unsigned> m_node2term_offset;				// ��¼ÿ������term�Ľڵ�������õ�nodeKey�����е�ƫ��
	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);


		/* A series of checking functions
		 */
	private:
		static void check4NNTermIndex(int typed_term_idx, const std::vector<ushort4>& knn_vec, unsigned short node_idx);
		static void checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort2>& node_graph, unsigned short node_idx);
		void compactedIndexSanityCheck();


		/* The accessing interface
		 * Depends on BuildIndex
		 */
	public:
		struct Node2TermMap {
			DeviceArrayView<unsigned> offset;
			DeviceArrayView<unsigned> term_index;
			TermTypeOffset term_offset;
		};

		//Return the outside-accessed index
		Node2TermMap GetNode2TermMap() const {
			Node2TermMap map;
			map.offset = m_node2term_offset.ArrayReadOnly();
			map.term_index = m_node2term_sorter.valid_sorted_value;
			map.term_offset = m_term_offset;
			return map;
		}
	};

}