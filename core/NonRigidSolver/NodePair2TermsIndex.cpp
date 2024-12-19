
#include <base/Constants.h>
#include "NodePair2TermsIndex.h"
#include <base/EncodeUtils.h>


SparseSurfelFusion::NodePair2TermsIndex::NodePair2TermsIndex() {
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
}
void SparseSurfelFusion::NodePair2TermsIndex::setColRow(unsigned col, unsigned row)
{
	imageClipCol = col;
	imageClipRow = row;
}

void SparseSurfelFusion::NodePair2TermsIndex::AllocateBuffer() {
	const unsigned int num_pixels = imageClipCol * imageClipRow;
	const unsigned int max_dense_depth_terms = num_pixels;
	const unsigned int max_node_graph_terms = Constants::maxNodesNum * Constants::nodesGraphNeigboursNum;
	const unsigned int max_density_terms = num_pixels;
	const unsigned int max_foreground_terms = num_pixels / 2; //Only part on boundary
	const unsigned int max_feature_terms = Constants::kMaxMatchedSparseFeature;
	const unsigned int max_cross_terms = Constants::MaxCrossViewMatchedPairs;

	const unsigned int kv_buffer_size = 6 * (max_dense_depth_terms + max_density_terms + max_foreground_terms + max_feature_terms + max_cross_terms) + 1 * max_node_graph_terms;

	//Allocate the key-value pair
	m_nodepair_keys.AllocateBuffer(kv_buffer_size);
	m_term_idx_values.AllocateBuffer(kv_buffer_size);

	//Allocate the sorter and compaction
	m_nodepair2term_sorter.AllocateBuffer(kv_buffer_size);
	m_segment_label.AllocateBuffer(kv_buffer_size);
	m_segment_label_prefixsum.AllocateBuffer(kv_buffer_size);
	const unsigned int max_unique_nodepair = Constants::kMaxNumNodePairs;
	m_half_nodepair_keys.AllocateBuffer(max_unique_nodepair);
	m_half_nodepair2term_offset.AllocateBuffer(max_unique_nodepair);

	//The buffer for symmetric index
	m_compacted_nodepair_keys.AllocateBuffer(2 * max_unique_nodepair);
	m_nodepair_term_range.AllocateBuffer(2 * max_unique_nodepair);
	m_symmetric_kv_sorter.AllocateBuffer(2 * max_unique_nodepair);

	//For blocked offset and length of each row
	m_blkrow_offset_array.AllocateBuffer(Constants::maxNodesNum + 1);
	m_blkrow_length_array.AllocateBuffer(Constants::maxNodesNum);

	//For offset measured in bin
	const int max_bins = divUp(Constants::maxNodesNum * 6, 32);
	m_binlength_array.AllocateBuffer(max_bins);
	m_binnonzeros_prefixsum.AllocateBuffer(max_bins + 1);
	m_binblocked_csr_rowptr.AllocateBuffer(32 * (max_bins + 1));

	//For the colptr of bin blocked csr format
	m_binblocked_csr_colptr.AllocateBuffer(6 * Constants::kMaxNumNodePairs);
}

void SparseSurfelFusion::NodePair2TermsIndex::ReleaseBuffer() {
	m_nodepair_keys.ReleaseBuffer();
	m_term_idx_values.ReleaseBuffer();

	m_segment_label.ReleaseBuffer();

	m_compacted_nodepair_keys.ReleaseBuffer();
	m_nodepair_term_range.ReleaseBuffer();
}

void SparseSurfelFusion::NodePair2TermsIndex::SetInputs(
	unsigned num_nodes,
	DeviceArrayView<ushort4> dense_image_knn,
	DeviceArrayView<ushort2> node_graph,
	DeviceArrayView<ushort4> foreground_mask_knn,
	DeviceArrayView<ushort4> sparse_feature_knn,
	DeviceArrayView<ushort4> cross_corr_knn)
{
	m_num_nodes = num_nodes;

	m_term2node.dense_image_knn = dense_image_knn;
	m_term2node.node_graph = node_graph;
	m_term2node.foreground_mask_knn = foreground_mask_knn;
	m_term2node.sparse_feature_knn = sparse_feature_knn;
	m_term2node.cross_corr_knn = cross_corr_knn;

	//build the offset of these terms
	size2offset(
		m_term_offset,
		dense_image_knn,
		node_graph,
		foreground_mask_knn,
		sparse_feature_knn,
		cross_corr_knn
	);
}

void SparseSurfelFusion::NodePair2TermsIndex::BuildHalfIndex(cudaStream_t stream) {
	buildTermKeyValue(stream);
	//m_nodepair_keys.Ptr(),//编码后的节点对，这里边肯定有重复的
	//m_term_idx_values.Ptr()//这个节点对来自未展平的knn序列的哪

	sortCompactTermIndex(stream);
	//key是不重复的节点对的编码
	// offset是这个不重复的节点对的编码在排序后的有重复编码的节点对序列中的位置，也就是原来的位置
	//m_half_nodepair_keys.Ptr(),
	//m_half_nodepair2term_offset.Ptr()
}

void SparseSurfelFusion::NodePair2TermsIndex::BuildSymmetricAndRowBlocksIndex(cudaStream_t stream) {
	buildSymmetricCompactedIndex(stream);

	computeBlockRowLength(stream);
	computeBinLength(stream);
	computeBinBlockCSRRowPtr(stream);

	nullifyBinBlockCSRColumePtr(stream);
	computeBinBlockCSRColumnPtr(stream);
}

unsigned SparseSurfelFusion::NodePair2TermsIndex::NumTerms() const {
	return
		m_term2node.dense_image_knn.Size()
		+ m_term2node.node_graph.Size()
		+ m_term2node.foreground_mask_knn.Size()
		+ m_term2node.sparse_feature_knn.Size()
		+ m_term2node.cross_corr_knn.Size();
}

unsigned SparseSurfelFusion::NodePair2TermsIndex::NumKeyValuePairs() const {
	return
		m_term2node.dense_image_knn.Size() * 6
		+ m_term2node.node_graph.Size() * 1
		+ m_term2node.foreground_mask_knn.Size() * 6
		+ m_term2node.sparse_feature_knn.Size() * 6
		+ m_term2node.cross_corr_knn.Size() * 6;
}

SparseSurfelFusion::NodePair2TermsIndex::NodePair2TermMap SparseSurfelFusion::NodePair2TermsIndex::GetNodePair2TermMap() const {
	NodePair2TermMap map;
	map.encoded_nodepair = m_symmetric_kv_sorter.valid_sorted_key;
	map.nodepair_term_range = m_symmetric_kv_sorter.valid_sorted_value;
	map.nodepair_term_index = m_nodepair2term_sorter.valid_sorted_value;
	map.term_offset = m_term_offset;

	//For bin blocked csr format
	map.blkrow_offset = m_blkrow_offset_array.ArrayView();
	map.binblock_csr_rowptr = m_binblocked_csr_rowptr.ArrayView();
	map.binblock_csr_colptr = m_binblocked_csr_colptr.Ptr(); //The size is not required, and not queried
	return map;
}

/* The method for sanity check
 */
void SparseSurfelFusion::NodePair2TermsIndex::CheckHalfIndex() {
	LOGGING(INFO) << "Sanity check of the pair2term map half index";

	//First download the input data
	std::vector<ushort2> h_node_graph;
	std::vector<ushort4> h_dense_depth_knn, h_density_map_knn, h_foreground_mask_knn, h_sparse_feature_knn, h_cross_corr_knn;
	m_term2node.dense_image_knn.Download(h_dense_depth_knn);
	m_term2node.node_graph.Download(h_node_graph);
	m_term2node.foreground_mask_knn.Download(h_foreground_mask_knn);
	m_term2node.sparse_feature_knn.Download(h_sparse_feature_knn);
	m_term2node.cross_corr_knn.Download(h_cross_corr_knn);

	//Next download the value
	std::vector<unsigned> half_nodepair, half_offset, map_term_index;
	m_half_nodepair_keys.ArrayView().Download(half_nodepair);
	m_half_nodepair2term_offset.ArrayView().Download(half_offset);
	m_nodepair2term_sorter.valid_sorted_value.download(map_term_index);

	//Iterate over pairs
	for (auto nodepair_idx = 0; nodepair_idx < half_nodepair.size(); nodepair_idx++)
	{
		for (auto j = half_offset[nodepair_idx]; j < half_offset[nodepair_idx + 1]; j++) {
			const unsigned int term_idx = map_term_index[j];
			const unsigned int encoded_pair = half_nodepair[nodepair_idx];
			TermType type;
			unsigned int in_type_offset;
			query_typed_index(term_idx, m_term_offset, type, in_type_offset);
			switch (type) {
			case TermType::DenseImage:
				check4NNTermIndex(in_type_offset, h_dense_depth_knn, encoded_pair);
				break;
			case TermType::Smooth:
				checkSmoothTermIndex(in_type_offset, h_node_graph, encoded_pair);
				break;

			case TermType::Foreground:
				check4NNTermIndex(in_type_offset, h_foreground_mask_knn, encoded_pair);
				break;
			case TermType::Feature:
				check4NNTermIndex(in_type_offset, h_sparse_feature_knn, encoded_pair);
				break;
			case TermType::CrossViewCorr:
				check4NNTermIndex(in_type_offset, h_cross_corr_knn, encoded_pair);
				break;
			case TermType::Invalid:
			default:
				LOGGING(FATAL) << "Can not be invalid types";
			}
		}
	}

	LOGGING(INFO) << "Check done";
}

void SparseSurfelFusion::NodePair2TermsIndex::CompactedIndexSanityCheck() {
	LOGGING(INFO) << "Sanity check of the pair2term map";

	//First download the input data
	std::vector<ushort2> h_node_graph;
	std::vector<ushort4> h_dense_depth_knn, h_density_map_knn, h_foreground_mask_knn, h_sparse_feature_knn, h_cross_corr_knn;
	m_term2node.dense_image_knn.Download(h_dense_depth_knn);
	m_term2node.node_graph.Download(h_node_graph);
	m_term2node.foreground_mask_knn.Download(h_foreground_mask_knn);
	m_term2node.sparse_feature_knn.Download(h_sparse_feature_knn);
	m_term2node.cross_corr_knn.Download(h_cross_corr_knn);

	//Download the required map data
	const auto pair2term = GetNodePair2TermMap();
	std::vector<unsigned> nodepair, map_term_index;
	std::vector<uint2> nodepair_term_range;
	pair2term.encoded_nodepair.Download(nodepair);
	pair2term.nodepair_term_index.Download(map_term_index);
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	//Basic check
	FUNCTION_CHECK_EQ(map_term_index.size(), NumKeyValuePairs());

	//Check each nodes
	for (auto nodepair_idx = 0; nodepair_idx < nodepair.size(); nodepair_idx++)
	{
		for (auto j = nodepair_term_range[nodepair_idx].x; j < nodepair_term_range[nodepair_idx].y; j++) {
			const auto term_idx = map_term_index[j];
			const auto encoded_pair = nodepair[nodepair_idx];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, m_term_offset, type, in_type_offset);
			switch (type) {
			case TermType::DenseImage:
				check4NNTermIndex(in_type_offset, h_dense_depth_knn, encoded_pair);
				break;
			case TermType::Smooth:
				checkSmoothTermIndex(in_type_offset, h_node_graph, encoded_pair);
				break;
			case TermType::Foreground:
				check4NNTermIndex(in_type_offset, h_foreground_mask_knn, encoded_pair);
				break;
			case TermType::Feature:
				check4NNTermIndex(in_type_offset, h_sparse_feature_knn, encoded_pair);
				break;
			case TermType::CrossViewCorr:
				check4NNTermIndex(in_type_offset, h_cross_corr_knn, encoded_pair);
				break;
			case TermType::Invalid:
			default:
				LOGGING(FATAL) << "Can not be invalid types";
			}
		}
	}

	LOGGING(INFO) << "Check done";
}

void SparseSurfelFusion::NodePair2TermsIndex::check4NNTermIndex(
	int typed_term_idx,
	const std::vector<ushort4>& knn_vec,
	unsigned encoded_nodepair
) {
	const auto knn = knn_vec[typed_term_idx];
	unsigned node_i, node_j;
	decode_nodepair(encoded_nodepair, node_i, node_j);
	auto node_idx = node_i;
	FUNCTION_CHECK(node_idx == knn.x || node_idx == knn.y || node_idx == knn.z || node_idx == knn.w);
	node_idx = node_j;
	FUNCTION_CHECK(node_idx == knn.x || node_idx == knn.y || node_idx == knn.z || node_idx == knn.w);
}

void SparseSurfelFusion::NodePair2TermsIndex::checkSmoothTermIndex(
	int smooth_term_idx,
	const std::vector<ushort2>& node_graph,
	unsigned encoded_nodepair
) {
	const auto graph_pair = node_graph[smooth_term_idx];
	unsigned node_i, node_j;
	decode_nodepair(encoded_nodepair, node_i, node_j);
	auto node_idx = node_i;
	FUNCTION_CHECK(node_idx == graph_pair.x || node_idx == graph_pair.y);
	node_idx = node_j;
	FUNCTION_CHECK(node_idx == graph_pair.x || node_idx == graph_pair.y);
}

void SparseSurfelFusion::NodePair2TermsIndex::IndexStatistics() {
	LOGGING(INFO) << "Performing some statistics on the index";
	const auto pair2term = GetNodePair2TermMap();
	std::vector<uint2> nodepair_term_range;
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	double avg_term = 0.0;
	double min_term = 1e5;
	double max_term = 0.0;
	for (auto i = 0; i < nodepair_term_range.size(); i++) {
		uint2 term_range = nodepair_term_range[i];
		auto term_size = double(term_range.y - term_range.x);
		if (term_size < min_term) min_term = term_size;
		if (term_size > max_term) max_term = term_size;
		avg_term += term_size;
	}
	avg_term /= nodepair_term_range.size();
	LOGGING(INFO) << "The average size of node pair term is " << avg_term;
	LOGGING(INFO) << "The max size of node pair term is " << max_term;
	LOGGING(INFO) << "The min size of node pair term is " << min_term;
}


void SparseSurfelFusion::NodePair2TermsIndex::CheckSmoothTermIndexCompleteness() {
	//Download the required map data
	const auto pair2term = GetNodePair2TermMap();
	std::vector<unsigned> nodepair, map_term_index;
	std::vector<uint2> nodepair_term_range;
	pair2term.encoded_nodepair.Download(nodepair);
	pair2term.nodepair_term_index.Download(map_term_index);
	pair2term.nodepair_term_range.Download(nodepair_term_range);

	//Basic check
	FUNCTION_CHECK_EQ(map_term_index.size(), NumKeyValuePairs());
	FUNCTION_CHECK_EQ(nodepair.size(), nodepair_term_range.size());

	//The global term offset
	const TermTypeOffset& term_offset = m_term_offset;

	//Iterate over all pairs
	unsigned non_smooth_pairs = 0;
	for (int nodepair_idx = 0; nodepair_idx < nodepair.size(); nodepair_idx++) {
		bool contains_smooth_term = false;	// 标志位

		const uint2& term_range = nodepair_term_range[nodepair_idx];
		for (unsigned int term_iter = term_range.x; term_iter < term_range.y; term_iter++) {
			const unsigned int term_idx = map_term_index[term_iter];
			unsigned typed_term_idx;
			TermType term_type;
			query_typed_index(term_idx, term_offset, term_type, typed_term_idx);

			if (term_type == TermType::Smooth || term_type == TermType::Feature || term_type == TermType::Foreground || term_type == TermType::CrossViewCorr) {
				contains_smooth_term = true;
				break;
			}
		}

		//Increase the counter
		if (!contains_smooth_term) {
			non_smooth_pairs++;
		}
	}

	//Output if there is pair without smooth term
	if (non_smooth_pairs > 0) {
		std::cout << "在 " << nodepair.size() << " 点对中, 非平滑项点对个数为 " << non_smooth_pairs << "个" << std::endl;
		//std::cout << "There are " << non_smooth_pairs << " contains no smooth term of all " << nodepair.size() << " pairs!";
	}
	else {
		LOGGING(INFO) << "The smooth term is complete";
	}
}



















