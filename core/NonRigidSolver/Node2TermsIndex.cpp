
#include <base/Logging.h>
#include <base/Constants.h>
#include "Node2TermsIndex.h"
#include <base/term_offset_types.h>
#include <vector>

SparseSurfelFusion::Node2TermsIndex::Node2TermsIndex() {
	memset(&m_term2node, 0, sizeof(m_term2node));
	memset(&m_term_offset, 0, sizeof(TermTypeOffset));
	m_num_nodes = 0;
}

void SparseSurfelFusion::Node2TermsIndex::AllocateBuffer() {
	//const auto& config = ConfigParser::Instance();
	//const auto num_pixels = config.clip_image_cols() * config.clip_image_rows();
	unsigned num_pixels = imageClipCol * imageClipRow;
	//printf("imageClipCol 的大小 %u \n", imageClipCol);
	//printf("imageClipRow 的大小 %u \n", imageClipRow);
	const unsigned int max_dense_image_terms = num_pixels;
	const unsigned int max_node_graph_terms = Constants::maxNodesNum * Constants::nodesGraphNeigboursNum;
	const unsigned int max_foreground_terms = num_pixels / 2; //Only part on boundary
	const unsigned int max_feature_terms = Constants::kMaxMatchedSparseFeature;
	const unsigned int max_cross_terms = Constants::MaxCrossViewMatchedPairs;

	//The total maximum size of kv buffer
	const auto kv_buffer_size = 4 * (max_dense_image_terms + max_node_graph_terms + max_foreground_terms + max_feature_terms + max_cross_terms);
	//printf("kv_buffer_size 的大小 %u \n", kv_buffer_size);

	//Allocate the key-value pair
	m_node_keys.AllocateBuffer(kv_buffer_size);
	m_term_idx_values.AllocateBuffer(kv_buffer_size);

	////Allocate the sorting and compaction buffer
	m_node2term_sorter.AllocateBuffer(kv_buffer_size);
	m_node2term_offset.AllocateBuffer(Constants::maxNodesNum + 1);
}

void SparseSurfelFusion::Node2TermsIndex::ReleaseBuffer() {
	m_node_keys.ReleaseBuffer();
	m_term_idx_values.ReleaseBuffer();
}

void SparseSurfelFusion::Node2TermsIndex::setColRow(unsigned col, unsigned row)
{
	imageClipCol = col;
	imageClipRow = row;
}

void SparseSurfelFusion::Node2TermsIndex::SetInputs(
	DeviceArrayView<ushort4> dense_image_knn,
	DeviceArrayView<ushort2> node_graph, unsigned num_nodes,
	DeviceArrayView<ushort4> foreground_mask_knn,
	DeviceArrayView<ushort4> sparse_feature_knn, 
	DeviceArrayView<ushort4> cross_corr_knn
) {
	m_term2node.dense_image_knn = dense_image_knn;
	m_term2node.node_graph = node_graph;
	m_term2node.foreground_mask_knn = foreground_mask_knn;
	m_term2node.sparse_feature_knn = sparse_feature_knn;
	m_term2node.cross_corr_knn = cross_corr_knn;
	m_num_nodes = num_nodes;

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

void SparseSurfelFusion::Node2TermsIndex::BuildIndex(cudaStream_t stream) {
	buildTermKeyValue(stream);		// 这一步是把自之前四个找到的knn结果放到一起，
	sortCompactTermIndex(stream);
}


/* The size query methods
 */
unsigned int SparseSurfelFusion::Node2TermsIndex::NumTerms() const
{
	return
		m_term2node.dense_image_knn.Size()
		+ m_term2node.node_graph.Size()
		+ m_term2node.foreground_mask_knn.Size()
		+ m_term2node.sparse_feature_knn.Size()
		+ m_term2node.cross_corr_knn.Size();
}


unsigned int SparseSurfelFusion::Node2TermsIndex::NumKeyValuePairs() const
{
	return
		m_term2node.dense_image_knn.Size() * 4
		+ m_term2node.node_graph.Size() * 2
		+ m_term2node.foreground_mask_knn.Size() * 4
		+ m_term2node.sparse_feature_knn.Size() * 4
		+ m_term2node.cross_corr_knn.Size() * 4;
}

/* A sanity check function for node2term maps
 */
void SparseSurfelFusion::Node2TermsIndex::compactedIndexSanityCheck() {
	LOGGING(INFO) << "Check of compacted node2term index";
	//First download the input data
	std::vector<ushort2> h_node_graph;
	std::vector<ushort4> h_dense_depth_knn, h_density_map_knn, h_foreground_mask_knn, h_sparse_feature_knn, h_cross_corr_knn;
	m_term2node.dense_image_knn.Download(h_dense_depth_knn);
	m_term2node.node_graph.Download(h_node_graph);
	m_term2node.foreground_mask_knn.Download(h_foreground_mask_knn);
	m_term2node.sparse_feature_knn.Download(h_sparse_feature_knn);
	m_term2node.cross_corr_knn.Download(h_cross_corr_knn);

	//Next download the maps
	const auto map = GetNode2TermMap();
	std::vector<unsigned> map_offset, map_term_index;
	map.offset.Download(map_offset);
	map.term_index.Download(map_term_index);

	//Basic check
	FUNCTION_CHECK_EQ(map_offset.size(), m_num_nodes + 1);
	FUNCTION_CHECK_EQ(map_term_index.size(), NumKeyValuePairs());

	//Check each nodes
	for (auto node_idx = 0; node_idx < m_num_nodes; node_idx++)
	{
		for (auto j = map_offset[node_idx]; j < map_offset[node_idx + 1]; j++) {
			const auto term_idx = map_term_index[j];
			TermType type;
			unsigned in_type_offset;
			query_typed_index(term_idx, map.term_offset, type, in_type_offset);
			switch (type) {
			case TermType::DenseImage:
				check4NNTermIndex(in_type_offset, h_dense_depth_knn, node_idx);
				break;
			case TermType::Smooth:
				checkSmoothTermIndex(in_type_offset, h_node_graph, node_idx);
				break;
			case TermType::Foreground:
				check4NNTermIndex(in_type_offset, h_foreground_mask_knn, node_idx);
				break;
			case TermType::Feature:
				check4NNTermIndex(in_type_offset, h_sparse_feature_knn, node_idx);
				break;
			case TermType::CrossViewCorr:
				check4NNTermIndex(in_type_offset, h_cross_corr_knn, node_idx);
				break;
			case TermType::Invalid:
			default:
				LOGGING(FATAL) << "Can not be invalid types";
			}
		}
	}

	LOGGING(INFO) << "Check done! Seems correct!";
}

void SparseSurfelFusion::Node2TermsIndex::check4NNTermIndex(
	int typed_term_idx,
	const std::vector<ushort4>& knn_vec,
	unsigned short node_idx
) {
	const auto knn = knn_vec[typed_term_idx];
	FUNCTION_CHECK(node_idx == knn.x || node_idx == knn.y || node_idx == knn.z || node_idx == knn.w);
}

void SparseSurfelFusion::Node2TermsIndex::checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort2>& node_graph, unsigned short node_idx)
{
	const auto node_pair = node_graph[smooth_term_idx];
	FUNCTION_CHECK(node_idx == node_pair.x || node_idx == node_pair.y);
}