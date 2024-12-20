#include <base/Constants.h>
#include "PreconditionerRhsBuilder.h"

void SparseSurfelFusion::PreconditionerRhsBuilder::AllocateBuffer() {
	m_block_preconditioner.AllocateBuffer(36 * Constants::maxNodesNum);
	m_preconditioner_inverse_handler = std::make_shared<BlockDiagonalPreconditionerInverse<6>>();
	m_preconditioner_inverse_handler->AllocateBuffer(6 * Constants::maxNodesNum);

	m_jt_residual.AllocateBuffer(6 * Constants::maxNodesNum);
}

void SparseSurfelFusion::PreconditionerRhsBuilder::ReleaseBuffer() {
	m_block_preconditioner.ReleaseBuffer();
	m_preconditioner_inverse_handler->ReleaseBuffer();

	m_jt_residual.ReleaseBuffer();
}

void SparseSurfelFusion::PreconditionerRhsBuilder::SetInputs(
	Node2TermMap node2term,
	DenseDepthTerm2Jacobian dense_depth_term,
	NodeGraphSmoothTerm2Jacobian smooth_term,
	DensityMapTerm2Jacobian density_map_term,
	ForegroundMaskTerm2Jacobian foreground_mask_term,
	Point2PointICPTerm2Jacobian sparse_feature_term,
	Point2PointICPTerm2Jacobian cross_corr_term,
	PenaltyConstants constants
) {
	m_node2term_map = node2term;

	m_term2jacobian_map.dense_depth_term = dense_depth_term;
	m_term2jacobian_map.smooth_term = smooth_term;
	m_term2jacobian_map.density_map_term = density_map_term;
	m_term2jacobian_map.foreground_mask_term = foreground_mask_term;
	m_term2jacobian_map.sparse_feature_term = sparse_feature_term;
	m_term2jacobian_map.cross_corr_term = cross_corr_term;

	m_penalty_constants = constants;
}


//The high level processing interface
void SparseSurfelFusion::PreconditionerRhsBuilder::ComputeDiagonalPreconditioner(cudaStream_t stream) {
	ComputeDiagonalBlocks(stream);
	ComputeDiagonalPreconditionerInverse(stream);
}



