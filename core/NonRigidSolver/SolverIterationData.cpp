#include <base/Constants.h>
#include "SolverIterationData.h"

/* The method for construction/destruction, buffer management
 */
SparseSurfelFusion::SolverIterationData::SolverIterationData() {
	allocateBuffer();
	m_updated_se3 = IterationInputFrom::WarpFieldInit;
	m_newton_iters = 0;

	//The flag for density and foreground term

	m_use_density = ConfigParser::UseDensityTerm();
	m_use_foreground = ConfigParser::UseForegroundTerm();
	m_use_cross_view = ConfigParser::UseCrossViewTerm();
}

SparseSurfelFusion::SolverIterationData::~SolverIterationData() {
	releaseBuffer();
}

void SparseSurfelFusion::SolverIterationData::allocateBuffer() {
	node_se3_0_.AllocateBuffer(Constants::maxNodesNum);
	node_se3_1_.AllocateBuffer(Constants::maxNodesNum);
	m_twist_update.AllocateBuffer((float)6 * Constants::maxNodesNum);
}

void SparseSurfelFusion::SolverIterationData::releaseBuffer() {
	node_se3_0_.ReleaseBuffer();
	node_se3_1_.ReleaseBuffer();
	m_twist_update.ReleaseBuffer();
}


/* The processing interface
 */
void SparseSurfelFusion::SolverIterationData::SetWarpFieldInitialValue(DeviceArrayView<DualQuaternion> init_node_se3) {
	node_se3_init_ = init_node_se3;
	m_updated_se3 = IterationInputFrom::WarpFieldInit;
	m_newton_iters = 0;

	//Correct the size of everything
	const auto num_nodes = init_node_se3.Size();
	node_se3_0_.ResizeArrayOrException(num_nodes);
	node_se3_1_.ResizeArrayOrException(num_nodes);
	m_twist_update.ResizeArrayOrException(6 * num_nodes);

	//Init the penalty constants
	setElasticPenaltyValue(0, m_penalty_constants);
}

SparseSurfelFusion::DeviceArrayView<SparseSurfelFusion::DualQuaternion> SparseSurfelFusion::SolverIterationData::CurrentWarpFieldInput() const {
	switch (m_updated_se3) {
	case IterationInputFrom::WarpFieldInit:
		return node_se3_init_;
	case IterationInputFrom::SE3_Buffer_0:
		return node_se3_0_.ArrayView();
	case IterationInputFrom::SE3_Buffer_1:
		return node_se3_1_.ArrayView();
	default:
		LOGGING(FATAL) << "Should never happen";
	}
}
SparseSurfelFusion::DeviceArray<SparseSurfelFusion::DualQuaternion> SparseSurfelFusion::SolverIterationData::CurrentWarpFieldSe3Interface()
{
	switch (m_updated_se3) {
	// 使用在非刚性对齐后面，不可能存在case IterationInputFrom::WarpFieldInit
	//case IterationInputFrom::WarpFieldInit:
	//	return node_se3_init_;
	case IterationInputFrom::SE3_Buffer_0:
		return node_se3_0_.Array();
	case IterationInputFrom::SE3_Buffer_1:
		return node_se3_1_.Array();
	default:
		LOGGING(FATAL) << "Should never happen";
	}
}
SparseSurfelFusion::DeviceArrayHandle<float> SparseSurfelFusion::SolverIterationData::CurrentWarpFieldUpdateBuffer() {
	return m_twist_update.ArraySlice();
}

void SparseSurfelFusion::SolverIterationData::SanityCheck() const {
	const auto num_nodes = node_se3_init_.Size();
	FUNCTION_CHECK_EQ(num_nodes, node_se3_0_.ArraySize());
	FUNCTION_CHECK_EQ(num_nodes, node_se3_1_.ArraySize());
	FUNCTION_CHECK_EQ(num_nodes * 6, m_twist_update.ArraySize());
}

void SparseSurfelFusion::SolverIterationData::updateIterationFlags() {
	//Update the flag
	if (m_updated_se3 == IterationInputFrom::SE3_Buffer_0) {
		m_updated_se3 = IterationInputFrom::SE3_Buffer_1;
	}
	else {
		//Either init or from buffer 1
		m_updated_se3 = IterationInputFrom::SE3_Buffer_0;
	}

	//Update the iteration counter
	m_newton_iters++;

	//The penalty for next iteration
	setElasticPenaltyValue(m_newton_iters, m_penalty_constants, m_use_density, m_use_foreground, m_use_cross_view);
}


void SparseSurfelFusion::SolverIterationData::setElasticPenaltyValue(
	int newton_iter,
	PenaltyConstants& constants,
	bool use_density,
	bool use_foreground,
	bool use_cross_view_corr
) {
	if (!Constants::kUseElasticPenalty) {
		constants.setDefaultValue();
		return;
	}

	if (newton_iter < Constants::kNumGlobalSolverItarations) {
		constants.setGlobalIterationValue(use_foreground, use_cross_view_corr);
	}
	else {
		constants.setLocalIterationValue(use_density, use_cross_view_corr);
	}
}

