#include "solver_constants.h"
#include "PenaltyConstants.h"


SparseSurfelFusion::PenaltyConstants::PenaltyConstants() {
	setDefaultValue();
}

void SparseSurfelFusion::PenaltyConstants::setDefaultValue() {
#if defined(USE_DENSE_SOLVER_MAPS)
	m_lambda_smooth = 2.5f;
	m_lambda_density = 0.0f;
	m_lambda_foreground = 0.0f;
	m_lambda_feature = 0.7f;
	m_lambda_corss = 0.1f;
#else
	m_lambda_smooth = 2.0f;
	m_lambda_density = 0.0f;
	m_lambda_foreground = 0.0f;
	m_lambda_feature = 0.0f;
	m_lambda_corss = 0.1f;
#endif
}

void SparseSurfelFusion::PenaltyConstants::setGlobalIterationValue(bool use_foreground, bool use_cross_view_corr) {
	m_lambda_smooth = 5.0f;					// 1:2.3f		3:30.0f
	m_lambda_density = 0.0f;				// 1:0.0f		3:0.0f
	if (use_foreground)						   				
		m_lambda_foreground = 2e-3f;		// 1:2e-3f		3:2e-3f
	else									   				
		m_lambda_foreground = 0.0f;			// 1:0.0f		3:0.0f
	m_lambda_feature = 1.0f;				// 1:1.0f		3:20.0f
	if (use_cross_view_corr)
		m_lambda_corss = 1.0f;
	else
		m_lambda_corss = 0.0f;
}

void SparseSurfelFusion::PenaltyConstants::setLocalIterationValue(bool use_density, bool use_cross_view_corr) {
	m_lambda_smooth = 5.0f;					// 1:2.3f		3:30.0f
	if (use_density)
		m_lambda_density = 1e-2f;			// 1:1e-2f		3:1e-2f
	else
		m_lambda_density = 0.0f;			// 1:0.0f		3:0.0f
	m_lambda_foreground = 0.0f;				// 1:0.0f		3:0.0f
	m_lambda_feature = 0.0f;				// 1:0.0f		3:50.0f
	if (use_cross_view_corr)
		m_lambda_corss = 1.0f;
	else
		m_lambda_corss = 0.0f;
}
