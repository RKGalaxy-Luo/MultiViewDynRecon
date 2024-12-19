#pragma once

#include <base/CommonUtils.h>
#include "solver_types.h"
#include "PenaltyConstants.h"
#include "Node2TermsIndex.h"
#include <memory>
#include <base/GlobalConfigs.h>

namespace SparseSurfelFusion {

	class ResidualEvaluator {
	private:

		const unsigned int imageWidth = FRAME_WIDTH - 2 * CLIP_BOUNDARY;		// Í¼Æ¬µÄ¿í
		const unsigned int imageHeight = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;		// Í¼Æ¬µÄ¸ß
		//The map from term to jacobian, will also be accessed on device
		Term2JacobianMaps m_term2jacobian_map;

		//The map from node to terms
		using Node2TermMap = Node2TermsIndex::Node2TermMap;
		Node2TermMap m_node2term_map;

		//The penalty constants
		PenaltyConstants m_penalty_constants;

	public:
		using Ptr = std::shared_ptr<ResidualEvaluator>;
		DEFAULT_CONSTRUCT_DESTRUCT(ResidualEvaluator);
		NO_COPY_ASSIGN(ResidualEvaluator);

		//Explicit allocation, release and input
		void AllocateBuffer();
		void ReleaseBuffer();


		//Input from external methods
		void SetInputs(
			Node2TermMap node2term,
			DenseDepthTerm2Jacobian dense_depth_term,
			NodeGraphSmoothTerm2Jacobian smooth_term,
			DensityMapTerm2Jacobian density_map_term = DensityMapTerm2Jacobian(),
			ForegroundMaskTerm2Jacobian foreground_mask_term = ForegroundMaskTerm2Jacobian(),
			Point2PointICPTerm2Jacobian sparse_feature_term = Point2PointICPTerm2Jacobian(),
			Point2PointICPTerm2Jacobian cross_corr_term = Point2PointICPTerm2Jacobian(),
			PenaltyConstants constants = PenaltyConstants()
		);

		//Compute the residual and sync to host
		float ComputeTotalResidualSynced(cudaStream_t stream = 0);


		/* The buffer and methods to compute term-wise residual
		 */
	private:
		DeviceBufferArray<float> m_termwise_residual;
	public:
		void ComputeResidualByTerms(cudaStream_t stream = 0);


		/* Collect the term-wise residual to node-wise residual
		 */
	private:
		DeviceBufferArray<float> m_nodewise_residual;
	public:
		void CollectResidualByNodes(cudaStream_t stream = 0);


		/* Collect the total residual
		 */
	private:
		float* m_residual_value_pagelock;
		DeviceBufferArray<float> m_residual_prefixsum;
		DeviceArray<unsigned char> m_prefixsum_buffer;
	public:
		void CollectTotalResidual(cudaStream_t stream = 0);
		float SyncQueryTotalResidualHost(cudaStream_t stream = 0);
	};

}
