#pragma once

//hsg

#include <base/Constants.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include <core/NonRigidSolver/PenaltyConstants.h>
#include <base/CommonUtils.h>
#include <base/DeviceReadWrite/DeviceArrayHandle.h>


namespace SparseSurfelFusion {

	class SolverIterationData {
	private:
		//The state to keep track currect input/output
		enum class IterationInputFrom {
			WarpFieldInit,
			SE3_Buffer_0,
			SE3_Buffer_1
		};

		//The input from warp field
		DeviceArrayView<DualQuaternion> node_se3_init_;

		//The double buffer are mainted in this class
		DeviceBufferArray<DualQuaternion> node_se3_0_;
		DeviceBufferArray<DualQuaternion> node_se3_1_;
		IterationInputFrom m_updated_se3;
		unsigned m_newton_iters;
		void updateIterationFlags();

		//Only need to keep one twist buffer
		DeviceBufferArray<float> m_twist_update;

		//The constants for different terms
		bool m_use_density;
		bool m_use_foreground;
		bool m_use_cross_view;
		PenaltyConstants m_penalty_constants;

		//Use elastic penalty or not
		static void setElasticPenaltyValue(
			int newton_iter,
			PenaltyConstants& constants,
			bool use_density = false, 
			bool use_foreground = false, 
			bool use_cross_view_corr = false
		);


		//Allocate and release these buffers
		void allocateBuffer();
		void releaseBuffer();

	public:
		explicit SolverIterationData();
		~SolverIterationData();
		NO_COPY_ASSIGN_MOVE(SolverIterationData);

		//The process interface
		void SetWarpFieldInitialValue(DeviceArrayView<DualQuaternion> init_node_se3);
		DeviceArrayView<DualQuaternion> CurrentWarpFieldInput() const;
		DeviceArray<DualQuaternion> CurrentWarpFieldSe3Interface();
		DeviceArrayHandle<float> CurrentWarpFieldUpdateBuffer();
		bool IsInitialIteration() const { return m_updated_se3 == IterationInputFrom::WarpFieldInit; }

		//The constants for current iteration
		PenaltyConstants CurrentPenaltyConstants() const { return m_penalty_constants; }
		bool IsGlobalIteration() const { return m_penalty_constants.Density() < 1e-7f; }
		bool ComputeJtJLazyEvaluation() const { return m_newton_iters >= Constants::kNumGlobalSolverItarations; };

		//External accessed sanity check method
		void SanityCheck() const;
		size_t NumNodes() const { return node_se3_init_.Size(); }

		//Required cuda access
		void ApplyWarpFieldUpdate(cudaStream_t stream = 0, float step = 1.0f);
	};

}
