//hsg 

#include "NonRigidSolver.h"


//void SparseSurfelFusion::NonRigidSolver::SolveSerial(cudaStream_t stream) {
//	solveMatrixFreeFixedIndexSerial(stream);
//	solveMaterializedFixedIndexSerial(stream);
//	solveMaterializedFixedIndexGlobalLocalSerial(stream);
//}


//void SparseSurfelFusion::NonRigidSolver::solveMaterializedFixedIndexGlobalLocalSerial(cudaStream_t stream) {
//	QueryPixelKNN(stream);
//	fullGlobalSolverIterationMaterializedFixedIndexSerial(stream);
//	for (auto i = 0; i < Constants::kNumGaussNewtonIterations - 1; i++) {
//		if (m_iteration_data.IsGlobalIteration())
//			materializedFixedIndexSolverGlobalIterationSerial(stream);
//		else
//			materializedFixedIndexSolverIterationSerial(stream);
//	}
//}