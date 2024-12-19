#pragma once

#include <core/NonRigidSolver/WarpField.h>
#include <core/Geometry/SurfelGeometry.h>
#include <core/Geometry/SurfelFusionHandler.h>
#include <core/Geometry/FusionRemainingSurfelMarker.h>
#include <core/Geometry/AppendSurfelProcessor.h>
#include <core/Geometry/DoubleBufferCompactor.h>
#include <ImageProcess/FrameProcessor.h>
#include <render/Renderer.h>
#include <core/Geometry/KNNSearch.h>

namespace SparseSurfelFusion {

	/* The geometry updater access to both surfel geometry (but not friend),
	 * Its task is to update the LIVE geometry and knn for both existing
	 * and newly appended surfels. The double buffer approach is implemented here
	 */
	class LiveGeometryUpdater {
	private:
		SurfelGeometry::Ptr m_surfel_geometry[MAX_CAMERA_COUNT][2];
		//SurfelGeometry::Ptr m_surfel_geometry_IndexMap;
		//For any processing iteration, this variable should be constant, only assign by external variable
		int m_updated_idx;
		float m_current_time;

		unsigned int devicesCount = 0;

		//The map from the renderer
		Renderer::FusionMaps m_fusion_maps[MAX_CAMERA_COUNT];

		//The skinning method from updater
		WarpField::LiveGeometryUpdaterInput m_warpfield_input;
		KNNSearch::Ptr m_live_node_skinner;

		//The observation from depth camera
		CameraObservation m_observation;
		mat34 m_world2camera[MAX_CAMERA_COUNT];
	public:
		using Ptr = std::shared_ptr<LiveGeometryUpdater>;
		LiveGeometryUpdater(SurfelGeometry::Ptr surfel_geometry[MAX_CAMERA_COUNT][2], Intrinsic* clipedintrinsic, const unsigned int devCount);
		~LiveGeometryUpdater();
		NO_COPY_ASSIGN_MOVE(LiveGeometryUpdater);

		//The process input
		void SetInputs(
			const Renderer::FusionMaps* maps,
			const CameraObservation& observation,
			const WarpField::LiveGeometryUpdaterInput& warpfield_input,
			const KNNSearch::Ptr& live_node_skinner,
			int updated_idx,
			float current_time,
			const mat34* world2camera
		);

		//The processing pipeline
		void TestFusion();
		void ProcessFusionSerial(unsigned& num_remaining_surfel, unsigned& num_appended_surfel, cudaStream_t stream = 0);

		/* The buffer and method for surfel fusion
		 */
	private:
		SurfelFusionHandler::Ptr m_surfel_fusion_handler;
	public:
		void FuseCameraObservationSync(cudaStream_t stream = 0);


		/* The buffer and method for cleaning the existing surfels
		 */
	private:
		FusionRemainingSurfelMarker::Ptr m_fusion_remaining_surfel_marker;
	public:
		void MarkRemainingSurfels(cudaStream_t stream = 0);
		RemainingLiveSurfelKNN GetRemainingLiveSurfelKNN() const;


		/* The buffer and method to process appended surfels
		 */
	private:
		AppendSurfelProcessor::Ptr m_appended_surfel_processor;
	public:
		void ProcessAppendedSurfels(cudaStream_t stream = 0);


		/* Compact the remaining surfel and appended surfel to another buffer
		 */
	private:
		DoubleBufferCompactor::Ptr m_surfel_compactor;
	public:
		void CompactSurfelToAnotherBufferSync(unsigned& num_remaining_surfel, unsigned& num_appended_surfel, cudaStream_t stream = 0);
		void TestCompactionKNNFirstIter(unsigned num_remaining_surfel, unsigned num_appended_surfel);


		/* The stream for fusion processing
		 */
	private:
		cudaStream_t m_fusion_stream[2];
		void initFusionStream();
		void releaseFusionStream();
	public:
		void ProcessFusionStreamed(unsigned& num_remaining_surfel, unsigned& num_appended_surfel);
	};


}