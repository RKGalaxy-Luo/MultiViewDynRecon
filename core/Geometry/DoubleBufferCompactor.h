//
// Created by wei on 5/7/18.
//

#pragma once

#include <base/CommonUtils.h>
#include <core/AlgorithmTypes.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <core/Geometry/fusion_types.h>
#include <core/Geometry/SurfelGeometry.h>

#include <memory>


namespace SparseSurfelFusion {
	


	/**
	 * \brief The compactor takes input from:
	 * 1. Original live surfel, their knn/weights, and validity indicator
	 * 2. Appended depth surfel, their knn/weights, and validity indicator
	 * The task of the compactor is to compact all these surfels to another
	 * buffer provided by OpenGL pipeline, and count the total number of compacted surfels. 
	 */
	class DoubleBufferCompactor {
	private:
		//The appended observation surfel from the depth/color image
		AppendedObservationSurfelKNN m_appended_surfel_knn;
		
		//The append surfel for reinit
		ReinitAppendedObservationSurfel m_reinit_append_surfel;

		//The surfel from the original model
		RemainingLiveSurfel m_remaining_surfel;
		RemainingSurfelKNN m_remaining_knn;

		//The geometry that shoule be compacted to
		SurfelGeometry::Ptr m_compact_to_geometry[MAX_CAMERA_COUNT][2];

		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];

		//The rows and cols used to decode the geometry
		const unsigned int m_image_rows = CLIP_HEIGHT;
		const unsigned int m_image_cols = CLIP_WIDTH;
		const unsigned int devicesCount = MAX_CAMERA_COUNT;

		unsigned int updatedGeometryIndex = 0;
	public:
		using Ptr = std::shared_ptr<DoubleBufferCompactor>;
		DoubleBufferCompactor();
		~DoubleBufferCompactor();
		NO_COPY_ASSIGN_MOVE(DoubleBufferCompactor);


		//The input from both append handler and geometry updater
		void SetFusionInputs(
			const RemainingLiveSurfelKNN& remaining_surfels,
			const AppendedObservationSurfelKNN& appended_surfels,
			const unsigned int compacted_to_idx,
			SurfelGeometry::Ptr compacted_geometry[MAX_CAMERA_COUNT][2]
		);
		
		//The input from geometry reiniter
		void SetReinitInputs(
			const RemainingLiveSurfel& remaining_surfels,
			const ReinitAppendedObservationSurfel& append_surfels,
			const unsigned int compacted_to_idx,
			SurfelGeometry::Ptr compact_to_geometry[MAX_CAMERA_COUNT][2]
		);
	
		//The main data for compaction, note that this method will sync
		//to query the size of remaining and appended sufel
		void PerformCompactionGeometryKNNSync(unsigned int& num_valid_remaining_surfels, unsigned int& num_valid_append_surfels, cudaStream_t stream = 0);
		void PerformComapctionGeometryOnlySync(unsigned int& num_valid_remaining_surfels, unsigned int& num_valid_append_surfels, unsigned int* number, mat34* world2camera, cudaStream_t stream = 0);
	};


} 