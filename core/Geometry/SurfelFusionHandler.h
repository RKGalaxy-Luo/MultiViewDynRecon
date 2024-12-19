#pragma once

#include <base/CameraObservation.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <core/AlgorithmTypes.h>
#include <math/MatUtils.h>
#include <core/Geometry/SurfelGeometry.h>
#include <render/Renderer.h>
#include <visualization/Visualizer.h>

namespace SparseSurfelFusion {
	
	// The task of surfel fusion handler is: given the fusion
	// geometry and the fusion maps, fuse it to existing geometry
	// and compute indicators for which surfels should remain and
	// which depth pixel will potentiall be appended to the surfel array
	// This kernel parallelize over images (not the surfel array)
	class SurfelFusionHandler {
	private:
		//Basic parameters
		const unsigned int clipedImageCols = CLIP_WIDTH;
		const unsigned int clipedImageRows = CLIP_HEIGHT;
		
		//The input from outside
		Renderer::FusionMaps m_fusion_maps[MAX_CAMERA_COUNT];
		SurfelGeometry::SurfelFusionInput m_fusion_geometry;
		CameraObservation m_observation;
		float m_current_time;
		mat34 m_world2camera[MAX_CAMERA_COUNT];
		bool m_use_atomic_append; //Use atomic append or not
		const unsigned int devicesCount = MAX_CAMERA_COUNT;
		const unsigned int clipedImageSize = CLIP_WIDTH * CLIP_HEIGHT;
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];	// �����ʼ��λ��

		//The buffer maintained by this class
		DeviceBufferArray<unsigned int> remainingSurfelIndicator;

		//debug �����ֳ��ں��˵ġ�ԭʼ�Ķ�����Ԫ����������ɾȥ
		DeviceBufferArray<float4> fusion;
		DeviceBufferArray<float4> remain;


	public:
		using Ptr = std::shared_ptr<SurfelFusionHandler>;
		SurfelFusionHandler();
		~SurfelFusionHandler();
		NO_COPY_ASSIGN(SurfelFusionHandler);

		//The input requires all CamearObservation
		void SetInputs(
			const Renderer::FusionMaps* maps,
			const CameraObservation& observation,
			const SurfelGeometry::SurfelFusionInput& geometry,
			float current_time,
			const mat34* world2camera,
			bool use_atomic_append
		);
		
		//The processing interface for fusion
		void ProcessFusion(cudaStream_t stream = 0);
		void BuildCandidateAppendedPixelsSync(cudaStream_t stream = 0);

		//The fusion pipeline for reinit
		void ProcessFusionReinit(cudaStream_t stream = 0);
		
		//The fused indicator
		struct FusionIndicator {
			DeviceArrayHandle<unsigned> remaining_surfel_indicator;
			DeviceArrayView<ushort4> appended_pixels;
		};
		FusionIndicator GetFusionIndicator();


		/* Process data fusion using compaction
		 */
	private:
		DeviceArray<unsigned int> appendedObservedSurfelIndicator;		// ��ǹ۲����ݺͲ�ֵ�����Ƿ���Ҫ���
		PrefixSum appendedObservedSurfelIndicatorPrefixSum;
		DeviceBufferArray<ushort4> compactedAppendedPixel;

		void prepareFuserArguments(void* fuser_ptr);
		void processFusionAppendCompaction(cudaStream_t stream = 0);
		void processFusionReinit(cudaStream_t stream = 0);
		void compactAppendedIndicator(cudaStream_t stream = 0);
	public:
		void ZeroInitializeRemainingIndicator(unsigned num_surfels, cudaStream_t stream = 0);
		DeviceArrayHandle<unsigned int> GetRemainingSurfelIndicator();
		DeviceArrayView<unsigned int> GetAppendedObservationCandidateIndicator() const;

		/* Process appedning using atomic operation
		 */
	private:
		unsigned int* atomicAppendedPixelIndex; //�����ӽ���ȫ������Ҫ��ӵĵ�

		//������¼�ں��������Ԫ�Ķ�����Ԫ����
		unsigned int* FusedDepthSurfelNum;
		unsigned int* RemainingLiveSurfelNum;
		
		
		DeviceBufferArray<ushort4> atomicAppendedObservationPixel;
		void processFusionAppendAtomic(cudaStream_t stream = 0);
		void queryAtomicAppendedPixelSize(cudaStream_t stream = 0);


		//The debug method for fusion pipeline
	private:
		void fusionStatistic(bool using_atomic = false);
		void confidenceStatistic();
	};
	
	
}

