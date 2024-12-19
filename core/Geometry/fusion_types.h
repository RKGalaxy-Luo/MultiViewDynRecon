#pragma once

#include<math/MatUtils.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>


namespace SparseSurfelFusion {
	
	//The struct as the input to compactor, should be
	//used for both the regular fusion and reinitialization
	struct AppendedObservationSurfelKNN {
		//The binary indicator for the validity of each surfel
		DeviceArrayView<unsigned> validityIndicator;
		const unsigned int* validityIndicatorPrefixsum;
		const float4* surfelVertexConfidence;
		const float4* surfelNormalRadius;
		const float4* surfelColorTime;
		const ushort4* surfelKnn;
		const float4* surfelKnnWeight;
	};

	//The input to compactor for reinit
	struct ReinitAppendedObservationSurfel {
		DeviceArrayView<unsigned int> validityIndicator;
		const unsigned int* validityIndicatorPrefixsum;
		cudaTextureObject_t observedVertexMap[MAX_CAMERA_COUNT];
		cudaTextureObject_t observedNormalMap[MAX_CAMERA_COUNT];
		cudaTextureObject_t observedColorMap[MAX_CAMERA_COUNT];

		DeviceArrayView2D<float4> interVertexMap[MAX_CAMERA_COUNT];
		DeviceArrayView2D<float4> interNormalMap[MAX_CAMERA_COUNT];
		DeviceArrayView2D<float4> interColorMap[MAX_CAMERA_COUNT];

		mat34 initialCameraSE3[MAX_CAMERA_COUNT];
	};


	struct RemainingLiveSurfel {
		//The binary indicator for whether surfel i should remain
		DeviceArrayView<unsigned int> remainingIndicator;
		const unsigned int* remainingIndicatorPrefixsum;
		const float4* liveVertexConfidence;
		const float4* liveNormalRadius;
		const float4* colorTime;
	};

	struct RemainingSurfelKNN {
		const ushort4* surfelKnn;
		const float4* surfelKnnWeight;
	};

	struct RemainingLiveSurfelKNN {
		RemainingLiveSurfel liveGeometry;
		RemainingSurfelKNN remainingKnn;
	};
}

