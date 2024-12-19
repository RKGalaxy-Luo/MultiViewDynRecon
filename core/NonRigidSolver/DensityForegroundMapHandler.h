#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

#include <core/AlgorithmTypes.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include "solver_types.h"
#include "ImageTermKNNFetcher.h"
#include <memory>
#include <render/Renderer.h>
#include <vector>

namespace SparseSurfelFusion {
	namespace device {

		struct DepthObservationForegroundInterface {
			cudaTextureObject_t foregroundMask[MAX_CAMERA_COUNT];				// uchar1 texture
			cudaTextureObject_t filteredForegroundMask[MAX_CAMERA_COUNT];		// float1 texture
			cudaTextureObject_t foregroundMaskGradientMap[MAX_CAMERA_COUNT];	// float2 texture

			cudaTextureObject_t densityMap[MAX_CAMERA_COUNT];					// float1 texture
			cudaTextureObject_t densityGradientMap[MAX_CAMERA_COUNT];			// float2 texture
		};

		struct GeometryMapForegroundInterface {
			Intrinsic intrinsic[MAX_CAMERA_COUNT];								// 内参矩阵
			mat34 world2Camera[MAX_CAMERA_COUNT];
			mat34 initialCameraSE3[MAX_CAMERA_COUNT];							// 初始化相机位姿矩阵
			mat34 initialCameraSE3Inverse[MAX_CAMERA_COUNT];

			cudaTextureObject_t referenceVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t referenceNormalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t indexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t normalizedRgbMap[MAX_CAMERA_COUNT];
			DeviceArrayView2D<KNNAndWeight> knnMap[MAX_CAMERA_COUNT];
		};

		__device__ unsigned int CalculateForegroundMapCameraViewFgr(const unsigned int idx, const unsigned int devicesCount);
		
		__device__ unsigned int CalculateDensityMapCameraViewFgr(const unsigned int idx, const unsigned int devicesCount);
		
		__global__ void computeDensityMapJacobian(
			DepthObservationForegroundInterface depthObservationForegroundInterface,
			GeometryMapForegroundInterface geometryMapForegroundInterface,
			const unsigned int width, const unsigned int height, const unsigned int devicesCount, const unsigned int totalTermsNum,
			//The queried pxiels and their weights
			const DeviceArrayView<ushort3> densityTermPixels,
			const ushort4* densityTermKnn,
			const float4* densityTermKnnWeight,
			//The warp field information
			const DualQuaternion* deviceWarpField,
			//Output
			TwistGradientOfScalarCost* gradient,
			float* residualArray);

		__global__ void computeForegroundMaskJacobian(
			DepthObservationForegroundInterface	depthObservationForegroundInterface,
			GeometryMapForegroundInterface geometryMapForegroundInterface,
			const unsigned int width, const unsigned int height, const unsigned int totalMaskPixels, const unsigned int devicesCount,
			//The queried pxiels and their weights
			const DeviceArrayView<ushort3> foreground_term_pixels,
			const ushort4* foreground_term_knn,
			const float4* foreground_term_knn_weight,
			//The warp field information
			const DualQuaternion* device_warp_field,
			//Output
			TwistGradientOfScalarCost* gradient,
			float* residual_array);


	};
	class DensityForegroundMapHandler {
	private:
		//The info from config
		int m_image_height;
		int m_image_width;
		unsigned int devicesCount = MAX_CAMERA_COUNT;
		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];
		device::DepthObservationForegroundInterface depthObservationForegroundInterface;	// 当前帧观察到的数据参数打包
		device::GeometryMapForegroundInterface geometryMapForegroundInterface;				// 重建集合体的参数打包

		//The info from solver
		DeviceArrayView<DualQuaternion> m_node_se3;

		//The pixel from the indexer
		ImageTermKNNFetcher::ImageTermPixelAndKNN m_potential_pixels_knn;

	public:
		using Ptr = std::shared_ptr<DensityForegroundMapHandler>;
		DensityForegroundMapHandler();
		~DensityForegroundMapHandler() = default;
		NO_COPY_ASSIGN_MOVE(DensityForegroundMapHandler);

		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();


		//Set input
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3,
			DeviceArray2D<KNNAndWeight>* knn_map,
			//The foreground mask terms
			cudaTextureObject_t* foreground_mask,
			cudaTextureObject_t* filtered_foreground_mask,
			cudaTextureObject_t* foreground_gradient_map,
			//The color density terms
			cudaTextureObject_t* density_map,
			cudaTextureObject_t* density_gradient_map,
			cudaTextureObject_t* normalized_rgb_map,
			mat34* world2camera,
			Intrinsic* Clipcolor,
			Renderer::SolverMaps* solver,
			//The potential pixels,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN& potential_pixels_knn
		);

		//Update the node se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);


		//The finder interface
		void FindValidColorForegroundMaskPixels(cudaStream_t color_stream = 0, cudaStream_t mask_stream = 0);//4.19这个函数里边要改以下，这函数还没用
		void FindPotentialForegroundMaskPixelSynced(cudaStream_t stream = 0);//4.19这个函数里边要改以下，这函数还没用


		/* Mark the valid pixel for both color and foreground mask
		 */
	private:
		//These should be 2D maps
		DeviceArray<unsigned> m_color_pixel_indicator_map;
		DeviceArray<unsigned> m_mask_pixel_indicator_map;
	public:
		void MarkValidColorForegroundMaskPixels(cudaStream_t stream);


		/* The compaction for maps
		 */
	private:
		PrefixSum m_color_pixel_indicator_prefixsum;
		PrefixSum m_mask_pixel_indicator_prefixsum;
		DeviceBufferArray<ushort3> m_valid_color_pixel, m_valid_mask_pixel;
		DeviceBufferArray<ushort4> m_valid_color_pixel_knn, m_valid_mask_pixel_knn;
		DeviceBufferArray<float4> m_valid_color_pixel_knn_weight, m_valid_mask_pixel_knn_weight;
		DeviceBufferArray<unsigned int> differentViewsForegroundMapOffset;

		//The pagelocked memory
		unsigned* m_num_mask_pixel;
	public:
		void getDiffoffsetDensityForegroundMapHandler(std::vector<unsigned>& diff);
		void CompactValidColorPixel(cudaStream_t stream = 0);
		void QueryCompactedColorPixelArraySize(cudaStream_t stream = 0);
		void CompactValidMaskPixel(cudaStream_t stream );
		void QueryCompactedMaskPixelArraySize(cudaStream_t stream = 0);


		/* Compute the gradient
		 */
	private:
		DeviceBufferArray<float> m_color_residual;
		DeviceBufferArray<TwistGradientOfScalarCost> m_color_twist_gradient;
		DeviceBufferArray<float> m_foreground_residual;
		DeviceBufferArray<TwistGradientOfScalarCost> m_foreground_twist_gradient;
		void computeDensityMapTwistGradient(std::vector<unsigned int> differenceOffsetImageKnnFetcher, cudaStream_t stream);
		void computeForegroundMaskTwistGradient(std::vector<unsigned int> differenceOffsetForegroundHandler, cudaStream_t stream);
	public:
		void ComputeTwistGradient(std::vector<unsigned> diffoffsetImage, vector<unsigned> diffoffsetForeground, cudaStream_t colorStream, cudaStream_t foregroundStream);
		void Term2JacobianMaps(
			DensityMapTerm2Jacobian& density_term2jacobian,
			ForegroundMaskTerm2Jacobian& foreground_term2jacobian
		);


		/* The access interface
		 */
	public:
		DeviceArrayView<ushort4> DensityTermKNN() const { return m_valid_color_pixel_knn.ArrayView(); }
		DeviceArrayView<ushort4> ForegroundMaskTermKNN() const { return m_valid_mask_pixel_knn.ArrayView(); }
	};

}