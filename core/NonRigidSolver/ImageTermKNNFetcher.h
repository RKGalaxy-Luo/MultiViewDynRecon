#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <core/AlgorithmTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
//#include "surfel_types.h"
#include "solver_types.h"
#include <base/data_transfer.h>
#include <vector>

#include <memory>

namespace SparseSurfelFusion {
	namespace device {
		struct ImageKnnFetcherInterface {
			cudaTextureObject_t IndexMap[MAX_CAMERA_COUNT];
			DeviceArrayView2D<KNNAndWeight> KnnMap[MAX_CAMERA_COUNT];
		};
	};

	class ImageTermKNNFetcher {
		//The info from config
		unsigned int m_image_height;
		unsigned int m_image_width;
		unsigned int devicesCount = MAX_CAMERA_COUNT;

		device::ImageKnnFetcherInterface imageKnnFetcherInterface;	// IndexMap和KnnMap的打包接口

	public:
		//The contructor group
		using Ptr = std::shared_ptr<ImageTermKNNFetcher>;
		ImageTermKNNFetcher();
		~ImageTermKNNFetcher();
		NO_COPY_ASSIGN_MOVE(ImageTermKNNFetcher);

		//The input from the solver
		void SetInputs(
			DeviceArray2D<KNNAndWeight> *knn_map,
			Renderer::SolverMaps* solverMap
		);

		/* The main processor methods for mark all potential valid pixels
		 */
	private:
		//A fixed size array to indicator the pixel validity
		DeviceArray<unsigned> m_potential_pixel_indicator;
		//这是用于存储单个视角的计算结果
		//DeviceArray<unsigned> m_potential_pixel_indicator_single[MAX_CAMERA_COUNT];



	public:
		//This method, only collect pixel that has non-zero index map value
		//All these pixels are "potentially" matched with depth pixel with appropriate SE3
		void MarkPotentialMatchedPixels(cudaStream_t stream);

		//After all the potential pixels are marked
	private:
		PrefixSum m_indicator_prefixsum;
		DeviceBufferArray<ushort3> m_potential_pixels;//需要从ushort2改成3，z是来自哪个indexmap
		DeviceBufferArray<ushort4> m_dense_image_knn;
		DeviceBufferArray<float4> m_dense_image_knn_weight;
		DeviceBufferArray<unsigned int> differenceViewOffset;//用来获得压缩后的数组中，不同indexmap的像素个数是多少
	public:
		void CompactPotentialValidPixels(cudaStream_t stream);

		void getDifferenceViewOffset(vector<unsigned> &diff);
		//这个是累积的值，因为是从累计数组得到的
	private:
		unsigned* m_num_potential_pixel;
	public:
		void SyncQueryCompactedPotentialPixelSize(cudaStream_t stream = 0);


		//Accessing interface
	public:
		struct ImageTermPixelAndKNN {
			DeviceArrayView<ushort3> pixels;//2改成了3
			DeviceArrayView<ushort4> node_knn;
			DeviceArrayView<float4> knn_weight;
		};
		ImageTermPixelAndKNN GetImageTermPixelAndKNN() const {
			ImageTermPixelAndKNN output;
			output.pixels = m_potential_pixels.ArrayReadOnly();
			output.node_knn = m_dense_image_knn.ArrayReadOnly();
			output.knn_weight = m_dense_image_knn_weight.ArrayReadOnly();
			return output;
		}
		DeviceArrayView<ushort4> DenseImageTermKNNArray() const { return m_dense_image_knn.ArrayView(); }


		//Sanity check
		void CheckDenseImageTermKNN(const DeviceArrayView<ushort4>& dense_depth_knn_gpu);

		//hsg debug
		void debugPotentialPixelIndicator(vector<unsigned> hostpotentialpixelindicator);
		void debugPotentialPixels(vector<ushort3> host);
		void debugDenseImageKnn(vector<ushort4> host);
		void debugdifferenceviewoffset(vector<unsigned> host);
		void debugoutputImageTermKNN(vector<unsigned> host1, vector<ushort3> host2, vector<ushort4> host3, vector<unsigned> host4);
	};


}