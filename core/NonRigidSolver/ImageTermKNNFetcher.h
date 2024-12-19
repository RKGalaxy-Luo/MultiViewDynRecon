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

		device::ImageKnnFetcherInterface imageKnnFetcherInterface;	// IndexMap��KnnMap�Ĵ���ӿ�

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
		//�������ڴ洢�����ӽǵļ�����
		//DeviceArray<unsigned> m_potential_pixel_indicator_single[MAX_CAMERA_COUNT];



	public:
		//This method, only collect pixel that has non-zero index map value
		//All these pixels are "potentially" matched with depth pixel with appropriate SE3
		void MarkPotentialMatchedPixels(cudaStream_t stream);

		//After all the potential pixels are marked
	private:
		PrefixSum m_indicator_prefixsum;
		DeviceBufferArray<ushort3> m_potential_pixels;//��Ҫ��ushort2�ĳ�3��z�������ĸ�indexmap
		DeviceBufferArray<ushort4> m_dense_image_knn;
		DeviceBufferArray<float4> m_dense_image_knn_weight;
		DeviceBufferArray<unsigned int> differenceViewOffset;//�������ѹ����������У���ͬindexmap�����ظ����Ƕ���
	public:
		void CompactPotentialValidPixels(cudaStream_t stream);

		void getDifferenceViewOffset(vector<unsigned> &diff);
		//������ۻ���ֵ����Ϊ�Ǵ��ۼ�����õ���
	private:
		unsigned* m_num_potential_pixel;
	public:
		void SyncQueryCompactedPotentialPixelSize(cudaStream_t stream = 0);


		//Accessing interface
	public:
		struct ImageTermPixelAndKNN {
			DeviceArrayView<ushort3> pixels;//2�ĳ���3
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