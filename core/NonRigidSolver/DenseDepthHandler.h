#pragma once
#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <core/AlgorithmTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <math/DualQuaternion/DualQuaternion.h>
#include "solver_types.h"
#include "ImageTermKNNFetcher.h"
#include <memory>
#include <render/Renderer.h>
#include <base/Constants.h>
#include "WarpField.h"



//4.13 ��һ�α���ʱ��ע�͵���freeindex.cu�е�������
//ע�͵���void ComputeAlignmentErrorMapFromNode


namespace SparseSurfelFusion {
	namespace device {
		struct ObservationDenseDepthHandlerInterface {
			cudaTextureObject_t vertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t normalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t filteredForegroundMap[MAX_CAMERA_COUNT];
		};

		struct GeometryMapDenseDepthHandlerInterface {
			cudaTextureObject_t referenceVertexMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t referenceNormalMap[MAX_CAMERA_COUNT];
			cudaTextureObject_t indexMap[MAX_CAMERA_COUNT];
			mat34 world2Camera[MAX_CAMERA_COUNT];
			mat34 camera2World[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
			mat34 InitialCameraSE3Inverse[MAX_CAMERA_COUNT];
			DeviceArrayView2D<KNNAndWeight> knnMap[MAX_CAMERA_COUNT];
			Intrinsic intrinsic[MAX_CAMERA_COUNT];
		};

		struct NodeAccumulatedErrorAndWeight {
			CudaTextureSurface alignmentErrorMap[MAX_CAMERA_COUNT];	// �������ͼ

		};

		__device__ unsigned int CalculateDensityMapCameraViewDpt(const unsigned int idx, const unsigned int devicesCount);


		//This version needs to first check whether the given vertex will result in a match. If there is
		//a correspondence, the fill in the jacobian and residual values, else mark the value to zero. 
		//The input is only depended on the SE(3) of the nodes, which can be updated without rebuild the index

		/**
		 * \brief ����汾��Ҫ���ȼ������Ķ����Ƿ�����ƥ�䡣����ж�Ӧ��ϵ���������ſɱȾ���Ͳв�ֵ������ֵ���Ϊ�㡣����������ڽڵ��SE(3)�������ڲ��ؽ�����������¸���SE.
		 */
		__global__ void computeDenseDepthJacobianKernel(
			ObservationDenseDepthHandlerInterface observedDenseDepthHandlerInterface,
			GeometryMapDenseDepthHandlerInterface geometryDenseDepthHandlerInterface,
			const unsigned int imgRows,
			const unsigned int imgCols,
			const unsigned int totalPotentialPixels,
			const unsigned int devicesCount,
			//The potential matched pixels and their knn
			DeviceArrayView<ushort3> potentialMatchedPixels,
			const ushort4* potentialMatchedKnn,
			const float4* potentialMatchedKnnWeight,
			//The deformation
			const DualQuaternion* nodeSE3,
			//The output
			TwistGradientOfScalarCost* twistGradient,
			float* residual
		);
	};

	class DenseDepthHandler {
	private:
		//The info from config
		int m_image_height = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;;
		int m_image_width = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
		//Intrinsic m_project_intrinsic[MAX_CAMERA_COUNT];

		unsigned int devicesCount = MAX_CAMERA_COUNT;

		mat34 InitialCameraSE3[MAX_CAMERA_COUNT];
		mat34 InitialCameraSE3Inv[MAX_CAMERA_COUNT];
		//The info from solver
		DeviceArrayView<DualQuaternion> m_node_se3;

		device::ObservationDenseDepthHandlerInterface observedDenseDepthHandlerInterface;	// �۲쵽�Ĳ���
		device::GeometryMapDenseDepthHandlerInterface geometryDenseDepthHandlerInterface;	// �ؽ��������

		//The info from image term fetcher
		ImageTermKNNFetcher::ImageTermPixelAndKNN m_potential_pixels_knn;

	public:
		using Ptr = std::shared_ptr<DenseDepthHandler>;
		DenseDepthHandler();
		~DenseDepthHandler() = default;
		NO_COPY_ASSIGN_MOVE(DenseDepthHandler);

		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();

		//Set input
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3,
			DeviceArray2D<KNNAndWeight>* knnMap,
			cudaTextureObject_t* depthVertexMap,
			cudaTextureObject_t* depthNormalMap,
			cudaTextureObject_t* foregroundMap,
			Renderer::SolverMaps* solverMap,
			mat34* world2camera,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN& pixelsKnn
		);

		//�����ʱ��ǵ��������
		void SetInitalData(Intrinsic * intr);


		//Update the se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);

		//The processing interface for free-index solver
		void FindCorrespondenceSynced(cudaStream_t stream = 0);
	private:
		//These two should be 2D maps, flatten as the compaction is required
		DeviceArray<ushort2> m_pixel_pair_maps; //The matched depth pixel pairs, used in index free version
		DeviceArray<unsigned> m_pixel_match_indicator; //The indicator for surfel pixel, used in both version
	private:
		//The method will project the model surfel into depth image, check the
		//correspondence between them and mark corresponded surfel pixels
		void MarkMatchedPixelPairs(cudaStream_t stream = 0);

		/* Compact the pixel pairs
		 */
	private:
		PrefixSum m_indicator_prefixsum;
		DeviceBufferArray<ushort4> m_valid_pixel_pairs;
		DeviceBufferArray<ushort4> m_dense_depth_knn;
		DeviceBufferArray<float4> m_dense_depth_knn_weight;
	private:
		void CompactMatchedPixelPairs(cudaStream_t stream = 0);
		void compactedPairSanityCheck(DeviceArrayView<ushort4> surfel_knn_array);
	public:
		void SyncQueryCompactedArraySize(cudaStream_t stream = 0);


		/* Compute the twist jacobian
		 */
	private:
		DeviceBufferArray<float> m_term_residual;
		DeviceBufferArray<TwistGradientOfScalarCost> m_term_twist_gradient;
	public:
		void ComputeJacobianTermsFreeIndex(cudaStream_t stream = 0);
		void ComputeJacobianTermsFixedIndex(std::vector<unsigned int> differenceOffsetImageKnnFetcher, cudaStream_t stream);
		DenseDepthTerm2Jacobian Term2JacobianMap() const;



		/* Compute the residual map and gather them into nodes. Different from previous residual
		 * The map will return non-zero value at valid pixels that doesnt have corresponded depth pixel
		 * The method is used in Reinit pipeline and visualization.
		 */
	private:
		device::NodeAccumulatedErrorAndWeight nodeAccumulatedErrorAndWeight;		// ��������������m_node_accumlate_error_temp��m_node_accumlate_weight_temp�׵�ַ��
	public:
		void ComputeAlignmentErrorMapDirect(const DeviceArrayView<DualQuaternion>& node_se3, cudaStream_t stream);
		cudaTextureObject_t GetAlignmentErrorMap(const unsigned int CameraID) const { return nodeAccumulatedErrorAndWeight.alignmentErrorMap[CameraID].texture; }


		/* Compute the error and accmulate them on nodes. May distribute them again on
		 * map for further use or visualization
		 */
	private:
		DeviceBufferArray<float> nodeAccumulatedError;		// �����ܵ�����(fabsf_diff_xyz * knnWeight)ȫ���ۼӵ����ھӽڵ�Node
		DeviceBufferArray<float> nodeAccumulatedWeight;		// ��������ۼƵĳ��ܵ������Ȩ��֮��
		DeviceBufferArray<float> nodeUnitedAlignmentError;

		DeviceBufferArray<unsigned int> nodeLargeNodeErrorNum;
		//Distribute the node error on maps
		void distributeNodeErrorOnMap(cudaStream_t stream = 0);
	public:

		/**
		 * \brief ��ýڵ�Ĺ�һ���ۻ����.
		 * 
		 * \return �ڵ�Ĺ�һ���ۻ����.
		 */
		DeviceArrayView<float> GetNodeUnitAccumulatedError() { return nodeUnitedAlignmentError.ArrayView(); }



		/**
		 * \brief ����ڵ㷶Χ�ڵ��������ڵ���۲쵽�����ͼ���жԱȣ��۲����.
		 * 
		 * \param node_se3 �Ǹ������Ľڵ�SE3
		 * \param stream cuda��
		 */
		void ComputeNodewiseError(const DeviceArrayView<DualQuaternion>& node_se3, bool printNodeError = false, cudaStream_t stream = 0);

		/**
		 * \brief ���ݽڵ���������ڵ�SE3�Լ�Live��ĵ�.
		 * 
		 * \param warpField Ť����
		 */
		void CorrectWarpFieldSE3(WarpField& warpField);

		/**
		 * \brief �ӽڵ�����л�ö������.
		 * 
		 * \param node_se3 �ڵ�Ǹ��Զ���se3
		 * \param stream cuda��
		 */
		void ComputeAlignmentErrorMapFromNode(const DeviceArrayView<DualQuaternion>& node_se3, cudaStream_t stream = 0);


		/* Accessing interface
		 */
	public:
		//The nodewise error
		NodeAlignmentError GetNodeAlignmentError() const {
			NodeAlignmentError error;
			error.nodeAccumlatedError = nodeAccumulatedError.ArrayView();
			error.nodeAccumlateWeight = nodeAccumulatedWeight.Ptr();
			return error;
		}
	};
}