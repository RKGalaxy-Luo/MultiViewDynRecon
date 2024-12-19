/*****************************************************************//**
 * \file   SurfelGeometry.h
 * \brief  ��Ԫ�ļ��ζ��󣬳��ܵ�
 * 
 * \author LUO
 * \date   February 1st 2024
 *********************************************************************/
#pragma once
#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/Constants.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>
#include <math/MatUtils.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace SparseSurfelFusion {
	namespace device {
		/**
		 * \brief �������Ԫ��Ϣ�ռ�������ֵ����ǰ��GeometrySurfel.
		 * 
		 * \param surfelArray ��õ���Ԫ��Ϣ
		 * \param canonicalVertexConfidence ��׼�򶥵㼰���Ŷ�
		 * \param canonicalNormalRadius ��׼���ּ���Ԫ�뾶
		 * \param liveVertexConfidence ʵʱ�򶥵㼰���Ŷ�
		 * \param liveNormalRadius ʵʱ���ּ��뾶
		 * \param colorTime ������ɫ����ʼ���ֺ͵�ǰ����ʱ��(֡��)
		 */
		__global__ void initializerCollectDepthSurfelKernel(DeviceArrayView<DepthSurfel> surfelArray, float4* canonicalVertexConfidence, float4* canonicalNormalRadius, float4* liveVertexConfidence, float4* liveNormalRadius, float4* colorTime/*,int*flag*/);


		__global__ void collectDepthSurfel(
			DeviceArrayHandle<DepthSurfel> surfelArray,
			DeviceArrayHandle<DepthSurfel> surfelArrayRef,
			float4* canonicalVertexConfidence,
			float4* canonicalNormalRadius,
			float4* liveVertexConfidence,
			float4* liveNormalRadius,
			float4* colorTime
		);

	}
	struct GLSurfelGeometryVBO; // ǰ������
	class SurfelNodeDeformer;
	class DoubleBufferCompactor;
	/**
	 * \brief ������Ԫ���νṹ.
	 */
	class SurfelGeometry {
	private:
		// �������Զ��ڴ���ж�д������������ӵ��(�������п��ٺ��ͷ�)
		DeviceSliceBufferArray<float4> CanonicalVertexConfidence;	// ��׼�ռ�Ķ��㼰���Ŷ�
		DeviceSliceBufferArray<float4> CanonicalNormalRadius;		// ��׼�ռ�ķ��߼��뾶
		DeviceSliceBufferArray<float4> LiveVertexConfidence;		// ʵʱ�ռ�Ķ��㼰���Ŷ�
		DeviceSliceBufferArray<float4> LiveNormalRadius;			// ʵʱ�ռ�ķ��߼��뾶
		DeviceSliceBufferArray<float4> ColorTime;					// ��ɫ��ʱ��

		friend struct GLSurfelGeometryVBO; // ��ͼ�ιܵ�(OpenGL��ӳ��)
		friend class SurfelNodeDeformer; //deform the vertex/normal given warp field
		friend class DoubleBufferCompactor; //compact from one buffer to another in double buffer setup
		// ӵ�и��ڴ棬�����Խ��ж�д�����ٺ��ͷ��ڴ�
		DeviceBufferArray<ushort4> surfelKNN;						// ��ԪKNN
		DeviceBufferArray<float4> surfelKNNWeight;					// ��ԪKNN��Ȩ��

		//�������������񻯵�buffer
		DeviceBufferArray<DepthSurfel> liveDepthSurfelBuffer;
		DeviceBufferArray<DepthSurfel> canonicalDepthSurfelBuffer;
		size_t validSurfelNum;
	public:
		using Ptr = std::shared_ptr<SurfelGeometry>;
		SurfelGeometry();
		~SurfelGeometry();

		NO_COPY_ASSIGN_MOVE(SurfelGeometry);

		/**
		 * \brief ������Ԫ����Ч����.
		 * 
		 * \return ��Ч��Ԫ������
		 */
		size_t ValidSurfelsNum() const { 
			return validSurfelNum; 
		}

		/**
		 * \brief ����������������Ĵ�С��Ԥ����Buffer����ֱ�ӱ���Ԥ����Buffer����array����buffer��ַ�����ҿ���һ��size��С��GPU(Array)����.
		 * 
		 * \param �������õ�Array��С
		 */
		void ResizeValidSurfelArrays(size_t size);

		/**
		 * \brief ʹ�ô������Ч����Ԫ����ʼ��Geometry.
		 * 
		 * \param validSurfelArray ��Ч����Ԫ����
		 * \param stream CUDA��ID
		 */
		void initGeometryFromCamera(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream = 0);


		//������surfelgeometryindexmap���������õ�
		//void updateSurfelGeometryIndexMap(SurfelGeometry::Ptr fusionsurfelgeometry);



		/**
		 * \brief ��ñ�׼���еĳ��ܶ����Լ����Ŷ�.
		 */
		DeviceArrayView<float4> GetCanonicalVertexConfidence() { return CanonicalVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getCanonicalVertexConfidence() { return CanonicalVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getLiveVertexConfidence() { return LiveVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getCanonicalNormalRadius() { return CanonicalNormalRadius.ArrayView(); }
		DeviceArrayView<float4> getLiveNormalRadius() { return LiveNormalRadius.ArrayView(); }
		DeviceArrayView<float4> getColorTime() { return ColorTime.ArrayView(); }
		
		/**
		 * \brief ��Ҫ�Ǽ�¼Geometry������(�����������ǿɶ�д��Array).
		 */
		struct GeometryAttributes {
			DeviceArrayHandle<float4> canonicalVertexConfidence;	// canonical��Ķ��㣬���Ŷ�
			DeviceArrayHandle<float4> canonicalNormalRadius;		// canonical��ķ��ߣ���Ԫ�뾶
			DeviceArrayHandle<float4> liveVertexConfidence;			// live��Ķ��㣬���Ŷ�
			DeviceArrayHandle<float4> liveNormalRadius;				// live��ķ��ߣ��뾶
			DeviceArrayHandle<float4> colorTime;					// ��ɫ -- ��һ֡������ʱ��
			//DeviceArrayHandle<int> flag;							// �����ĸ��ӽ�
		};
		
		/**
		 * \brief ������Ԫ���ԣ�GeometryAttributes�ڲ��������ǿɶ�д��ArrayHandle������Ҳ�൱��ָ�����ʽ.
		 */
		GeometryAttributes Geometry();

		/**
		 * \brief ��ʼ����Ԫ���Σ�����GeometryAttributes�д洢����Ԫ���Եĵ�ַ��ֱ�ӽ�surfelArray�е�ֵ����SurfelGeometry�е�����.
		 * 
		 * \param geometry ��Ԫ���ԣ�����������Ԫ˽�б�����ָ�룬����ֱ�Ӷ��ڴ���в���
		 * \param surfelArray ͨ�������CUDA�����õ���Ч�ĳ�����Ԫ����Ҫͨ��cuda���丳��������Ԫ˽�б�������
		 * \param CUDA��ID
		 */
		void initSurfelGeometry(GeometryAttributes geometry, const DeviceArrayView<DepthSurfel>& surfelArray, cudaStream_t stream = 0);


		/* The fusion handler will write to these elements, while maintaining
		* a indicator that which surfel should be compacted to another buffer.
		 */
		struct SurfelFusionInput {
			DeviceArrayHandle<float4> liveVertexConfidence;
			DeviceArrayHandle<float4> liveNormalRadius;
			DeviceArrayHandle<float4> colorTime;
			DeviceArrayView<ushort4> surfelKnn;
			DeviceArrayView<float4> surfelKnnWeight;
		};
		SurfelFusionInput SurfelFusionAccess();



		/**
		 * \brief ���ݴ���ṹ�壺���ͳ��ܵ����ݸ���Ƥ������¶��ַ���ɹ��޸ġ�.
		 */
		struct SkinnerInput {
			DeviceArrayView<float4> canonicalVerticesConfidence;	// �ɶ����ܵ�λ����Ϣ
			DeviceArrayHandle<ushort4> denseSurfelsKnn;				// ��д�������ԪKNN��������Ϣ
			DeviceArrayHandle<float4> denseSurfelsKnnWeight;		// ��д�������ԪKNN��Ȩ����Ϣ
		};
		/**
		 * \brief ��SkinnerInput��SurfelGeometry�У�����Ƥ�йص�KNN Index��Weight���ݣ�������SkinnerInput������ֱ��ӳ�䵽SurfelGeometry��Ӧ����.
		 * 
		 * \return ����Skinner���ݰ�
		 */
		SkinnerInput BindSurfelGeometrySkinnerInfo();




		/**
		 * \brief �Ǹ�������������룬����������Ԫ��KNN��Ȩ�أ�����ֻ�������Թ��任������¶��ַ.
		 */
		struct NonRigidSolverInput {
			DeviceArrayView<ushort4> surfelKnn;			// ��Ԫ��KNN
			DeviceArrayView<float4> surfelKnnWeight;	// ��ԪKNN��Ȩ��
		};
		/**
		 * \brief �󶨷Ǹ��Ա任���������.
		 * 
		 * \return ���ش���õķǸ��Ա任�Ĳ���
		 */
		NonRigidSolverInput BindNonRigidSolverInfo() const;

		struct OpticalFlowGuideInput {
			DeviceArray<float4> denseLiveSurfelsVertex;
			DeviceArray<float4> denseLiveSurfelsNormal;
			DeviceArray<ushort4> surfelKnn;			// ��Ԫ��KNN
			DeviceArray<float4> surfelKnnWeight;	// ��ԪKNN��Ȩ��
		};

		OpticalFlowGuideInput BindOpticalFlowGuideInfo();

		/**
		 * \brief indexmap��Ҫ�Ķ���
		 *
		 * \return ����ֻ����live�㡣
		 */

		DeviceArrayView<ushort4> SurfelKNNArray() const { return surfelKNN.ArrayView(); }

		//�����������񻯵�
		unsigned int collectLiveandCanDepthSurfel(cudaStream_t stream = 0);
		DeviceArrayView<DepthSurfel> getLiveDepthSurfels();
		DeviceArrayView<DepthSurfel> getCanonicalDepthSurfels();
		DeviceArray<DepthSurfel> getLiveDepthSurfelArrayPtr();
		DeviceArray<DepthSurfel> getCanonicalDepthSurfelArrayPtr();
	};


}


