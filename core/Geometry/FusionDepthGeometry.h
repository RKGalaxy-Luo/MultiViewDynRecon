#pragma once
#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/Constants.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
namespace SparseSurfelFusion {
	struct GLfusionDepthSurfelVBO; // ǰ������
	class FusionDepthGeometry
	{
	private:
		// �������Զ��ڴ���ж�д������������ӵ��(�������п��ٺ��ͷ�)
		DeviceSliceBufferArray<float4> CanonicalVertexConfidence;	// ��׼�ռ�Ķ��㼰���Ŷ�
		DeviceSliceBufferArray<float4> CanonicalNormalRadius;		// ��׼�ռ�ķ��߼��뾶
		DeviceSliceBufferArray<float4> ColorTime;					// ��ɫ��ʱ��

		friend struct GLfusionDepthSurfelVBO; // ��ͼ�ιܵ�(OpenGL��ӳ��)
		size_t validSurfelNum;
	public:
		using Ptr = std::shared_ptr<FusionDepthGeometry>;
		FusionDepthGeometry();
		~FusionDepthGeometry();

		NO_COPY_ASSIGN_MOVE(FusionDepthGeometry);


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
		void initGeometryFromMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream = 0);

		DeviceArrayView<float4> getCanonicalVertexConfidence() { return CanonicalVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getcanonicalNormalRadius() { return CanonicalNormalRadius.ArrayView(); }
		DeviceArrayView<float4> getColorTime() { return ColorTime.ArrayView(); }

		/**
		 * \brief ��Ҫ�Ǽ�¼Geometry������(�����������ǿɶ�д��Array).
		 */
		struct FusionDepthSurfelGeometryAttributes {
			DeviceArrayHandle<float4> canonicalVertexConfidence;	// canonical��Ķ��㣬���Ŷ�
			DeviceArrayHandle<float4> canonicalNormalRadius;		// canonical��ķ��ߣ���Ԫ�뾶
			DeviceArrayHandle<float4> colorTime;					// ��ɫ -- ��һ֡������ʱ��
		};

		/**
		 * \brief ������Ԫ���ԣ�GeometryAttributes�ڲ��������ǿɶ�д��ArrayHandle������Ҳ�൱��ָ�����ʽ.
		 */
		FusionDepthSurfelGeometryAttributes Geometry();


		/**
		 * \brief ��ʼ����Ԫ���Σ�����GeometryAttributes�д洢����Ԫ���Եĵ�ַ��ֱ�ӽ�surfelArray�е�ֵ����SurfelGeometry�е�����.
		 *
		 * \param geometry ��Ԫ���ԣ�����������Ԫ˽�б�����ָ�룬����ֱ�Ӷ��ڴ���в���
		 * \param surfelArray ͨ�������CUDA�����õ���Ч�ĳ�����Ԫ����Ҫͨ��cuda���丳��������Ԫ˽�б�������
		 * \param CUDA��ID
		 */
		void initSurfelGeometry(FusionDepthSurfelGeometryAttributes geometry, const DeviceArrayView<DepthSurfel>& surfelArray, cudaStream_t stream = 0);
	};
}
