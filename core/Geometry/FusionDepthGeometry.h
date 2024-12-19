#pragma once
#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/Constants.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
namespace SparseSurfelFusion {
	struct GLfusionDepthSurfelVBO; // 前向声明
	class FusionDepthGeometry
	{
	private:
		// 下述可以对内存进行读写操作，但并不拥有(即不进行开辟和释放)
		DeviceSliceBufferArray<float4> CanonicalVertexConfidence;	// 标准空间的顶点及置信度
		DeviceSliceBufferArray<float4> CanonicalNormalRadius;		// 标准空间的法线及半径
		DeviceSliceBufferArray<float4> ColorTime;					// 颜色及时间

		friend struct GLfusionDepthSurfelVBO; // 从图形管道(OpenGL中映射)
		size_t validSurfelNum;
	public:
		using Ptr = std::shared_ptr<FusionDepthGeometry>;
		FusionDepthGeometry();
		~FusionDepthGeometry();

		NO_COPY_ASSIGN_MOVE(FusionDepthGeometry);


		/**
		 * \brief 设置面元的有效数量.
		 *
		 * \return 有效面元的数量
		 */
		size_t ValidSurfelsNum() const {
			return validSurfelNum;
		}

		/**
		 * \brief 调整所有容器数组的大小：预分配Buffer不够直接报错，预分配Buffer够则将array赋上buffer地址，并且开辟一个size大小的GPU(Array)缓存.
		 *
		 * \param 所需设置的Array大小
		 */
		void ResizeValidSurfelArrays(size_t size);

		/**
		 * \brief 使用处理后有效的面元，初始化Geometry.
		 *
		 * \param validSurfelArray 有效的面元数组
		 * \param stream CUDA流ID
		 */
		void initGeometryFromMergedDepthSurfel(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream = 0);

		DeviceArrayView<float4> getCanonicalVertexConfidence() { return CanonicalVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getcanonicalNormalRadius() { return CanonicalNormalRadius.ArrayView(); }
		DeviceArrayView<float4> getColorTime() { return ColorTime.ArrayView(); }

		/**
		 * \brief 主要是记录Geometry的属性(传出的属性是可读写的Array).
		 */
		struct FusionDepthSurfelGeometryAttributes {
			DeviceArrayHandle<float4> canonicalVertexConfidence;	// canonical域的顶点，置信度
			DeviceArrayHandle<float4> canonicalNormalRadius;		// canonical域的法线，面元半径
			DeviceArrayHandle<float4> colorTime;					// 颜色 -- 上一帧看到的时间
		};

		/**
		 * \brief 返回面元属性，GeometryAttributes内部包含的是可读写的ArrayHandle，传入也相当于指针的形式.
		 */
		FusionDepthSurfelGeometryAttributes Geometry();


		/**
		 * \brief 初始化面元几何，传入GeometryAttributes中存储的面元属性的地址，直接将surfelArray中的值传给SurfelGeometry中的属性.
		 *
		 * \param geometry 面元属性，包含上述面元私有变量的指针，可以直接对内存进行操作
		 * \param surfelArray 通过纹理和CUDA计算获得的有效的稠密面元，需要通过cuda将其赋给上述面元私有变量属性
		 * \param CUDA流ID
		 */
		void initSurfelGeometry(FusionDepthSurfelGeometryAttributes geometry, const DeviceArrayView<DepthSurfel>& surfelArray, cudaStream_t stream = 0);
	};
}
