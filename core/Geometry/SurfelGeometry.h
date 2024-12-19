/*****************************************************************//**
 * \file   SurfelGeometry.h
 * \brief  面元的几何对象，稠密点
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
		 * \brief 将深度面元信息收集，并赋值给当前的GeometrySurfel.
		 * 
		 * \param surfelArray 获得的面元信息
		 * \param canonicalVertexConfidence 标准域顶点及置信度
		 * \param canonicalNormalRadius 标准域发现及面元半径
		 * \param liveVertexConfidence 实时域顶点及置信度
		 * \param liveNormalRadius 实时域发现及半径
		 * \param colorTime 顶点颜色及开始发现和当前发现时间(帧数)
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
	struct GLSurfelGeometryVBO; // 前向声明
	class SurfelNodeDeformer;
	class DoubleBufferCompactor;
	/**
	 * \brief 稠密面元几何结构.
	 */
	class SurfelGeometry {
	private:
		// 下述可以对内存进行读写操作，但并不拥有(即不进行开辟和释放)
		DeviceSliceBufferArray<float4> CanonicalVertexConfidence;	// 标准空间的顶点及置信度
		DeviceSliceBufferArray<float4> CanonicalNormalRadius;		// 标准空间的法线及半径
		DeviceSliceBufferArray<float4> LiveVertexConfidence;		// 实时空间的顶点及置信度
		DeviceSliceBufferArray<float4> LiveNormalRadius;			// 实时空间的法线及半径
		DeviceSliceBufferArray<float4> ColorTime;					// 颜色及时间

		friend struct GLSurfelGeometryVBO; // 从图形管道(OpenGL中映射)
		friend class SurfelNodeDeformer; //deform the vertex/normal given warp field
		friend class DoubleBufferCompactor; //compact from one buffer to another in double buffer setup
		// 拥有该内存，即可以进行读写，开辟和释放内存
		DeviceBufferArray<ushort4> surfelKNN;						// 面元KNN
		DeviceBufferArray<float4> surfelKNNWeight;					// 面元KNN的权重

		//这俩是用来网格化的buffer
		DeviceBufferArray<DepthSurfel> liveDepthSurfelBuffer;
		DeviceBufferArray<DepthSurfel> canonicalDepthSurfelBuffer;
		size_t validSurfelNum;
	public:
		using Ptr = std::shared_ptr<SurfelGeometry>;
		SurfelGeometry();
		~SurfelGeometry();

		NO_COPY_ASSIGN_MOVE(SurfelGeometry);

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
		void initGeometryFromCamera(const DeviceArrayView<DepthSurfel>& validSurfelArray, cudaStream_t stream = 0);


		//用来给surfelgeometryindexmap更新数据用的
		//void updateSurfelGeometryIndexMap(SurfelGeometry::Ptr fusionsurfelgeometry);



		/**
		 * \brief 获得标准域中的稠密顶点以及置信度.
		 */
		DeviceArrayView<float4> GetCanonicalVertexConfidence() { return CanonicalVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getCanonicalVertexConfidence() { return CanonicalVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getLiveVertexConfidence() { return LiveVertexConfidence.ArrayView(); }
		DeviceArrayView<float4> getCanonicalNormalRadius() { return CanonicalNormalRadius.ArrayView(); }
		DeviceArrayView<float4> getLiveNormalRadius() { return LiveNormalRadius.ArrayView(); }
		DeviceArrayView<float4> getColorTime() { return ColorTime.ArrayView(); }
		
		/**
		 * \brief 主要是记录Geometry的属性(传出的属性是可读写的Array).
		 */
		struct GeometryAttributes {
			DeviceArrayHandle<float4> canonicalVertexConfidence;	// canonical域的顶点，置信度
			DeviceArrayHandle<float4> canonicalNormalRadius;		// canonical域的法线，面元半径
			DeviceArrayHandle<float4> liveVertexConfidence;			// live域的顶点，置信度
			DeviceArrayHandle<float4> liveNormalRadius;				// live域的法线，半径
			DeviceArrayHandle<float4> colorTime;					// 颜色 -- 上一帧看到的时间
			//DeviceArrayHandle<int> flag;							// 来自哪个视角
		};
		
		/**
		 * \brief 返回面元属性，GeometryAttributes内部包含的是可读写的ArrayHandle，传入也相当于指针的形式.
		 */
		GeometryAttributes Geometry();

		/**
		 * \brief 初始化面元几何，传入GeometryAttributes中存储的面元属性的地址，直接将surfelArray中的值传给SurfelGeometry中的属性.
		 * 
		 * \param geometry 面元属性，包含上述面元私有变量的指针，可以直接对内存进行操作
		 * \param surfelArray 通过纹理和CUDA计算获得的有效的稠密面元，需要通过cuda将其赋给上述面元私有变量属性
		 * \param CUDA流ID
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
		 * \brief 数据打包结构体：发送稠密点数据给蒙皮器【暴露地址，可供修改】.
		 */
		struct SkinnerInput {
			DeviceArrayView<float4> canonicalVerticesConfidence;	// 可读稠密点位置信息
			DeviceArrayHandle<ushort4> denseSurfelsKnn;				// 可写入稠密面元KNN点索引信息
			DeviceArrayHandle<float4> denseSurfelsKnnWeight;		// 可写入稠密面元KNN点权重信息
		};
		/**
		 * \brief 绑定SkinnerInput与SurfelGeometry中，与蒙皮有关的KNN Index和Weight数据，后续对SkinnerInput操作可直接映射到SurfelGeometry相应变量.
		 * 
		 * \return 返回Skinner数据包
		 */
		SkinnerInput BindSurfelGeometrySkinnerInfo();




		/**
		 * \brief 非刚性求解器的输入，包含稠密面元的KNN和权重，传入只读数据以供变换，不暴露地址.
		 */
		struct NonRigidSolverInput {
			DeviceArrayView<ushort4> surfelKnn;			// 面元的KNN
			DeviceArrayView<float4> surfelKnnWeight;	// 面元KNN的权重
		};
		/**
		 * \brief 绑定非刚性变换的输入参数.
		 * 
		 * \return 返回打包好的非刚性变换的参数
		 */
		NonRigidSolverInput BindNonRigidSolverInfo() const;

		struct OpticalFlowGuideInput {
			DeviceArray<float4> denseLiveSurfelsVertex;
			DeviceArray<float4> denseLiveSurfelsNormal;
			DeviceArray<ushort4> surfelKnn;			// 面元的KNN
			DeviceArray<float4> surfelKnnWeight;	// 面元KNN的权重
		};

		OpticalFlowGuideInput BindOpticalFlowGuideInfo();

		/**
		 * \brief indexmap需要的东西
		 *
		 * \return 返回只读的live点。
		 */

		DeviceArrayView<ushort4> SurfelKNNArray() const { return surfelKNN.ArrayView(); }

		//这是用来网格化的
		unsigned int collectLiveandCanDepthSurfel(cudaStream_t stream = 0);
		DeviceArrayView<DepthSurfel> getLiveDepthSurfels();
		DeviceArrayView<DepthSurfel> getCanonicalDepthSurfels();
		DeviceArray<DepthSurfel> getLiveDepthSurfelArrayPtr();
		DeviceArray<DepthSurfel> getCanonicalDepthSurfelArrayPtr();
	};


}


