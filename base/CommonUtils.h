/*****************************************************************//**
 * \file   CommonUtils.h
 * \brief  ��Ҫ�Ǽ�¼һЩ���ú����������ڴ���䣬���ٵ�
 * 
 * \author LUO
 * \date   January 9th 2024
 *********************************************************************/
#pragma once
#include "CommonTypes.h"
#include <cuda_texture_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>
#include <exception>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <base/DeviceReadWrite/DeviceBufferArray.h>


 //  �����Ͳ�����ʽ����/��ֵ
#define NO_COPY_ASSIGN(TypeName)                        \
    TypeName(const TypeName&) = delete;                 \
    TypeName& operator=(const TypeName&) = delete

// �����Ͳ�������ʽ����/��ֵ/�ƶ�
#define NO_COPY_ASSIGN_MOVE(TypeName)                   \
    TypeName(const TypeName&) = delete;                 \
    TypeName& operator=(const TypeName&) = delete;      \
    TypeName(TypeName&&) = delete;                      \
    TypeName& operator=(TypeName&&) = delete

// ����Ĭ���ƶ�
#define DEFAULT_MOVE(TypeName)                          \
	TypeName(TypeName&&) noexcept = default;            \
	TypeName& operator=(TypeName&&) noexcept = default

// �����͹��������������ΪĬ��
#define DEFAULT_CONSTRUCT_DESTRUCT(TypeName)            \
    TypeName() = default;                               \
    ~TypeName() = default

namespace SparseSurfelFusion {
	
    /**
     * \brief ��a��b������ͨ��ģ�庯��
     * \param a ��������a
     * \param b ��������b
     * \return 
     */
    template <typename T> // __forceinline__�ڱ����ʱ��ͰѺ����ŵ���Ӧ���õ�λ�ã��������㷨���̣������˱����ļ��Ĵ�С
    __host__ __device__ __forceinline__ void swap(T& a, T& b) noexcept
    {
        T c(a); a = b; b = c;
    }
    //hsg

#if defined(__CUDA_ARCH__) //�����������������豸֮��Ĵ���·��
    template<typename T>
    __device__ __forceinline__ T fetch1DLinear(cudaTextureObject_t texObj, int x) {
        return tex1Dfetch<T>(texObj, x);
    }

    template<typename T>
    __device__ __forceinline__ T fetch1DArray(cudaTextureObject_t texObj, float x) {
        return tex1D<T>(texObj, x);
    }
#else
    template<typename T>
    __host__ __forceinline__ T fetch1DLinear(cudaTextureObject_t texObj, int x) {
        throw new std::runtime_error("The texture object can only be accessed on device! ");
    }

    template<typename T>
    __host__ __forceinline__ T fetch1DArray(cudaTextureObject_t texObj, float x) {
        throw new std::runtime_error("The texture object can only be accessed on device! ");
    }
#endif

    __host__ __device__ __forceinline__
        unsigned int decodeVertexCameraView(const float4& colorViewTime) {
        for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
            if (i < colorViewTime.y && colorViewTime.y <= i + 1) {
                return i;
            }
        }
        return 0xFFFFFFFF;
    }

    __host__ __device__ __forceinline__
        unsigned int decodeSolverMapCameraView(const float4& colorViewTime) {
        for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
            if (i < colorViewTime.w && colorViewTime.w <= i + 1) {
                return i;
            }
        }
        return 0xFFFFFFFF;
    }

    __host__ __device__ __forceinline__
        unsigned int decodeFusionMapCameraView(const float4& colorViewTime) {
        for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
            if (i < colorViewTime.y && colorViewTime.y <= i + 1) {
                return i;
            }
        }
        return 0xFFFFFFFF;
    }

    __host__ __device__ __forceinline__
        unsigned int decodeNodesCameraView(const float4& nodeVertex) {
        for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
            if (i < nodeVertex.w && nodeVertex.w <= i + 1) {
                return i;
            }
        }
        return 0xFFFFFFFF;
    }

    __host__ __device__ __forceinline__
        unsigned int decodeCameraView(const float& cameraView) {
        for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
            if (i < cameraView && cameraView <= i + 1) {
                return i;
            }
        }
        return 0;
    }

    /**
     * \brief �ڲεĵ���
     * \param intrinsic ������ڲξ���
     * \return 
     */
    __host__ __forceinline__ IntrinsicInverse inverse(const Intrinsic& intrinsic) {
        IntrinsicInverse intr_inv;
        intr_inv.principal_x = intrinsic.principal_x;
        intr_inv.principal_y = intrinsic.principal_y;
        intr_inv.inv_focal_x = 1.0f / intrinsic.focal_x;
        intr_inv.inv_focal_y = 1.0f / intrinsic.focal_y;
        return intr_inv;
    }


    /**
     * \brief ��ʼ��cuda�����������ĺ����������API
     * \param selected_device ѡ��ʹ�õ�GPU��ID.
     */
    CUcontext initCudaContext(int selected_device = 0);



    /**
     * \brief �ڳ������ʱ���cuda������
     * \param context cuda������
     */
    void destroyCudaContext(CUcontext context);

/*************************************** ΪGPU����1ά��2ά�����ڴ� ***************************************/


    /**
     * \brief ����Ĭ�ϵ�2D���������ӣ�����ָ��cudaArray_tӦ���Ƕ���ͨ���������������͵���������.
     *
     * \param descriptor cuda����������.
     */
    void createDefault2DTextureDescriptor(cudaTextureDesc& descriptor);

    /**
     * \brief ����Ĭ�ϵ�2D��Դ�����ӣ�����ָ��cudaArray_tӦ���Ƕ���ͨ���������������͵���������.
     *
     * \param descriptor cuda��Դ������
     * \param cudaArray �������Դ����
     */
    void createDefault2DResourceDescriptor(cudaResourceDesc& descriptor, cudaArray_t& cudaArray);

    /**
     * \brief ����������������ڴ棬��������Ϊuint16(16λ�޷�������).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param texture ��Ҫ���䴴����ͼ������(����cudaTextureObject_t����)
     * \param cudaArray ��Ҫ����������������(����cudaArray_t����)
     */
    void createDepthTexture(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaArray_t& cudaArray);
    
    /**
      * \brief ��������������������ڴ棬��������Ϊuint16(16λ�޷�������).
      *
      * \param rows ͼ��ĸ�
      * \param cols ͼ��Ŀ�
      * \param texture ��Ҫ���䴴����ͼ������(����cudaTextureObject_t����)
      * \param surface ��Ҫ���䴴����ͼ�����(����cudaSurfaceObject_t����)
      * \param cudaArray ��Ҫ������������(����)����(����cudaArray_t����)
      */
    void createDepthTextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);
    
    /**
     * \brief ��������������������ڴ棬��������Ϊuint16(16λ�޷�������).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param collect ��Ҫ�����������������(����CudaTextureSurface����)
     */
    void createDepthTextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& collect);



    /**
     * \brief ��������2άFloat1���͵����������ڴ�(Ϊƽ����������2D float1����(�ͱ���)).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param texture ��Ҫ����������ڴ�(��cudaTextureObject_t��ʽ����)
     * \param surface ��Ҫ����ı����ڴ�(��cudaSurfaceObject_t��ʽ����)
     * \param cudaArray ���ݵĵ�ַ
     */
    void createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief ��������2άFloat1���͵����������ڴ�(Ϊƽ����������2D float1����(�ͱ���)).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param textureCollect ��Ҫ����������������ݵ�ַ(��CudaTextureSurface��ʽ����)
     */
    void createFloat1TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);

    /**
     * \brief ��������2άFloat2���͵����������ڴ�(Ϊ�ݶ�ͼ����2D float2����(�ͱ���)).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param texture ��Ҫ����������ڴ�(��cudaTextureObject_t��ʽ����)
     * \param surface ��Ҫ����ı����ڴ�(��cudaSurfaceObject_t��ʽ����)
     * \param cudaArray ���ݵĵ�ַ
     */
    void createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief ��������2άFloat2���͵����������ڴ�(Ϊ�ݶ�ͼ����2D float2����(�ͱ���)).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param textureCollect ��Ҫ����������������ݵ�ַ(��CudaTextureSurface��ʽ����)
     */
    void createFloat2TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);

    /**
     * \brief ��������2άUChar1���͵�������������(Ϊ������Maskͼ�񴴽�2D uchar1����(�ͱ���)).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param texture ��Ҫ����������ڴ�(��cudaTextureObject_t��ʽ����)
     * \param surface ��Ҫ����ı����ڴ�(��cudaSurfaceObject_t��ʽ����)
     * \param cudaArray ���ݵĵ�ַ
     */
    void createUChar1TextureSurface(const unsigned rows, const unsigned cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);

    /**
     * \brief ��������2άUChar1���͵�������������(Ϊ������Maskͼ�񴴽�2D uchar1����(�ͱ���)).
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param textureCollect ��Ҫ����������������ݵ�ַ(��CudaTextureSurface��ʽ����)
     */
    void createUChar1TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface& textureCollect);

    /**
     * \brief �����άͼ���float4��������cudaTextureObject_t�ͱ���cudaSurfaceObject_t�ڴ�.
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param texture ��Ҫ���ٵ������ڴ�
     * \param surface ��Ҫ���ٵı����ڴ�
     * \param cudaArray �������������(����)����
     */
    void createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);


    void createUnsignedTextureSurface(const unsigned int rows, const unsigned int cols, cudaTextureObject_t& texture, cudaSurfaceObject_t& surface, cudaArray_t& cudaArray);
    void createUnsignedTextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);



    /**
     * \brief �����άͼ���float4��������CudaTextureSurface�ڴ�.
     *
     * \param rows ͼ��ĸ�
     * \param cols ͼ��Ŀ�
     * \param textureCollect ��Ҫ���ٵ���������ڴ�
     */
    void createFloat4TextureSurface(const unsigned int rows, const unsigned int cols, CudaTextureSurface& textureCollect);

    /**
     * \brief �ͷ�2D����
     *
     * \param textureCollect ��Ҫ�ͷŵ�����(����CudaTextureSurface����)
     */
    void releaseTextureCollect(CudaTextureSurface& textureCollect);

    /**
     * \brief 2D�����ѯ��������ѯ2D Array��2D�����ڴ���2��ά���µĳߴ�(CUDA API��ʹ�ýṹ��cudaExtent����3D Array��3D�����ڴ�������ά���ϵĳߴ�)
     *
     * \param texture ��Ҫ��ѯ��2D����
     * \param width ������2ά�µĳߴ�(�˴����)
     * \param height ������2ά�µĳߴ�(�˴����)
     */
    void query2DTextureExtent(cudaTextureObject_t texture, unsigned int& width, unsigned int& height);

    template<typename T> class DeviceBufferArray;

    //hsg ԭ����common_texture_utils ���
    /**
    * \����һά���Ը�������ͨ��fetch1DLinear����,ʹ��������Ϊunderline�ڴ�
    */
    cudaTextureObject_t create1DLinearTexture(const DeviceArray<float>& array);

    cudaTextureObject_t create1DLinearTexture(const DeviceBufferArray<float>& array);

}