/*****************************************************************//**
 * \file   device_intrinsics.h
 * \brief  ��������һ���߳����еĿ��ٹ�Լ�ӷ������ͷ�ļ�ֻ����.cu�ļ��а�������Ϊ��������PTX�����ʱ����߱�������Ҫ�Ż���λ�����
 * 
 * \author LUO
 * \date   February 4th 2024
 *********************************************************************/
#pragma once
#include <vector_functions.h>

namespace SparseSurfelFusion {
        /**
         * \brief ɨ�����(float����).
         * 
         * \param x ��ǰ�߳����е�һ���̵߳�����
         * \param offset �߳����е�ƫ����(bitλ����)
         * \return ͬһ���߳���������ӵĽ��
         */
        __device__ __forceinline__ float shfl_add(float x, int offset) {
            float result = 0;
            //__asm__ ��asm��������һ�����������ʽ���κ����������ʽ����������ͷ���ز�����
            //volatile �ǿ�ѡ�ģ�������������������GCC ��������Ӧ�Ը���������Ż�
            asm volatile (
                "{.reg .f32 r0;"
                ".reg .pred p;"
                "shfl.sync.up.b32 r0|p, %1, %2, 0, 0;"
                "@p add.f32 r0, r0, %1;"
                "mov.f32 %0, r0;}"
                : "=f"(result) : "f"(x), "r"(offset));
            //result = __shfl_up_sync(0xffffffff, x, offset, 32);

            return result;
        }


    //__device__ __forceinline__
    //    float shfl_add(float x, int offset)
    //{
    //    float result = 0;

    //    // ʹ��ԭ�Ӳ����滻
    //    int* address_as_int = reinterpret_cast<int*>(&result);
    //    int old = atomic_add_sync(address_as_int, __float_as_int(x));

    //    return result;
    //}




    /**
     * \brief �߳�����ɨ�����(float����).
     * 
     * \param data �߳�����һ������(bitλ�������)
     * \return ͬһ���߳���������ӵĽ��
     */
     __device__ __forceinline__ float warp_scan(float data) {
         data = shfl_add(data, 1);
         data = shfl_add(data, 2);
         data = shfl_add(data, 4);
         data = shfl_add(data, 8);
         data = shfl_add(data, 16);
         return data;
    }

    /**
     * \brief ɨ�����(int ����).
     * 
     * \param x ��ǰ�߳����е�һ���̵߳�����
     * \param offset �߳����е�ƫ����(bitλ����)
     * \return ͬһ���߳���������ӵĽ��
     */
    __device__ __forceinline__ int shfl_add(int x, int offset)
    {
        int result = 0;
        asm volatile (
            "{.reg .s32 r0;"
            ".reg .pred p;"
            "shfl.sync.up.b32 r0|p, %1, %2, 0, 0;"
            "@p add.s32 r0, r0, %1;"
            "mov.s32 %0, r0;}"
            : "=r"(result) : "r"(x), "r"(offset));


        return result;
    }

    
    __device__ __forceinline__ int warp_scan(int data)
    {
        data = shfl_add(data, 1);
        data = shfl_add(data, 2);
        data = shfl_add(data, 4);
        data = shfl_add(data, 8);
        data = shfl_add(data, 16);
        return data;
    }
     //GPT 
     //__device__ __forceinline__
     //    int warp_scan(int data)
     //{
     //    int lane_id = threadIdx.x % warpSize;
     //    int mask = 0xffffffff << lane_id;

     //    // �𲽺ϲ����
     //    data += __shfl_up_sync(mask, data, 1);
     //    data += __shfl_up_sync(mask, data, 2);
     //    data += __shfl_up_sync(mask, data, 4);
     //    data += __shfl_up_sync(mask, data, 8);
     //    data += __shfl_up_sync(mask, data, 16);

     //    return data;
     //}

     //__device__ __forceinline__
     //    float warp_scan(float data)
     //{
     //    int lane_id = threadIdx.x % warpSize;
     //    int mask = 0xffffffff << lane_id;

     //    // �𲽺ϲ����
     //    data += __shfl_up_sync(mask, data, 1);
     //    data += __shfl_up_sync(mask, data, 2);
     //    data += __shfl_up_sync(mask, data, 4);
     //    data += __shfl_up_sync(mask, data, 8);
     //    data += __shfl_up_sync(mask, data, 16);

     //    return data;
     //}

     //__device__ __forceinline__
     //    unsigned int warp_scan(unsigned int data)
     //{
     //    int lane_id = threadIdx.x % warpSize;
     //    int mask = 0xffffffff << lane_id;

     //    // �𲽺ϲ����
     //    data += __shfl_up_sync(mask, data, 1);
     //    data += __shfl_up_sync(mask, data, 2);
     //    data += __shfl_up_sync(mask, data, 4);
     //    data += __shfl_up_sync(mask, data, 8);
     //    data += __shfl_up_sync(mask, data, 16);

     //    return data;
     //}


    /**
     * \brief ɨ�����(unsigned int ����).
     * 
     * \param x ��ǰ�߳����е�һ���̵߳�����
     * \param offset �߳����е�ƫ����(bitλ����)
     * \return ͬһ���߳���������ӵĽ��
     */
    __device__ __forceinline__ unsigned int shfl_add(unsigned int x, unsigned int offset)
    {
        unsigned int result = 0;
        asm volatile (
            "{.reg .u32 r0;"
            ".reg .pred p;"
            "shfl.sync.up.b32 r0|p, %1, %2, 0, 0;"
            "@p add.u32 r0, r0, %1;"
            "mov.u32 %0, r0;}"
            : "=r"(result) : "r"(x), "r"(offset));
        //result = __shfl_up_sync(0xffffffff, x, offset, 32);

        return result;
    }

    /**
     * \brief �߳�����ɨ�����(unsigned int����).
     * 
     * \param data �߳�����һ������(bitλ�������)
     * \return ͬһ���߳���������ӵĽ��
     */
    __device__ __forceinline__ unsigned int warp_scan(unsigned int data)
    {
        data = shfl_add(data, 1u);
        data = shfl_add(data, 2u);
        data = shfl_add(data, 4u);
        data = shfl_add(data, 8u);
        data = shfl_add(data, 16u);
        return data;
    }







}
