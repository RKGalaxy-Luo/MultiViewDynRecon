#pragma once

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

//cublas的宏
#ifndef cublasSafeCall
#define cublasSafeCall(err)    surfelwarp::__cublasSafeCall(err, __FILE__, __LINE__)
#endif

//cuda驱动api宏
#ifndef cuSafeCall
#define cuSafeCall(err) surfelwarp::__cuSafeCall(err, __FILE__, __LINE__)
#endif

namespace surfelwarp {

	//cublas错误函数的实际处理程序
	static inline void __cublasSafeCall(cublasStatus_t err, const char* file, const int line)
	{
		if (CUBLAS_STATUS_SUCCESS != err) {
			fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", __FILE__, __LINE__, err);
			cudaDeviceReset();
			std::exit(1);
		}
	}
	//如果出错，显示错误名称
	static inline void __cuSafeCall(CUresult err, const char* file, const int line) {
		if (err != CUDA_SUCCESS) {
			//查询错误名称和字符串
			const char* error_name;
			cuGetErrorName(err, &error_name);
			const char* error_string;
			cuGetErrorString(err, &error_string);
			fprintf(stderr, "CUDA driver error %s: %s in the line %d of file %s \n", error_name, error_string, line, file);
			cudaDeviceReset();
			std::exit(1);
		}
	}

}