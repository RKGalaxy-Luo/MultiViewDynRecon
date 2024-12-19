#include "environment_test.h"


void GPUConfig::getGPUConfig()
{
	//printf("Hello World from CPU\n");
	//hello_world_from_gpu << <1, 5 >> > ();


	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor == 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("Device %d: %s\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA驱动版本:                                              %d.%d\n",
			driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                            %d.%d\n",
			runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("GPU计算能力:                                               %d.%d\n",
			deviceProp.major, deviceProp.minor);
		printf("每个线程块(Block)中可用Registers大小:                      %d\n",
			deviceProp.regsPerBlock);
		printf("每个线程块(Block)中Local Memory大小:                       %u bytes\n",
			deviceProp.regsPerBlock);//Local Memory大小不定
		printf("每个线程块(Block)中Shared Memory大小:                      %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("常量内存(Global Memory)大小:                               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("1D纹理内存(Global Memory)大小:                             %u bytes\n",
			deviceProp.maxTexture1D);
		printf("2D纹理内存(Global Memory)大小:                             %d x %d \n",
			deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
		printf("3D纹理内存(Global Memory)大小:                             %d x %d x %d\n",
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("全局内存(Global Memory)的大小:                             %zu bytes\n",
			deviceProp.totalGlobalMem);
		printf("线程束(Warp)大小:                                          %d\n",
			deviceProp.warpSize);
		printf("流多处理器(SM)数量:                                        %d\n",
			deviceProp.multiProcessorCount);
		printf("每个流处理器(SM)最大线程数量:                              %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("每个线程块(Block)最大线程数量:                             %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("每个线程块(Block)中每个维度的最大数量:                     %d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("每个网格(Grid)中每个维度的最大数量:                        %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("GPU储存二维数组中每行之间最大偏移量:                       %zu bytes\n",
			deviceProp.memPitch);
		printf("纹理数据在内存中的基地址需要按照多少字节对齐:              %zu bytes\n",
			deviceProp.texturePitchAlignment);
		printf("时钟频率:                                                  %.2f GHz\n",
			deviceProp.clockRate * 1e-6f);
		printf("内存时钟频率:                                              %.0f MHz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("内存总线宽度:                                              %d-bit\n",
			deviceProp.memoryBusWidth);
	}

	//system("pause");

	/*
	三重尖括号里的参数表明的是相关的执行配置，用来表明使用多少线程来执行内核函数，
	在本例子中有5个GPU线程被系统所调用。
	*/
	cudaDeviceReset();
	/*
	执行完成后调用cudaDeviceReset()函数释放和清空与当前进程运行相关的资源。
	*/
	return;
}
