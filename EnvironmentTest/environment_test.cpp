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
		printf("CUDA�����汾:                                              %d.%d\n",
			driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA����ʱ�汾:                                            %d.%d\n",
			runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("GPU��������:                                               %d.%d\n",
			deviceProp.major, deviceProp.minor);
		printf("ÿ���߳̿�(Block)�п���Registers��С:                      %d\n",
			deviceProp.regsPerBlock);
		printf("ÿ���߳̿�(Block)��Local Memory��С:                       %u bytes\n",
			deviceProp.regsPerBlock);//Local Memory��С����
		printf("ÿ���߳̿�(Block)��Shared Memory��С:                      %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("�����ڴ�(Global Memory)��С:                               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("1D�����ڴ�(Global Memory)��С:                             %u bytes\n",
			deviceProp.maxTexture1D);
		printf("2D�����ڴ�(Global Memory)��С:                             %d x %d \n",
			deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
		printf("3D�����ڴ�(Global Memory)��С:                             %d x %d x %d\n",
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("ȫ���ڴ�(Global Memory)�Ĵ�С:                             %zu bytes\n",
			deviceProp.totalGlobalMem);
		printf("�߳���(Warp)��С:                                          %d\n",
			deviceProp.warpSize);
		printf("���ദ����(SM)����:                                        %d\n",
			deviceProp.multiProcessorCount);
		printf("ÿ����������(SM)����߳�����:                              %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("ÿ���߳̿�(Block)����߳�����:                             %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("ÿ���߳̿�(Block)��ÿ��ά�ȵ��������:                     %d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("ÿ������(Grid)��ÿ��ά�ȵ��������:                        %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("GPU�����ά������ÿ��֮�����ƫ����:                       %zu bytes\n",
			deviceProp.memPitch);
		printf("�����������ڴ��еĻ���ַ��Ҫ���ն����ֽڶ���:              %zu bytes\n",
			deviceProp.texturePitchAlignment);
		printf("ʱ��Ƶ��:                                                  %.2f GHz\n",
			deviceProp.clockRate * 1e-6f);
		printf("�ڴ�ʱ��Ƶ��:                                              %.0f MHz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("�ڴ����߿��:                                              %d-bit\n",
			deviceProp.memoryBusWidth);
	}

	//system("pause");

	/*
	���ؼ�������Ĳ�������������ص�ִ�����ã���������ʹ�ö����߳���ִ���ں˺�����
	�ڱ���������5��GPU�̱߳�ϵͳ�����á�
	*/
	cudaDeviceReset();
	/*
	ִ����ɺ����cudaDeviceReset()�����ͷź�����뵱ǰ����������ص���Դ��
	*/
	return;
}
