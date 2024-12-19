#include "environment_test.h"


/*
修饰符__global__表明这个函数是将会从CPU中调用，在GPU中进行执行。
并借此函数来启动内核函数。
*/
__global__ void hello_world_from_gpu(void)
{
	printf("Hello World from GPU\n");
	return;
}






