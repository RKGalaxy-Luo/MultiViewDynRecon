#include "environment_test.h"


/*
���η�__global__������������ǽ����CPU�е��ã���GPU�н���ִ�С�
����˺����������ں˺�����
*/
__global__ void hello_world_from_gpu(void)
{
	printf("Hello World from GPU\n");
	return;
}






