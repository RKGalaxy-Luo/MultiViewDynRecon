#pragma once
#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace GPUConfig {
	void getGPUConfig();
}

class ConstructionTest {
public:
	ConstructionTest(int A) {
		std::cout << "���ﹹ����test"<< A << std::endl;
	}
	~ConstructionTest() {
		std::cout << "����������test" << std::endl;
	}
};
