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
		std::cout << "这里构造了test"<< A << std::endl;
	}
	~ConstructionTest() {
		std::cout << "这里析构了test" << std::endl;
	}
};
