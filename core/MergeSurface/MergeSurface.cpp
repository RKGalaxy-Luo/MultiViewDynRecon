/*****************************************************************//**
 * \file   MergeSurface.cpp
 * \brief  ʹ�ò�ֵ��������ķ�ʽ�ں���������
 * 
 * \author LUOJIAXUAN
 * \date   June 14th 2024
 *********************************************************************/
#include "MergeSurface.h"

SparseSurfelFusion::MergeSurface::MergeSurface(const Intrinsic* intrinsics)
{
	MergedDepthSurfels.AllocateBuffer(MAX_SURFEL_COUNT);
	OverlappingSurfelsCountMap.AllocateBuffer(CLIP_WIDTH * CLIP_HEIGHT);
	OverlappingSurfelsCountMap.ResizeArrayOrException(CLIP_WIDTH * CLIP_HEIGHT);
	OverlappingOrderMap.AllocateBuffer(MAX_SURFEL_COUNT);
	surfelProjectedPixelPos.AllocateBuffer(MAX_SURFEL_COUNT);
	CHECKCUDA(cudaStreamCreate(&stream));

	for (int i = 0; i < CamerasCount; i++) {
		CameraParameter[2 * i].SE3 = Constants::GetInitialCameraSE3(i);
		CameraParameter[2 * i + 1].SE3 = Constants::GetInitialCameraSE3(i);
	}

	for (int i = 0; i < CamerasCount; i++) {
		CameraParameter[2 * i].ID = i;
		CameraParameter[2 * i + 1].ID = i;
		if (i != CamerasCount - 1) {
			// ��ֵ���Ϊ��������ڲ�ȡƽ��ֵ
			CameraParameter[2 * i].intrinsic = intrinsics[i];
			CameraParameter[2 * i + 1].intrinsic.focal_x = (intrinsics[i].focal_x + intrinsics[i + 1].focal_x) / 2.0f;
			CameraParameter[2 * i + 1].intrinsic.focal_y = (intrinsics[i].focal_y + intrinsics[i + 1].focal_y) / 2.0f;
			CameraParameter[2 * i + 1].intrinsic.principal_x = (intrinsics[i].principal_x + intrinsics[i + 1].principal_x) / 2.0f;
			CameraParameter[2 * i + 1].intrinsic.principal_y = (intrinsics[i].principal_y + intrinsics[i + 1].principal_y) / 2.0f;
		}
		else {
			// ��ֵ���Ϊ��������ڲ�ȡƽ��ֵ
			CameraParameter[2 * i].intrinsic = intrinsics[i];
			CameraParameter[2 * i + 1].intrinsic.focal_x = (intrinsics[i].focal_x + intrinsics[0].focal_x) / 2.0f;
			CameraParameter[2 * i + 1].intrinsic.focal_y = (intrinsics[i].focal_y + intrinsics[0].focal_y) / 2.0f;
			CameraParameter[2 * i + 1].intrinsic.principal_x = (intrinsics[i].principal_x + intrinsics[0].principal_x) / 2.0f;
			CameraParameter[2 * i + 1].intrinsic.principal_y = (intrinsics[i].principal_y + intrinsics[0].principal_y) / 2.0f;
		}
	}
}

SparseSurfelFusion::MergeSurface::~MergeSurface()
{
	MergedDepthSurfels.ReleaseBuffer();
	OverlappingOrderMap.ReleaseBuffer();
	OverlappingSurfelsCountMap.ReleaseBuffer();
	surfelProjectedPixelPos.ReleaseBuffer();
	for (int i = 0; i < MAX_CAMERA_COUNT; i++) {
		CHECKCUDA(cudaStreamDestroy(stream));
	}
}

void SparseSurfelFusion::MergeSurface::MergeAllSurfaces(DeviceBufferArray<DepthSurfel>* depthSurfels)
{
	//auto start = std::chrono::high_resolution_clock::now();						// ��¼��ʼʱ���
	if (CamerasCount == 1) {
		MergedDepthSurfels.ResizeArrayOrException(depthSurfels[0].ArraySize());
		CHECKCUDA(cudaMemcpyAsync(MergedDepthSurfels.Ptr(), depthSurfels[0].Ptr(), sizeof(DepthSurfel) * depthSurfels[0].ArraySize(), cudaMemcpyDeviceToDevice));
	}
	else {
		// �����һ֡������
		MergedDepthSurfels.ResizeArrayOrException(0);
		for (int i = 0; i < CamerasCount; i++) {
			// �����һ���ӽǵ�����
			CHECKCUDA(cudaMemsetAsync(OverlappingOrderMap.Array().ptr(), 0, sizeof(unsigned short) * CamerasCount, stream));
			CHECKCUDA(cudaMemsetAsync(OverlappingSurfelsCountMap.Array().ptr(), 0, sizeof(unsigned int) * ClipedImageSize, stream));
			CalculateTSDFMergedSurfels(CameraParameter[2 * i], CameraParameter[2 * ((i + 1) % CamerasCount)], CameraParameter[2 * i + 1], depthSurfels[i].Array(), depthSurfels[(i + 1) % CamerasCount].Array(), stream);
		}
		CollectNotMergedSurfels(depthSurfels, stream);
	}

	CHECKCUDA(cudaStreamSynchronize(stream));
	//auto end = std::chrono::high_resolution_clock::now();							// ��¼����ʱ���
	//std::chrono::duration<double, std::milli> duration = end - start;				// ����ִ��ʱ�䣨��msΪ��λ��
	//std::cout << "���ӽ�Surface�ں�: " << duration.count() << " ms" << std::endl;		// ���
}
