#include <base/ColorTypeTransfer.h>
#include "PatchColliderBuildFeature.h"

template<int PatchHalfSize = 10>
__device__ __forceinline__ void SparseSurfelFusion::buildDCTPatchFeature(cudaTextureObject_t normalizedRGB, int centerX, int centerY, GPCPatchFeature<18>& feature)
{
	const float pi = 3.1415926f;
	const int left_x = centerX - PatchHalfSize;	// ��ǰPatch�����Ͻ�x����
	const int top_y = centerY - PatchHalfSize;	// ��ǰPatch�����Ͻ�y����

	// YCbCr��Ҳ��ΪYCC����һ����ɫ����ϵͳ����������ͼ�����Ƶ�е�ɫ�ʱ�ʾ������һ������ - ɫ�ȱ���
	// �������۶����ȱ仯�����У�����ɫ�ȱ仯��̫���У�ʹ������-ɫ�ȱ�������ڱ���ͼ��������ͬʱ�������ݴ���ʹ洢������

	// ��Patch���ص��ֲ��ڴ���
	float3 patch_ycrcb[2 * PatchHalfSize][2 * PatchHalfSize];
	for (int y = top_y; y < top_y + 2 * PatchHalfSize; y++) {
		for (int x = left_x; x < left_x + 2 * PatchHalfSize; x++)
		{
			// ��ȡ����
			const float4 rgba = tex2D<float4>(normalizedRGB, x, y);

			// ת��ʽ
			float3 ycrcb;
			normalized_rgba2ycrcb(rgba, ycrcb);
			ycrcb.x *= 255;
			ycrcb.y *= 255;
			ycrcb.z *= 255;

			// ע��y�������ڵ�һ��
			patch_ycrcb[y - top_y][x - left_x] = ycrcb;
		}
	}

	// DCT����ѭ�� 4��4ϵ������feature.Feature(һά��ʽ�洢)
	for (int n0 = 0; n0 < 4; n0++) {
		for (int n1 = 0; n1 < 4; n1++) {
			float dct_sum = 0.0f;
			for (int y = 0; y < 2 * PatchHalfSize; y++) {
				for (int x = 0; x < 2 * PatchHalfSize; x++) {
					// ��ȡͼ���������ݣ�ֻ�������ȷ���x
					const float3 ycrcb = patch_ycrcb[y][x];
					dct_sum += ycrcb.x 
						* cosf(pi * (x + 0.5f) * n0 / (2 * PatchHalfSize)) 
						* cosf(pi * (y + 0.5f) * n1 / (2 * PatchHalfSize));
				}
			}
			// ����Patch
			feature.Feature[n0 * 4 + n1] = dct_sum / float(PatchHalfSize);
		}
	}

	// ���������ӵĳ߶�
	for (int k = 0; k < 4; k++) {
		feature.Feature[k] *= 0.7071067811865475f;		// ��ǰ4��������(0,1,2,3)������1/��2
		feature.Feature[k * 4] *= 0.7071067811865475f;	// ��0��4��8��12�������ӽ����˶���߶ȵ�������Ϊ��DCT�任�У���Щλ���ϵ�ϵ�����нϸߵ���������Ҫ������г߶ȵ����Ա���ƽ��
	}


	// �������������������ɫ��y��ɫ��z��ƽ��ֵ
	float cr_sum = 0.0f;
	float cb_sum = 0.0f;
	for (int y = 0; y < 2 * PatchHalfSize; y++) {
		for (int x = 0; x < 2 * PatchHalfSize; x++) {
			// ��ȡͼ����������
			const float3 ycrcb = patch_ycrcb[y][x];
			cr_sum += ycrcb.y;
			cb_sum += ycrcb.z;
		}
	}
	feature.Feature[16] = cr_sum / (2 * PatchHalfSize);
	feature.Feature[17] = cb_sum / (2 * PatchHalfSize);

}
