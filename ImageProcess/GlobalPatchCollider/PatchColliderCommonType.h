/*****************************************************************//**
 * \file   PatchColliderCommonType.h
 * \brief  ����GPC�㷨���õ����ݽṹ
 * 
 * \author LUO
 * \date   March 12th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <base/FileOperation/Stream.h>

namespace SparseSurfelFusion {

	/**
	 * \brief ����GPC Patch�����Ľṹ��.
	 */
	template<int FeatureDim = 18>
	struct GPCPatchFeature {
		float Feature[FeatureDim];		// GPC Patch����
	};

	/**
	 * \brief GPC���ڵ�Ľṹ��.
	 */
	template<int FeatureDim = 18>
	struct GPCNode {
		float Coefficient[FeatureDim];	// GPC Patch������ϵ��
		float Boundary;					// �߽磺��������ڵ㻹���ҽڵ����ֵ
		int LeftChild;					// ���ӽڵ�
		int RightChild;					// �Һ��ӽڵ�

		/**
		 * \brief GPCNodeϵ����GPC�������˵Ĳ���.
		 * 
		 * \param feature ����GPC Patch����
		 * \return ��˽��
		 */
		__host__ __device__ __forceinline__ float dot(const GPCPatchFeature<FeatureDim>& feature) const {
			float dotValue = 0.0f;
			for (int i = 0; i < FeatureDim; i++) {
				dotValue += feature.Feature[i] * Coefficient[i];
			}
			return dotValue;
		}

		/**
		 * \brief �����������棬��ÿһ������ά�ȵ�ϵ�����Լ��߽�����ҽڵ�����ļ���.
		 * 
		 * \param stream д�����ݵ����ļ���
		 */
		inline void Save(Stream* stream) const {
			for (int i = 0; i < FeatureDim; i++) {
				stream->SerializeWrite<float>(Coefficient[i]);
			}
			stream->SerializeWrite<float>(Boundary);
			stream->SerializeWrite<int>(LeftChild);
			stream->SerializeWrite<int>(RightChild);
		}
		/**
		 * \brief ���ļ����м���GPCNodeÿһ������������.
		 * 
		 * \param stream �Ӹ��ļ����ж�ȡ����
		 * \return 
		 */
		inline bool Load(Stream* stream) {
			for (int i = 0; i < FeatureDim; i++) {
				stream->SerializeRead<float>(&Coefficient[i]);
			}
			stream->SerializeRead<float>(&Boundary);
			stream->SerializeRead<int>(&LeftChild);
			stream->SerializeRead<int>(&RightChild);
			return true;
		}
	};

	/**
	 * \brief GPC���Ľṹ��.
	 */
	template<int FeatureDim = 18>
	struct GPCTree {
		GPCNode<FeatureDim>* Nodes;			// ���Ľڵ�
		unsigned NodesNum;					// �ڵ������
		unsigned MaxLevel;					// ��������

		/**
		 * \brief Ѱ�Ҹ���patch��Ҷ��(�����patchƥ���patch).
		 * 
		 * \param patchFeature ������ҪѰ�ҵ�patch����
		 * \return �봫��patch��ƥ���patch��index
		 */
		__host__ __device__ __forceinline__ unsigned int leafForPatch(const GPCPatchFeature<FeatureDim>& patchFeature) const {
			unsigned int node_idx = 0, prev_idx = 0;

			// ������ѭ��������޶�
			for (int i = 0; i < MaxLevel; i++) {
				prev_idx = node_idx;
				// ���ؽڵ㣬�˴�������
				const GPCNode<FeatureDim>& node = Nodes[node_idx];

				// ������һ��������λ��
				if (node.dot(patchFeature) < node.Boundary) {
					node_idx = node.RightChild;
				}
				else {
					node_idx = node.LeftChild;
				}

				// ����Ƿ����
				if (node_idx == 0 || node_idx >= NodesNum) break;
			}

			//prev_idx �Ǿ�����Ч���ڵ������
			return prev_idx;
		}
	};

}
