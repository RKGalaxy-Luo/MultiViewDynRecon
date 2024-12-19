/*****************************************************************//**
 * \file   PatchColliderCommonType.h
 * \brief  声明GPC算法常用的数据结构
 * 
 * \author LUO
 * \date   March 12th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <base/FileOperation/Stream.h>

namespace SparseSurfelFusion {

	/**
	 * \brief 存有GPC Patch特征的结构体.
	 */
	template<int FeatureDim = 18>
	struct GPCPatchFeature {
		float Feature[FeatureDim];		// GPC Patch特征
	};

	/**
	 * \brief GPC树节点的结构体.
	 */
	template<int FeatureDim = 18>
	struct GPCNode {
		float Coefficient[FeatureDim];	// GPC Patch特征的系数
		float Boundary;					// 边界：决定是左节点还是右节点的阈值
		int LeftChild;					// 左孩子节点
		int RightChild;					// 右孩子节点

		/**
		 * \brief GPCNode系数与GPC特征相点乘的操作.
		 * 
		 * \param feature 传入GPC Patch特征
		 * \return 点乘结果
		 */
		__host__ __device__ __forceinline__ float dot(const GPCPatchFeature<FeatureDim>& feature) const {
			float dotValue = 0.0f;
			for (int i = 0; i < FeatureDim; i++) {
				dotValue += feature.Feature[i] * Coefficient[i];
			}
			return dotValue;
		}

		/**
		 * \brief 内联函数保存，将每一个特征维度的系数，以及边界和左右节点存入文件流.
		 * 
		 * \param stream 写入数据到该文件流
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
		 * \brief 从文件流中加载GPCNode每一个参数的数据.
		 * 
		 * \param stream 从该文件流中读取数据
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
	 * \brief GPC树的结构体.
	 */
	template<int FeatureDim = 18>
	struct GPCTree {
		GPCNode<FeatureDim>* Nodes;			// 树的节点
		unsigned NodesNum;					// 节点的数量
		unsigned MaxLevel;					// 树最大层数

		/**
		 * \brief 寻找给定patch的叶子(与给定patch匹配的patch).
		 * 
		 * \param patchFeature 传入需要寻找的patch特征
		 * \return 与传入patch相匹配的patch的index
		 */
		__host__ __device__ __forceinline__ unsigned int leafForPatch(const GPCPatchFeature<FeatureDim>& patchFeature) const {
			unsigned int node_idx = 0, prev_idx = 0;

			// 主搜索循环由深度限定
			for (int i = 0; i < MaxLevel; i++) {
				prev_idx = node_idx;
				// 加载节点，此处开销大
				const GPCNode<FeatureDim>& node = Nodes[node_idx];

				// 决定下一步搜索的位置
				if (node.dot(patchFeature) < node.Boundary) {
					node_idx = node.RightChild;
				}
				else {
					node_idx = node.LeftChild;
				}

				// 检查是否完成
				if (node_idx == 0 || node_idx >= NodesNum) break;
			}

			//prev_idx 是具有有效树节点的索引
			return prev_idx;
		}
	};

}
