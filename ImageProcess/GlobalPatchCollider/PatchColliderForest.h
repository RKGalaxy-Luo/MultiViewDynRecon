/*****************************************************************//**
 * \file   PatchColliderForest.h
 * \brief  GPC森林
 * 
 * \author LUO
 * \date   March 12th 2024
 *********************************************************************/
#pragma once
#include <memory>

#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/FileOperation/Stream.h>
#include <base/FileOperation/Serializer.h>

#include "PatchColliderCommonType.h"

namespace SparseSurfelFusion {
	/**
	 * \brief GPC森林类：FeatureDim是特征维度，TreesNum是森林中树的数量.
	 */
	template<int FeatureDim = 18, int TreesNum = 5>
	class PatchColliderForest {
	private:
		// GPC节点的CPU内存
		std::vector<GPCNode<FeatureDim>> TreeNodesHost[TreesNum];

		// GPC节点的GPU内存
		DeviceArray<GPCNode<FeatureDim>> TreeNodesDevice[TreesNum];

		//这些是给所有的树的
		unsigned MaxNodesNum;		// 树的最大节点数
		unsigned MaxLevel;			// 树的最大层数

	public:
		/**
		 * \brief 访问树中索引为index的节点.
		 * 
		 * \param index 节点索引
		 * \return GPC节点
		 */
		inline std::vector<GPCNode<FeatureDim>>& NodesForTree(int index) {
			return TreeNodesHost[index];
		}

	public:
		// 通过主机上的指针访问
		using Ptr = std::shared_ptr<PatchColliderForest>;

		/**
		 * \brief 构造函数，参数初始化为0.
		 * 
		 */

		PatchColliderForest();
		/**
		 * \brief 默认析构函数.
		 * 
		 */
		~PatchColliderForest() = default;

		/**
		 * \brief 用于GPU访问的GPC森林结构体.
		 */
		struct GPCForestDevice {
			GPCTree<FeatureDim> trees[TreesNum];
		};

		/**
		 * \brief 将数据加载到GPU上，并返回GPU数据结构.
		 * 
		 * \return 带有数据的GPU数据结构
		 */
		GPCForestDevice OnDevice() const;

		/**
		 * \brief 更新森林树的最大层数.
		 * 
		 * \param maxLevel 最大层数
		 */
		void UpdateSearchLevel(int maxLevel);

		/**
		 * \brief 将数据保存到文件流中.
		 * 
		 * \param stream 文件流
		 */
		void Save(Stream* stream) const;
		/**
		 * \brief 将数据从文件流中加载，从文件中获取树最大节点，最大层数，以及树的节点数据.
		 * 
		 * \param stream 文件流
		 * \return 是否加载成功
		 */
		bool Load(Stream* stream);
		/**
		 * \brief 将节点数据上传到GPU.
		 * 
		 */
		void UploadToDevice();


	};


}

/************************************************* 方法实现 *************************************************/
template<int FeatureDim, int TreesNum>
SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::PatchColliderForest() : MaxNodesNum(0), MaxLevel(0)
{
	// 对所有实体进行零初始化
	for (int i = 0; i < TreesNum; i++) {
		TreeNodesHost[i].clear();
		TreeNodesDevice[i] = DeviceArray<GPCNode<FeatureDim>>(nullptr, 0);
	}
}


template<int FeatureDim, int TreesNum>
typename SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::OnDevice() const
{
	// typename帮助编译器解析，以免编译器不知道GPCForestDevice是类型名字

	// 检查内存是否已上传至gpu内存
	if (TreeNodesDevice[0].size() == 0) {
		LOGGING(FATAL) << "森林尚未上传至GPU设备";
	}
	// 现在我们可以安全地赋值指针了
	GPCForestDevice forestDevice;
	// 构造树
	GPCTree<FeatureDim> tree;
	tree.MaxLevel = MaxLevel;
	for (int i = 0; i < TreesNum; i++) {
		tree.NodesNum = TreeNodesDevice[i].size();
		tree.Nodes = (GPCNode<FeatureDim>*) TreeNodesDevice[i].ptr();
		forestDevice.trees[i] = tree;
	}
	return forestDevice;
}

template<int FeatureDim, int TreesNum>
inline void SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::UpdateSearchLevel(int maxLevel)
{
	MaxLevel = maxLevel;
}

template<int FeatureDim, int TreesNum>
void SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::Save(Stream* stream) const
{
	stream->SerializeWrite(MaxNodesNum);
	stream->SerializeWrite(MaxLevel);
	// 所有树的节点都要写入
	for (int i = 0; i < TreesNum; i++) {
		stream->SerializeWrite(TreeNodesHost[i]);
	}
}

template<int FeatureDim, int TreesNum>
bool SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::Load(Stream* stream)
{
	stream->SerializeRead(&MaxNodesNum);
	stream->SerializeRead(&MaxLevel);
	MaxNodesNum = 0;		// 为何置零
	for (int i = 0; i < TreesNum; i++) {
		stream->SerializeRead(&(TreeNodesHost[i]));
		if (TreeNodesHost[i].size() > MaxNodesNum) {
			MaxNodesNum = TreeNodesHost[i].size();
		}
	}
	return true;
}

template<int FeatureDim, int TreesNum>
void SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::UploadToDevice()
{
	for (int i = 0; i < TreesNum; i++) {
		TreeNodesDevice[i].upload(TreeNodesHost[i]);
	}
}

