/*****************************************************************//**
 * \file   PatchColliderForest.h
 * \brief  GPCɭ��
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
	 * \brief GPCɭ���ࣺFeatureDim������ά�ȣ�TreesNum��ɭ������������.
	 */
	template<int FeatureDim = 18, int TreesNum = 5>
	class PatchColliderForest {
	private:
		// GPC�ڵ��CPU�ڴ�
		std::vector<GPCNode<FeatureDim>> TreeNodesHost[TreesNum];

		// GPC�ڵ��GPU�ڴ�
		DeviceArray<GPCNode<FeatureDim>> TreeNodesDevice[TreesNum];

		//��Щ�Ǹ����е�����
		unsigned MaxNodesNum;		// �������ڵ���
		unsigned MaxLevel;			// ����������

	public:
		/**
		 * \brief ������������Ϊindex�Ľڵ�.
		 * 
		 * \param index �ڵ�����
		 * \return GPC�ڵ�
		 */
		inline std::vector<GPCNode<FeatureDim>>& NodesForTree(int index) {
			return TreeNodesHost[index];
		}

	public:
		// ͨ�������ϵ�ָ�����
		using Ptr = std::shared_ptr<PatchColliderForest>;

		/**
		 * \brief ���캯����������ʼ��Ϊ0.
		 * 
		 */

		PatchColliderForest();
		/**
		 * \brief Ĭ����������.
		 * 
		 */
		~PatchColliderForest() = default;

		/**
		 * \brief ����GPU���ʵ�GPCɭ�ֽṹ��.
		 */
		struct GPCForestDevice {
			GPCTree<FeatureDim> trees[TreesNum];
		};

		/**
		 * \brief �����ݼ��ص�GPU�ϣ�������GPU���ݽṹ.
		 * 
		 * \return �������ݵ�GPU���ݽṹ
		 */
		GPCForestDevice OnDevice() const;

		/**
		 * \brief ����ɭ������������.
		 * 
		 * \param maxLevel ������
		 */
		void UpdateSearchLevel(int maxLevel);

		/**
		 * \brief �����ݱ��浽�ļ�����.
		 * 
		 * \param stream �ļ���
		 */
		void Save(Stream* stream) const;
		/**
		 * \brief �����ݴ��ļ����м��أ����ļ��л�ȡ�����ڵ㣬���������Լ����Ľڵ�����.
		 * 
		 * \param stream �ļ���
		 * \return �Ƿ���سɹ�
		 */
		bool Load(Stream* stream);
		/**
		 * \brief ���ڵ������ϴ���GPU.
		 * 
		 */
		void UploadToDevice();


	};


}

/************************************************* ����ʵ�� *************************************************/
template<int FeatureDim, int TreesNum>
SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::PatchColliderForest() : MaxNodesNum(0), MaxLevel(0)
{
	// ������ʵ��������ʼ��
	for (int i = 0; i < TreesNum; i++) {
		TreeNodesHost[i].clear();
		TreeNodesDevice[i] = DeviceArray<GPCNode<FeatureDim>>(nullptr, 0);
	}
}


template<int FeatureDim, int TreesNum>
typename SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::OnDevice() const
{
	// typename���������������������������֪��GPCForestDevice����������

	// ����ڴ��Ƿ����ϴ���gpu�ڴ�
	if (TreeNodesDevice[0].size() == 0) {
		LOGGING(FATAL) << "ɭ����δ�ϴ���GPU�豸";
	}
	// �������ǿ��԰�ȫ�ظ�ֵָ����
	GPCForestDevice forestDevice;
	// ������
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
	// �������Ľڵ㶼Ҫд��
	for (int i = 0; i < TreesNum; i++) {
		stream->SerializeWrite(TreeNodesHost[i]);
	}
}

template<int FeatureDim, int TreesNum>
bool SparseSurfelFusion::PatchColliderForest<FeatureDim, TreesNum>::Load(Stream* stream)
{
	stream->SerializeRead(&MaxNodesNum);
	stream->SerializeRead(&MaxLevel);
	MaxNodesNum = 0;		// Ϊ������
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

