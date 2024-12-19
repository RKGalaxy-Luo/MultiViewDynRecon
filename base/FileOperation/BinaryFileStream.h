/*****************************************************************//**
 * \file   BinaryFileStream.h
 * \brief  ��ȡ�������ļ���Pythonѵ��ģ���ļ���
 * 
 * \author LUO
 * \date   March 11th 2024
 *********************************************************************/
#pragma once
#include "Stream.h"

namespace SparseSurfelFusion {
	/**
	 * \brief �������ļ����̳�Stream������.
	 */
	class BinaryFileStream : public Stream
	{
	private:
		std::FILE* fileStream;
	public:
		using Ptr = std::shared_ptr<BinaryFileStream>;
		/**
		 * \brief �ļ�������ģʽ���Ƕ��ļ�����д�ļ�.
		 */
		enum class FileOperationMode {
			ReadOnly,	// �ļ�ֻ��
			WriteOnly	// �ļ�ֻ��
		};
		/**
		 * \brief ���캯��(ֻ����ʽ����)���Ӹ����ļ�·���У���д�������ļ�.
		 * 
		 * \param filePath �������ļ�·��
		 * \param mode �ļ�����ģʽ��Ĭ��ֻ��
		 */
		explicit BinaryFileStream(const char* filePath, FileOperationMode mode = FileOperationMode::ReadOnly);
		/**
		 * \brief ������������������ʱ�����˸����������������.
		 * 
		 */
		~BinaryFileStream() override;
		/**
		 * \brief ��������������Ǵ��ļ���fileStream�ж�ȡָ�����������ݣ�������ȡ�����ݴ洢���������ڴ�λ��(��д��Stream�е��麯��).
		 * 
		 * \param ptr �Ӹ������ļ��� stream �ж�ȡ���ݣ�������ȡ�����ݴ洢�� ptr ָ����ڴ�λ��
		 * \param bytes ָ��Ҫ��ȡ�������������
		 * \return ����ʵ�ʳɹ���ȡ������������
		 */
		size_t Read(void* ptr, size_t bytes) override;
		/**
		 * \brief ��ptrָ����ڴ�λ��д�뵽fileStream��������ÿ���������С����Ϊ1�������ɹ�д����������Ƿ���Ҫд����������Сһ��.
		 * 
		 * \param ptr ���ݵĵ�ַ
		 * \param bytes ���ݵ�Byte����
		 * \return �Ƿ�ɹ�ȫ��д��
		 */
		bool Write(const void* ptr, size_t bytes) override;
	};
}


