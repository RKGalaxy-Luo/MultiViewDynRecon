/*****************************************************************//**
 * \file   BinaryFileStream.cpp
 * \brief  �������ļ�����д
 * 
 * \author LUO
 * \date   March 11th 2024
 *********************************************************************/
#include "BinaryFileStream.h"

SparseSurfelFusion::BinaryFileStream::BinaryFileStream(const char* filePath, FileOperationMode mode)
{
	std::FILE* filePtr = nullptr;	// �ļ�ָ��
	if (mode == FileOperationMode::ReadOnly) {
		filePtr = std::fopen(filePath, "rb");
	}
	else if (mode == FileOperationMode::WriteOnly) {
		filePtr = std::fopen(filePath, "wb");
	}
	fileStream = filePtr;	// �����ʼ�ָ��
}

SparseSurfelFusion::BinaryFileStream::~BinaryFileStream()
{
	std::fclose(fileStream);	// �ر��ļ���
	fileStream = nullptr;		// �ļ����ָ�븳ֵΪ��
}

size_t SparseSurfelFusion::BinaryFileStream::Read(void* ptr, size_t bytes)
{
	size_t dataBytesSize = std::fread(ptr, 1, bytes, fileStream);
	return dataBytesSize;
}

bool SparseSurfelFusion::BinaryFileStream::Write(const void* ptr, size_t bytes)
{
	// �ɹ�д���Byte��
	size_t successfullyWriteBytes = std::fwrite(ptr, 1, bytes, fileStream);
	// ���ȫ���ɹ�д�룬����true
	if (successfullyWriteBytes == bytes) return true;
	else return false;
}
