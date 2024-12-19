/*****************************************************************//**
 * \file   BinaryFileStream.cpp
 * \brief  二进制文件流读写
 * 
 * \author LUO
 * \date   March 11th 2024
 *********************************************************************/
#include "BinaryFileStream.h"

SparseSurfelFusion::BinaryFileStream::BinaryFileStream(const char* filePath, FileOperationMode mode)
{
	std::FILE* filePtr = nullptr;	// 文件指针
	if (mode == FileOperationMode::ReadOnly) {
		filePtr = std::fopen(filePath, "rb");
	}
	else if (mode == FileOperationMode::WriteOnly) {
		filePtr = std::fopen(filePath, "wb");
	}
	fileStream = filePtr;	// 更新问价指针
}

SparseSurfelFusion::BinaryFileStream::~BinaryFileStream()
{
	std::fclose(fileStream);	// 关闭文件流
	fileStream = nullptr;		// 文件句柄指针赋值为空
}

size_t SparseSurfelFusion::BinaryFileStream::Read(void* ptr, size_t bytes)
{
	size_t dataBytesSize = std::fread(ptr, 1, bytes, fileStream);
	return dataBytesSize;
}

bool SparseSurfelFusion::BinaryFileStream::Write(const void* ptr, size_t bytes)
{
	// 成功写入的Byte数
	size_t successfullyWriteBytes = std::fwrite(ptr, 1, bytes, fileStream);
	// 如果全部成功写入，返回true
	if (successfullyWriteBytes == bytes) return true;
	else return false;
}
