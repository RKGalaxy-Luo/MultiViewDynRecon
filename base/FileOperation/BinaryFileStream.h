/*****************************************************************//**
 * \file   BinaryFileStream.h
 * \brief  读取二进制文件，Python训练模型文件等
 * 
 * \author LUO
 * \date   March 11th 2024
 *********************************************************************/
#pragma once
#include "Stream.h"

namespace SparseSurfelFusion {
	/**
	 * \brief 二进制文件流继承Stream抽象类.
	 */
	class BinaryFileStream : public Stream
	{
	private:
		std::FILE* fileStream;
	public:
		using Ptr = std::shared_ptr<BinaryFileStream>;
		/**
		 * \brief 文件操作的模式，是读文件还是写文件.
		 */
		enum class FileOperationMode {
			ReadOnly,	// 文件只读
			WriteOnly	// 文件只存
		};
		/**
		 * \brief 构造函数(只能显式构造)，从给定文件路径中，读写二进制文件.
		 * 
		 * \param filePath 二进制文件路径
		 * \param mode 文件操作模式，默认只读
		 */
		explicit BinaryFileStream(const char* filePath, FileOperationMode mode = FileOperationMode::ReadOnly);
		/**
		 * \brief 析构函数，基类析构时调用了该派生类的析构函数.
		 * 
		 */
		~BinaryFileStream() override;
		/**
		 * \brief 这个函数的作用是从文件流fileStream中读取指定数量的数据，并将读取的数据存储到给定的内存位置(重写了Stream中的虚函数).
		 * 
		 * \param ptr 从给定的文件流 stream 中读取数据，并将读取的数据存储到 ptr 指向的内存位置
		 * \param bytes 指定要读取的数据项的数量
		 * \return 返回实际成功读取的数据项数量
		 */
		size_t Read(void* ptr, size_t bytes) override;
		/**
		 * \brief 从ptr指向的内存位置写入到fileStream流，并将每个数据项大小设置为1，并检查成功写入的数据量是否与要写的数据量大小一致.
		 * 
		 * \param ptr 数据的地址
		 * \param bytes 数据的Byte数量
		 * \return 是否成功全部写入
		 */
		bool Write(const void* ptr, size_t bytes) override;
	};
}


