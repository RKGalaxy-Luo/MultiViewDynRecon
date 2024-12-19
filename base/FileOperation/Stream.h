/*****************************************************************//**
 * \file   Stream.h
 * \brief  文件流读写抽象类，处理文件流读写
 * 
 * \author LUO
 * \date   March 2024
 *********************************************************************/
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <streambuf>

#include <base/CommonTypes.h>

namespace SparseSurfelFusion {
	class Stream {
	public:
		// 虚函数的调用是通过对象的虚函数表来实现的，而在对象创建时，虚函数表尚未被构造，因此无法在构造函数中进行虚函数的调用
		/**
		 * \brief 默认构造函数，抽象类构造函数不能是虚函数().
		 * 
		 */
		explicit Stream() = default;
		// 通过基类指针或引用删除一个派生类对象时，如果基类的析构函数不是虚函数，只会调用基类的析构函数而不会调用派生类的析构函数，这可能导致资源泄漏或未能正确清理对象
		/**
		 * \brief 默认析构函数，通常抽象类析构函数推荐是虚函数.
		 * 
		 */
		virtual ~Stream() = default;

		/**
		 * \brief 文件读取接口，将数据从文件流中读出，写到ptr。"=0"表示是纯虚函数：它在基类中没有默认的实现，而是要求派生类必须实现该函数.
		 * 
		 * \param ptr 数据的地址
		 * \param bytes 数据的Byte数
		 * \return 是否成功全部写入
		 */
		virtual size_t Read(void* ptr, size_t bytes) = 0;
		/**
		 * \brief 序列化读取，将文件流中的数据读入output中.
		 * 
		 * \param output 将文件流数据读入此处
		 * \return 是否读取文件数据成功
		 */
		template<typename T> 
		inline bool SerializeRead(T * output);

		/**
		 * \brief 文件写入接口，将数据从ptr读出，写到文件，"=0"表示是纯虚函数：它在基类中没有默认的实现，而是要求派生类必须实现该函数.
		 * 
		 * \param ptr 数据从ptr中读出
		 * \param bytes 数据的大小，单位：byte
		 * \return 是否将数据成功写入文件流
		 */
		virtual bool Write(const void* ptr, size_t bytes) = 0;
		/**
		 * \brief 序列化写入，将对象object数据写入文件流中.
		 * 
		 * \param object 数据的对象
		 */
		template<typename T> 
		inline void SerializeWrite(const T & object);
	};
}
