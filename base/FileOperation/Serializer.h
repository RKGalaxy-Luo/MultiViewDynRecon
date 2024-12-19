/*****************************************************************//**
 * \file   Serializer.h
 * \brief  序列化处理器，用于序列化处理
 * 
 * \author LUO
 * \date   March 11th 2024
 *********************************************************************/
#pragma once
#include <vector>
#include <iostream>
#include <type_traits>
#include <base/Logging.h>
#include <base/CommonTypes.h>
#include <base/FileOperation/Stream.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>

namespace SparseSurfelFusion {
	/**
	 * \brief 序列化处理器通用类.
	 */
	template<typename T>
	struct SerializeHandler;

	/**
	 * \brief 检查是否是Plain - Old - Data，Plain 代表一个对象是一个普通类型，Old 代表一个对象可以与 C 兼容。
	 *		  一个类、结构、共用体对象或非构造类型对象能通过二进制拷贝(如 memcpy())后还能保持其数据不变正常使用的就是POD类型的对象.
	 */
	template <typename T>
	struct CheckPOD {
		static const bool value = std::is_pod<T>::value;	// 判断是否是POD类型
	};

	/**
	 * \brief 【帮助序列化处理程序】简单判断是执行Then类型中的函数(condition = true)，还是执行Else类型中的函数(condition = false).
	 */
	template<bool condition, typename Then, typename Else, typename T>
	struct IfThenElse;
	/**
	 * \brief condition = true，执行Then类型的函数Write和Read.
	 * 
	 * \param stream 文件流
	 * \param object 传入的数据对象
	 */
	template<typename Then, typename Else, typename T>
	struct IfThenElse<true, Then, Else, T> {
		inline static void Write(Stream* stream, const T& object) {
			Then::Write(stream, object);
		}
		inline static bool Read(Stream* stream, T* object) {
			return Then::Read(stream, object);
		}
	};
	/**
	 * \brief condition = false，执行Else类型的函数Write和Read.
	 * 
	 * \param stream 文件流
	 * \param object 传入的数据对象
	 */
	template<typename Then, typename Else, typename T>
	struct IfThenElse<false, Then, Else, T> {
		inline static void Write(Stream* stream, const T& object) {
			Else::Write(stream, object);
		}
		inline static bool Read(Stream* stream, T* object) {
			return Else::Read(stream, object);
		}
	};

	/**
	 * \brief POD序列化处理器，T表示要序列化的类型.
	 */
	template<typename T>
	struct PODSerializeHandler {
		/**
		 * \brief 将T类型的数据data写入文件流.
		 * 
		 * \param stream 需要写入的文件流
		 * \param data 传入的T类型的data
		 */
		inline static void Write(Stream* stream, const T& data) {
			stream->Write(&data, sizeof(T));
		}
		/**
		 * \brief 将T类型的数据data从文件流中读出来.
		 * 
		 * \param stream 从文件流中读出
		 * \param data 指向读出数据的指针
		 * \return 检查是否正确读出
		 */
		inline static bool Read(Stream* stream, T* data) {
			return stream->Read(data, sizeof(T));
		}
	};

	/**
	 * \brief 序列化POD Vector，T是vector中存储的数据类型.
	 */
	template<typename T>
	struct PODVectorSerializeHandler {
		/**
		 * \brief 将vector中的数据写入文件流中.
		 * 
		 * \param stream 需要写入的文件流
		 * \param vec 数据vector
		 */
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			uint64_t vectorSize = static_cast<uint64_t>(vec.size());
			stream->Write(&vectorSize, sizeof(vectorSize));
			if (vectorSize != 0) {
				stream->Write(&vec[0], sizeof(T) * vec.size());
			}
		}
		/**
		 * \brief 将文件流中的数据读到vector中.
		 * 
		 * \param stream 从文件流中读出
		 * \param vec 指向vector的指针，将数据读到vector中
		 * \return 是否成功读出
		 */
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			uint64_t rawVectorSize;
			// 读取大小并检查读取是否成功
			if (!(stream->Read(&rawVectorSize, sizeof(uint64_t)))) {
				return false;
			}
			// 如果传入的vector是一个空指针，则分配一个内存
			if (vec == nullptr) {
				vec = new std::vector<T>();
			}

			// 预留空间
			size_t vectorSize = static_cast<size_t>(rawVectorSize);
			vec->resize(vectorSize);	// 调整容器大小

			// 从文件流中读取数据到vector的data中
			if (rawVectorSize != 0) {
				return stream->Read(vec->data(), sizeof(T) * vectorSize);
			}
			else {
				LOGGING(INFO) << "原始文件流数据为空";
				return true;
			}
		}
	};

	/**
	 * \brief 通用的方式来序列化和反序列化存储在 std::vector 中的元素.
	 */
	template<typename T>
	struct ComposedVectorSerializeHandler
	{
		/**
		 * \brief 向流中写入vector的数据.
		 * 
		 * \param stream 文件流
		 * \param vec 数据vector
		 */
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			uint64_t vectorSize = static_cast<uint64_t>(vec.size());
			stream->Write(&vectorSize, sizeof(vectorSize));
			if (vectorSize == 0) return;
			// 使用自定义的序列化处理器
			for (int i = 0; i < vec.size(); i++) {
				SerializeHandler<T>::Write(stream, vec[i]);	// 将数据一个一个写入
			}
		}
		/**
		 * \brief 从文件流中读取数据赋值给vector.
		 * 
		 * \param stream 读取数据的文件流
		 * \param vec 获得数据的vector
		 * \return 是否读取成功
		 */
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			uint64_t rawVectorSize;
			//Read the size and check the read is success
			if (!(stream->Read(&rawVectorSize, sizeof(uint64_t)))) {
				return false;
			}

			// 如果传入的vector是一个空指针，则分配一个内存
			if (vec == nullptr) {
				vec = new std::vector<T>();
			}

			// 预留空间
			size_t vec_size = static_cast<size_t>(rawVectorSize);
			vec->resize(vec_size);

			// 检查这是否是一个空Vector
			if (vec->size() == 0) return true;

			// 为每个元素加载元素
			for (int i = 0; i < vec->size(); i++) {
				SerializeHandler<T>::Read(stream, &((*vec)[i]));
			}
		}
	};

	/**
	 * \brief 非容器类型的序列化处理器.
	 */
	template<typename T>
	struct SerializeHandler {
		/**
		 * \brief 读取非容器类型的数据，写入到Stream中.
		 * 
		 * \param stream 将数据写入文件流中
		 * \param object 传入的数据
		 */
		inline static void Write(Stream* stream, const T& object) {
			if (CheckPOD<T>::value)	PODSerializeHandler<T>::Write(stream, object);
			else LOGGING(FATAL) << "没有实现非POD序列化的文件存入。";
		}

		/**
		 * \brief 读取Stream中的数据，写入非容器类型的对象中.
		 * 
		 * \param stream 从Stream中读取数据
		 * \param object 将数据写入此地址
		 * \return 是否成功读取文件流中的数据
		 */
		inline static bool Read(Stream* stream, T* object) {
			if (CheckPOD<T>::value)	PODSerializeHandler<T>::Read(stream, object);
			else LOGGING(FATAL) << "没有实现非POD序列化的文件读取。";
		}
	};

	/**
	 * \brief 处理vector容器类的序列化处理器.
	 */
	template<typename T>
	struct SerializeHandler<std::vector<T>> {
		/**
		 * \brief 将数据写入stream【判断是否是POD类型的对象，如果是则直接使用PODVector序列化处理器即可，如果不是则使用ComposedVector序列化处理器】.
		 * 
		 * \param stream 写入的数据流
		 * \param vec 从这里读取数据写入stream
		 */
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			IfThenElse< CheckPOD<T>::value, PODVectorSerializeHandler<T>, ComposedVectorSerializeHandler<T>, std::vector<T> >::Write(stream, vec);
		}

		/**
		 * \brief 从stream中读取数据，存入vector中【判断是否是POD类型的对象，如果是则直接使用PODVector序列化处理器即可，如果不是则使用ComposedVector序列化处理器】.
		 * 
		 * \param stream 从steam中读取数据
		 * \param vec 将stream数据存入vec
		 * \return 返回是否读取数据成功
		 */
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			return IfThenElse< CheckPOD<T>::value, PODVectorSerializeHandler<T>, ComposedVectorSerializeHandler<T>, std::vector<T> >::Read(stream, vec);
		}
	};

	/**
	 * \brief 读取或写入GPU中存储的vector数据的序列化处理器.
	 * 
	 */
	template <typename T>
	struct SerializeHandler<DeviceArray<T>> {
		/**
		 * \brief 将GPU中的数据(DeviceArray)写入stream中.
		 * 
		 * \param stream 将vec中数据写入该文件流
		 * \param vec GPU中DeviceArray类型的数据
		 */
		inline static void Write(Stream* stream, const DeviceArray<T>& vec) {
			std::vector<T> hostVector;
			vec.download(hostVector);
			SerializeHandler<std::vector<T>>::Write(stream, hostVector);
		}
		/**
		 * \brief 将stream的数据读出，并写入到GPU中(DeviceArray).
		 * 
		 * \param stream 将数据从该stream中读取
		 * \param vec 将stream中的数据读取后写入GPU中(存成DeviceArray类型)
		 * \return 是否读取数据成功(上传到GPU之前)
		 */
		inline static bool Read(Stream* stream, DeviceArray<T>* vec) {
			std::vector<T> hostVector;
			const bool readSuccess = SerializeHandler<std::vector<T>>::Read(stream, &hostVector);
			if (!readSuccess) return false;

			// 读取数据成功，上传到GPU
			vec->upload(hostVector);
			return true;
		}
	};

	/**
	 * \brief 读取或写入GPU中存储的vector数据的序列化处理器.
	 * 
	 */
	template<typename T>
	struct SerializeHandler<DeviceArrayView<T>> {
		/**
		 * \brief 读取GPU中的数据(DeviceArrayView类型)，并将其写入到stream中.
		 * 
		 * \param stream 将vec中数据写入该文件流
		 * \param vec GPU中DeviceArrayView类型的数据
		 */
		inline static void Write(Stream* stream, const DeviceArrayView<T>& vec) {
			std::vector<T> hostVector;
			vec.Download(hostVector);
			SerializeHandler<std::vector<T>>::Write(stream, hostVector);
		}
		/**
		 * \brief 将stream的数据读取并写入到GPU中(DeviceArrayView类型)【DeviceArrayView是只读类型，写入是非法的】.
		 * 
		 * \param stream 将数据从该stream中读取【DeviceArrayView是只读类型，写入是非法的】
		 * \param vec 将stream中的数据读取后写入GPU中(存成DeviceArrayView类型)【DeviceArrayView是只读类型，写入是非法的】
		 * \return 是否读取数据成功(上传到GPU之前)【DeviceArrayView是只读类型，写入是非法的】
		 */
		inline static bool Read(Stream* stream, DeviceArrayView<T>* vec) {
			LOGGING(FATAL) << "不能将数据写入DeviceArrayView，DeviceArrayView是只读类型。请加载到std::vector或DeviceArray";
			return false;
		}
	};
}

template<typename T>
inline bool SparseSurfelFusion::Stream::SerializeRead(T* output)
{
	return SerializeHandler<T>::Read(this, output);
}

template<typename T>
inline void SparseSurfelFusion::Stream::SerializeWrite(const T& object)
{
	SerializeHandler<T>::Write(this, object);
}
