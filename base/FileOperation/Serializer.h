/*****************************************************************//**
 * \file   Serializer.h
 * \brief  ���л����������������л�����
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
	 * \brief ���л�������ͨ����.
	 */
	template<typename T>
	struct SerializeHandler;

	/**
	 * \brief ����Ƿ���Plain - Old - Data��Plain ����һ��������һ����ͨ���ͣ�Old ����һ����������� C ���ݡ�
	 *		  һ���ࡢ�ṹ������������ǹ������Ͷ�����ͨ�������ƿ���(�� memcpy())���ܱ��������ݲ�������ʹ�õľ���POD���͵Ķ���.
	 */
	template <typename T>
	struct CheckPOD {
		static const bool value = std::is_pod<T>::value;	// �ж��Ƿ���POD����
	};

	/**
	 * \brief ���������л�������򡿼��ж���ִ��Then�����еĺ���(condition = true)������ִ��Else�����еĺ���(condition = false).
	 */
	template<bool condition, typename Then, typename Else, typename T>
	struct IfThenElse;
	/**
	 * \brief condition = true��ִ��Then���͵ĺ���Write��Read.
	 * 
	 * \param stream �ļ���
	 * \param object ��������ݶ���
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
	 * \brief condition = false��ִ��Else���͵ĺ���Write��Read.
	 * 
	 * \param stream �ļ���
	 * \param object ��������ݶ���
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
	 * \brief POD���л���������T��ʾҪ���л�������.
	 */
	template<typename T>
	struct PODSerializeHandler {
		/**
		 * \brief ��T���͵�����dataд���ļ���.
		 * 
		 * \param stream ��Ҫд����ļ���
		 * \param data �����T���͵�data
		 */
		inline static void Write(Stream* stream, const T& data) {
			stream->Write(&data, sizeof(T));
		}
		/**
		 * \brief ��T���͵�����data���ļ����ж�����.
		 * 
		 * \param stream ���ļ����ж���
		 * \param data ָ��������ݵ�ָ��
		 * \return ����Ƿ���ȷ����
		 */
		inline static bool Read(Stream* stream, T* data) {
			return stream->Read(data, sizeof(T));
		}
	};

	/**
	 * \brief ���л�POD Vector��T��vector�д洢����������.
	 */
	template<typename T>
	struct PODVectorSerializeHandler {
		/**
		 * \brief ��vector�е�����д���ļ�����.
		 * 
		 * \param stream ��Ҫд����ļ���
		 * \param vec ����vector
		 */
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			uint64_t vectorSize = static_cast<uint64_t>(vec.size());
			stream->Write(&vectorSize, sizeof(vectorSize));
			if (vectorSize != 0) {
				stream->Write(&vec[0], sizeof(T) * vec.size());
			}
		}
		/**
		 * \brief ���ļ����е����ݶ���vector��.
		 * 
		 * \param stream ���ļ����ж���
		 * \param vec ָ��vector��ָ�룬�����ݶ���vector��
		 * \return �Ƿ�ɹ�����
		 */
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			uint64_t rawVectorSize;
			// ��ȡ��С������ȡ�Ƿ�ɹ�
			if (!(stream->Read(&rawVectorSize, sizeof(uint64_t)))) {
				return false;
			}
			// ��������vector��һ����ָ�룬�����һ���ڴ�
			if (vec == nullptr) {
				vec = new std::vector<T>();
			}

			// Ԥ���ռ�
			size_t vectorSize = static_cast<size_t>(rawVectorSize);
			vec->resize(vectorSize);	// ����������С

			// ���ļ����ж�ȡ���ݵ�vector��data��
			if (rawVectorSize != 0) {
				return stream->Read(vec->data(), sizeof(T) * vectorSize);
			}
			else {
				LOGGING(INFO) << "ԭʼ�ļ�������Ϊ��";
				return true;
			}
		}
	};

	/**
	 * \brief ͨ�õķ�ʽ�����л��ͷ����л��洢�� std::vector �е�Ԫ��.
	 */
	template<typename T>
	struct ComposedVectorSerializeHandler
	{
		/**
		 * \brief ������д��vector������.
		 * 
		 * \param stream �ļ���
		 * \param vec ����vector
		 */
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			uint64_t vectorSize = static_cast<uint64_t>(vec.size());
			stream->Write(&vectorSize, sizeof(vectorSize));
			if (vectorSize == 0) return;
			// ʹ���Զ�������л�������
			for (int i = 0; i < vec.size(); i++) {
				SerializeHandler<T>::Write(stream, vec[i]);	// ������һ��һ��д��
			}
		}
		/**
		 * \brief ���ļ����ж�ȡ���ݸ�ֵ��vector.
		 * 
		 * \param stream ��ȡ���ݵ��ļ���
		 * \param vec ������ݵ�vector
		 * \return �Ƿ��ȡ�ɹ�
		 */
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			uint64_t rawVectorSize;
			//Read the size and check the read is success
			if (!(stream->Read(&rawVectorSize, sizeof(uint64_t)))) {
				return false;
			}

			// ��������vector��һ����ָ�룬�����һ���ڴ�
			if (vec == nullptr) {
				vec = new std::vector<T>();
			}

			// Ԥ���ռ�
			size_t vec_size = static_cast<size_t>(rawVectorSize);
			vec->resize(vec_size);

			// ������Ƿ���һ����Vector
			if (vec->size() == 0) return true;

			// Ϊÿ��Ԫ�ؼ���Ԫ��
			for (int i = 0; i < vec->size(); i++) {
				SerializeHandler<T>::Read(stream, &((*vec)[i]));
			}
		}
	};

	/**
	 * \brief ���������͵����л�������.
	 */
	template<typename T>
	struct SerializeHandler {
		/**
		 * \brief ��ȡ���������͵����ݣ�д�뵽Stream��.
		 * 
		 * \param stream ������д���ļ�����
		 * \param object ���������
		 */
		inline static void Write(Stream* stream, const T& object) {
			if (CheckPOD<T>::value)	PODSerializeHandler<T>::Write(stream, object);
			else LOGGING(FATAL) << "û��ʵ�ַ�POD���л����ļ����롣";
		}

		/**
		 * \brief ��ȡStream�е����ݣ�д����������͵Ķ�����.
		 * 
		 * \param stream ��Stream�ж�ȡ����
		 * \param object ������д��˵�ַ
		 * \return �Ƿ�ɹ���ȡ�ļ����е�����
		 */
		inline static bool Read(Stream* stream, T* object) {
			if (CheckPOD<T>::value)	PODSerializeHandler<T>::Read(stream, object);
			else LOGGING(FATAL) << "û��ʵ�ַ�POD���л����ļ���ȡ��";
		}
	};

	/**
	 * \brief ����vector����������л�������.
	 */
	template<typename T>
	struct SerializeHandler<std::vector<T>> {
		/**
		 * \brief ������д��stream���ж��Ƿ���POD���͵Ķ����������ֱ��ʹ��PODVector���л����������ɣ����������ʹ��ComposedVector���л���������.
		 * 
		 * \param stream д���������
		 * \param vec �������ȡ����д��stream
		 */
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			IfThenElse< CheckPOD<T>::value, PODVectorSerializeHandler<T>, ComposedVectorSerializeHandler<T>, std::vector<T> >::Write(stream, vec);
		}

		/**
		 * \brief ��stream�ж�ȡ���ݣ�����vector�С��ж��Ƿ���POD���͵Ķ����������ֱ��ʹ��PODVector���л����������ɣ����������ʹ��ComposedVector���л���������.
		 * 
		 * \param stream ��steam�ж�ȡ����
		 * \param vec ��stream���ݴ���vec
		 * \return �����Ƿ��ȡ���ݳɹ�
		 */
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			return IfThenElse< CheckPOD<T>::value, PODVectorSerializeHandler<T>, ComposedVectorSerializeHandler<T>, std::vector<T> >::Read(stream, vec);
		}
	};

	/**
	 * \brief ��ȡ��д��GPU�д洢��vector���ݵ����л�������.
	 * 
	 */
	template <typename T>
	struct SerializeHandler<DeviceArray<T>> {
		/**
		 * \brief ��GPU�е�����(DeviceArray)д��stream��.
		 * 
		 * \param stream ��vec������д����ļ���
		 * \param vec GPU��DeviceArray���͵�����
		 */
		inline static void Write(Stream* stream, const DeviceArray<T>& vec) {
			std::vector<T> hostVector;
			vec.download(hostVector);
			SerializeHandler<std::vector<T>>::Write(stream, hostVector);
		}
		/**
		 * \brief ��stream�����ݶ�������д�뵽GPU��(DeviceArray).
		 * 
		 * \param stream �����ݴӸ�stream�ж�ȡ
		 * \param vec ��stream�е����ݶ�ȡ��д��GPU��(���DeviceArray����)
		 * \return �Ƿ��ȡ���ݳɹ�(�ϴ���GPU֮ǰ)
		 */
		inline static bool Read(Stream* stream, DeviceArray<T>* vec) {
			std::vector<T> hostVector;
			const bool readSuccess = SerializeHandler<std::vector<T>>::Read(stream, &hostVector);
			if (!readSuccess) return false;

			// ��ȡ���ݳɹ����ϴ���GPU
			vec->upload(hostVector);
			return true;
		}
	};

	/**
	 * \brief ��ȡ��д��GPU�д洢��vector���ݵ����л�������.
	 * 
	 */
	template<typename T>
	struct SerializeHandler<DeviceArrayView<T>> {
		/**
		 * \brief ��ȡGPU�е�����(DeviceArrayView����)��������д�뵽stream��.
		 * 
		 * \param stream ��vec������д����ļ���
		 * \param vec GPU��DeviceArrayView���͵�����
		 */
		inline static void Write(Stream* stream, const DeviceArrayView<T>& vec) {
			std::vector<T> hostVector;
			vec.Download(hostVector);
			SerializeHandler<std::vector<T>>::Write(stream, hostVector);
		}
		/**
		 * \brief ��stream�����ݶ�ȡ��д�뵽GPU��(DeviceArrayView����)��DeviceArrayView��ֻ�����ͣ�д���ǷǷ��ġ�.
		 * 
		 * \param stream �����ݴӸ�stream�ж�ȡ��DeviceArrayView��ֻ�����ͣ�д���ǷǷ��ġ�
		 * \param vec ��stream�е����ݶ�ȡ��д��GPU��(���DeviceArrayView����)��DeviceArrayView��ֻ�����ͣ�д���ǷǷ��ġ�
		 * \return �Ƿ��ȡ���ݳɹ�(�ϴ���GPU֮ǰ)��DeviceArrayView��ֻ�����ͣ�д���ǷǷ��ġ�
		 */
		inline static bool Read(Stream* stream, DeviceArrayView<T>* vec) {
			LOGGING(FATAL) << "���ܽ�����д��DeviceArrayView��DeviceArrayView��ֻ�����͡�����ص�std::vector��DeviceArray";
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
