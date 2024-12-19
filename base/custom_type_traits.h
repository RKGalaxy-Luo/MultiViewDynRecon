#pragma once
#include <type_traits>
#include <string>
#include <exception>

namespace SparseSurfelFusion {

	template <typename T>
	struct is_pod {
		static const bool value = std::is_pod<T>::value;
	};

	//ǰ���������ڱ���У�һ���������������ķ����������������ǵľ���ʵ�֡�
	//��������Ŀ�����ñ�����֪����Щ���������Ĵ��ڣ��Ա��������ط�ʹ������
	class Stream;

	//�����м��غͱ����ͨ�ú���
	template <typename T>
	inline bool streamLoad(Stream* stream, T* object) {
		throw new std::runtime_error("The stream load function is not implemented");
	}

	template <typename T>
	inline void streamSave(Stream* stream, const T& object) {
		throw new std::runtime_error("The stream save function is not implemented");
	}

	template <typename T>
	struct has_outclass_saveload {
		static const bool value = false;
	};

	template<typename T>
	struct has_inclass_saveload {
		static const bool value = false;
	};
}