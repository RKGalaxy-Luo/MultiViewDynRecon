#pragma once
#include <type_traits>
#include <string>
#include <exception>

namespace SparseSurfelFusion {

	template <typename T>
	struct is_pod {
		static const bool value = std::is_pod<T>::value;
	};

	//前向声明：在编程中，一种声明变量或函数的方法，但不定义它们的具体实现。
	//这样做的目的是让编译器知道这些变量或函数的存在，以便在其他地方使用它们
	class Stream;

	//从流中加载和保存的通用函数
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