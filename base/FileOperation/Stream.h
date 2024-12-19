/*****************************************************************//**
 * \file   Stream.h
 * \brief  �ļ�����д�����࣬�����ļ�����д
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
		// �麯���ĵ�����ͨ��������麯������ʵ�ֵģ����ڶ��󴴽�ʱ���麯������δ�����죬����޷��ڹ��캯���н����麯���ĵ���
		/**
		 * \brief Ĭ�Ϲ��캯���������๹�캯���������麯��().
		 * 
		 */
		explicit Stream() = default;
		// ͨ������ָ�������ɾ��һ�����������ʱ�����������������������麯����ֻ����û������������������������������������������ܵ�����Դй©��δ����ȷ�������
		/**
		 * \brief Ĭ������������ͨ�����������������Ƽ����麯��.
		 * 
		 */
		virtual ~Stream() = default;

		/**
		 * \brief �ļ���ȡ�ӿڣ������ݴ��ļ����ж�����д��ptr��"=0"��ʾ�Ǵ��麯�������ڻ�����û��Ĭ�ϵ�ʵ�֣�����Ҫ�����������ʵ�ָú���.
		 * 
		 * \param ptr ���ݵĵ�ַ
		 * \param bytes ���ݵ�Byte��
		 * \return �Ƿ�ɹ�ȫ��д��
		 */
		virtual size_t Read(void* ptr, size_t bytes) = 0;
		/**
		 * \brief ���л���ȡ�����ļ����е����ݶ���output��.
		 * 
		 * \param output ���ļ������ݶ���˴�
		 * \return �Ƿ��ȡ�ļ����ݳɹ�
		 */
		template<typename T> 
		inline bool SerializeRead(T * output);

		/**
		 * \brief �ļ�д��ӿڣ������ݴ�ptr������д���ļ���"=0"��ʾ�Ǵ��麯�������ڻ�����û��Ĭ�ϵ�ʵ�֣�����Ҫ�����������ʵ�ָú���.
		 * 
		 * \param ptr ���ݴ�ptr�ж���
		 * \param bytes ���ݵĴ�С����λ��byte
		 * \return �Ƿ����ݳɹ�д���ļ���
		 */
		virtual bool Write(const void* ptr, size_t bytes) = 0;
		/**
		 * \brief ���л�д�룬������object����д���ļ�����.
		 * 
		 * \param object ���ݵĶ���
		 */
		template<typename T> 
		inline void SerializeWrite(const T & object);
	};
}
