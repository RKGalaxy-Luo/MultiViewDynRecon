/*****************************************************************//**
 * \file   CommonTypes.h
 * \brief  ����һЩ���������ͣ���������������������
 * 
 * \author LUO
 * \date   January 9th 2024
 *********************************************************************/
#pragma once
#include "GlobalConfigs.h"
#include "DeviceAPI/convenience.cuh"
#include "DeviceAPI/device_array.hpp"
#include "DeviceAPI/kernel_containers.hpp"
#include "DeviceAPI/safe_call.hpp"
/*	�� CUDA 5.5 �� Eigen 3.3 ��ʼ�������� CUDA �ں���ʹ�� Eigen �ľ���������������ʵ�̶ֹ���С�����ڴ������С����ʱ�ر����á�
	Ĭ������£��� Eigen �ı�ͷ�������� nvcc ����� .cu �ļ���ʱ������� Eigen �ĺ����ͷ��������� device host �ؼ�����Ϊǰ׺��
	�Ӷ����Դ��������豸����������ǡ�����ͨ���ڰ����κ� Eigen �ı�ͷ֮ǰ���� EIGEN_NO_CUDA �����ô�֧�֡�
	�� .cu �ļ�����������ʹ�� Eigen ʱ������������ڽ���ĳЩ���档*/
//CUDA��ʹ��Eigen��
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif
#include <Eigen/Eigen>//����Dense��Sparse:������Eigen���ȫ������
//CUDA����
#include <vector_functions.h>
#include <vector>
//**************************************hsg
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
typedef pcl::PointCloud<pcl::PointXYZ>    PointCloud3f;
typedef pcl::PointCloud<pcl::Normal>      PointCloudNormal;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud3fRGB;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr    PointCloud3f_Pointer;
typedef pcl::PointCloud<pcl::Normal>::Ptr      PointCloudNormal_Pointer;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud3fRGB_Pointer;

//**************************************hsg

namespace SparseSurfelFusion {
	using Matrix3f = Eigen::Matrix3f;
	using Vector3f = Eigen::Vector3f;
	using Matrix4f = Eigen::Matrix4f;
	using Vector4f = Eigen::Vector4f;
	using Matrix6f = Eigen::Matrix<float, 6, 6>;
	using Vector6f = Eigen::Matrix<float, 6, 1>;
	using MatrixXf = Eigen::MatrixXf;
	using VectorXf = Eigen::VectorXf;
	using Isometry3f = Eigen::Isometry3f;
	/* host��device���ʵ� gpu ����
	*/
	//��DeviceArray����DeviceArrayPCL
	template<typename T>
	using DeviceArray = DeviceArrayPCL<T>;
	//��DeviceArray2D����DeviceArray2DPCL
	template<typename T>
	using DeviceArray2D = DeviceArray2DPCL<T>;

	namespace device {
		//device���� gpu ����������
		//����Device��ַ��ָ��
		template<typename T>
		using DevicePtr = DevPtr<T>;

		//����Device��ַ�Լ����ݴ�С
		template<typename T>
		using PtrSize = PtrSzPCL<T>;

		//����Device��ַ�Լ������ֽ�Ϊ��λ������������֮��Ĳ��������Խ�DeviceArray����ת��PtrStep�����ں˺�����ֱ�ӷ���(DeviceArray�ں˺����в���ֱ�ӷ���)
		template<typename T>
		using PtrStep = PtrStepPCL<T>;

		//����Device��ַ�����ݵĴ�С�Լ������ֽ�Ϊ��λ������������֮��Ĳ���
		template<typename T>
		using PtrStepSize = PtrStepSzPCL<T>;
	}

	/**
	 * ����ڲ�
	 */
	struct Intrinsic
	{
		/**
		 * \brief ������host��device�������Intrinsic��������
		 * \return
		 */
		__host__ __device__ Intrinsic()
			: principal_x(0), principal_y(0),
			focal_x(0), focal_y(0) {}

		__host__ __device__ Intrinsic(
			const float focal_x_, const float focal_y_,
			const float principal_x_, const float principal_y_
		) : principal_x(principal_x_), principal_y(principal_y_),
			focal_x(focal_x_), focal_y(focal_y_) {}

		//����float4   [ֱ������float4(),��������ֱ���Ҹ�ֵ -> float4 A = (Intrinsic) B]
		__host__ operator float4() {
			return make_float4(principal_x, principal_y, focal_x, focal_y);
		}

		// ����ڲ�
		float principal_x, principal_y;
		float focal_x, focal_y;
	};

	/**
	 * ����ڲε���
	 */
	struct IntrinsicInverse
	{
		//Ĭ�ϸ���ֵ0
		__host__ __device__ IntrinsicInverse() : principal_x(0), principal_y(0), inv_focal_x(0), inv_focal_y(0) {}

		// ����ڲ�
		float principal_x, principal_y; //����ڲ����ĵ�
		float inv_focal_x, inv_focal_y; //�������ĵ���
	};

	/**
	 * \brief ��������������ϣ�������cudaSurfaceObject_t��cudaTextureObject_t�������ͣ��Լ���Ӧ����cudaArray.
	 */
	struct CudaTextureSurface {
		cudaTextureObject_t texture;	//�����ڴ棬�ɶ�����д�����ص�ֱ�ӿ��Խ���Ӳ����ֵ
		cudaSurfaceObject_t surface;	//�����ڴ棬�ɶ���д��������OpenGL�е�Resource�ռ����ӳ����߹�����ͨ��CPU
		cudaArray_t cudaArray;			//����һ��Array������ݣ�����cudaArray_t����Ҳ����������GPU��ʵ�ʵ�����
	};

	//����ȡ�������������������ģ�ͬconvenience.cuh�е�getGridDim()����
	using pcl::gpu::divUp;

	/**
	 * \brief �����ṹ���¼����.
	 */
	struct PixelCoordinate {
		unsigned int row;	// ͼ��ĸߣ����ص�y����
		unsigned int col;	// ͼ��Ŀ����ص�x����
		__host__ __device__ PixelCoordinate() : row(0), col(0) {}
		__host__ __device__ PixelCoordinate(const unsigned row_, const unsigned col_)
			: row(row_), col(col_) {}

		__host__ __device__ const unsigned int& x() const { return col; }
		__host__ __device__ const unsigned int& y() const { return row; }
		__host__ __device__ unsigned int& x() { return col; }
		__host__ __device__ unsigned int& y() { return row; }
	};

	/**
	 * \brief �����ͼ�񹹽���surfel�ṹӦ�����豸�Ϸ���.
	 */
	struct DepthSurfel {
		PixelCoordinate pixelCoordinate;	// pixelCoordinate��Ԫ��������
		float4 VertexAndConfidence;			// VertexAndConfidence (x, y, z)Ϊ���֡�е�λ�ã�(w)Ϊ���Ŷ�ֵ��
		float4 NormalAndRadius;				// NormalAndRadius (x, y, z)�ǹ�һ�����߷���w�ǰ뾶
		float4 ColorAndTime;				// ColorAndTime (x)�Ǹ�������RGBֵ;(z)Ϊ���һ�ι۲�ʱ��;(w)Ϊ��ʼ��ʱ��
		bool isMerged;						// ��¼�Ƿ��ںϹ���true�Ǳ��ںϹ��˵���Ԫ�����ٲ��������ںϣ�false��δ���ںϹ�����Ԫ�����Բ��������ӽǵ��ں�
		short CameraID;						// ��¼������һ�����
	};

	struct KNNAndWeight {
		ushort4 knn;		// �ٽ�4�����ID
		float4 weight;		// �ٽ�4�����Ȩ��

		__host__ __device__ void setInvalid() {
			knn.x = knn.y = knn.z = knn.w = 0xFFFF;
			weight.x = weight.y = weight.z = weight.w = 0.0f;
		}
	};

	/**
	 * \brief ����ɫ�Ķ���.
	 */
	struct ColorVertex {
		float3 coor;	// ��ά����
		float3 color;	// RGB
	};

	/**
	 * \brief ���ڼ�¼�羵ƥ���.
	 */
	struct CrossViewCorrPairs {
		ushort4 PixelPairs;
		ushort2 PixelViews;
	};
	enum DatasetType {
		// 640_400_120��_1.0m
		LowRes_1000mm_calibration,      // �ͷֱ��ʣ�1m���궨���ݼ�
		LowRes_1000mm_boxing_slow,      // �ͷֱ��ʣ�1m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		LowRes_1000mm_boxing_fast,      // �ͷֱ��ʣ�1m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		LowRes_1000mm_doll_slow,        // �ͷֱ��ʣ�1m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		LowRes_1000mm_doll_fast,        // �ͷֱ��ʣ�1m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		LowRes_1000mm_coat_slow,        // �ͷֱ��ʣ�1m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		LowRes_1000mm_coat_fast,        // �ͷֱ��ʣ�1m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		LowRes_1000mm_claphands_slow,   // �ͷֱ��ʣ�1m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		LowRes_1000mm_claphands_fast,   // �ͷֱ��ʣ�1m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����

		// 1280_720_120��_1.0m
		HighRes_1000mm_boxing_slow,     // �߷ֱ��ʣ�1m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		HighRes_1000mm_boxing_fast,     // �߷ֱ��ʣ�1m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		HighRes_1000mm_doll_slow,       // �߷ֱ��ʣ�1m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		HighRes_1000mm_doll_fast,       // �߷ֱ��ʣ�1m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		HighRes_1000mm_coat_slow,       // �߷ֱ��ʣ�1m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		HighRes_1000mm_coat_fast,       // �߷ֱ��ʣ�1m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		HighRes_1000mm_claphands_slow,  // �߷ֱ��ʣ�1m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		HighRes_1000mm_claphands_fast,  // �߷ֱ��ʣ�1m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����


		// 640_400_120��_1.5m
		LowRes_1500mm_calibration,      // �ͷֱ��ʣ�1.5m���궨���ݼ�
		LowRes_1500mm_boxing_slow,      // �ͷֱ��ʣ�1.5m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		LowRes_1500mm_boxing_fast,      // �ͷֱ��ʣ�1.5m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		LowRes_1500mm_doll_slow,        // �ͷֱ��ʣ�1.5m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		LowRes_1500mm_doll_fast,        // �ͷֱ��ʣ�1.5m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		LowRes_1500mm_coat_slow,        // �ͷֱ��ʣ�1.5m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		LowRes_1500mm_coat_fast,        // �ͷֱ��ʣ�1.5m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		LowRes_1500mm_claphands_slow,   // �ͷֱ��ʣ�1.5m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		LowRes_1500mm_claphands_fast,   // �ͷֱ��ʣ�1.5m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		LowRes_1500mm_ChallengeTest_A,
		LowRes_1500mm_ChallengeTest_B,

		// 1280_720_120��_1.5m
		HighRes_1500mm_boxing_slow,     // �߷ֱ��ʣ�1.5m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		HighRes_1500mm_boxing_fast,     // �߷ֱ��ʣ�1.5m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		HighRes_1500mm_doll_slow,       // �߷ֱ��ʣ�1.5m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		HighRes_1500mm_doll_fast,       // �߷ֱ��ʣ�1.5m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		HighRes_1500mm_coat_slow,       // �߷ֱ��ʣ�1.5m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		HighRes_1500mm_coat_fast,       // �߷ֱ��ʣ�1.5m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		HighRes_1500mm_claphands_slow,  // �߷ֱ��ʣ�1.5m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		HighRes_1500mm_claphands_fast,  // �߷ֱ��ʣ�1.5m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����


		// 640_400_120��_2.0m
		LowRes_2000mm_calibration,      // �ͷֱ��ʣ�2.0m���궨���ݼ�
		LowRes_2000mm_boxing_slow,      // �ͷֱ��ʣ�2.0m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		LowRes_2000mm_boxing_fast,      // �ͷֱ��ʣ�2.0m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		LowRes_2000mm_doll_slow,        // �ͷֱ��ʣ�2.0m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		LowRes_2000mm_doll_fast,        // �ͷֱ��ʣ�2.0m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		LowRes_2000mm_coat_slow,        // �ͷֱ��ʣ�2.0m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		LowRes_2000mm_coat_fast,        // �ͷֱ��ʣ�2.0m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		LowRes_2000mm_claphands_slow,   // �ͷֱ��ʣ�2.0m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		LowRes_2000mm_claphands_fast,   // �ͷֱ��ʣ�2.0m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����

		// 1280_720_120��_2.0m
		HighRes_2000mm_boxing_slow,     // �߷ֱ��ʣ�2.0m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		HighRes_2000mm_boxing_fast,     // �߷ֱ��ʣ�2.0m��[����]���˳�ȭ + ���� + ����ҡ�֣���Ҫչʾ������
		HighRes_2000mm_doll_slow,       // �߷ֱ��ʣ�2.0m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		HighRes_2000mm_doll_fast,       // �߷ֱ��ʣ�2.0m��[����]������ż + ������ż + ��Ͷ��ż ����Ҫչʾ�Ǹ��ԣ�
		HighRes_2000mm_coat_slow,       // �߷ֱ��ʣ�2.0m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		HighRes_2000mm_coat_fast,       // �߷ֱ��ʣ�2.0m��[����]��ק���� + ������ + �������ף���Ҫչʾ�������˱任��
		HighRes_2000mm_claphands_slow,  // �߷ֱ��ʣ�2.0m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		HighRes_2000mm_claphands_fast,  // �߷ֱ��ʣ�2.0m��[����]�������� + ���˹����⣨��Ҫչʾ���˶�����
		LowRes_2000mm_ChallengeTest_A,
		LowRes_2000mm_ChallengeTest_B,
		LowRes_2000mm_ChallengeTest_C,
	};

	void DatasetSwitch(DatasetType type, std::string& res, std::string& dis, std::string& speed, std::string& action);
}