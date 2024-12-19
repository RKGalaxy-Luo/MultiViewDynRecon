/*****************************************************************//**
 * \file   ConfigParser.h
 * \brief  ��Ҫ�漰����һЩ����ڲΣ�ͼƬ���õȻ��������Ĵ���
 * 
 * \author LUO
 * \date   January 25th 2024
 *********************************************************************/
#pragma once
#include <base/CommonTypes.h>
#include <ImageProcess/FrameIO/GetFrameFromCamera.h>
#include <ImageProcess/FrameIO/GetImageFromLocal.h>
namespace SparseSurfelFusion {
    /**
     * \brief ��Ҫ�漰����һЩ����ڲΣ�ͼƬ���õȻ��������Ĵ���.
     */
    class ConfigParser {

    private:
        int deviceCount;                                        // �������
        Intrinsic ColorIntrinsic[MAX_CAMERA_COUNT];             // RGBͼ����ڲ�
        Intrinsic DepthIntrinsic[MAX_CAMERA_COUNT];             // Depthͼ����ڲ�
        Intrinsic ClipColorIntrinsic[MAX_CAMERA_COUNT];         // ����ͼ����ڲ�
        

        unsigned int maxDepthMillimeter = MAX_DEPTH_THRESHOLD;  // ������[��λ��mm]
        unsigned int minDepthMillimeter = MIN_DEPTH_THRESHOLD;  // ��С���[��λ��mm]

        const unsigned int rawImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
        const unsigned int rawImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;

        GetFrameFromCamera::Ptr cameras;                        // ��¼cameras����

        bool m_use_periodic_reinit;			//������������������
        int m_reinit_period;				//������������ʱ��
        void setDefaultPeroidsValue() {     //����Ĭ�ϵ����ڲ���
            m_use_periodic_reinit = true;
            m_reinit_period = REINITIALIZATION_TIME;  // By default, no periodic
        };		

    public:
        bool use_periodic_reinit() const { return m_use_periodic_reinit; }	//�������ڲ���m_use_periodic_reinit
        int reinit_period() const { return m_reinit_period; }				//�������ڲ���m_reinit_period	
        static bool UseForegroundTerm() {
#ifdef REBUILD_WITHOUT_BACKGROUND
            return false;
#else
            return true;
#endif // REBUILD_WITHOUT_BACKGROUND

        }
        static bool UseDensityTerm() { return true;  }

        static bool UseCrossViewTerm() { return true; }
    public:

        using Ptr = std::shared_ptr<ConfigParser>;

        /**
         * \brief �����������.
         * 
         * \param Cameras ����������Ĳ���
         */
        ConfigParser(GetFrameFromCamera::Ptr Cameras) {
            cameras = Cameras;
            deviceCount = Cameras->getCameraCount();
            for (int i = 0; i < deviceCount; i++) {
                ColorIntrinsic[i] = Cameras->getColorIntrinsic(i);
                DepthIntrinsic[i] = Cameras->getDepthIntrinsic(i);
                ClipColorIntrinsic[i] = Cameras->getClipColorIntrinsic(i);
            }
            setDefaultPeroidsValue();
        }



        /**
         * \brief ���������(��λ��m).
         * 
         * \return ������(��λ��m).
         */
        float getMaxDepthMeter() {
            float maxDepthMeter = maxDepthMillimeter / 1000.0f;
            return maxDepthMeter;
        }
        /**
         * \brief �����С���(��λ��m).
         * 
         * \return ��С���(��λ��m).
         */
        float getMinDepthMeter() {
            float minDepthMeter = minDepthMillimeter / 1000.0f;
            return minDepthMeter;
        }

        /**
         * \brief �ӱ������ݼ��л�ȡ����.
         * 
         * \param LocalData ��������
         */
        explicit ConfigParser(GetImageFromLocal LocalData) {
            setDefaultPeroidsValue();
        }

        /**
         * \brief ��ý�������ͷ������.
         * 
         * \return ��������ͷ����
         */
        int getDeviceCount() {
            return deviceCount;
        }

        /**
         * \brief ���ÿ��RGB����ͷ���ڲ�����.
         * 
         * \return �ڲ�����
         */
        Intrinsic getColorIntrinsic(unsigned int CameraID) {
            return ColorIntrinsic[CameraID];
        }

        /**
         * \brief ���ÿ�����������ڲξ���.
         * 
         * \return �ڲξ���
         */
        Intrinsic getDepthIntrinsic(unsigned int CameraID) {
            return DepthIntrinsic[CameraID];
        }

        /**
         * \brief ���ÿ�����ú��RGB������ڲξ�����Ҫ�ǹ����б仯.
         * 
         * \return ���ú��RGB������ڲξ�������
         */
        Intrinsic getClipColorIntrinsic(unsigned int CameraID) {
            return ClipColorIntrinsic[CameraID];
        }
        /**
         * \brief ��ü��õ���������ڲε�����.
         * 
         * \return ���õ���������ڲε�����
         */
        Intrinsic* getClipedIntrinsicArray() {
            return ClipColorIntrinsic;
        }
        /**
         * \brief ��ü��ú��ͼ��ĸ�.
         * 
         * \return ���ú��ͼ��ĸ�
         */
        unsigned int getImageClipedRows() {
            return rawImageRowsCliped;
        }
        /**
         * \brief ��ü��ú�ͼ��Ŀ�.
         * 
         * \return ���ú�ͼ��Ŀ�
         */
        unsigned int getImageClipedCols() {
            return rawImageColsCliped;
        }
        
        bool ShouldDoReinitConfig(const unsigned int frameIndex) {
            // Check the config
            if (!use_periodic_reinit()) {
                return false;
            }

            // Check the peroid
            const int period = reinit_period();
            FUNCTION_CHECK(period > 0);
            if (frameIndex > 0 && (frameIndex % period) == 0) {
                return true;
            }
            else
                return false;
        }

    }; 
}

