/*****************************************************************//**
 * \file   ConfigParser.h
 * \brief  主要涉及的是一些相机内参，图片剪裁等基本参数的处理
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
     * \brief 主要涉及的是一些相机内参，图片剪裁等基本参数的处理.
     */
    class ConfigParser {

    private:
        int deviceCount;                                        // 相机数量
        Intrinsic ColorIntrinsic[MAX_CAMERA_COUNT];             // RGB图像的内参
        Intrinsic DepthIntrinsic[MAX_CAMERA_COUNT];             // Depth图像的内参
        Intrinsic ClipColorIntrinsic[MAX_CAMERA_COUNT];         // 剪裁图像的内参
        

        unsigned int maxDepthMillimeter = MAX_DEPTH_THRESHOLD;  // 最大深度[单位：mm]
        unsigned int minDepthMillimeter = MIN_DEPTH_THRESHOLD;  // 最小深度[单位：mm]

        const unsigned int rawImageRowsCliped = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
        const unsigned int rawImageColsCliped = FRAME_WIDTH - 2 * CLIP_BOUNDARY;

        GetFrameFromCamera::Ptr cameras;                        // 记录cameras参数

        bool m_use_periodic_reinit;			//开启周期性重新启动
        int m_reinit_period;				//周期性重启的时间
        void setDefaultPeroidsValue() {     //设置默认的周期参数
            m_use_periodic_reinit = true;
            m_reinit_period = REINITIALIZATION_TIME;  // By default, no periodic
        };		

    public:
        bool use_periodic_reinit() const { return m_use_periodic_reinit; }	//返回周期参数m_use_periodic_reinit
        int reinit_period() const { return m_reinit_period; }				//返回周期参数m_reinit_period	
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
         * \brief 传入相机参数.
         * 
         * \param Cameras 传入获得相机的参数
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
         * \brief 获得最大深度(单位：m).
         * 
         * \return 最大深度(单位：m).
         */
        float getMaxDepthMeter() {
            float maxDepthMeter = maxDepthMillimeter / 1000.0f;
            return maxDepthMeter;
        }
        /**
         * \brief 获得最小深度(单位：m).
         * 
         * \return 最小深度(单位：m).
         */
        float getMinDepthMeter() {
            float minDepthMeter = minDepthMillimeter / 1000.0f;
            return minDepthMeter;
        }

        /**
         * \brief 从本地数据集中获取数据.
         * 
         * \param LocalData 本地数据
         */
        explicit ConfigParser(GetImageFromLocal LocalData) {
            setDefaultPeroidsValue();
        }

        /**
         * \brief 获得接入摄像头的数量.
         * 
         * \return 返回摄像头数量
         */
        int getDeviceCount() {
            return deviceCount;
        }

        /**
         * \brief 获得每个RGB摄像头的内参数组.
         * 
         * \return 内参数组
         */
        Intrinsic getColorIntrinsic(unsigned int CameraID) {
            return ColorIntrinsic[CameraID];
        }

        /**
         * \brief 获得每个深度相机的内参矩阵.
         * 
         * \return 内参矩阵
         */
        Intrinsic getDepthIntrinsic(unsigned int CameraID) {
            return DepthIntrinsic[CameraID];
        }

        /**
         * \brief 获得每个剪裁后的RGB相机的内参矩阵，主要是光心有变化.
         * 
         * \return 剪裁后的RGB相机的内参矩阵数组
         */
        Intrinsic getClipColorIntrinsic(unsigned int CameraID) {
            return ClipColorIntrinsic[CameraID];
        }
        /**
         * \brief 获得剪裁的所有相机内参的数组.
         * 
         * \return 剪裁的所有相机内参的数组
         */
        Intrinsic* getClipedIntrinsicArray() {
            return ClipColorIntrinsic;
        }
        /**
         * \brief 获得剪裁后的图像的高.
         * 
         * \return 剪裁后的图像的高
         */
        unsigned int getImageClipedRows() {
            return rawImageRowsCliped;
        }
        /**
         * \brief 获得剪裁后图像的宽.
         * 
         * \return 剪裁后图像的宽
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

