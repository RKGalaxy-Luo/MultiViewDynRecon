/*****************************************************************//**
 * \file   GetOneCamera.cpp
 * \brief  ����һ��Orbbec����ͷ���࣬������ͷ�ںϣ�ÿһ������ͷ����һ���������
 * 
 * \author LUO
 * \date   January 20th 2024
 *********************************************************************/
#include "GetOneCamera.h"

GetOneCamera::GetOneCamera(std::shared_ptr<ob::Pipeline> AStreamPipe, int PipeID, std::shared_ptr<ThreadPool> CameraThread) 
    : pipe(AStreamPipe) , CameraID(PipeID) , pool(CameraThread)
{

    ColorImageName = "RGB_Image_" + cv::format("%.1d", CameraID);
    DepthImageName = "Depth_Image_" + cv::format("%.1d", CameraID);
    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();
    //��ȡ�����������б�
    auto depthProfileList = pipe->getStreamProfileList(OB_SENSOR_DEPTH);
    std::shared_ptr<ob::VideoStreamProfile> depthProfile = nullptr;

    if (depthProfileList) {
        // ����ȴ�������Ĭ�������ļ�������ͨ�������ļ���������
        depthProfile = depthProfileList->getVideoStreamProfile(FRAME_WIDTH, OB_HEIGHT_ANY, OB_FORMAT_ANY, 30);
        //depthProfile = std::const_pointer_cast<ob::StreamProfile>(depthProfileList->getProfile(OB_PROFILE_DEFAULT))->as<ob::VideoStreamProfile>();
    }
    config->enableStream(depthProfile);

    // ��ȡ��ɫ����������б�
    try {
        auto colorProfileList = pipe->getStreamProfileList(OB_SENSOR_COLOR);
        std::shared_ptr<ob::VideoStreamProfile> colorProfile = nullptr;
        if (colorProfileList) {
            // �򿪡���ɫ����������Ĭ�������ļ�������ͨ�������ļ���������
            colorProfile = colorProfileList->getVideoStreamProfile(FRAME_WIDTH, OB_HEIGHT_ANY, OB_FORMAT_ANY, 30);
            //colorProfile = std::const_pointer_cast<ob::StreamProfile>(colorProfileList->getProfile(OB_PROFILE_DEFAULT))->as<ob::VideoStreamProfile>();
        }
        config->enableStream(colorProfile);
    }
    catch (ob::Error& e) {
        std::cerr << "��ǰ�豸��֧����ɫ������!" << std::endl;
    }

    config->setAlignMode(ALIGN_D2C_HW_MODE);    //Ӳ������

    //����pipeline�����������ã�������ͼ�񶼴���֮�������������lambda���ʽ
    pipe->start(config, [=](std::shared_ptr<ob::FrameSet> frameSet) {
        std::lock_guard<std::mutex> lock(frameMutex); // ��ס������ԴColorImage��DepthImage���ڽ��и�ֵ��ʱ�򲻿��Դ��
        if (frameSet->colorFrame() != nullptr) {
            colorFrames = frameSet->colorFrame();
        }
        if (frameSet->depthFrame() != nullptr) {
            depthFrames = frameSet->depthFrame();
        }
    });
}

GetOneCamera::GetOneCamera(std::string DataPath, int cameraID, std::shared_ptr<ThreadPool> CameraThread) : CameraID(cameraID)
{

}

void GetOneCamera::processFrames()
{
    while (!stop) {
        Frames.clear(); // �����ʱ�������
        {   // ���ܻ�������������
            std::lock_guard<std::mutex> lock(frameMutex);                       // ����߳�ͬʱ����colorFrames��depthFrames
            if (colorFrames != nullptr) Frames.emplace_back(colorFrames);
            if (depthFrames != nullptr) Frames.emplace_back(depthFrames);
        }
        {   // ���ܻ�������������
            std::lock_guard<std::mutex> lk(imageMutex);                         // ���ｫFrames���ݸ�srcFrames��������ͼ����ʾ�߳�
            srcFrames = Frames;
            activateShowThread.notify_one();
        }
    }
}

void GetOneCamera::convertToImage()
{
    std::vector<std::shared_ptr<ob::Frame>> frames;
    while (!stop) {
        {
            std::unique_lock<std::mutex> lk(imageMutex);
            activateShowThread.wait(lk, [this] { return !srcFrames.empty() || stop; });
            if (stop) break;
            frames = srcFrames;
            srcFrames.clear();
        }
        for (int i = 0; i < frames.size(); i++) {
            if (frames[i] == nullptr) break;
            FrameToMat(frames[i]);
        }
    }
}

void GetOneCamera::FrameToMat(std::shared_ptr<ob::Frame> frame)
{
    if (frame->type() == OB_FRAME_COLOR) {
        cv::Mat rstMat; // �����������ͷ���
        auto videoFrame = frame->as<ob::VideoFrame>();
        switch (videoFrame->format()) {
            case OB_FORMAT_MJPG: {
                cv::Mat rawMat(1, videoFrame->dataSize(), CV_8UC1, videoFrame->data());
                rstMat = cv::imdecode(rawMat, 1);
            } break;
            case OB_FORMAT_NV21: {
                cv::Mat rawMat(videoFrame->height() * 3 / 2, videoFrame->width(), CV_8UC1, videoFrame->data());
                cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_NV21);
            } break;
            case OB_FORMAT_YUYV:
            case OB_FORMAT_YUY2: {
                cv::Mat rawMat(videoFrame->height(), videoFrame->width(), CV_8UC2, videoFrame->data());
                cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_YUY2);
            } break;
            case OB_FORMAT_RGB: {
                cv::Mat rawMat(videoFrame->height(), videoFrame->width(), CV_8UC3, videoFrame->data());
                cv::cvtColor(rawMat, rstMat, cv::COLOR_RGB2BGR);
            } break;
            case OB_FORMAT_UYVY: {
                cv::Mat rawMat(videoFrame->height(), videoFrame->width(), CV_8UC2, videoFrame->data());
                cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_UYVY);
            } break;
            default:
                break;
        }
        {
            std::lock_guard<std::mutex> lock(showMutex);
            CurrentColorImage = rstMat; // ���
        }
        if(!colorPrepared) colorPrepared = true;
    }
    else if (frame->type() == OB_FRAME_DEPTH) {
        cv::Mat rstMat;// �����������ͷ���
        
        auto videoFrame = frame->as<ob::VideoFrame>();
        if (videoFrame->format() == OB_FORMAT_Y16) {
            cv::Mat cvtMat;
            cv::Mat rawMat = cv::Mat(videoFrame->height(), videoFrame->width(), CV_16UC1, videoFrame->data());
            // ���֡����ֵ�˳߶ȵõ��Ժ���Ϊ��λ�ľ���
            float scale = videoFrame->as<ob::DepthFrame>()->getValueScale();
            // MIN_DEPTH_THRESHOLD �� ���ص� �� MAX_DEPTH_THRESHOLD
            cv::threshold(rawMat, rstMat, MAX_DEPTH_THRESHOLD * 1.0 / scale, 0, cv::THRESH_TOZERO_INV);
            cv::threshold(rstMat, rstMat, MIN_DEPTH_THRESHOLD * 1.0 / scale, 0, cv::THRESH_TOZERO);
        }
        {
            std::lock_guard<std::mutex> lock(showMutex);
            CurrentDepthImage = rstMat; // ���
        }
        if(!depthPrepared) depthPrepared = true;
    }
    else if (frame->type() == OB_FRAME_IR || frame->type() == OB_FRAME_IR_LEFT || frame->type() == OB_FRAME_IR_RIGHT) {
        cv::Mat rstMat;// �����������ͷ���
        auto videoFrame = frame->as<ob::VideoFrame>();
        if (videoFrame->format() == OB_FORMAT_Y16) {
            cv::Mat cvtMat;
            cv::Mat rawMat = cv::Mat(videoFrame->height(), videoFrame->width(), CV_16UC1, videoFrame->data());
            rawMat.convertTo(cvtMat, CV_8UC1, 1.0 / 16.0f);
            cv::cvtColor(cvtMat, rstMat, cv::COLOR_GRAY2RGB);
        }
        else if (videoFrame->format() == OB_FORMAT_Y8) {
            cv::Mat rawMat = cv::Mat(videoFrame->height(), videoFrame->width(), CV_8UC1, videoFrame->data());
            cv::cvtColor(rawMat * 2, rstMat, cv::COLOR_GRAY2RGB);
        }
        else if (videoFrame->format() == OB_FORMAT_MJPG) {
            cv::Mat rawMat(1, videoFrame->dataSize(), CV_8UC1, videoFrame->data());
            rstMat = cv::imdecode(rawMat, 1);
            cv::cvtColor(rstMat * 2, rstMat, cv::COLOR_GRAY2RGB);
        }
        {
            std::lock_guard<std::mutex> lock(showMutex);
            CurrentIRImage = rstMat; // ���
        }
    }
    else {
        std::cout << "Ŀǰ�����ǳ�RGB��Depth�Լ�IR֮���ͼ������" << std::endl;
        LOGGING(FATAL) << "Ŀǰ�����ǳ�RGB��Depth�Լ�IR֮���ͼ������";
    }

}

void GetOneCamera::CameraStop()
{
    stop = true;
    colorFrames.reset();
    depthFrames.reset();
    pipe->stop();
}

cv::Mat GetOneCamera::GetCurrentColorImage()
{
    std::lock_guard<std::mutex> lock(showMutex);
    return CurrentColorImage.clone();
}

cv::Mat GetOneCamera::GetCurrentDepthImage()
{
    std::lock_guard<std::mutex> lock(showMutex);
    return CurrentDepthImage.clone();
}

cv::Mat GetOneCamera::GetCurrentIRImage()
{
    std::lock_guard<std::mutex> lock(showMutex);
    return CurrentIRImage.clone();
}

cv::String GetOneCamera::GetColorWindowName()
{
    return ColorImageName;
}

cv::String GetOneCamera::GetDepthWindowName()
{
    return DepthImageName;
}
