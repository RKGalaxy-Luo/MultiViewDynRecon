#include "ExtractFeaturePoints.h"

void ExtractFeaturePoints::ExtractFeaturePointsORB(cv::Mat Image1, cv::Mat Image2, std::vector<cv::Point2f>& matchpoints1, std::vector<cv::Point2f>& matchpoints2)
{
    std::cout << "��ʼ����ORB������ƥ��..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    int featureNumber = MAX_FEATURE_POINT_COUNT;
    std::vector<cv::KeyPoint> img_keypoints, origin_keypoints;
    cv::Mat img_descriptors, origin_descriptors;
    //����ORB�㷨��ȡ������
    cv::Ptr<cv::ORB> detector = cv::ORB::create(featureNumber, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    detector->detect(Image1, img_keypoints);
    detector->compute(Image1, img_keypoints, img_descriptors);
    detector->detect(Image2, origin_keypoints);
    detector->compute(Image2, origin_keypoints, origin_descriptors);
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); //����������Ϊ���ƶȶ���
    //ƥ��������matches����
    std::vector<cv::DMatch> matches;
    matcher.match(origin_descriptors, img_descriptors, matches);//ƥ��ɹ��ĵ�
    if (matches.size() < 4) {
        LOGGING(FATAL) << "����ƥ���С��4�����޷�ƥ��";
    }
    //����ƥ������
    std::vector<int> photoIdxs(matches.size()), standardIdxs(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        standardIdxs[i] = matches[i].queryIdx;//ȡ����ѯͼƬ��ƥ��ĵ�Ե�������id�ţ���ôqueryIdxs��trainIdxs��Ϊ257
        photoIdxs[i] = matches[i].trainIdx;//ȡ��ѵ��ͼƬ��ƥ��ĵ�Ե�������id�ţ�
    }
    //��keyPointת����Point2f
    cv::KeyPoint::convert(img_keypoints, matchpoints1, photoIdxs);
    cv::KeyPoint::convert(origin_keypoints, matchpoints2, standardIdxs);//KeyPoint��������תpoint2f����
    // ��ȡ�������ʱ���
    auto end = std::chrono::high_resolution_clock::now();
    // �����������ʱ�䣨���룩
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // �����������ʱ��
    std::cout << "ORB������ƥ������ʱ��: " << duration << " ����     ��ѡ��������" << matchpoints1.size() << std::endl;
    //cv::Mat matchImage;
    //drawMatches(Image1, origin_keypoints, Image2, img_keypoints, matches, matchImage);
    //cv::imshow("matchImage", matchImage);
    //cv::waitKey(10);
}

void ExtractFeaturePoints::ExtractFeaturePointsBEBLID(cv::Mat Image1, cv::Mat Image2, std::vector<cv::Point2f>& matchpoints1, std::vector<cv::Point2f>& matchpoints2)
{
    std::cout << "��ʼ����BEBLID������ƥ��..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img1, img2;
    cvtColor(Image1, img1, cv::COLOR_RGB2GRAY);
    cvtColor(Image2, img2, cv::COLOR_RGB2GRAY);
    cv::Ptr<cv::ORB> detector = cv::ORB::create(MAX_FEATURE_POINT_COUNT, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    // Detect features in both images
    std::vector<cv::KeyPoint> points1, points2;
    detector->detect(img1, points1);
    detector->detect(img2, points2);

    // Use 32 bytes per descriptor and configure the scale factor for ORB detector
    cv::Ptr<BEBLID> descriptor = BEBLID::create(512, 0.75);
    // Describe the detected features i both images
    cv::Mat descriptors1, descriptors2;
    descriptor->compute(img1, points1, descriptors1);
    descriptor->compute(img2, points2, descriptors2);

    // Match the generated descriptors for img1 and img2 using brute force matching
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // If there is not enough matches exit
    if (matches.size() < 4) {
        LOGGING(FATAL) << "����ƥ���С��4�����޷�ƥ��";
    }

    // Take only the matched points that will be used to calculate the
    // transformation between both images
    std::vector<cv::Point2d> matched_pts1, matched_pts2;

    for (cv::DMatch match : matches) {
        matched_pts1.push_back(points1[match.queryIdx].pt);
        matched_pts2.push_back(points2[match.trainIdx].pt);
    }

    // Find the homography that transforms a point in the first image to a point in the second image.
    cv::Mat inliers;
    cv::Mat H = findHomography(matched_pts1, matched_pts2, cv::RANSAC, 3, inliers);
    for (int i = 0; i < matched_pts1.size(); i++) {
        if (inliers.at<uchar>(i, 0))
        {
            matchpoints1.push_back(matched_pts1[i]);
            matchpoints2.push_back(matched_pts2[i]);
        }
    }
    // ��ȡ�������ʱ���
    auto end = std::chrono::high_resolution_clock::now();
    // �����������ʱ�䣨���룩
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // �����������ʱ��
    std::cout << "BEBLID������ƥ������ʱ�䣺 " << duration << " ����     ��ѡ��������" << matchpoints1.size()<< std::endl;

}
