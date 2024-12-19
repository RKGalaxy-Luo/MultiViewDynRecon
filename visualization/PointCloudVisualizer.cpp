/*****************************************************************//**
 * \file   PointCloudVisualizer.cpp
 * \brief  可视化点云及面元
 * 
 * \author Administrator
 * \date   November 2024
 *********************************************************************/
#include <base/FileOperation/Stream.h>
#include <base/FileOperation/Serializer.h>
#include <base/FileOperation/BinaryFileStream.h>
#include <base/data_transfer.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <base/common_point_cloud_utils.h>
#include <visualization/Visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>


/* The point cloud drawing methods
*/
void SparseSurfelFusion::Visualizer::DrawPointCloud(const PointCloud3f_Pointer &point_cloud) {
    const std::string window_title = "simple point cloud viewer";

    pcl::visualization::PCLVisualizer viewer(window_title);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 255, 0, 0);
    viewer.addPointCloud(point_cloud, handler, "point cloud");
    viewer.addCoordinateSystem(2.0, "point cloud", 0);
    viewer.setBackgroundColor(0.0f, 0.0f, 0.0f, 0.0f);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "point cloud");
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void SparseSurfelFusion::Visualizer::DrawPointCloudcanandlive(const PointCloud3f_Pointer& can, const PointCloud3f_Pointer& live)
{
    //const std::string window_title = "左can 右live";

    //pcl::visualization::PCLVisualizer viewer(window_title);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(can, 255, 255, 255);
    //viewer.addPointCloud(can, "point cloud");
    //viewer.addCoordinateSystem(2.0, "point cloud", 0);
    //viewer.setBackgroundColor(0.05, 0.05, 0.05, 1);
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");
    //while (!viewer.wasStopped()) {
    //    viewer.spinOnce();
    //}
}

void SparseSurfelFusion::Visualizer::DrawPointCloud(const SparseSurfelFusion::DeviceArray<float4> &point_cloud) {
    const auto h_point_cloud = downloadPointCloud(point_cloud);
    DrawPointCloud(h_point_cloud);
}
void SparseSurfelFusion::Visualizer::DrawPointCloudcanandlive(const DeviceArray<float4>& can, const DeviceArray<float4>& live)
{
    const auto h_can_point_cloud = downloadPointCloud(can);
    const auto h_live_point_cloud = downloadPointCloud(live);
    DrawPointCloudDoubleView(h_can_point_cloud, h_live_point_cloud);
}

void SparseSurfelFusion::Visualizer::DrawPointCloud(const SparseSurfelFusion::DeviceArrayView<float4> &cloud) {
    CHECKCUDA(cudaDeviceSynchronize());
    DeviceArray<float4> cloud_array = DeviceArray<float4>((float4 *) cloud.RawPtr(), cloud.Size());
    DrawPointCloud(cloud_array);
}

void SparseSurfelFusion::Visualizer::DrawPointCloudcanandlive(const DeviceArrayView<float4>& can, const DeviceArrayView<float4>& live)
{
    DeviceArray<float4> can_cloud_array = DeviceArray<float4>((float4*)can.RawPtr(), can.Size());
    DeviceArray<float4> live_cloud_array = DeviceArray<float4>((float4*)live.RawPtr(), live.Size());
    DrawPointCloudcanandlive(can_cloud_array, live_cloud_array);

}

void SparseSurfelFusion::Visualizer::DrawPointCloud(const DeviceArray2D<float4> &vertex_map) {
    const auto point_cloud = downloadPointCloud(vertex_map);
    DrawPointCloud(point_cloud);
}

void SparseSurfelFusion::Visualizer::DrawPointCloud(cudaTextureObject_t vertex_map) {
    CHECKCUDA(cudaDeviceSynchronize());
    const auto point_cloud = downloadPointCloud(vertex_map);
    DrawPointCloud(point_cloud);
}

void SparseSurfelFusion::Visualizer::DrawPointCloud(
        const DeviceArray<DepthSurfel> &surfel_array
) {

    PointCloud3f_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);

    DrawPointCloud(point_cloud);
}

void SparseSurfelFusion::Visualizer::DrawPointCloud(const DeviceArrayView<DepthSurfel> surfel_array)
{
    PointCloud3f_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);

    DrawPointCloud(point_cloud);
}

void SparseSurfelFusion::Visualizer::SavePointCloud(const std::vector<float4> &point_vec, const std::string &path) {
    std::ofstream file_output;
    file_output.open(path);
    file_output << "OFF" << std::endl;
    file_output << point_vec.size() << " " << 0 << " " << 0 << std::endl;
    for (int node_iter = 0; node_iter < point_vec.size(); node_iter++) {
        file_output << point_vec[node_iter].x * 1000
                    << " " << point_vec[node_iter].y * 1000 << " "
                    << point_vec[node_iter].z * 1000
                    << std::endl;
    }
}

void SparseSurfelFusion::Visualizer::SavePointCloud(cudaTextureObject_t vertex_map, const std::string &path) {
    std::vector<float4> point_vec;
    downloadPointCloud(vertex_map, point_vec);
    std::ofstream file_output;
    file_output.open(path);
    file_output << "OFF" << std::endl;
    file_output << point_vec.size() << " " << 0 << " " << 0 << std::endl;
    for (int node_iter = 0; node_iter < point_vec.size(); node_iter++) {
        file_output << point_vec[node_iter].x * 1000
                    << " " << point_vec[node_iter].y * 1000 << " "
                    << point_vec[node_iter].z * 1000
                    << std::endl;
    }
}


void SparseSurfelFusion::Visualizer::SavePointCloud(const DeviceArrayView<float4> cloud, const std::string &path) {
    DeviceArray<float4> point_cloud((float4 *) cloud.RawPtr(), cloud.Size());
    std::vector<float4> point_vec;
    point_cloud.download(point_vec);
    SavePointCloud(point_vec, path);
}

/* The point cloud with normal
*/
void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
        const PointCloud3fRGB_Pointer&point_cloud
        ,const PointCloudNormal_Pointer & normal_cloud

) {
    const std::string window_title = "3D Viewer";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> handler(point_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud, "sample cloud");
    //viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud, normal_cloud, 30, 60.0f);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

}

void SparseSurfelFusion::Visualizer::DrawPointCloudDoubleView(
    const PointCloud3fRGB_Pointer& point_cloud0, 
    const PointCloudNormal_Pointer& normal_cloud0, 
    const PointCloud3fRGB_Pointer& point_cloud1, 
    const PointCloudNormal_Pointer& normal_cloud1
)
{
    //const std::string window_title = "3D Viewer";
    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    //viewer->setBackgroundColor(0, 0, 0);
    ////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> handler(point_cloud, 0, 255, 0);
    //viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud, "sample cloud");
    ////viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud, normal_cloud, 30, 60.0f);
    //while (!viewer->wasStopped()) {
    //    viewer->spinOnce(100);
    //}

    //**********************
    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));

    int v1(0);
    int v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    //// The color we will be using
    //float bckgr_gray_level = 0.0;  // Black
    //float txt_gray_lvl = 1.0 - bckgr_gray_level;

    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h(point_cloud0, 255, 255, 255);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h(point_cloud1, 255, 255, 255);
    //viewer->addPointCloud(point_cloud0, cloud0_in_color_h, "cloud_in_v1", v1);
    //viewer->addPointCloud(point_cloud1, cloud1_in_color_h, "cloud_in_v2", v2);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud0, "sample cloud0", v1);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud1, "sample cloud1", v2);

    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->setBackgroundColor(0, 0, 0, v2);

    viewer->setSize(2560, 1440);  // Visualiser window size
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}




/* The point cloud with normal
*/
void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
    const PointCloud3f_Pointer& point_cloud
    , const PointCloudNormal_Pointer& normal_cloud

) {
    const std::string window_title = "3D Viewer";
#ifdef WITH_PCL
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(point_cloud, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud, normal_cloud, 30, 60.0f);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#elif defined(WITH_CILANTRO)
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("point cloud", *point_cloud.get(),
        cilantro::RenderingProperties()
        .setPointColor(1.0, 1.0, 1.0)
        .setPointSize(2.0)
        .setDrawNormals(true)
        .setNormalLength(60.0f)
        .setLineDensityFraction(1.0f / 30.0f));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}


void SparseSurfelFusion::Visualizer::DrawPointCloudDoubleView(
    cudaTextureObject_t vertex_map0, 
    cudaTextureObject_t vertex_map1
)
{
    const auto point_cloud0 = downloadPointCloud(vertex_map0);
    const auto point_cloud1 = downloadPointCloud(vertex_map1);
    DrawPointCloudDoubleView(point_cloud0, point_cloud1);
}


void SparseSurfelFusion::Visualizer::DrawPointCloud2OneView(
    cudaTextureObject_t vertex_map0, 
    cudaTextureObject_t vertex_map1
) {
    const auto point_cloud0 = downloadPointCloud(vertex_map0);
    const auto point_cloud1 = downloadPointCloud(vertex_map1);
    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));

    //int v1(0);
    //viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h(point_cloud0, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h(point_cloud1, 255, 0, 0);
    viewer->addPointCloud(point_cloud0, cloud0_in_color_h, "cloud_in_v1");
    viewer->addPointCloud(point_cloud1, cloud1_in_color_h, "cloud_in_v2");
    //viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->setSize(1920, 1080);  // Visualiser window size
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void SparseSurfelFusion::Visualizer::DrawPointCloud2OneView(
    cudaTextureObject_t vertex_map0,
    cudaTextureObject_t vertex_map1,
    mat34 SE3
) {
    const auto point_cloud0 = downloadPointCloud(vertex_map0);
    const auto point_cloud1 = downloadPointCloudUseSE3(vertex_map1,SE3);
    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));

    //int v1(0);
    //viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h(point_cloud0, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h(point_cloud1, 255, 0, 0);
    viewer->addPointCloud(point_cloud0, cloud0_in_color_h, "cloud_in_v1");
    viewer->addPointCloud(point_cloud1, cloud1_in_color_h, "cloud_in_v2");
    //viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->setSize(1920, 1080);  // Visualiser window size
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void SparseSurfelFusion::Visualizer::DrawPointCloudDoubleView(
    const PointCloud3f_Pointer& point_cloud0,
    const PointCloud3f_Pointer& point_cloud1
    )
{
    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));

    int v1(0);
    int v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    // The color we will be using
    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h(point_cloud0, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h(point_cloud1, 0, 255, 0);
    viewer->addPointCloud(point_cloud0, cloud0_in_color_h, "cloud_in_v1", v1);
    viewer->addPointCloud(point_cloud1, cloud1_in_color_h, "cloud_in_v2", v2);

    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->setBackgroundColor(0, 0, 0, v2);

    viewer->setSize(2560, 1440);  // Visualiser window size
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}


template<typename TPointInput, typename TNormalsInput>
void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormals_Generic(TPointInput &points, TNormalsInput &normals) {
#ifdef WITH_PCL
    const auto point_cloud = downloadPointCloud(points);
    const auto normal_cloud = downloadNormalCloud(normals);
    DrawPointCloudWithNormal(point_cloud, normal_cloud);
#elif defined(WITH_CILANTRO)
    auto point_cloud = surfelwarp::downloadPointCloud(points);
    surfelwarp::downloadNormalCloud(normals, point_cloud);
    DrawPointCloudWithNormal(point_cloud);
#endif
}

void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
        const DeviceArray<float4> &vertex,
        const DeviceArray<float4> &normal
) {
    DrawPointCloudWithNormals_Generic(vertex, normal);

}

void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
        const DeviceArrayView<float4> &vertex_cloud,
        const DeviceArrayView<float4> &normal_cloud
) {
    FUNCTION_CHECK(vertex_cloud.Size() == normal_cloud.Size());
    DeviceArray<float4> vertex_array((float4 *) vertex_cloud.RawPtr(), vertex_cloud.Size());
    DeviceArray<float4> normal_array((float4 *) normal_cloud.RawPtr(), normal_cloud.Size());
    DrawPointCloudWithNormal(vertex_array, normal_array);
}

void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
        const DeviceArray2D<float4> &vertex_map,
        const DeviceArray2D<float4> &normal_map
) {
    DrawPointCloudWithNormals_Generic(vertex_map, normal_map);
}

void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
        cudaTextureObject_t vertex_map,
        cudaTextureObject_t normal_map
) {
    DrawPointCloudWithNormals_Generic(vertex_map, normal_map);
}

void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
        const DeviceArray<DepthSurfel> &surfel_array
) {
#ifdef WITH_PCL
    PointCloud3fRGB_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);
    DrawPointCloudWithNormal(point_cloud, normal_cloud);
#elif defined(WITH_CILANTRO)
    PointCloud3f_Pointer point_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud);
    DrawPointCloudWithNormal(point_cloud);
#endif
}


void SparseSurfelFusion::Visualizer::DrawPointCloudWithNormal(
    const DeviceArray<DepthSurfel>& surfel_array0, 
    const DeviceArray<DepthSurfel>& surfel_array1
) {

    PointCloud3fRGB_Pointer point_cloud0, point_cloud1;
    PointCloudNormal_Pointer normal_cloud0, normal_cloud1;
    downloadPointNormalCloud(surfel_array0, point_cloud0, normal_cloud0);
    downloadPointNormalCloud(surfel_array1, point_cloud1, normal_cloud1);
    DrawPointCloudDoubleView(point_cloud0, normal_cloud0, point_cloud1, normal_cloud1);

}



void SparseSurfelFusion::Visualizer::SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map) {
    //Download it
#ifdef WITH_PCL
    const auto point_cloud = downloadPointCloud(vertex_map);
    const auto normal_cloud = downloadNormalCloud(normal_map);
#elif defined(WITH_CILANTRO)
    auto point_cloud = downloadPointCloud(vertex_map);
    downloadNormalCloud(normal_map, point_cloud);
#endif

    //Construct the output stream
    BinaryFileStream output_fstream("pointnormal", BinaryFileStream::FileOperationMode::WriteOnly);

    //Prepare the test data
    std::vector<float4> save_vec;
    for (auto i = 0; i < point_cloud->points.size(); i++) {
#ifdef WITH_PCL
        save_vec.push_back(
                make_float4(point_cloud->points[i].x, point_cloud->points[i].y, point_cloud->points[i].z, 0));
        save_vec.push_back(make_float4(
                normal_cloud->points[i].normal_x,
                normal_cloud->points[i].normal_y,
                normal_cloud->points[i].normal_z,
                0));
#elif defined(WITH_CILANTRO)
        save_vec.push_back(
                make_float4(point_cloud->points(0, i),
                            point_cloud->points(1, i),
                            point_cloud->points(2, i), 0));
        save_vec.push_back(make_float4(
                point_cloud->normals(0, i),
                point_cloud->normals(1, i),
                point_cloud->normals(2, i), 0));
#endif

    }

    //Save it
    //PODVectorSerializeHandler<int>::Write(&output_fstream, save_vec);
    //SerializeHandler<std::vector<int>>::Write(&output_fstream, save_vec);
    //output_fstream.Write<std::vector<int>>(save_vec);
    output_fstream.SerializeWrite<std::vector<float4>>(save_vec);
}


/* The colored point cloud drawing method
 */
void SparseSurfelFusion::Visualizer::DrawColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud) {
    std::string window_title = "3D Viewer";
#ifdef WITH_PCL
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#elif defined(WITH_CILANTRO)
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("point cloud", *point_cloud.get(),
                                                     cilantro::RenderingProperties().setPointSize(3.0));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}

void SparseSurfelFusion::Visualizer::SaveColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud,
                                                   const std::string &path) {
    std::ofstream file_output;
    file_output.open(path);
    const auto &points = point_cloud->points;

    file_output << "COFF" << std::endl;
    file_output << points.size() << " " << 0 << " " << 0 << std::endl;
    for (auto i = 0; i < points.size(); i++) {
#ifdef WITH_PCL
        const auto point = points[i];
        file_output << point.x
                    << " " << point.y
                    << " " << point.z
                    << " " << point.r / 255.f
                    << " " << point.g / 255.f
                    << " " << point.b / 255.f
                    << std::endl;
#elif defined(WITH_CILANTRO)
        file_output << points(0, i) << points(1, i) << points(2, i)
                    << point_cloud->colors(0, i) << point_cloud->colors(1, i) << point_cloud->colors(2, i)
                    << std::endl;
#endif
    }
    file_output.close();
}

void SparseSurfelFusion::Visualizer::DrawColoredPointCloud(
        const SparseSurfelFusion::DeviceArray<float4> &vertex,
        const SparseSurfelFusion::DeviceArray<float4> &color_time
) {
    auto point_cloud = downloadColoredPointCloud(vertex, color_time);
    DrawColoredPointCloud(point_cloud);
}

void SparseSurfelFusion::Visualizer::DrawColoredPointCloud(
        const SparseSurfelFusion::DeviceArrayView<float4> &vertex,
        const SparseSurfelFusion::DeviceArrayView<float4> &color_time
) {
    DeviceArray<float4> vertex_array((float4 *) vertex.RawPtr(), vertex.Size());
    DeviceArray<float4> color_time_array((float4 *) color_time.RawPtr(), color_time.Size());
    DrawColoredPointCloud(vertex_array, color_time_array);
}

void SparseSurfelFusion::Visualizer::DrawColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map) {
    auto cloud = downloadColoredPointCloud(vertex_map, color_time_map, true);
    DrawColoredPointCloud(cloud);
}

void SparseSurfelFusion::Visualizer::SaveColoredPointCloud(
        cudaTextureObject_t vertex_map,
        cudaTextureObject_t color_time_map,
        const std::string &path
) {
    auto cloud = downloadColoredPointCloud(vertex_map, color_time_map, true);
    SaveColoredPointCloud(cloud, path);
}

/* The method to draw matched cloud pair
 */
void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2
) {
    std::string window_title = "Matched Viewer";
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.0f, 0.0f, 0.0f);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 2");

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1, const PointCloud3f_Pointer& cloud_2, const PointCloud3f_Pointer& cloud_3)
{
    std::string window_title = "Matched Viewer";

    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_3(cloud_3, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_3, handler_3, "cloud 3");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1, const PointCloud3f_Pointer& cloud_2, const PointCloud3f_Pointer& cloud_3, const PointCloud3f_Pointer& cloud_4, const PointCloud3f_Pointer& cloud_5)
{
    std::string window_title = "Matched Viewer";

    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.999f, 0.999f, 0.999f);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 2");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_3(cloud_3, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_3, handler_3, "cloud 3");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 3");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_4(cloud_4, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_4, handler_4, "cloud 4");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 4");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_5(cloud_5, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_5, handler_5, "cloud 5");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 5");

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}


void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2,
        const Eigen::Matrix4f &from1To2
) {
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(cloud_1, from1To2);
    DrawMatchedCloudPair(transformed_cloud_1, cloud_2);
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const SparseSurfelFusion::DeviceArray<float4> &cloud_2,
        const SparseSurfelFusion::Matrix4f &from1To2
) {
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    DrawMatchedCloudPair(h_cloud_1, h_cloud_2, from1To2);
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const DeviceArrayView<float4> &cloud_2,
        const Matrix4f &from1To2
) {
    DrawMatchedCloudPair(
            cloud_1,
            DeviceArray<float4>((float4 *) cloud_2.RawPtr(), cloud_2.Size()),
            from1To2
    );
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        cudaTextureObject_t cloud_2,
        const SparseSurfelFusion::Matrix4f &from1To2
) {
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    DrawMatchedCloudPair(h_cloud_1, h_cloud_2, from1To2);
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(const DeviceArrayView<DepthSurfel>& observedVertex, const DeviceArrayView<DepthSurfel>& referenceVertex, const mat34& Observed2Reference)
{
    const auto cloud_1 = downloadColoredPointCloud(observedVertex, Observed2Reference, "overvedVertex");
    const auto cloud_2 = downloadColoredPointCloud(referenceVertex, Observed2Reference.identity(), "canonical");
    DrawMatchedRGBCloudPair(cloud_1, cloud_2);
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPairWithLine(const PointCloud3f_Pointer& cloud_1, const PointCloud3f_Pointer& cloud_2, const float4* canVertexHost, const float4* observedVertexHost, const unsigned int vertexNum)
{
    std::string window_title = "Matched Viewer";
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.999f, 0.999f, 0.999f);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 2");
    for (int i = 0; i < vertexNum; i++) {
        if (vertexNum > 8192 && i % 2 == 0) {
            pcl::PointXYZ startPoint(canVertexHost[i].x * 1000.0f, canVertexHost[i].y * 1000.0f, canVertexHost[i].z * 1000.0f);
            pcl::PointXYZ endPoint(observedVertexHost[i].x * 1000.0f, observedVertexHost[i].y * 1000.0f, observedVertexHost[i].z * 1000.0f);
            viewer->addLine(startPoint, endPoint, 0.0f, 0.0f, 0.0f, "line" + std::to_string(i));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "line" + std::to_string(i));
        }
        else if (vertexNum <= 8192) {
            pcl::PointXYZ startPoint(canVertexHost[i].x * 1000.0f, canVertexHost[i].y * 1000.0f, canVertexHost[i].z * 1000.0f);
            pcl::PointXYZ endPoint(observedVertexHost[i].x * 1000.0f, observedVertexHost[i].y * 1000.0f, observedVertexHost[i].z * 1000.0f);
            viewer->addLine(startPoint, endPoint, 0.0f, 0.0f, 0.0f, "line" + std::to_string(i));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "line" + std::to_string(i));
        }
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPairWithLine(const PointCloud3f_Pointer& preVertex, const PointCloud3f_Pointer& currVertex, const PointCloud3f_Pointer& machedVertex_0, const PointCloud3f_Pointer& machedVertex_1, const float4* machedVertex_0_Host, const float4* machedVertex_1_Host, const unsigned int matchedPairsNum)
{
    std::string window_title = "Matched Viewer";
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(preVertex, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(preVertex, handler_1, "preVertex");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "preVertex");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(currVertex, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(currVertex, handler_2, "currVertex");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "currVertex");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_3(machedVertex_0, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(machedVertex_0, handler_3, "machedVertex_0");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "machedVertex_0");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_4(machedVertex_1, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(machedVertex_1, handler_4, "machedVertex_1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "machedVertex_1");

    for (int i = 0; i < matchedPairsNum; i++) {
        if (matchedPairsNum > 8192 && 8000 <= i && i <= 10000) {
            pcl::PointXYZ startPoint(machedVertex_0_Host[i].x * 1000.0f, machedVertex_0_Host[i].y * 1000.0f, machedVertex_0_Host[i].z * 1000.0f);
            pcl::PointXYZ endPoint(machedVertex_1_Host[i].x * 1000.0f, machedVertex_1_Host[i].y * 1000.0f, machedVertex_1_Host[i].z * 1000.0f);
            viewer->addLine(startPoint, endPoint, 1.0, 1.0, 1.0, "line" + std::to_string(i));
        }
        else if (matchedPairsNum <= 8192) {
            pcl::PointXYZ startPoint(machedVertex_0_Host[i].x * 1000.0f, machedVertex_0_Host[i].y * 1000.0f, machedVertex_0_Host[i].z * 1000.0f);
            pcl::PointXYZ endPoint(machedVertex_1_Host[i].x * 1000.0f, machedVertex_1_Host[i].y * 1000.0f, machedVertex_1_Host[i].z * 1000.0f);
            viewer->addLine(startPoint, endPoint, 1.0, 1.0, 1.0, "line" + std::to_string(i));
        }
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}


void SparseSurfelFusion::Visualizer::SaveMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    auto color_cloud_1 = addColorToPointCloud(cloud_1, make_uchar4(245, 0, 0, 255));
    auto color_cloud_2 = addColorToPointCloud(cloud_2, make_uchar4(200, 200, 200, 255));
    SaveColoredPointCloud(color_cloud_1, cloud_1_name);
    SaveColoredPointCloud(color_cloud_2, cloud_2_name);
}

void SparseSurfelFusion::Visualizer::SaveMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2,
        const Eigen::Matrix4f &from1To2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(cloud_1, from1To2);
    SaveMatchedCloudPair(transformed_cloud_1, cloud_2, cloud_1_name, cloud_2_name);
}


void SparseSurfelFusion::Visualizer::SaveMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const DeviceArray<float4> &cloud_2,
        const Eigen::Matrix4f &from1To2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    SaveMatchedCloudPair(
            h_cloud_1,
            h_cloud_2,
            from1To2,
            cloud_1_name, cloud_2_name
    );
}


void SparseSurfelFusion::Visualizer::SaveMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const DeviceArrayView<float4> &cloud_2,
        const Eigen::Matrix4f &from1To2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    SaveMatchedCloudPair(
            cloud_1,
            DeviceArray<float4>((float4 *) cloud_2.RawPtr(), cloud_2.Size()),
            from1To2,
            cloud_1_name, cloud_2_name
    );
}


/* The method to draw mached color point cloud
 */
void SparseSurfelFusion::Visualizer::DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer &cloud_1,
                                                     const PointCloud3fRGB_Pointer &cloud_2
) {
    std::string window_title = "3D Viewer";
#ifdef WITH_PCL
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_1(cloud_1);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud_1, handler_1, "cloud_1");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_2(cloud_2);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud_2, handler_2, "cloud_2");

    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_1");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#else
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("cloud 1", *cloud_1.get(),
                                                     cilantro::RenderingProperties().setPointSize(3.0));
    viewer.addObject<cilantro::PointCloudRenderable>("cloud 2", *cloud_1.get(),
                                                     cilantro::RenderingProperties().setPointSize(3.0));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}


void SparseSurfelFusion::Visualizer::DrawMatchedRGBCloudPair(
        const PointCloud3fRGB_Pointer &cloud_1,
        const PointCloud3fRGB_Pointer &cloud_2,
        const Eigen::Matrix4f &from1To2
) {
    PointCloud3fRGB_Pointer transformed_cloud_1 = transformPointCloudRGB(cloud_1, from1To2);

    //Hand in to drawer
	DrawMatchedRGBCloudPair(transformed_cloud_1, cloud_2);
}

void SparseSurfelFusion::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map,
        const DeviceArrayView<float4> &surfel_array,
        const DeviceArrayView<float4> &color_time_array,
        const Eigen::Matrix4f &camera2world
) {
    auto cloud_1 = downloadColoredPointCloud(vertex_map, color_time_map, true);
    auto cloud_2 = downloadColoredPointCloud(
            DeviceArray<float4>((float4 *) surfel_array.RawPtr(), surfel_array.Size()),
            DeviceArray<float4>((float4 *) color_time_array.RawPtr(), color_time_array.Size())
    );
	DrawMatchedRGBCloudPair(cloud_1, cloud_2, camera2world);
}

void SparseSurfelFusion::Visualizer::Debugworld2Camera(
    cudaTextureObject_t vertex_map,
    cudaTextureObject_t depth_vertex_map,
    mat34 world2camera
) {
    const auto point_cloud = downloadPointCloudUseSE3(vertex_map, world2camera);
    //const auto point_cloud = downloadPointCloud(vertex_map);
    const auto point_cloud_depth = downloadPointCloud(depth_vertex_map);
    
    //const auto point_cloud_raw = downloadPointCloud(vertex_map);
    //const auto point_cloud_depth_raw = downloadPointCloud(depth_vertex_map);

    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    
    //int v1(0);
    //int v2(1);
    //viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    //viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h(point_cloud, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h(point_cloud_depth, 255, 0, 0);
    viewer->addPointCloud(point_cloud, cloud0_in_color_h, "白色 使用World2Camera后的live");
    viewer->addPointCloud(point_cloud_depth, cloud1_in_color_h, "红色 深度面元");

    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h_raw(point_cloud_raw, 255, 255, 255);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h_raw(point_cloud_depth_raw, 255, 0, 0);
    //viewer->addPointCloud(point_cloud_raw, cloud0_in_color_h_raw, "白色 未使用的live", v2);
    //viewer->addPointCloud(point_cloud_depth_raw, cloud1_in_color_h_raw, "红色 深度面元", v2);


    //viewer->setBackgroundColor(0, 0, 0, v1);
    //viewer->setBackgroundColor(0, 0, 0, v2);

    //viewer->setSize(2560, 1440);  // Visualiser window size
    viewer->setSize(1920, 1080);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
   
}

void SparseSurfelFusion::Visualizer::DebugCamera2World(
    cudaTextureObject_t depth_vertex_map0,
    cudaTextureObject_t depth_vertex_map1,
    mat34 camera2world0,
    mat34 camera2world1,
    mat34 SE3//1-0
) {
    //const auto point_cloud_0 = downloadPointCloud(depth_vertex_map0);
    //const auto point_cloud_1 = downloadPointCloudUseSE3(depth_vertex_map1,SE3);

    const auto point_cloud_0_use = downloadPointCloudUseSE3(depth_vertex_map0, camera2world0);
    const auto point_cloud_1_use = downloadPointCloudUseSE3AndWorld2Camera(depth_vertex_map1, SE3, camera2world1);

    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    //int v1(0);
    //int v2(1);
    //viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    //viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h(point_cloud_0, 255, 255, 255);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h(point_cloud_1, 255, 0, 0);
    //viewer->addPointCloud(point_cloud_0, cloud0_in_color_h, "白色 0",v1);
    //viewer->addPointCloud(point_cloud_1, cloud1_in_color_h, "红色 1",v1);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h_raw(point_cloud_0_use, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h_raw(point_cloud_1_use, 255, 0, 0);
    viewer->addPointCloud(point_cloud_0_use, cloud0_in_color_h_raw, "白色 0");
    viewer->addPointCloud(point_cloud_1_use, cloud1_in_color_h_raw, "红色 1");


    //viewer->setBackgroundColor(0, 0, 0, v1);
    //viewer->setBackgroundColor(0, 0, 0, v2);

    viewer->setSize(2560, 1440);  // Visualiser window size
    //viewer->setSize(1920, 1080);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

}
void SparseSurfelFusion::Visualizer::DebugCamera2WorldUseFirstFrame(
    DeviceArrayView<float4> firstframedata,
    cudaTextureObject_t depth_vertex_map0,
    cudaTextureObject_t depth_vertex_map1,
    mat34 camera2world0,
    mat34 camera2world1,
    mat34 SE3
) {
    PointCloud3f_Pointer point_cloud(new PointCloud3f);
    std::vector<float4> h_vertex;
    firstframedata.Download(h_vertex);
    setPointCloudSize(point_cloud, firstframedata.Size());
    for (auto idx = 0; idx < firstframedata.Size(); idx++) {
        setPoint(h_vertex[idx].x, h_vertex[idx].y, h_vertex[idx].z, point_cloud, idx);
    }
    //mat34 camera2world0_from1 = camera2world1*SE3;


    //使用camera2world的
    const auto point_cloud_0_use = downloadPointCloudUseSE3(depth_vertex_map0, camera2world0);
    const auto point_cloud_1_use = downloadPointCloudUseSE3AndWorld2Camera(depth_vertex_map1, SE3, camera2world0);
    //只使用SE3
    const auto point_cloud_0_nouse = downloadPointCloud(depth_vertex_map0);
    const auto point_cloud_1_nouse = downloadPointCloudUseSE3(depth_vertex_map1, SE3);


    const std::string window_title = "3D Viewer  ";
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));

    int v1(0);
    int v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h_raw(point_cloud_0_use, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h_raw(point_cloud_1_use, 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_firstframe(point_cloud, 255, 0,0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud0_in_color_h_raw_nouse(point_cloud_0_nouse, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_in_color_h_raw_nouse(point_cloud_1_nouse, 0, 0, 255);


    //左边窗口是使用了camera2world的
    viewer->addPointCloud(point_cloud_0_use, cloud0_in_color_h_raw, "绿色 0",v1);
    viewer->addPointCloud(point_cloud_1_use, cloud1_in_color_h_raw, "蓝色 1",v1);
    viewer->addPointCloud(point_cloud, cloud_firstframe, "红色 第0帧目标面元",v1);


    viewer->addPointCloud(point_cloud_0_nouse, cloud0_in_color_h_raw_nouse, "绿色 0 nouse", v2);
    viewer->addPointCloud(point_cloud_1_nouse, cloud1_in_color_h_raw_nouse, "蓝色 1 nouse", v2);
    viewer->addPointCloud(point_cloud, cloud_firstframe, "红色 第0帧目标面元 nouse", v2);


    viewer->setSize(2560, 1440);  // Visualiser window size
    //viewer->setSize(1920, 1080);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}


void SparseSurfelFusion::Visualizer::DrawFusedAndRemainSurfelCloud(
    DeviceArrayView<float4> surfel_vertex,
    unsigned remainingnumber,
    unsigned number0view
) {
    PointCloud3f_Pointer remain_cloud(new PointCloud3f);
    PointCloud3f_Pointer append_cloud0(new PointCloud3f);
    PointCloud3f_Pointer append_cloud1(new PointCloud3f);

    hsgseparateDownloadPointCloud(surfel_vertex, remainingnumber, number0view, remain_cloud, append_cloud0, append_cloud1);

    std::string window_title = "3D Viewer";
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(remain_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(remain_cloud, handler_1, "保留的");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(append_cloud0, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(append_cloud0, handler_2, "0的");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_3(append_cloud1, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(append_cloud1, handler_3, "1的");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

  

}



void SparseSurfelFusion::Visualizer::DrawFusedSurfelCloud(
        DeviceArrayView<float4> surfel_vertex,
        DeviceArrayView<unsigned> fused_indicator
) {
    FUNCTION_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());
    CHECKCUDA(cudaDeviceSynchronize());
    //Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);
    //Download it
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);
    //Ok draw it
    //DrawMatchedCloudPair(fused_cloud, unfused_cloud, Eigen::Matrix4f::Identity());
    DrawMatchedCloudPair(fused_cloud, unfused_cloud);

}

void SparseSurfelFusion::Visualizer::DrawFusedSurfelCloud(DeviceArrayView<float4> surfel_vertex, DeviceArrayView<unsigned> fused_indicator, cudaTextureObject_t observeMap, const Matrix4f& world2camera)
{
    FUNCTION_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    //Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);

    PointCloud3f_Pointer ObservePCL = downloadPointCloud(observeMap);
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(ObservePCL, world2camera);
    DrawMatchedCloudPair(fused_cloud, unfused_cloud, transformed_cloud_1);
}

void SparseSurfelFusion::Visualizer::DrawFusedSurfelCloud(DeviceArrayView<float4> surfel_vertex, DeviceArrayView<unsigned> fused_indicator, cudaTextureObject_t observeMap_0, const Matrix4f& initSE3_0, cudaTextureObject_t observeMap_1, const Matrix4f& initSE3_1, cudaTextureObject_t observeMap_2, const Matrix4f& initSE3_2)
{
    FUNCTION_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    //Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);
    PointCloud3f_Pointer ObservePCL_0 = downloadPointCloud(observeMap_0);
    PointCloud3f_Pointer cloud_0 = transformPointCloud(ObservePCL_0, initSE3_0);
    PointCloud3f_Pointer ObservePCL_1 = downloadPointCloud(observeMap_1);
    PointCloud3f_Pointer cloud_1 = transformPointCloud(ObservePCL_1, initSE3_1);
    PointCloud3f_Pointer ObservePCL_2 = downloadPointCloud(observeMap_2);
    PointCloud3f_Pointer cloud_2 = transformPointCloud(ObservePCL_2, initSE3_2);
    DrawMatchedCloudPair(fused_cloud, unfused_cloud, cloud_0, cloud_1, cloud_2);
}

void SparseSurfelFusion::Visualizer::DrawFusedSurfelCloud(DeviceArrayView<float4> surfel_vertex, cudaTextureObject_t observeMap_0, const Matrix4f& initSE3_0, cudaTextureObject_t observeMap_1, const Matrix4f& initSE3_1, cudaTextureObject_t observeMap_2, const Matrix4f& initSE3_2)
{
    CHECKCUDA(cudaDeviceSynchronize());
    DeviceArray<float4> surfel_vertex_array = DeviceArray<float4>((float4*)surfel_vertex.RawPtr(), surfel_vertex.Size());

    PointCloud3f_Pointer currCloudArrayHost = downloadPointCloud(surfel_vertex_array);
    PointCloud3f_Pointer ObservePCL_0 = downloadPointCloud(observeMap_0);
    PointCloud3f_Pointer cloud_0 = transformPointCloud(ObservePCL_0, initSE3_0);
    PointCloud3f_Pointer ObservePCL_1 = downloadPointCloud(observeMap_1);
    PointCloud3f_Pointer cloud_1 = transformPointCloud(ObservePCL_1, initSE3_1);
    PointCloud3f_Pointer ObservePCL_2 = downloadPointCloud(observeMap_2);
    PointCloud3f_Pointer cloud_2 = transformPointCloud(ObservePCL_2, initSE3_2);
    DrawMatchedCloudPair(currCloudArrayHost, currCloudArrayHost, cloud_0, cloud_1, cloud_2);
}

void SparseSurfelFusion::Visualizer::DrawFusedSurfelCloud(
    SparseSurfelFusion::DeviceArrayView<float4> surfel_vertex,
        unsigned num_remaining_surfels
) {
    FUNCTION_CHECK(surfel_vertex.Size() >= num_remaining_surfels);

    //Construct the host cloud
    PointCloud3f_Pointer remaining_cloud(new PointCloud3f);
    PointCloud3f_Pointer appended_cloud(new PointCloud3f);

    //Download it
    separateDownloadPointCloud(surfel_vertex, num_remaining_surfels, remaining_cloud, appended_cloud);

    //Ok draw it
    DrawMatchedCloudPair(remaining_cloud, appended_cloud);
}

void SparseSurfelFusion::Visualizer::DrawFusedAppendedSurfelCloud(
    SparseSurfelFusion::DeviceArrayView<float4> surfel_vertex,
    SparseSurfelFusion::DeviceArrayView<unsigned int> fused_indicator,
        cudaTextureObject_t depth_vertex_map,
    SparseSurfelFusion::DeviceArrayView<unsigned int> append_indicator,
        const SparseSurfelFusion::Matrix4f &world2camera
) {
    FUNCTION_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    //Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);

    //Download it
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_indicator);

    //Draw it
    DrawMatchedCloudPair(fused_cloud, h_append_surfels, world2camera);
}

void SparseSurfelFusion::Visualizer::DrawAppendedSurfelCloud(
        DeviceArrayView<float4> surfel_vertex,
        cudaTextureObject_t depth_vertex_map,
        DeviceArrayView<unsigned int> append_indicator,
        const SparseSurfelFusion::Matrix4f &world2camera
) {
    auto h_surfels = downloadPointCloud(DeviceArray<float4>((float4 *) surfel_vertex.RawPtr(), surfel_vertex.Size()));
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_indicator);
    DrawMatchedCloudPair(h_surfels, h_append_surfels, world2camera);
}


void SparseSurfelFusion::Visualizer::DrawAppendedSurfelCloud(
        DeviceArrayView<float4> surfel_vertex,
        cudaTextureObject_t depth_vertex_map0,
        cudaTextureObject_t depth_vertex_map1,
        DeviceArrayView<ushort3> append_pixel,
        const SparseSurfelFusion::Matrix4f &world2camera
) {
    auto h_surfels = downloadPointCloud(DeviceArray<float4>((float4 *) surfel_vertex.RawPtr(), surfel_vertex.Size()));
    auto h_append_surfels = downloadPointCloud(depth_vertex_map0, depth_vertex_map1, append_pixel);
    DrawMatchedCloudPair(h_surfels, h_append_surfels, world2camera);
}

void SparseSurfelFusion::Visualizer::DrawVertexMapAsPointCloud(cudaTextureObject_t vertexMap, cudaTextureObject_t observeMap)
{
    CHECKCUDA(cudaDeviceSynchronize());
    PointCloud3f_Pointer VertexMapPCL = downloadPointCloud(vertexMap);
    PointCloud3f_Pointer ObservePCL = downloadPointCloud(observeMap);
    DrawMatchedCloudPair(VertexMapPCL, ObservePCL);
}

void SparseSurfelFusion::Visualizer::DrawFusedProcessInCameraView(cudaTextureObject_t LiveVertex, mat34 initialSE3Inverse, cudaTextureObject_t observeMap)
{
    CHECKCUDA(cudaDeviceSynchronize());
    PointCloud3f_Pointer VertexMapPCL = downloadPointCloud(LiveVertex);
    PointCloud3f_Pointer ObservePCL = downloadPointCloud(observeMap);
    DrawMatchedCloudPair(VertexMapPCL, ObservePCL, toEigen(initialSE3Inverse));
}

void SparseSurfelFusion::Visualizer::DrawFusedProcessInCanonicalField(const DeviceArrayView<float4>& canVertex, const DeviceArrayView<float4>& liveVertex)
{
    CHECKCUDA(cudaDeviceSynchronize());
    DeviceArray<float4> can_cloud_array = DeviceArray<float4>((float4*)canVertex.RawPtr(), canVertex.Size());
    DeviceArray<float4> live_cloud_array = DeviceArray<float4>((float4*)liveVertex.RawPtr(), liveVertex.Size());
    const auto h_can_point_cloud = downloadPointCloud(can_cloud_array);
    const auto h_live_point_cloud = downloadPointCloud(live_cloud_array);
    DrawMatchedCloudPair(h_can_point_cloud, h_live_point_cloud);
}

void SparseSurfelFusion::Visualizer::DrawMatchedReferenceAndObseveredPointsPair(const DeviceArrayView<float4>& canVertex, const DeviceArrayView<float4>& observedVertex)
{
    CHECKCUDA(cudaDeviceSynchronize());
    DeviceArray<float4> can_cloud_array = DeviceArray<float4>((float4*)canVertex.RawPtr(), canVertex.Size());
    DeviceArray<float4> live_cloud_array = DeviceArray<float4>((float4*)observedVertex.RawPtr(), observedVertex.Size());
    const auto h_can_point_cloud = downloadPointCloud(can_cloud_array);
    const auto h_observed_point_cloud = downloadPointCloud(live_cloud_array);
    std::vector<float4> canVertexHost(canVertex.Size());
    std::vector<float4> observedVertexHost(observedVertex.Size());
    canVertex.Download(canVertexHost);
    observedVertex.Download(observedVertexHost);
    DrawMatchedCloudPairWithLine(h_can_point_cloud, h_observed_point_cloud, canVertexHost.data(), observedVertexHost.data(), observedVertex.Size());
}

void SparseSurfelFusion::Visualizer::DrawMatchedReferenceAndObseveredPointsPair(const DeviceArrayView<float4>& correctedVertex, const DeviceArrayView<float4>& observedVertex, mat34 world2camera)
{
    CHECKCUDA(cudaDeviceSynchronize());
    DeviceArray<float4> corrected_cloud_array = DeviceArray<float4>((float4*)correctedVertex.RawPtr(), correctedVertex.Size());
    DeviceArray<float4> observe_cloud_array = DeviceArray<float4>((float4*)observedVertex.RawPtr(), observedVertex.Size());
    PointCloud3f_Pointer correctedCloudArrayHost = downloadPointCloud(corrected_cloud_array);
    PointCloud3f_Pointer observedCloudArrayHost = downloadPointCloud(observe_cloud_array);
    observedCloudArrayHost = transformPointCloud(observedCloudArrayHost, toEigen(world2camera));

    std::vector<float4> correctedVertexHost(correctedVertex.Size());
    std::vector<float4> observedVertexHost(observedVertex.Size());
    correctedVertex.Download(correctedVertexHost);
    observedVertex.Download(observedVertexHost);

    DrawMatchedCloudPairWithLine(correctedCloudArrayHost, observedCloudArrayHost, correctedVertexHost.data(), observedVertexHost.data(), observedVertex.Size());
}

void SparseSurfelFusion::Visualizer::DrawMatchedReferenceAndObseveredPointsPair(cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, mat34 camPos, const DeviceArrayView<float4>& matchedVertex_0, const DeviceArrayView<float4>& matchedVertex_1)
{
    CHECKCUDA(cudaDeviceSynchronize());
    PointCloud3f_Pointer preCloudArrayHost = downloadPointCloud(preVertex);
    PointCloud3f_Pointer transformed_preCloudArrayHost = transformPointCloud(preCloudArrayHost, toEigen(camPos));

    PointCloud3f_Pointer currCloudArrayHost = downloadPointCloud(currVertex);
    PointCloud3f_Pointer transformed_currCloudArrayHost = transformPointCloud(currCloudArrayHost, toEigen(camPos));
    DeviceArray<float4> machedCloudArray_0 = DeviceArray<float4>((float4*)matchedVertex_0.RawPtr(), matchedVertex_0.Size());
    DeviceArray<float4> machedCloudArray_1 = DeviceArray<float4>((float4*)matchedVertex_1.RawPtr(), matchedVertex_1.Size());
    const auto machedCloudArray_0_Host = downloadPointCloud(machedCloudArray_0);
    const auto machedCloudArray_1_Host = downloadPointCloud(machedCloudArray_1);
    std::vector<float4> matchedVertex_0_Host(matchedVertex_0.Size());
    std::vector<float4> matchedVertex_1_Host(matchedVertex_1.Size());
    matchedVertex_0.Download(matchedVertex_0_Host);
    matchedVertex_1.Download(matchedVertex_1_Host);

    DrawMatchedCloudPairWithLine(transformed_preCloudArrayHost, transformed_currCloudArrayHost, machedCloudArray_0_Host, machedCloudArray_1_Host, matchedVertex_0_Host.data(), matchedVertex_1_Host.data(), matchedVertex_0.Size());

}

void SparseSurfelFusion::Visualizer::DrawOpticalFlowMapDenseAlignment(cudaTextureObject_t PreVertexMap, cudaTextureObject_t CurrVertexMap, const DeviceArrayView2D<mat34>& vertexSe3Map, const unsigned int mapCols, const unsigned int mapRows)
{
    CHECKCUDA(cudaDeviceSynchronize());
    // 光流引导得到的当前帧的点
    PointCloud3f_Pointer Pre2CurrCloudArrayHost = downloadPointCloudUseSE3Map(PreVertexMap, vertexSe3Map);
    PointCloud3f_Pointer currCloudArrayHost = downloadPointCloud(CurrVertexMap);
    DrawMatchedCloudPair(Pre2CurrCloudArrayHost, currCloudArrayHost);
}

void SparseSurfelFusion::Visualizer::DrawUnitNodeError(DeviceArrayView<float4> nodeCoor, DeviceArrayView<float> nodeError)
{
    CHECKCUDA(cudaDeviceSynchronize());
    PointCloud3fRGB_Pointer pointCloud(new PointCloud3fRGB());

    std::vector<float4> nodeCoorHost(nodeCoor.Size());
    nodeCoor.Download(nodeCoorHost);
    std::vector<float> nodeErrorHost(nodeCoor.Size());
    nodeError.Download(nodeErrorHost);
    for (int i = 0; i < nodeCoor.Size(); i++) {
        pcl::PointXYZRGB colorPoint;
        colorPoint.x = nodeCoorHost[i].x * 1000.0f;
        colorPoint.y = nodeCoorHost[i].y * 1000.0f;
        colorPoint.z = nodeCoorHost[i].z * 1000.0f;
        colorPoint.r = nodeErrorHost[i] * 255.0f;
        colorPoint.g = nodeErrorHost[i] * 255.0f;
        colorPoint.b = nodeErrorHost[i] * 255.0f;
        pointCloud->push_back(colorPoint);
    }
    DrawColoredPointCloud(pointCloud);
}

void SparseSurfelFusion::Visualizer::DrawCrossCorrPairs(DeviceArray<DepthSurfel> surfels, cudaTextureObject_t vertexMap_0, mat34 initialPose_0, cudaTextureObject_t vertexMap_1, mat34 initialPose_1, cudaTextureObject_t vertexMap_2, mat34 initialPose_2, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs)
{
    std::string window_title = "Cross View Matching";
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.999f, 0.999f, 0.999f);
    CHECKCUDA(cudaDeviceSynchronize());

    PointCloud3fRGB_Pointer point_cloud_rgb;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfels, point_cloud_rgb, normal_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_rgb, "colored cloud");

    const unsigned int pairsNum = crossCorrPairs.Size();
    DeviceArray<float3> Point_1;
    DeviceArray<float3> Point_2;
    Point_1.create(pairsNum);
    Point_2.create(pairsNum);
    TransCrossViewPairs2SameCoordinateSystem(vertexMap_0, initialPose_0, vertexMap_1, initialPose_1, vertexMap_2, initialPose_2, crossCorrPairs, Point_1, Point_2);
    CHECKCUDA(cudaDeviceSynchronize());
    std::vector<float3> Point_1_Host(pairsNum);
    std::vector<float3> Point_2_Host(pairsNum);
    Point_1.download(Point_1_Host);
    Point_2.download(Point_2_Host);
    PointCloud3f_Pointer PointCloud_1(new PointCloud3f);
    PointCloud3f_Pointer PointCloud_2(new PointCloud3f);
    for (int i = 0; i < pairsNum; i++) {
        pcl::PointXYZ point_xyz_1, point_xyz_2;
        point_xyz_1.x = Point_1_Host[i].x * 1000.0f;
        point_xyz_1.y = Point_1_Host[i].y * 1000.0f;
        point_xyz_1.z = Point_1_Host[i].z * 1000.0f;
        PointCloud_1->points.push_back(point_xyz_1);

        point_xyz_2.x = Point_2_Host[i].x * 1000.0f;
        point_xyz_2.y = Point_2_Host[i].y * 1000.0f;
        point_xyz_2.z = Point_2_Host[i].z * 1000.0f;
        PointCloud_2->points.push_back(point_xyz_2);

        viewer->addLine(point_xyz_1, point_xyz_2, 0.0, 0.0, 0.0, "line" + std::to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "line" + std::to_string(i));
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(PointCloud_1, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(PointCloud_1, handler_1, "cloud 1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(PointCloud_2, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(PointCloud_2, handler_2, "cloud 2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 2");

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void SparseSurfelFusion::Visualizer::DrawInterpolatedSurfels(DeviceArray<DepthSurfel> surfels, DeviceArrayView2D<float4> InterVertexMap_0, DeviceArrayView2D<uchar> markInterVertex_0, mat34 initialPose_0, DeviceArrayView2D<float4> InterVertexMap_1, DeviceArrayView2D<uchar> markInterVertex_1, mat34 initialPose_1, DeviceArrayView2D<float4> InterVertexMap_2, DeviceArrayView2D<uchar> markInterVertex_2, mat34 initialPose_2, cudaStream_t stream)
{
    std::string window_title = "Cross View Matching";
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0.3f, 0.3f, 0.3f);
    CHECKCUDA(cudaDeviceSynchronize());

    PointCloud3fRGB_Pointer point_cloud_rgb;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfels, point_cloud_rgb, normal_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_rgb, "colored cloud");

    DeviceArray2D<float4> interpolatedVertexMap_0, interpolatedVertexMap_1, interpolatedVertexMap_2;
    interpolatedVertexMap_0.create(InterVertexMap_0.Rows(), InterVertexMap_0.Cols());
    interpolatedVertexMap_1.create(InterVertexMap_1.Rows(), InterVertexMap_1.Cols());
    interpolatedVertexMap_2.create(InterVertexMap_2.Rows(), InterVertexMap_2.Cols());

    TransInterpolatedSurfels2SameCoordinateSystem(
        InterVertexMap_0, markInterVertex_0, interpolatedVertexMap_0, initialPose_0,
        InterVertexMap_1, markInterVertex_1, interpolatedVertexMap_1, initialPose_1,
        InterVertexMap_2, markInterVertex_2, interpolatedVertexMap_2, initialPose_2
    );

    PointCloud3f_Pointer interPointCloud_0 = downloadPointCloud(interpolatedVertexMap_0);
    PointCloud3f_Pointer interPointCloud_1 = downloadPointCloud(interpolatedVertexMap_1);
    PointCloud3f_Pointer interPointCloud_2 = downloadPointCloud(interpolatedVertexMap_2);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_0(interPointCloud_0, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(interPointCloud_0, handler_0, "cloud 1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(interPointCloud_1, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(interPointCloud_1, handler_1, "cloud 2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 2");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(interPointCloud_2, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(interPointCloud_2, handler_2, "cloud 3");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud 3");

    interpolatedVertexMap_0.release();
    interpolatedVertexMap_1.release();
    interpolatedVertexMap_2.release();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}








