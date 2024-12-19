//
// Created by wei on 2/20/18.
//

#pragma once

#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/DeviceReadWrite/DeviceArrayView.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <math/MatUtils.h>

namespace SparseSurfelFusion {


	/**
     * \brief This class is for DEBUG visualization. Any methods 
     *        in this class should NOT be used in real-time code.
     */
    class Visualizer {
    public:
        using Ptr = std::shared_ptr<Visualizer>;
	    
		DEFAULT_CONSTRUCT_DESTRUCT(Visualizer);
		NO_COPY_ASSIGN_MOVE(Visualizer);

		/* The depth image drawing methods
		 */
		static void DrawDepthImage(const cv::Mat& depth_img);
        static void SaveDepthImage(const cv::Mat& depth_img, const std::string& path);
		static void DrawDepthImage(const DeviceArray2D<unsigned short>& depth_img);
		static void SaveDepthImage(const DeviceArray2D<unsigned short>& depth_img, const std::string& path);
    	static void DrawDepthImage(cudaTextureObject_t depth_img);
		static void SaveDepthImage(cudaTextureObject_t depth_img, const std::string& path);

		/* The color image drawing methods
		*/
        static void DrawRGBImage(const cv::Mat& rgb_img);
        static void SaveRGBImage(const cv::Mat& rgb_img, const std::string& path);
		static void DrawRGBImage(const DeviceArray<uchar3>& rgb_img, const int rows, const int cols);
		static void SaveRGBImage(const DeviceArray<uchar3>& rgb_img, const int rows, const int cols, const std::string& path);
    	static void DrawNormalizeRGBImage(cudaTextureObject_t rgb_img);
		static void SaveNormalizeRGBImage(cudaTextureObject_t rgb_img, const std::string& path);
		static void DrawColorTimeMap(cudaTextureObject_t color_time_map);
	    static void DrawNormalMap(cudaTextureObject_t normal_map);
	    
	    /* The gray scale image drawing for filtered
	     */
	    static void DrawGrayScaleImage(const cv::Mat& gray_scale_img);
		static void DrawGrayScaleImage(const cv::Mat& gray_scale_img, std::string& windowName);
	    static void SaveGrayScaleImage(const cv::Mat& gray_scale_img, const std::string& path);
	    static void DrawGrayScaleImage(cudaTextureObject_t gray_scale_img, float scale = 1.0f);
		static void DrawAlignmentErrorMap(cudaTextureObject_t errorMap, cudaTextureObject_t maskTexture, std::string windowName, float scale = 1.0f);
	    static void SaveGrayScaleImage(cudaTextureObject_t gray_scale_img, const std::string& path, float scale = 1.0f);

		static void DrawGradientMap(cudaTextureObject_t gradientMap, std::string orientation);
	    
		/* The segmentation mask drawing methods
		*/
		static void MarkSegmentationMask(
			const std::vector<unsigned char>& mask, 
			cv::Mat& rgb_img,
			const unsigned sample_rate = 2
		);
		static void DrawSegmentMask(
			const std::vector<unsigned char>& mask, 
			cv::Mat& rgb_img, 
			const unsigned sample_rate = 2
		);
		static void SaveSegmentMask(
			const std::vector<unsigned char>& mask, 
			cv::Mat& rgb_img,
			const std::string& path, 
			const unsigned sample_rate = 2
		);
		static void DrawSegmentMask(
			cudaTextureObject_t mask, 
			cudaTextureObject_t normalized_rgb_img,
			const unsigned sample_rate = 2
		);
		static void SaveSegmentMask(
			cudaTextureObject_t mask, 
			cudaTextureObject_t normalized_rgb_img, 
			const std::string& path, 
			const unsigned sample_rate = 2
		);
	    static void SaveRawSegmentMask(
		    cudaTextureObject_t mask,
		    const std::string& path
	    );
	    static void DrawRawSegmentMask(
		    cudaTextureObject_t mask
	    );
		static void DrawRawSegmentMask(
			const unsigned int view,
			cudaTextureObject_t mask
		);
		static void DrawFilteredSegmentMask(
			cudaTextureObject_t filteredMask
		);

		/* The binary meanfield drawing methods
		*/
		static void DrawBinaryMeanfield(cudaTextureObject_t meanfield_q);
		static void SaveBinaryMeanfield(cudaTextureObject_t meanfield_q, const std::string& path);
	    
	    
	    /* Visualize the valid geometry maps as binary mask
	     */
	    static void DrawValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize);
	    static void SaveValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize, const std::string& path);
	    
	    static cv::Mat GetValidityMapCV(cudaTextureObject_t index_map, int validity_halfsize);
	    //Mark the validity of each index map pixel and save them to flatten indicator
	    //Assume pre-allcoated indicator
	    static void MarkValidIndexMapValue(
		    cudaTextureObject_t index_map,
		    int validity_halfsize,
		    DeviceArray<unsigned char> flatten_validity_indicator
	    );
	    
	    
	    
	    /* The correspondence
	     */
	    static void DrawImagePairCorrespondence(
		    cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1,
		    const DeviceArray<ushort4>& correspondence
	    );


		/* The point cloud drawing methods
		*/
		//static void Draw2PointCloudWithDiffColor(const PointCloud3fRGB_Pointer& point_cloud0, const PointCloud3fRGB_Pointer& point_cloud1);
		static void DrawPointCloud(const PointCloud3f_Pointer& point_cloud);
		static void DrawPointCloudcanandlive(const PointCloud3f_Pointer& can, const PointCloud3f_Pointer& live);
	    static void DrawPointCloud(const DeviceArray<float4>& point_cloud);
		static void DrawPointCloudcanandlive(const DeviceArray<float4>& can, const DeviceArray<float4>& live);
	    static void DrawPointCloud(const DeviceArrayView<float4>& point_cloud);
		static void DrawPointCloudcanandlive(const DeviceArrayView<float4>& can, const DeviceArrayView<float4>& live);
		static void DrawPointCloud(const DeviceArray2D<float4>& vertex_map);
	    static void DrawPointCloud(const DeviceArray<DepthSurfel>& surfel_array);
		static void DrawPointCloud(const DeviceArrayView<DepthSurfel> surfel_array);
		static void DrawPointCloud(cudaTextureObject_t vertex_map);
	    static void SavePointCloud(const std::vector<float4>& point_cloud, const std::string& path);
		static void SavePointCloud(cudaTextureObject_t veretx_map, const std::string& path);
	    static void SavePointCloud(const DeviceArrayView<float4> point_cloud, const std::string& path);


		/* The point cloud with normal
		 */
		static void DrawPointCloudWithNormal(const PointCloud3fRGB_Pointer& point_cloud, const PointCloudNormal_Pointer& normal_cloud);
		/* The point cloud with normal
		 */
		static void DrawPointCloudWithNormal(const PointCloud3f_Pointer& point_cloud, const PointCloudNormal_Pointer& normal_cloud);

		//hsg 双视角的
		static void DrawPointCloudDoubleView(const PointCloud3f_Pointer& point_cloud0, const PointCloud3f_Pointer& point_cloud1);

		static void DrawPointCloudDoubleView(
			const PointCloud3fRGB_Pointer& point_cloud0,
			const PointCloudNormal_Pointer& normal_cloud0,
			const PointCloud3fRGB_Pointer& point_cloud1,
			const PointCloudNormal_Pointer& normal_cloud1
		);

		static void DrawPointCloudDoubleView(cudaTextureObject_t vertex_map0, cudaTextureObject_t vertex_map1);

	    static void DrawPointCloudWithNormal(
		    const DeviceArray<float4>& vertex_cloud,
		    const DeviceArray<float4>& normal_cloud
	    );
	    static void DrawPointCloudWithNormal(
		    const DeviceArrayView<float4>& vertex_cloud,
		    const DeviceArrayView<float4>& normal_cloud
	    );
		static void DrawPointCloudWithNormal(
			const DeviceArray2D<float4>& vertex_map,
			const DeviceArray2D<float4>& normal_map
		);

		//这个函数是把两个视角下的三维点显示在一起  可用于debug重新生成的vertexconfidence谁否正确
		static void DrawPointCloud2OneView(cudaTextureObject_t vertex_map0, cudaTextureObject_t vertex_map1);
		static void DrawPointCloud2OneView(cudaTextureObject_t vertex_map0, cudaTextureObject_t vertex_map1, mat34 SE3);

		static void DrawPointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map);
	    static void DrawPointCloudWithNormal(const DeviceArray<DepthSurfel>& surfel_array);
		static void DrawPointCloudWithNormal(const DeviceArray<DepthSurfel>& surfel_array0,const DeviceArray<DepthSurfel>& surfel_array1);
	    static void SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map);
	    
	    
	    /* The colored point cloud
	     */
	    static void DrawColoredPointCloud(const PointCloud3fRGB_Pointer& point_cloud);
	    static void SaveColoredPointCloud(const PointCloud3fRGB_Pointer& point_cloud, const std::string& path);
	    static void DrawColoredPointCloud(const DeviceArray<float4>& vertex, const DeviceArray<float4>& color_time);
	    static void DrawColoredPointCloud(const DeviceArrayView<float4>& vertex, const DeviceArrayView<float4>& color_time);
	    static void DrawColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map);
	    static void SaveColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map, const std::string& path);
	    
	    /* The matched point cloud
	     */
	    static void DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1,
	                                     const PointCloud3f_Pointer& cloud_2);
		static void DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1,
										 const PointCloud3f_Pointer& cloud_2,
										 const PointCloud3f_Pointer& cloud_3);
		static void DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1,
										 const PointCloud3f_Pointer& cloud_2,
										 const PointCloud3f_Pointer& cloud_3, 
										 const PointCloud3f_Pointer& cloud_4, 
										 const PointCloud3f_Pointer& cloud_5);
	    static void DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1,
	                                     const PointCloud3f_Pointer& cloud_2,
	                                     const Eigen::Matrix4f & from1To2);
	    static void DrawMatchedCloudPair(
										 cudaTextureObject_t cloud_1,
										 const DeviceArray<float4>& cloud_2,
										 const Matrix4f& from1To2);
		static void DrawMatchedCloudPair(cudaTextureObject_t cloud_1,
										 const DeviceArrayView<float4>& cloud_2,
										 const Matrix4f& from1To2);
	    static void DrawMatchedCloudPair(cudaTextureObject_t cloud_1,
	                                     cudaTextureObject_t cloud_2,
	                                     const Matrix4f& from1To2);
	
		static void DrawMatchedCloudPair(const DeviceArrayView<DepthSurfel>& observedVertex,
										 const DeviceArrayView<DepthSurfel>& referenceVertex,
										 const mat34& Observed2Reference);

		static void DrawMatchedCloudPairWithLine(
										 const PointCloud3f_Pointer& cloud_1,
										 const PointCloud3f_Pointer& cloud_2, 
										 const float4* canVertexHost, 
										 const float4* observedVertexHost,
										 const unsigned int vertexNum);

		static void DrawMatchedCloudPairWithLine(
										 const PointCloud3f_Pointer& preVertex,
										 const PointCloud3f_Pointer& currVertex,
										 const PointCloud3f_Pointer& machedVertex_0,
										 const PointCloud3f_Pointer& machedVertex_1,
										 const float4* machedVertex_0_Host,
										 const float4* machedVertex_1_Host,
										 const unsigned int matchedPairsNum);

	    static void SaveMatchedCloudPair(
		    const PointCloud3f_Pointer& cloud_1,
		    const PointCloud3f_Pointer& cloud_2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	    static void SaveMatchedCloudPair(
		    const PointCloud3f_Pointer & cloud_1,
		    const PointCloud3f_Pointer & cloud_2,
		    const Eigen::Matrix4f & from1To2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	    static void SaveMatchedCloudPair(
		    cudaTextureObject_t cloud_1,
		    const DeviceArray<float4>& cloud_2,
		    const Matrix4f& from1To2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	    static void SaveMatchedCloudPair(
		    cudaTextureObject_t cloud_1,
		    const DeviceArrayView<float4>& cloud_2,
		    const Matrix4f& from1To2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	
	
	    /* The method to draw matched color-point cloud
	     */
	    static void DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer& cloud_1,
	                                        const PointCloud3fRGB_Pointer& cloud_2);
	    static void DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer& cloud_1,
	                                        const PointCloud3fRGB_Pointer& cloud_2,
	                                        const Eigen::Matrix4f& from1To2);

		static void DrawMatchedCloudPair(
		    cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map,
		    const DeviceArrayView<float4>& surfel_array, const DeviceArrayView<float4>& color_time_array,
		    const Eigen::Matrix4f& camera2world
	    );

		static void Debugworld2Camera(
			cudaTextureObject_t vertex_map,
			cudaTextureObject_t depth_vertex_map,

			mat34 world2camera
		);
		static void DebugCamera2World(
			cudaTextureObject_t depth_vertex_map0,
			cudaTextureObject_t depth_vertex_map1,
			mat34 camera2world0,
			mat34 camera2world1,
			mat34 SE3
		);
		static void DebugCamera2WorldUseFirstFrame(
			DeviceArrayView<float4> firstframedata,
			cudaTextureObject_t depth_vertex_map0,
			cudaTextureObject_t depth_vertex_map1,
			mat34 camera2world0,
			mat34 camera2world1,
			mat34 SE3
		);





		static void DrawFusedAndRemainSurfelCloud(
			DeviceArrayView<float4> surfel_vertex,
			unsigned remainingnumber,
			unsigned number0view
		);


		/**
		 * \brief 红色是第一个参数，白色是第二个参数.
		 *
		 * \param surfel_vertex
		 * \param fused_indicator
		 */
		static void DrawFusedSurfelCloud(
			DeviceArrayView<float4> surfel_vertex,
			DeviceArrayView<unsigned> fused_indicator
		);
		/**
		 * \brief 绘制reinit帧Live域的点，Live域中能融合的点，观察到的点.
		 * 
		 * \param surfel_vertex Live域的点
		 * \param fused_indicator Live域中能被融合的点
		 * \param depth_vertex_map 当前帧实际观察到的点
		 * \param world2camera 观察到的点转到Live域
		 */
		static void DrawFusedSurfelCloud(
			DeviceArrayView<float4> surfel_vertex,
			DeviceArrayView<unsigned> fused_indicator,
			cudaTextureObject_t depth_vertex_map,
			const Matrix4f& world2camera
		);

		static void DrawFusedSurfelCloud(
			DeviceArrayView<float4> surfel_vertex,
			DeviceArrayView<unsigned> fused_indicator,
			cudaTextureObject_t observeMap_0,
			const Matrix4f& initSE3_0,
			cudaTextureObject_t observeMap_1,
			const Matrix4f& initSE3_1,
			cudaTextureObject_t observeMap_2,
			const Matrix4f& initSE3_2
		);

		static void DrawFusedSurfelCloud(
			DeviceArrayView<float4> surfel_vertex,
			cudaTextureObject_t observeMap_0,
			const Matrix4f& initSE3_0,
			cudaTextureObject_t observeMap_1,
			const Matrix4f& initSE3_1,
			cudaTextureObject_t observeMap_2,
			const Matrix4f& initSE3_2
		);

	    static void DrawFusedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    unsigned num_remaining_surfels
	    );
	    
	    static void DrawFusedAppendedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    DeviceArrayView<unsigned> fused_indicator,
		    cudaTextureObject_t depth_vertex_map,
		    DeviceArrayView<unsigned> append_indicator,
		    const Matrix4f& world2camera
	    );
	    
	    static void DrawAppendedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    cudaTextureObject_t depth_vertex_map,
		    DeviceArrayView<unsigned> append_indicator,
		    const Matrix4f& world2camera
	    );
	    static void DrawAppendedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    cudaTextureObject_t depth_vertex_map0,
			cudaTextureObject_t depth_vertex_map1,
		    DeviceArrayView<ushort3> append_pixel,
		    const Matrix4f& world2camera
	    );


		static void DrawVertexMapAsPointCloud(cudaTextureObject_t vertexMap, cudaTextureObject_t observeMap);

		static void DrawFusedProcessInCameraView(cudaTextureObject_t LiveVertex, mat34 initialSE3Inverse, cudaTextureObject_t observeMap);

		static void DrawFusedProcessInCanonicalField(const DeviceArrayView<float4>& canVertex, const DeviceArrayView<float4>& liveVertex);

		// 绘制匹配点，并且连线，红色点是Renference点，绿色点是Observe的点，绘制观察帧所匹配的Reference域的点
		static void DrawMatchedReferenceAndObseveredPointsPair(const DeviceArrayView<float4>& canVertex, const DeviceArrayView<float4>& observedVertex);
		static void DrawMatchedReferenceAndObseveredPointsPair(const DeviceArrayView<float4>& correctedVertex, const DeviceArrayView<float4>& observedVertex, mat34 world2camera);
		static void DrawMatchedReferenceAndObseveredPointsPair(cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, mat34 camPos, const DeviceArrayView<float4>& matchedVertex_0, const DeviceArrayView<float4>& matchedVertex_1);

		// 根据solverMapIndexMap绘制canonicalVertex中的有效点
		static void DrawFilteredSolverMapCanonicalVertex(cudaTextureObject_t canonicalVertex, cudaTextureObject_t solverMapIndexMap, const unsigned int mapCols, const unsigned int mapRows, cudaStream_t stream = 0);

		// 红色是Pre通过光流对准的，绿色是Curr观察的
		static void DrawOpticalFlowMapDenseAlignment(cudaTextureObject_t PreVertexMap, cudaTextureObject_t CurrVertexMap, const DeviceArrayView2D<mat34>& vertexSe3Map, const unsigned int mapCols, const unsigned int mapRows);

		/**
		 * \brief 绘制节点误差，颜色越深，误差越大.
		 * 
		 * \param nodeCoor 节点坐标
		 * \param nodeError 节点误差
		 */
		static void DrawUnitNodeError(DeviceArrayView<float4> nodeCoor, DeviceArrayView<float> nodeError);

		/**
		 * \brief 绘制跨视角找到的匹配点.
		 * 
		 * \param vertexMap 稠密点map
		 * \param crossCorrPairs 跨视角的匹配点
		 */
		static void DrawCrossCorrPairs(
			DeviceArray<DepthSurfel> surfels,
			cudaTextureObject_t vertexMap_0, mat34 initialPose_0,
			cudaTextureObject_t vertexMap_1, mat34 initialPose_1,
			cudaTextureObject_t vertexMap_2, mat34 initialPose_2,
			DeviceArrayView<CrossViewCorrPairs> crossCorrPairs
		);

		static void TransCrossViewPairs2SameCoordinateSystem(
			cudaTextureObject_t vertexMap_0, mat34 initialPose_0,
			cudaTextureObject_t vertexMap_1, mat34 initialPose_1,
			cudaTextureObject_t vertexMap_2, mat34 initialPose_2,
			DeviceArrayView<CrossViewCorrPairs> crossCorrPairs,
			DeviceArray<float3> Points_1, DeviceArray<float3> Points_2,
			cudaStream_t stream = 0
		);

		static void DrawInterpolatedSurfels(
			DeviceArray<DepthSurfel> surfels,
			DeviceArrayView2D<float4> InterVertexMap_0, DeviceArrayView2D<uchar> markInterVertex_0, mat34 initialPose_0,
			DeviceArrayView2D<float4> InterVertexMap_1, DeviceArrayView2D<uchar> markInterVertex_1, mat34 initialPose_1,
			DeviceArrayView2D<float4> InterVertexMap_2, DeviceArrayView2D<uchar> markInterVertex_2, mat34 initialPose_2,
			cudaStream_t stream = 0
		);

		static void TransInterpolatedSurfels2SameCoordinateSystem(
			DeviceArrayView2D<float4> InterVertexMap_0, DeviceArrayView2D<uchar> markInterVertex_0, DeviceArray2D<float4> vertexMap_0, mat34 initialPose_0,
			DeviceArrayView2D<float4> InterVertexMap_1, DeviceArrayView2D<uchar> markInterVertex_1, DeviceArray2D<float4> vertexMap_1, mat34 initialPose_1,
			DeviceArrayView2D<float4> InterVertexMap_2, DeviceArrayView2D<uchar> markInterVertex_2, DeviceArray2D<float4> vertexMap_2, mat34 initialPose_2,
			cudaStream_t stream = 0
		);
    private:
        template<typename TPointInput, typename TNormalsInput>
        static void DrawPointCloudWithNormals_Generic(TPointInput& points, TNormalsInput& normals);
    };
}
