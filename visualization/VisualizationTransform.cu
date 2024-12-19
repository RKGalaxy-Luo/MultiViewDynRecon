//#include "common/common_texture_utils.h"
#include "Visualizer.h"

#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

namespace SparseSurfelFusion { 
	namespace device {

		__global__ void markValidIndexMapPixelKernel(
			cudaTextureObject_t index_map,
			int validity_halfsize,
			unsigned img_rows, unsigned img_cols,
			unsigned char* flatten_validity_indicator
		) {
			const auto x_center = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y_center = threadIdx.y + blockDim.y * blockIdx.y;
			if(x_center >= img_cols || y_center >= img_rows) return;
			const auto offset = x_center + y_center * img_cols;

			//Only depend on this pixel
			if(validity_halfsize <= 0) {
				const auto surfel_index = tex2D<unsigned>(index_map, x_center, y_center);
				unsigned char validity = 0;
				if(surfel_index != 0xFFFFFFFF) validity = 1;

				//Write it and return
				flatten_validity_indicator[offset] = validity;
				return;
			}

			//Should perform a window search as the halfsize is at least 1
			unsigned char validity = 1;
			for(auto y = y_center - validity_halfsize; y <= y_center + validity_halfsize; y++) {
				for(auto x = x_center - validity_halfsize; x <= x_center + validity_halfsize; x++) {
					if(tex2D<unsigned>(index_map, x, y) == 0xFFFFFFFF) validity = 0;
				}
			}

			//Save it
			flatten_validity_indicator[offset] = validity;
		}


		__global__ void DrawFilteredCanonicalVertexKernel(cudaTextureObject_t canonicalVertex, cudaTextureObject_t solverMapIndexMap, float4* fliteredVertex, const unsigned int mapCols, const unsigned int mapRows) {
			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= mapCols || y >= mapRows) return;
			const unsigned int flattenIdx = x + y * mapCols;
			const unsigned int indexMap = tex2D<unsigned int>(solverMapIndexMap, x, y);
			if (indexMap != 0xFFFFFFFF) {
				const float4 vertex = tex2D<float4>(canonicalVertex, x, y);
				fliteredVertex[flattenIdx] = vertex;
			}
			else {
				fliteredVertex[flattenIdx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
		}

		__global__ void TransCrossViewPairs(
			cudaTextureObject_t vertexMap_0, mat34 intialPose_0,
			cudaTextureObject_t vertexMap_1, mat34 intialPose_1,
			cudaTextureObject_t vertexMap_2, mat34 intialPose_2,
			DeviceArrayView<CrossViewCorrPairs> crossCorrPairs,
			float3* points_1, float3* points_2,
			const unsigned int pairsNum) {
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= pairsNum) return;
			ushort4 CorrPair = crossCorrPairs[idx].PixelPairs;
			float4 p1, p2;
			if (crossCorrPairs[idx].PixelViews.x == 0) {
				p1 = tex2D<float4>(vertexMap_0, CorrPair.x, CorrPair.y); 
				points_1[idx] = intialPose_0.rot * p1 + intialPose_0.trans; 
			}
			else if (crossCorrPairs[idx].PixelViews.x == 1) {
				p1 = tex2D<float4>(vertexMap_1, CorrPair.x, CorrPair.y); 
				points_1[idx] = intialPose_1.rot * p1 + intialPose_1.trans;
			}
			else { 
				p1 = tex2D<float4>(vertexMap_2, CorrPair.x, CorrPair.y);
				points_1[idx] = intialPose_2.rot * p1 + intialPose_2.trans;
			}

			if (crossCorrPairs[idx].PixelViews.y == 0) {
				p2 = tex2D<float4>(vertexMap_0, CorrPair.z, CorrPair.w); 
				points_2[idx] = intialPose_0.rot * p2 + intialPose_0.trans;
			}
			else if (crossCorrPairs[idx].PixelViews.y == 1) {
				p2 = tex2D<float4>(vertexMap_1, CorrPair.z, CorrPair.w); 
				points_2[idx] = intialPose_1.rot * p2 + intialPose_1.trans;
			}
			else { 
				p2 = tex2D<float4>(vertexMap_2, CorrPair.z, CorrPair.w); 
				points_2[idx] = intialPose_2.rot * p2 + intialPose_2.trans;
			}
		}

		__global__ void TransInterpolatedSurfels(
			DeviceArrayView2D<float4> InterVertexMap_0, DeviceArrayView2D<uchar> markInterVertex_0, PtrStepSize<float4> vertexMap_0, mat34 initialPose_0,
			DeviceArrayView2D<float4> InterVertexMap_1, DeviceArrayView2D<uchar> markInterVertex_1, PtrStepSize<float4> vertexMap_1, mat34 initialPose_1,
			DeviceArrayView2D<float4> InterVertexMap_2, DeviceArrayView2D<uchar> markInterVertex_2, PtrStepSize<float4> vertexMap_2, mat34 initialPose_2,
			const unsigned int cols, const unsigned int rows) {

			const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
			const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= cols || y >= rows) return;
/************************************* йс╫г1 *************************************/
			uchar mark_1 = markInterVertex_0(y, x);
			if (mark_1 == (uchar)1) {
				float3 vertex = initialPose_0.rot * InterVertexMap_0(y, x) + initialPose_0.trans;
				vertexMap_0.ptr(y)[x] = make_float4(vertex.x, vertex.y, vertex.z, InterVertexMap_0(y, x).w);
			}
			else {
				vertexMap_0.ptr(y)[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
/************************************* йс╫г2 *************************************/
			uchar mark_2 = markInterVertex_1(y, x);
			if (mark_2 == (uchar)1) {
				float3 vertex = initialPose_1.rot * InterVertexMap_1(y, x) + initialPose_1.trans;
				vertexMap_1.ptr(y)[x] = make_float4(vertex.x, vertex.y, vertex.z, InterVertexMap_1(y, x).w);
			}
			else {
				vertexMap_1.ptr(y)[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
/************************************* йс╫г3 *************************************/
			uchar mark_3 = markInterVertex_2(y, x);
			if (mark_3 == (uchar)1) {
				float3 vertex = initialPose_2.rot * InterVertexMap_2(y, x) + initialPose_2.trans;
				vertexMap_2.ptr(y)[x] = make_float4(vertex.x, vertex.y, vertex.z, InterVertexMap_2(y, x).w);
			}
			else {
				vertexMap_2.ptr(y)[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
		}


	} // device
} // SparseSurfelFusion

void SparseSurfelFusion::Visualizer::DrawValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize) {
	//Query the map
	cv::Mat validity_map = GetValidityMapCV(index_map, validity_halfsize);
	
	//Draw it
	DrawRGBImage(validity_map);
}


void SparseSurfelFusion::Visualizer::SaveValidIndexMap(
	cudaTextureObject_t index_map,
	int validity_halfsize,
	const std::string &path
) {
	//Query the map
	cv::Mat validity_map = GetValidityMapCV(index_map, validity_halfsize);
	
	//Save it
	cv::imwrite(path, validity_map);
}

cv::Mat SparseSurfelFusion::Visualizer::GetValidityMapCV(cudaTextureObject_t index_map, int validity_halfsize) {
	//Query the size
	unsigned width, height;
	query2DTextureExtent(index_map, width, height);
	
	//Malloc
	DeviceArray<unsigned char> flatten_validity_indicator;
	flatten_validity_indicator.create(width * height);
	
	//Mark the validity
	MarkValidIndexMapValue(index_map, validity_halfsize, flatten_validity_indicator);
	
	//Download it and transfer it into cv::Mat
	std::vector<unsigned char> h_validity_array;
	flatten_validity_indicator.download(h_validity_array);
	
	//The validity map
	cv::Mat validity_map = cv::Mat(height, width, CV_8UC1);
	unsigned num_valid_pixel = 0;
	for(auto y = 0; y < height; y++) {
		for(auto x = 0; x < width; x++) {
			const auto offset = x + y * width;
			if(h_validity_array[offset] > 0) {
				validity_map.at<unsigned char>(y, x) = 255;
				num_valid_pixel++;
			} else {
				validity_map.at<unsigned char>(y, x) = 0;
			}
		}
	}
	
	//Log the number of valid pixel
	//LOG(INFO) << "The number of valid pixel in the index map of rendered geometry with validity halfsize " << validity_halfsize << " is " << num_valid_pixel;
	return validity_map;
}

void SparseSurfelFusion::Visualizer::MarkValidIndexMapValue(
	cudaTextureObject_t index_map,
	int validity_halfsize,
	SparseSurfelFusion::DeviceArray<unsigned char> flatten_validity_indicator
) {
	//Query the size
	unsigned width, height;
	query2DTextureExtent(index_map, width, height);

	//Do it
	dim3 blk(16, 16);
	dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
	device::markValidIndexMapPixelKernel<<<grid, blk>>>(
		index_map, 
		validity_halfsize, 
		height, width, 
		flatten_validity_indicator.ptr()
	);

	//Always sync and check error
	CHECKCUDA(cudaDeviceSynchronize());

}



void SparseSurfelFusion::Visualizer::DrawFilteredSolverMapCanonicalVertex(cudaTextureObject_t canonicalVertex, cudaTextureObject_t solverMapIndexMap, const unsigned int mapCols, const unsigned int mapRows, cudaStream_t stream)
{
	CHECKCUDA(cudaDeviceSynchronize());
	DeviceArray<float4> filteredVertex;
	filteredVertex.create(mapCols * mapRows);
	dim3 block(16, 16);
	dim3 grid(divUp(mapCols, block.x), divUp(mapRows, block.y));
	device::DrawFilteredCanonicalVertexKernel << <grid, block, 0, stream >> > (canonicalVertex, solverMapIndexMap, filteredVertex.ptr(), mapCols, mapRows);
	CHECKCUDA(cudaDeviceSynchronize());
	DrawPointCloud(filteredVertex);

	filteredVertex.release();
}

void SparseSurfelFusion::Visualizer::TransCrossViewPairs2SameCoordinateSystem(cudaTextureObject_t vertexMap_0, mat34 initialPose_0, cudaTextureObject_t vertexMap_1, mat34 initialPose_1, cudaTextureObject_t vertexMap_2, mat34 initialPose_2, DeviceArrayView<CrossViewCorrPairs> crossCorrPairs, DeviceArray<float3> Points_1, DeviceArray<float3> Points_2, cudaStream_t stream)
{
	const unsigned int pairsNum = crossCorrPairs.Size();
	dim3 block(64);
	dim3 grid(divUp(pairsNum, block.x));
	device::TransCrossViewPairs << <grid, block, 0, stream >> > (vertexMap_0, initialPose_0, vertexMap_1, initialPose_1, vertexMap_2, initialPose_2, crossCorrPairs, Points_1.ptr(), Points_2.ptr(), pairsNum);
}

void SparseSurfelFusion::Visualizer::TransInterpolatedSurfels2SameCoordinateSystem(DeviceArrayView2D<float4> InterVertexMap_0, DeviceArrayView2D<uchar> markInterVertex_0, DeviceArray2D<float4> vertexMap_0, mat34 initialPose_0, DeviceArrayView2D<float4> InterVertexMap_1, DeviceArrayView2D<uchar> markInterVertex_1, DeviceArray2D<float4> vertexMap_1, mat34 initialPose_1, DeviceArrayView2D<float4> InterVertexMap_2, DeviceArrayView2D<uchar> markInterVertex_2, DeviceArray2D<float4> vertexMap_2, mat34 initialPose_2, cudaStream_t stream)
{
	const unsigned int Cols = InterVertexMap_0.Cols();
	const unsigned int Rows = InterVertexMap_0.Rows();

	dim3 block(16, 16);
	dim3 grid(divUp(Cols, block.x), divUp(Rows, block.y));
	device::TransInterpolatedSurfels << <grid, block, 0, stream >> > (
		InterVertexMap_0, markInterVertex_0, vertexMap_0, initialPose_0,
		InterVertexMap_1, markInterVertex_1, vertexMap_1, initialPose_1,
		InterVertexMap_2, markInterVertex_2, vertexMap_2, initialPose_2,
		Cols, Rows
	);
}