
#include "ImageTermKNNFetcher.h"

SparseSurfelFusion::ImageTermKNNFetcher::ImageTermKNNFetcher() {

	m_image_height = FRAME_HEIGHT - 2 * CLIP_BOUNDARY;
	m_image_width = FRAME_WIDTH - 2 * CLIP_BOUNDARY;
	memset(&imageKnnFetcherInterface, 0, sizeof(device::ImageKnnFetcherInterface));
	//The malloc part
	const unsigned int numPixels = devicesCount * m_image_height * m_image_width;

	m_potential_pixel_indicator.create(numPixels);

	//For compaction
	m_indicator_prefixsum.InclusiveSum(numPixels);
	m_potential_pixels.AllocateBuffer(numPixels);
	m_dense_image_knn.AllocateBuffer(numPixels);
	m_dense_image_knn_weight.AllocateBuffer(numPixels);
	differenceViewOffset.AllocateBuffer(devicesCount);

#ifdef CUDA_DEBUG_SYNC_CHECK
	CHECKCUDA(cudaDeviceSynchronize());
#endif 



	//The page-locked memory
	CHECKCUDA(cudaMallocHost((void**)&m_num_potential_pixel, sizeof(unsigned)));
}

SparseSurfelFusion::ImageTermKNNFetcher::~ImageTermKNNFetcher() {
	m_potential_pixel_indicator.release();


	m_potential_pixels.ReleaseBuffer();
	m_dense_image_knn.ReleaseBuffer();
	m_dense_image_knn_weight.ReleaseBuffer();

	CHECKCUDA(cudaFreeHost(m_num_potential_pixel));
}

void SparseSurfelFusion::ImageTermKNNFetcher::SetInputs(
	DeviceArray2D<KNNAndWeight>* knn_map,
	Renderer::SolverMaps* solverMap
) {
	for (int i = 0; i < devicesCount; i++) {
		imageKnnFetcherInterface.KnnMap[i] = knn_map[i];
		imageKnnFetcherInterface.IndexMap[i] = solverMap[i].index_map;
	}
}


//Methods for sanity check
void SparseSurfelFusion::ImageTermKNNFetcher::CheckDenseImageTermKNN(const SparseSurfelFusion::DeviceArrayView<ushort4>& dense_depth_knn_gpu) {
	LOGGING(INFO) << "Check the image term knn against dense depth knn";

	//Should be called after sync
	FUNCTION_CHECK_EQ(m_dense_image_knn.ArraySize(), m_potential_pixels.ArraySize());
	FUNCTION_CHECK_EQ(m_dense_image_knn.ArraySize(), m_dense_image_knn_weight.ArraySize());
	FUNCTION_CHECK_EQ(m_dense_image_knn.ArraySize(), dense_depth_knn_gpu.Size());

	//Download the data
	std::vector<ushort4> potential_pixel_knn_array, dense_depth_knn_array;
	dense_depth_knn_gpu.Download(dense_depth_knn_array);
	m_dense_image_knn.ArrayReadOnly().Download(potential_pixel_knn_array);

	//Iterates
	for (auto i = 0; i < dense_depth_knn_array.size(); i++) {
		const auto& pixel_knn = potential_pixel_knn_array[i];
		const auto& depth_knn = dense_depth_knn_array[i];
		FUNCTION_CHECK(pixel_knn.x == depth_knn.x);
		FUNCTION_CHECK(pixel_knn.y == depth_knn.y);
		FUNCTION_CHECK(pixel_knn.z == depth_knn.z);
		FUNCTION_CHECK(pixel_knn.w == depth_knn.w);
	}

	//Seems correct
	LOGGING(INFO) << "Check done! Seems correct!";
}

void SparseSurfelFusion::ImageTermKNNFetcher::debugPotentialPixelIndicator(vector<unsigned> hostpotentialpixelindicator)
{
	m_potential_pixel_indicator.download(hostpotentialpixelindicator);
	for (int i = 0; i < m_potential_pixel_indicator.size(); i = i + 300) {
		std::cout << "m_potential_pixel_indicator里的部分数据：" << hostpotentialpixelindicator[i] << std::endl;
	}
}

void SparseSurfelFusion::ImageTermKNNFetcher::debugPotentialPixels(vector<ushort3> host)
{
	m_potential_pixels.Array().download(host);
	std::cout << "m_potential_pixels.ArraySize() : " << m_potential_pixels.ArraySize() << std::endl;
	for (int i = 0; i < m_potential_pixels.ArraySize(); i = i + 300) {
		std::cout << "m_potential_pixels里的部分数据(x,y,i)：" << host[i].x <<"  " << host[i].y << "  " << host[i].z << std::endl;
	}
	
}

void SparseSurfelFusion::ImageTermKNNFetcher::debugDenseImageKnn(vector<ushort4> host)
{
	m_dense_image_knn.Array().download(host);
	for (int i = 0; i < m_dense_image_knn.ArraySize(); i = i + 300) {
		std::cout << "m_dense_image_knn里的部分数据(x,y,z,w)：" << host[i].x << "  " << host[i].y << "  " << host[i].z << "  " << host[i].w << std::endl;
	}
}

void SparseSurfelFusion::ImageTermKNNFetcher::debugdifferenceviewoffset(vector<unsigned> host)
{
	differenceViewOffset.Array().download(host);
	for (int i = 0; i < differenceViewOffset.ArraySize(); i++)
	{
		std::cout << "differenceViewOffset的部分数据：" << host[i]<< std::endl;
	}
}

void SparseSurfelFusion::ImageTermKNNFetcher::debugoutputImageTermKNN(vector<unsigned> host1, vector<ushort3> host2, vector<ushort4> host3, vector<unsigned> host4)
{
	m_potential_pixel_indicator.download(host1);
	m_potential_pixels.Array().download(host2);
	m_dense_image_knn.Array().download(host3);
	differenceViewOffset.Array().download(host4);
	//for (int i = 0; i < m_potential_pixel_indicator.size(); i = i + 300)
	//{
	//	std::cout << "m_potential_pixel_indicator里的部分数据：" << host1[i] << std::endl;
	//}
	//for (int i = 0; i < m_potential_pixels.ArraySize(); i = i + 300)
	//{
	//	std::cout << "m_potential_pixels里的部分数据(x,y,i)：" << host2[i].x << "  " << host2[i].y << "  " << host2[i].z << std::endl;
	//	std::cout << "m_dense_image_knn里的部分数据(x,y,z,w)：" << host3[i].x << "  " << host3[i].y << "  " << host3[i].z << "  " << host3[i].w << std::endl;
	//}
	//printf("differenceViewOffset.ArraySize() = %zu \n", differenceViewOffset.ArraySize());
	//for (int i = 0; i < differenceViewOffset.ArraySize(); i++)
	//{
	//	std::cout << "differenceViewOffset的部分数据：" << host4[i] << std::endl;
	//}
}

void SparseSurfelFusion::ImageTermKNNFetcher::getDifferenceViewOffset(vector<unsigned> &diff)
{
	differenceViewOffset.Array().download(diff);
}



