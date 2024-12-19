#include "PatchColliderRGBCorrespondence.h"

template<int FeatureDim, int TreesNum>
__device__ __forceinline__ unsigned int SparseSurfelFusion::device::SearchGPCForest(const GPCPatchFeature<FeatureDim>& feature, const typename PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice& forest)
{
	/**
	 * ��ϣ�㷨�Ļ���Ŀ���ǽ������С�����루����Key��ת��Ϊ�̶���С�����֣�ͨ����һ������������Ϊ����ϣֵ�����ߡ���ϣ�룩��
	 * �ص������ת��������Ҫȷ����ʱ�����key�����˺�С�ĸı䣬ҲҪ����һ����ȫ��ͬ�Ĺ�ϣֵ�������֣�.
	 * ͨ������ϣֵ��Ҷ�ӽڵ��������г˷����㣬���Խ�Ҷ�ӽڵ������ֵ��һ���ϴ�ĳ�����ˣ��Ӷ�����Ҷ�ӽڵ������Թ�ϣֵ�Ĺ��ס�
	 * ��������Ŀ��������Ҷ�ӽڵ�������Ӱ������ʹ�ò�ͬ��Ҷ�ӽڵ���Զ����յĹ�ϣֵ��������Ĳ���.
	 * ���������һ����Խϴ�������������ṩ�Ϻõķֲ��Ժ���ɢ�ԣ��Ӷ����ӹ�ϣֵ�������.
	 */
	unsigned int hash = 0;
	for (int i = 0; i < TreesNum; i++) {
		const GPCTree<FeatureDim>& tree = forest.trees[i];
		const unsigned int leaf = tree.leafForPatch(feature);
		hash = hash * 67421 + leaf;	// 67421 -> ��Խϴ������
	}
	return hash;

}

__host__ __device__ __forceinline__ unsigned int SparseSurfelFusion::device::EncodePixelPair(int rgbX, int rgbY, bool isPrevious)
{
	unsigned int encode = rgbX + rgbY * 1024;	// Ԥ��Patch�������ص�����(x,y)��������1024
	if (isPrevious) {
		encode = encode & (~(1 << 31));		// 1����31λ����ȡ����01111111 11111111 11111111 11111111��Ȼ�� & encode  -->  ������λΪ0
	}
	else {
		encode = encode | (1 << 31);		// 1����31λ��Ȼ�� | encode  -->  ������λΪ1(ǰ���ǣ�rgbX + rgbY * 1024С��2^31-1)
	}
	return encode;
}

__host__ __device__ __forceinline__  void SparseSurfelFusion::device::DecodePixelPair(const unsigned int encode, int& rgbX, int& rgbY, bool& isPrevious)
{
	if ((encode & (1 << 31)) != 0) {	// ע������˳�����ű������
		isPrevious = false;
	}
	else isPrevious = true;

	unsigned int RestoreData = encode & (~(1 << 31));	// �����ݵ����λ���Ϊ0����ԭ����
	rgbX = RestoreData % 1024;
	rgbY = RestoreData / 1024;
}

template<int PatchHalfSize, int TreesNum>
__global__ void SparseSurfelFusion::device::buildColliderKeyValueKernel(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, const typename PatchColliderForest<18, TreesNum>::GPCForestDevice forest, const int stride, const int keyValueRows, const int keyValueCols, unsigned int* keys, unsigned int* values)
{
	const unsigned int KeyValueX = threadIdx.x + blockDim.x * blockIdx.x;	// KeyValueMap��x����
	const unsigned int KeyValueY = threadIdx.y + blockDim.y * blockIdx.y;	// KeyValueMap��y����

	if (KeyValueX >= keyValueCols || KeyValueY >= keyValueRows)	return;

	const unsigned int rgbCenterX = PatchHalfSize + KeyValueX * stride;		// ���rgbͼ���Patch�����ĵ��x
	const unsigned int rgbCenterY = PatchHalfSize + KeyValueY * stride;		// ���rgbͼ���Patch�����ĵ��y

	GPCPatchFeature<18> PatchFeature_0, PatchFeature_1;										// ����GPC��Patch��ȡ�����ķ���
	buildDCTPatchFeature<PatchHalfSize>(rgb_0, rgbCenterX, rgbCenterY, PatchFeature_0);		// ��DCT��ȡPatch����
	buildDCTPatchFeature<PatchHalfSize>(rgb_1, rgbCenterX, rgbCenterY, PatchFeature_1);		// ��DCT��ȡPatch����

	const unsigned int key_0 = SearchGPCForest<18, TreesNum>(PatchFeature_0, forest);		// ��ѵ���õ�forest�����ȥ��Ȼ���Patch���������нڵ���䣬������һ��hashֵ����Ϊÿ��Patch�ļ�(Key)
	const unsigned int key_1 = SearchGPCForest<18, TreesNum>(PatchFeature_1, forest);		// ��ѵ���õ�forest�����ȥ��Ȼ���Patch���������нڵ���䣬������һ��hashֵ����Ϊÿ��Patch�ļ�(Key)

	const unsigned int value_0 = EncodePixelPair(rgbCenterX, rgbCenterY, true);		// ��Patch���ĵ��������룬���������Keyɸѡ�õ�ƥ�����ҵ���ӦPatch��λ��
	const unsigned int value_1 = EncodePixelPair(rgbCenterX, rgbCenterY, false);	// ��Patch���ĵ��������룬���������Keyɸѡ�õ�ƥ�����ҵ���ӦPatch��λ��

	const unsigned int offset = 2 * (KeyValueX + keyValueCols * KeyValueY);			// �������ݴ�һ�Σ�
	keys[offset + 0] = key_0;	// ��PreviousͼƬPatch����ת���ɼ�ֵ������keys��
	keys[offset + 1] = key_1;	// ��CurrentͼƬPatch����ת���ɼ�ֵ������keys��

	values[offset + 0] = value_0;	 // ��key��ӦPatch�����ĵ�洢���������������λ��ƥ���λ��
	values[offset + 1] = value_1;	 // ��key��ӦPatch�����ĵ�洢���������������λ��ƥ���λ��
}
__global__ void SparseSurfelFusion::device::markCorrespondenceCandidateKernel(const PtrSize<const unsigned int> SortedTreeLeafKey, const unsigned int* SortedPixelValue, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, unsigned int* CandidateIndicator)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= SortedTreeLeafKey.size) return;

	unsigned int isCandidate = 0;	// ���������ж��Ƿ��Ǻ�ѡ�ĵ�

	// �Ƿ��ǵ�һ��Key��������ǰ�߲�ͬ��Key��Key��ͬ����������Patch��ƥ���Patch��Key��ͬ��ƥ�䡿������һ����ͬ˵����ǰ��������һ��������
	if (idx == 0 || SortedTreeLeafKey[idx] != SortedTreeLeafKey[idx - 1]) {
		const unsigned int hashedKey = SortedTreeLeafKey[idx];
		unsigned int matchedPixelKeyNum = 1;	// ��¼ƥ������ص����

		unsigned int end = idx + 2;	// ����������Pixel
		if (end >= SortedTreeLeafKey.size)	end = SortedTreeLeafKey.size - 1;	// ������������Լ�������SortedTreeLeafKey�ķ�Χ����endΪSortedTreeLeafKey���һ��Ԫ��

		for (int j = idx + 1; j <= end; j++) {	// �����������
			if (SortedTreeLeafKey[j] == hashedKey)	matchedPixelKeyNum++;	// �������ص��hashֵ��ͬ��˵����������ƥ��
		}

		// ��鵱ǰƥ������ص��Ƿ����Բ�ͬ��Image
		if (matchedPixelKeyNum == 2) {	// ��Ȼ����������������������ֺ�������hashֵҲһ����ͬ���ų���Ҫ��Ȼ�޷�����ѡ����һ�����ص���Ϊƥ������
			int x_0, y_0;	// ��Previous�����꣬��ǰPatch���ĵ������(�����������ĸ�ͼ�������)
			int x_1, y_1;	// ��Current�����꣬��ǰPatch���ĵ������(�����������ĸ�ͼ�������)
			bool checkPixelFrom_0, checkPixelFrom_1;	// ��ȡ��ǰpixel����������ͼƬ
			const unsigned int encodedPixelCoor_0 = SortedPixelValue[idx + 0];	// �����������
			const unsigned int encodedPixelCoor_1 = SortedPixelValue[idx + 1];	// �����������
			DecodePixelPair(encodedPixelCoor_0, x_0, y_0, checkPixelFrom_0);	// ����������һ��ͼƬ
			DecodePixelPair(encodedPixelCoor_1, x_1, y_1, checkPixelFrom_1);	// ����������һ��ͼƬ

			// ���Բ�ͬ��ͼƬ
			if ((checkPixelFrom_0 && !checkPixelFrom_1) || (!checkPixelFrom_0 && checkPixelFrom_1)) {
				if (checkPixelFrom_0 == false) {	// ���encodedPixelCoor_0��Current������㵱ǰ��Current Image������
					DecodePixelPair(encodedPixelCoor_0, x_1, y_1, checkPixelFrom_1);	// ����Current Patch���ĵ������
					DecodePixelPair(encodedPixelCoor_1, x_0, y_0, checkPixelFrom_0);	// ����Previous Patch���ĵ������
				}
				else {	// ���encodedPixelCoor_0��Previous���������һ��Current Image������
					DecodePixelPair(encodedPixelCoor_0, x_0, y_0, checkPixelFrom_0);	// ����Current Patch���ĵ������
					DecodePixelPair(encodedPixelCoor_1, x_1, y_1, checkPixelFrom_1);	// ����Current Patch���ĵ������
				}

				// �����ж�ֻ��Ϊ�˻��CurrentͼƬ�ϵ�Patch Center�����ж��Ƿ���ǰ���ϣ���ǰ����Ϊ��Ч�ĵ�
				const unsigned char isPreForeground = tex2D<unsigned char>(preForeground, x_0, y_0);
				const unsigned char isForeground = tex2D<unsigned char>(foreground, x_1, y_1);
				float squaredDis = squared_distance(tex2D<float4>(preVertex, x_0, y_0), tex2D<float4>(currVertex, x_1, y_1));
				float dotNormal = dotxyz(tex2D<float4>(preNormal, x_0, y_0), tex2D<float4>(currNormal, x_1, y_1));

				if (isForeground == (unsigned char)1 && isPreForeground == (unsigned char)1 && squaredDis < 2.5e-3f && dotNormal >= 0.8f) {
					// ͬʱ���㣺
					// �� ��һ�����ֵ�������(Key����һ����ͬ)��
					// �� ֻ������ƥ��㣻
					// �� ������ƥ������Բ�ͬͼ��
					// �� �������Patch��Ӧ��Current Imageͼ�������������ǰ���ϲ���ƥ�����Previous Image��ǰ����
					// �� ����ƥ���������5cm
					// �� ����ƥ��㷨�߼нǲ�����37��
					isCandidate = 1;
				}
			}
		}
	}
	// �����¼���������������һ��ͼƬ�ģ�ֻ��һ������������Key�����ж��ĸ�������������1
	CandidateIndicator[idx] = isCandidate;
}

__global__ void SparseSurfelFusion::device::collectCandidatePixelPairKernel(const PtrSize<const unsigned int> CandidateIndicator, const unsigned int* SortedPixelValue, const unsigned int* PrefixSumIndicator, ushort4* PixelPairArray, PtrStepSize<ushort4> CorrMap)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= CandidateIndicator.size)	return;

	if (CandidateIndicator[idx] == 1) {	// ������Patch��������Ӧ��λ��
		ushort4 PixelPair;	// ��ƥ�����Ϣ����
		int x, y;			// ������룺��������һ��ͼ�񣻢�Patch���ĵ������
		bool isPrevious;	// �Ƿ�Ϊǰһ֡ͼ��ı�־

		const unsigned int encodedPixel_0 = SortedPixelValue[idx + 0];	  // �������ƥ��Patch���ĵ�����ı���(��ʱû���ж�������һ��ͼ��)
		const unsigned int encodedPixel_1 = SortedPixelValue[idx + 1];	  // �������ƥ��Patch���ĵ�����ı���(��ʱû���ж�������һ��ͼ��)

		//x��y����һ֡�еĵ㣬z��w�ǵ�ǰ֡
		DecodePixelPair(encodedPixel_0, x, y, isPrevious);	// ������룬��������������ʵ�λ��
		if (isPrevious) {
			PixelPair.x = x;
			PixelPair.y = y;
		}
		else {
			PixelPair.z = x;
			PixelPair.w = y;
		}
		DecodePixelPair(encodedPixel_1, x, y, isPrevious);	// �ڶ��ν�����룬isPrevious�϶�����һ���ǲ�ͬ�ģ�Mark��ʱ�����
		if (isPrevious) {
			PixelPair.x = x;
			PixelPair.y = y;
		}
		else {
			PixelPair.z = x;
			PixelPair.w = y;
		}
		// �������ν��룬�Լ�����ͬͼ���ƥ��Patch�Ե����ĵ�����������(x, y, z, w)��Ӧ��λ��
		CorrMap.ptr(PixelPair.w)[PixelPair.z] = PixelPair;		// ��Ե���뵱ǰ֡���ض�Ӧ��λ��
		const unsigned int offset = PrefixSumIndicator[idx] - 1;	// ��ʱ�ǵڼ�������Key
		PixelPairArray[offset] = PixelPair;	// ��ƥ������
	}
}

void SparseSurfelFusion::PatchColliderRGBCorrespondence::FindCorrespondence(cudaStream_t stream)
{
	CHECKCUDA(cudaMemsetAsync(sparseCorrPairsMap.ptr(), 0xFFFF, sizeof(ushort4) * clipedImageSize, stream));
	dim3 KeyValueBlock(8, 8);
	dim3 KeyValueGrid(divUp(KeyValueMapCols, KeyValueBlock.x), divUp(KeyValueMapRows, KeyValueBlock.y));
	const PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice DeviceForest = Forest.OnDevice();	// ��GPU�ϴ洢��ɭ������
	// ��ȡrgbPrevious��rgbCurrent��Patch������������forest��ñ�ʾPatch������hash������Key��������ӦPatch��������ͼƬ��Patch�������ص��������Value
	device::buildColliderKeyValueKernel<PatchRadius, TreesNum> << <KeyValueGrid, KeyValueBlock, 0, stream >> > (
		rgbPrevious, rgbCurrent, 
		DeviceForest, 
		PatchStride, 
		KeyValueMapRows, 
		KeyValueMapCols, 
		TreeLeafKey, 
		PixelValue
	);

	CollideSort.Sort(TreeLeafKey, PixelValue, stream);	// ����Ӷ�����ͬKey(��ͬDCT����)��Patch������һ��

	CHECKCUDA(cudaStreamSynchronize(stream));

	// ���Candidate
	dim3 IndicatorBlock(64);
	dim3 IndicatorGrid(divUp(CollideSort.valid_sorted_key.size(), IndicatorBlock.x));
	device::markCorrespondenceCandidateKernel << <IndicatorGrid, IndicatorBlock, 0, stream >> >(
		CollideSort.valid_sorted_key, CollideSort.valid_sorted_value.ptr(), 
		foregroundPrevious, foregroundCurrent, 
		previousVertexMap, currentVertexMap, 
		previousNormalMap, currentNormalMap, 
		CandidatePixelPairsIndicator.ptr()
	);

	Prefixsum.InclusiveSum(CandidatePixelPairsIndicator, stream);
	// valid_prefixsum_array�����һ��Ԫ�ش����һ���ж��ٸ���Ч��(���ظ�)��Ԫ��
	const unsigned int* TotalPairs = Prefixsum.valid_prefixsum_array + Prefixsum.valid_prefixsum_array.size() - 1;
	// ֻ�������һ������Ϊ�˿���CorrespondencePixels��Array Size
	CHECKCUDA(cudaMemcpyAsync(CandidatePairsNumPagelock, TotalPairs, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	// ��ƥ��ĵ����CorrespondencePixels�е�Buffer��
	device::collectCandidatePixelPairKernel << <IndicatorGrid, IndicatorBlock, 0, stream >> > (CandidatePixelPairsIndicator, CollideSort.valid_sorted_value.ptr(), Prefixsum.valid_prefixsum_array.ptr(), CorrespondencePixels.Ptr(), sparseCorrPairsMap);
	
	CHECKCUDA(cudaStreamSynchronize(stream));	// ��ͬ������ΪҪ���õ�CorrespondencePixels������ĺ˺���������ʹ��

	CorrespondencePixels.ResizeArrayOrException(*CandidatePairsNumPagelock);	// ��Buffer��ָ�븳��Array��������Array��С

}