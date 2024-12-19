#include "PatchColliderRGBCorrespondence.h"

template<int FeatureDim, int TreesNum>
__device__ __forceinline__ unsigned int SparseSurfelFusion::device::SearchGPCForest(const GPCPatchFeature<FeatureDim>& feature, const typename PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice& forest)
{
	/**
	 * 哈希算法的基本目标是将任意大小的输入（比如Key）转换为固定大小的数字（通常是一个整数，被称为“哈希值”或者“哈希码）。
	 * 重点是这个转换过程需要确保及时输入的key发生了很小的改变，也要产生一个完全不同的哈希值。（数字）.
	 * 通过将哈希值与叶子节点索引进行乘法运算，可以将叶子节点的索引值与一个较大的常数相乘，从而扩大叶子节点索引对哈希值的贡献。
	 * 这样做的目的是增加叶子节点索引的影响力，使得不同的叶子节点可以对最终的哈希值产生更大的差异.
	 * 这个常数是一个相对较大的素数，可以提供较好的分布性和离散性，从而增加哈希值的随机性.
	 */
	unsigned int hash = 0;
	for (int i = 0; i < TreesNum; i++) {
		const GPCTree<FeatureDim>& tree = forest.trees[i];
		const unsigned int leaf = tree.leafForPatch(feature);
		hash = hash * 67421 + leaf;	// 67421 -> 相对较大的素数
	}
	return hash;

}

__host__ __device__ __forceinline__ unsigned int SparseSurfelFusion::device::EncodePixelPair(int rgbX, int rgbY, bool isPrevious)
{
	unsigned int encode = rgbX + rgbY * 1024;	// 预估Patch中心像素点坐标(x,y)均不超过1024
	if (isPrevious) {
		encode = encode & (~(1 << 31));		// 1左移31位，并取反：01111111 11111111 11111111 11111111，然后 & encode  -->  编码首位为0
	}
	else {
		encode = encode | (1 << 31);		// 1左移31位，然后 | encode  -->  编码首位为1(前提是：rgbX + rgbY * 1024小于2^31-1)
	}
	return encode;
}

__host__ __device__ __forceinline__  void SparseSurfelFusion::device::DecodePixelPair(const unsigned int encode, int& rgbX, int& rgbY, bool& isPrevious)
{
	if ((encode & (1 << 31)) != 0) {	// 注意运算顺序，括号必须加上
		isPrevious = false;
	}
	else isPrevious = true;

	unsigned int RestoreData = encode & (~(1 << 31));	// 将数据的最高位输出为0，还原数据
	rgbX = RestoreData % 1024;
	rgbY = RestoreData / 1024;
}

template<int PatchHalfSize, int TreesNum>
__global__ void SparseSurfelFusion::device::buildColliderKeyValueKernel(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, const typename PatchColliderForest<18, TreesNum>::GPCForestDevice forest, const int stride, const int keyValueRows, const int keyValueCols, unsigned int* keys, unsigned int* values)
{
	const unsigned int KeyValueX = threadIdx.x + blockDim.x * blockIdx.x;	// KeyValueMap的x分量
	const unsigned int KeyValueY = threadIdx.y + blockDim.y * blockIdx.y;	// KeyValueMap的y分量

	if (KeyValueX >= keyValueCols || KeyValueY >= keyValueRows)	return;

	const unsigned int rgbCenterX = PatchHalfSize + KeyValueX * stride;		// 获得rgb图像的Patch的中心点的x
	const unsigned int rgbCenterY = PatchHalfSize + KeyValueY * stride;		// 获得rgb图像的Patch的中心点的y

	GPCPatchFeature<18> PatchFeature_0, PatchFeature_1;										// 声明GPC中Patch提取特征的方法
	buildDCTPatchFeature<PatchHalfSize>(rgb_0, rgbCenterX, rgbCenterY, PatchFeature_0);		// 用DCT提取Patch特征
	buildDCTPatchFeature<PatchHalfSize>(rgb_1, rgbCenterX, rgbCenterY, PatchFeature_1);		// 用DCT提取Patch特征

	const unsigned int key_0 = SearchGPCForest<18, TreesNum>(PatchFeature_0, forest);		// 将训练好的forest加入进去，然后对Patch的特征进行节点分配，并返回一个hash值，作为每个Patch的键(Key)
	const unsigned int key_1 = SearchGPCForest<18, TreesNum>(PatchFeature_1, forest);		// 将训练好的forest加入进去，然后对Patch的特征进行节点分配，并返回一个hash值，作为每个Patch的键(Key)

	const unsigned int value_0 = EncodePixelPair(rgbCenterX, rgbCenterY, true);		// 将Patch中心点的坐标编码，方便后续对Key筛选得到匹配点后，找到对应Patch的位置
	const unsigned int value_1 = EncodePixelPair(rgbCenterX, rgbCenterY, false);	// 将Patch中心点的坐标编码，方便后续对Key筛选得到匹配点后，找到对应Patch的位置

	const unsigned int offset = 2 * (KeyValueX + keyValueCols * KeyValueY);			// 两个数据存一次，
	keys[offset + 0] = key_0;	// 将Previous图片Patch特征转化成键值，存入keys中
	keys[offset + 1] = key_1;	// 将Current图片Patch特征转化成键值，存入keys中

	values[offset + 0] = value_0;	 // 将key对应Patch的中心点存储下来，方便后续定位到匹配点位置
	values[offset + 1] = value_1;	 // 将key对应Patch的中心点存储下来，方便后续定位到匹配点位置
}
__global__ void SparseSurfelFusion::device::markCorrespondenceCandidateKernel(const PtrSize<const unsigned int> SortedTreeLeafKey, const unsigned int* SortedPixelValue, cudaTextureObject_t preForeground, cudaTextureObject_t foreground, cudaTextureObject_t preVertex, cudaTextureObject_t currVertex, cudaTextureObject_t preNormal, cudaTextureObject_t currNormal, unsigned int* CandidateIndicator)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= SortedTreeLeafKey.size) return;

	unsigned int isCandidate = 0;	// 参数用来判断是否是候选的点

	// 是否是第一个Key或者是与前者不同的Key【Key相同代表这两个Patch是匹配的Patch，Key不同则不匹配】，与上一个不同说明当前特征是以一个新特征
	if (idx == 0 || SortedTreeLeafKey[idx] != SortedTreeLeafKey[idx - 1]) {
		const unsigned int hashedKey = SortedTreeLeafKey[idx];
		unsigned int matchedPixelKeyNum = 1;	// 记录匹配的像素点个数

		unsigned int end = idx + 2;	// 往后检查两个Pixel
		if (end >= SortedTreeLeafKey.size)	end = SortedTreeLeafKey.size - 1;	// 如果往后两个以及超出了SortedTreeLeafKey的范围，则end为SortedTreeLeafKey最后一个元素

		for (int j = idx + 1; j <= end; j++) {	// 往后检索两个
			if (SortedTreeLeafKey[j] == hashedKey)	matchedPixelKeyNum++;	// 两个像素点的hash值相同，说明这两个点匹配
		}

		// 检查当前匹配的像素点是否来自不同的Image
		if (matchedPixelKeyNum == 2) {	// 虽然往后检查两个，但是如果出现后两个的hash值也一样，同样排除，要不然无法决策选择哪一个像素点作为匹配像素
			int x_0, y_0;	// 存Previous的坐标，当前Patch中心点的坐标(不做来自于哪个图像的区分)
			int x_1, y_1;	// 存Current的坐标，当前Patch中心点的坐标(不做来自于哪个图像的区分)
			bool checkPixelFrom_0, checkPixelFrom_1;	// 获取当前pixel来自于哪张图片
			const unsigned int encodedPixelCoor_0 = SortedPixelValue[idx + 0];	// 被编码的坐标
			const unsigned int encodedPixelCoor_1 = SortedPixelValue[idx + 1];	// 被编码的坐标
			DecodePixelPair(encodedPixelCoor_0, x_0, y_0, checkPixelFrom_0);	// 解算来自哪一张图片
			DecodePixelPair(encodedPixelCoor_1, x_1, y_1, checkPixelFrom_1);	// 解算来自哪一张图片

			// 来自不同的图片
			if ((checkPixelFrom_0 && !checkPixelFrom_1) || (!checkPixelFrom_0 && checkPixelFrom_1)) {
				if (checkPixelFrom_0 == false) {	// 如果encodedPixelCoor_0是Current，则解算当前的Current Image的坐标
					DecodePixelPair(encodedPixelCoor_0, x_1, y_1, checkPixelFrom_1);	// 解算Current Patch中心点的坐标
					DecodePixelPair(encodedPixelCoor_1, x_0, y_0, checkPixelFrom_0);	// 解算Previous Patch中心点的坐标
				}
				else {	// 如果encodedPixelCoor_0是Previous，则解算另一个Current Image的坐标
					DecodePixelPair(encodedPixelCoor_0, x_0, y_0, checkPixelFrom_0);	// 解算Current Patch中心点的坐标
					DecodePixelPair(encodedPixelCoor_1, x_1, y_1, checkPixelFrom_1);	// 解算Current Patch中心点的坐标
				}

				// 上述判断只是为了获得Current图片上的Patch Center，并判断是否在前景上，在前景则为有效的点
				const unsigned char isPreForeground = tex2D<unsigned char>(preForeground, x_0, y_0);
				const unsigned char isForeground = tex2D<unsigned char>(foreground, x_1, y_1);
				float squaredDis = squared_distance(tex2D<float4>(preVertex, x_0, y_0), tex2D<float4>(currVertex, x_1, y_1));
				float dotNormal = dotxyz(tex2D<float4>(preNormal, x_0, y_0), tex2D<float4>(currNormal, x_1, y_1));

				if (isForeground == (unsigned char)1 && isPreForeground == (unsigned char)1 && squaredDis < 2.5e-3f && dotNormal >= 0.8f) {
					// 同时满足：
					// ① 第一个出现的新特征(Key与上一个不同)；
					// ② 只有两个匹配点；
					// ③ 这两个匹配点来自不同图像；
					// ④ 这个特征Patch对应在Current Image图像的中心坐标在前景上并且匹配点在Previous Image的前景上
					// ⑤ 两个匹配点相差不超过5cm
					// ⑥ 两个匹配点法线夹角不超过37°
					isCandidate = 1;
				}
			}
		}
	}
	// 这里记录并不包含这个是哪一个图片的，只是一旦出现新特征Key，则判断四个条件，满足置1
	CandidateIndicator[idx] = isCandidate;
}

__global__ void SparseSurfelFusion::device::collectCandidatePixelPairKernel(const PtrSize<const unsigned int> CandidateIndicator, const unsigned int* SortedPixelValue, const unsigned int* PrefixSumIndicator, ushort4* PixelPairArray, PtrStepSize<ushort4> CorrMap)
{
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= CandidateIndicator.size)	return;

	if (CandidateIndicator[idx] == 1) {	// 发现了Patch新特征对应的位置
		ushort4 PixelPair;	// 将匹配点信息存入
		int x, y;			// 解算编码：①来自哪一个图像；②Patch中心点的坐标
		bool isPrevious;	// 是否为前一帧图像的标志

		const unsigned int encodedPixel_0 = SortedPixelValue[idx + 0];	  // 获得两个匹配Patch中心点坐标的编码(此时没有判断来自哪一幅图像)
		const unsigned int encodedPixel_1 = SortedPixelValue[idx + 1];	  // 获得两个匹配Patch中心点坐标的编码(此时没有判断来自哪一幅图像)

		//x，y是上一帧中的点，z，w是当前帧
		DecodePixelPair(encodedPixel_0, x, y, isPrevious);	// 解算编码，并将数据填入合适的位置
		if (isPrevious) {
			PixelPair.x = x;
			PixelPair.y = y;
		}
		else {
			PixelPair.z = x;
			PixelPair.w = y;
		}
		DecodePixelPair(encodedPixel_1, x, y, isPrevious);	// 第二次解算编码，isPrevious肯定与上一次是不同的，Mark的时候决定
		if (isPrevious) {
			PixelPair.x = x;
			PixelPair.y = y;
		}
		else {
			PixelPair.z = x;
			PixelPair.w = y;
		}
		// 上述两次解码，以及将不同图像的匹配Patch对的中心点坐标填入了(x, y, z, w)对应的位置
		CorrMap.ptr(PixelPair.w)[PixelPair.z] = PixelPair;		// 边缘存入当前帧像素对应的位置
		const unsigned int offset = PrefixSumIndicator[idx] - 1;	// 此时是第几个特征Key
		PixelPairArray[offset] = PixelPair;	// 将匹配点存入
	}
}

void SparseSurfelFusion::PatchColliderRGBCorrespondence::FindCorrespondence(cudaStream_t stream)
{
	CHECKCUDA(cudaMemsetAsync(sparseCorrPairsMap.ptr(), 0xFFFF, sizeof(ushort4) * clipedImageSize, stream));
	dim3 KeyValueBlock(8, 8);
	dim3 KeyValueGrid(divUp(KeyValueMapCols, KeyValueBlock.x), divUp(KeyValueMapRows, KeyValueBlock.y));
	const PatchColliderForest<FeatureDim, TreesNum>::GPCForestDevice DeviceForest = Forest.OnDevice();	// 在GPU上存储的森林数据
	// 提取rgbPrevious和rgbCurrent的Patch，并将其送入forest获得表示Patch特征的hash数存入Key，并将对应Patch归属哪张图片和Patch中心像素点坐标存入Value
	device::buildColliderKeyValueKernel<PatchRadius, TreesNum> << <KeyValueGrid, KeyValueBlock, 0, stream >> > (
		rgbPrevious, rgbCurrent, 
		DeviceForest, 
		PatchStride, 
		KeyValueMapRows, 
		KeyValueMapCols, 
		TreeLeafKey, 
		PixelValue
	);

	CollideSort.Sort(TreeLeafKey, PixelValue, stream);	// 排序从而将相同Key(相同DCT特征)的Patch放在了一起

	CHECKCUDA(cudaStreamSynchronize(stream));

	// 标记Candidate
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
	// valid_prefixsum_array中最后一个元素存的是一共有多少个有效的(不重复)的元素
	const unsigned int* TotalPairs = Prefixsum.valid_prefixsum_array + Prefixsum.valid_prefixsum_array.size() - 1;
	// 只拷贝最后一个数，为了开辟CorrespondencePixels的Array Size
	CHECKCUDA(cudaMemcpyAsync(CandidatePairsNumPagelock, TotalPairs, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	// 将匹配的点存入CorrespondencePixels中的Buffer中
	device::collectCandidatePixelPairKernel << <IndicatorGrid, IndicatorBlock, 0, stream >> > (CandidatePixelPairsIndicator, CollideSort.valid_sorted_value.ptr(), Prefixsum.valid_prefixsum_array.ptr(), CorrespondencePixels.Ptr(), sparseCorrPairsMap);
	
	CHECKCUDA(cudaStreamSynchronize(stream));	// 流同步，因为要调用的CorrespondencePixels在上面的核函数中正在使用

	CorrespondencePixels.ResizeArrayOrException(*CandidatePairsNumPagelock);	// 将Buffer的指针赋给Array，并调整Array大小

}