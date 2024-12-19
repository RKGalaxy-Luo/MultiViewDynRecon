/*****************************************************************//**
 * \file   Constants.cpp
 * \brief  维护主机存取常量，算法中一些常量的结构体，基本常数(算法的超参数).
 * 
 * \author LUO
 * \date   January 29th 2024
 *********************************************************************/
#include "Constants.h"

// 双边滤波：空间距离权重(4.5f)
const float SparseSurfelFusion::Constants::kFilterSigmaS = 4.5f;					
// 双边滤波：像素值权重(30.0f)
const float SparseSurfelFusion::Constants::kFilterSigmaR = 30.0f;						
// 前景滤波的sigma值(5.0f)
const float SparseSurfelFusion::Constants::kForegroundSigma = 5.0f;

// 渲染图放大倍数(因为相邻两个像素可能投影到渲染图中同一个像素点，因此需要超采样)
const int SparseSurfelFusion::Constants::superSampleScale = SUPER_SAMPLE_SCALE;			
// 最大面元数量(500000)
const unsigned int SparseSurfelFusion::Constants::maxSurfelsNum = MAX_SURFEL_COUNT;		
const unsigned int SparseSurfelFusion::Constants::maxMeshTrianglesNum = MAX_MESH_TRIANGLE_COUNT;

// 对原始稠密点压缩尺度(压缩后点的最大数量不超过原始点的1/maxCompactedInputPointsScale)  -->  【平均每一个体素至少有5个稠密点对应它】
const float SparseSurfelFusion::Constants::maxCompactedInputPointsScale = 0.2f;	
// 每个相机下采样最大的下采样点的数量(maxSurfelsNum / maxCompactedInputPointsScale)
const unsigned int SparseSurfelFusion::Constants::maxSubsampledSparsePointsNum = maxSurfelsNum * maxCompactedInputPointsScale;

// 从kMaxSubsampleFrom候选节点中最多选择1个节点
const unsigned int SparseSurfelFusion::Constants::maxSubsampleFrom = 5;
// 最大的节点数
const unsigned int SparseSurfelFusion::Constants::maxNodesNum = MAX_NODE_COUNT;
// 节点图中，一个节点有8个邻居节点
const unsigned int SparseSurfelFusion::Constants::nodesGraphNeigboursNum = 8;

// 节点间的平均距离0.025m
const float SparseSurfelFusion::Constants::NodeRadius = NODE_RADIUS;
// 体素的长度，0.7 * 节点距离  -->  确保每一个节点都能对应一个体素，不存在一个节点对应两个体素
const float SparseSurfelFusion::Constants::VoxelSize = 0.7f * Constants::NodeRadius;
// 节点采样半径 0.85 × 0.01  -->  确保不会有特别近的节点，保证下采样节点稀疏且均匀
const float SparseSurfelFusion::Constants::NodeSamplingRadius = 0.85f * Constants::NodeRadius;  
// 融合不同相机曲面的TSDF截断阈值 0.1m
const float SparseSurfelFusion::Constants::TSDFThreshold = TSDF_THRESHOLD;
// 最大融合面元的数量
const unsigned int SparseSurfelFusion::Constants::MaxOverlappingSurfel = 1;
// 最大截断距离，如果融合两点超过这个距离，则不对这两个点融合
const float SparseSurfelFusion::Constants::MaxTruncatedSquaredDistance = 0.05f * 0.05f;

//稳定面元的置信阈值
const int SparseSurfelFusion::Constants::kStableSurfelConfidenceThreshold = 10;
//The recent time threshold for rendering solver maps
const int SparseSurfelFusion::Constants::kRenderingRecentTimeThreshold = 3;

// 最大前后两帧匹配点的数量
const unsigned SparseSurfelFusion::Constants::kMaxMatchedSparseFeature = 50000;	

// 最大候选Surfels的数量
const unsigned SparseSurfelFusion::Constants::kMaxNumSurfelCandidates = maxSurfelsNum / 3;

// 是否使用弹性惩罚
const bool SparseSurfelFusion::Constants::kUseElasticPenalty = true;
// 全局求解器迭代次数
const int SparseSurfelFusion::Constants::kNumGlobalSolverItarations = 3;
// 高斯牛顿迭代次数
const int SparseSurfelFusion::Constants::kNumGaussNewtonIterations = 6;

const unsigned SparseSurfelFusion::Constants::kMaxNumNodePairs = 60 * Constants::maxNodesNum;

const int SparseSurfelFusion::Constants::LUTparent_Host[8][27] = 
{
	{0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13},
	{1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14},
	{3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16},
	{4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17},
	{9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22},
	{10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23},
	{12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25},
	{13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26}
};
const int SparseSurfelFusion::Constants::LUTchild_Host[8][27] = 
{
	{7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7},
	{6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6},
	{5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5},
	{4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4},
	{3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3},
	{2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2},
	{1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1},
	{0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0}
};

const float SparseSurfelFusion::Constants::CrossViewPairsSquaredDisThreshold = 1e-2f;		// 7cm

const unsigned int SparseSurfelFusion::Constants::MaxCrossViewMatchedPairs = 0.5f * kMaxMatchedSparseFeature;



