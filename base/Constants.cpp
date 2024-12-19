/*****************************************************************//**
 * \file   Constants.cpp
 * \brief  ά��������ȡ�������㷨��һЩ�����Ľṹ�壬��������(�㷨�ĳ�����).
 * 
 * \author LUO
 * \date   January 29th 2024
 *********************************************************************/
#include "Constants.h"

// ˫���˲����ռ����Ȩ��(4.5f)
const float SparseSurfelFusion::Constants::kFilterSigmaS = 4.5f;					
// ˫���˲�������ֵȨ��(30.0f)
const float SparseSurfelFusion::Constants::kFilterSigmaR = 30.0f;						
// ǰ���˲���sigmaֵ(5.0f)
const float SparseSurfelFusion::Constants::kForegroundSigma = 5.0f;

// ��Ⱦͼ�Ŵ���(��Ϊ�����������ؿ���ͶӰ����Ⱦͼ��ͬһ�����ص㣬�����Ҫ������)
const int SparseSurfelFusion::Constants::superSampleScale = SUPER_SAMPLE_SCALE;			
// �����Ԫ����(500000)
const unsigned int SparseSurfelFusion::Constants::maxSurfelsNum = MAX_SURFEL_COUNT;		
const unsigned int SparseSurfelFusion::Constants::maxMeshTrianglesNum = MAX_MESH_TRIANGLE_COUNT;

// ��ԭʼ���ܵ�ѹ���߶�(ѹ�������������������ԭʼ���1/maxCompactedInputPointsScale)  -->  ��ƽ��ÿһ������������5�����ܵ��Ӧ����
const float SparseSurfelFusion::Constants::maxCompactedInputPointsScale = 0.2f;	
// ÿ������²��������²����������(maxSurfelsNum / maxCompactedInputPointsScale)
const unsigned int SparseSurfelFusion::Constants::maxSubsampledSparsePointsNum = maxSurfelsNum * maxCompactedInputPointsScale;

// ��kMaxSubsampleFrom��ѡ�ڵ������ѡ��1���ڵ�
const unsigned int SparseSurfelFusion::Constants::maxSubsampleFrom = 5;
// ���Ľڵ���
const unsigned int SparseSurfelFusion::Constants::maxNodesNum = MAX_NODE_COUNT;
// �ڵ�ͼ�У�һ���ڵ���8���ھӽڵ�
const unsigned int SparseSurfelFusion::Constants::nodesGraphNeigboursNum = 8;

// �ڵ���ƽ������0.025m
const float SparseSurfelFusion::Constants::NodeRadius = NODE_RADIUS;
// ���صĳ��ȣ�0.7 * �ڵ����  -->  ȷ��ÿһ���ڵ㶼�ܶ�Ӧһ�����أ�������һ���ڵ��Ӧ��������
const float SparseSurfelFusion::Constants::VoxelSize = 0.7f * Constants::NodeRadius;
// �ڵ�����뾶 0.85 �� 0.01  -->  ȷ���������ر���Ľڵ㣬��֤�²����ڵ�ϡ���Ҿ���
const float SparseSurfelFusion::Constants::NodeSamplingRadius = 0.85f * Constants::NodeRadius;  
// �ںϲ�ͬ��������TSDF�ض���ֵ 0.1m
const float SparseSurfelFusion::Constants::TSDFThreshold = TSDF_THRESHOLD;
// ����ں���Ԫ������
const unsigned int SparseSurfelFusion::Constants::MaxOverlappingSurfel = 1;
// ���ضϾ��룬����ں����㳬��������룬�򲻶����������ں�
const float SparseSurfelFusion::Constants::MaxTruncatedSquaredDistance = 0.05f * 0.05f;

//�ȶ���Ԫ��������ֵ
const int SparseSurfelFusion::Constants::kStableSurfelConfidenceThreshold = 10;
//The recent time threshold for rendering solver maps
const int SparseSurfelFusion::Constants::kRenderingRecentTimeThreshold = 3;

// ���ǰ����֡ƥ��������
const unsigned SparseSurfelFusion::Constants::kMaxMatchedSparseFeature = 50000;	

// ����ѡSurfels������
const unsigned SparseSurfelFusion::Constants::kMaxNumSurfelCandidates = maxSurfelsNum / 3;

// �Ƿ�ʹ�õ��Գͷ�
const bool SparseSurfelFusion::Constants::kUseElasticPenalty = true;
// ȫ���������������
const int SparseSurfelFusion::Constants::kNumGlobalSolverItarations = 3;
// ��˹ţ�ٵ�������
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



