#pragma once
// ���񻯶���
#define CHECK_MESH_BUILD_TIME_COST	// �鿴�����ؽ�������ÿ������ʱ������

#define CUB_IGNORE_DEPRECATED_API

#define FORCE_UNIT_NORMALS 1

#define MAX_MESH_TRIANGLE_COUNT 1000000	// �����������������

#define STACKCAPACITY 2000

#define CONVTIMES 2

#define NORMALIZE 0

#define DIMENSION 3

#define MAX_THREADS 10

#define MAX_DEPTH_OCTREE 7	// octree������

#define MAX_MESH_STREAM 5	// ���ִ��mesh�����cuda������

#define F_DATA_RES ((1 << (MAX_DEPTH_OCTREE + 1)) - 1)					// 2^(maxDepth + 1) - 1
#define F_DATA_RES_SQUARE F_DATA_RES * F_DATA_RES						// 2047^2

#define D_LEVEL_MAX_NODE 8 * MAX_SURFEL_COUNT							// maxDepth��ڵ������Ӧ����8 * MAX_SURFEL_COUNT
#define TOTAL_NODEARRAY_MAX_COUNT MAX_SURFEL_COUNT * 10					// NodeArray��������
#define TOTAL_VERTEXARRAY_MAX_COUNT 8* TOTAL_NODEARRAY_MAX_COUNT		// NodeArray�������� * 8(8������)
#define TOTAL_EDGEARRAY_MAX_COUNT 12 * D_LEVEL_MAX_NODE					// NodeArray��maxDepth���нڵ����� * 12
#define TOTAL_FACEARRAY_MAX_COUNT 6 * TOTAL_NODEARRAY_MAX_COUNT			// NodeArray�������� * 6(6����)

#define RESOLUTION (1 << (MAX_DEPTH_OCTREE + 1)) - 1	// �ֱ���

#define COARSER_DIVERGENCE_LEVEL_NUM 4												// ������Խڵ�����ڵ����[0, LevelNum]
#define TOTAL_FINER_NODE_NUM 6 * MAX_SURFEL_COUNT									// [maxDepth - LevelNum, maxDepth]��Ľڵ�����
#define TOTAL_COARSER_NODE_NUM TOTAL_NODEARRAY_MAX_COUNT - TOTAL_FINER_NODE_NUM		// [1, maxDepth - LevelNum - 1]��Ľڵ�����

#define EPSILON float(1e-6)