/*****************************************************************//**
 * \file   OctNode.cuh
 * \brief  �㷨�����ݽṹ
 * 
 * \author LUOJIAXUAN
 * \date   May 2nd 2024
 *********************************************************************/
#pragma once
#include "Geometry.h"

namespace SparseSurfelFusion{
    /**
     * \brief ��¼�˲����ڵ���������ͣ���С 276 Bytes.
     */
    class OctNode {
    public:
        int key;            // �ڵ�ļ�key
        int pidx;           // �ڵ���SortedArray�еĵ�һ��Ԫ�ص�index
        int pnum;           // �뵱ǰ�ڵ�key��ͬ�ĳ��ܵ������
        int parent;         // 1�����ڵ�
        int children[8];    // 8�����ӽڵ�
        int neighs[27];     // 27���ھӽڵ�
        // record the start in maxDepth NodeArray the first node at maxDepth is index 0
        int didx;           // ��ǰ�ڵ����������ΪD���ӽڵ��һ���㣬idx��С�ĵ�
        int dnum;           // ��ǰ�ڵ����������ΪD���ӽڵ��������������Ч�����Ч��

        int vertices[8];    // ����¼��idxƫ��һλ����¼�ڵ�Ķ�����Ϣ�������Ϣ����VertexArray�����У�����ֻ�Ǽ�¼�Žڵ���VertexArray�е�index

        // (real idx) + 1,
        // idx start from (0 + 1)
        int edges[12];      // ����cube�ıߣ�ʵ��idx��洢����1��ƫ��

        // (real idx) + 1
        // idx start from (0 + 1)
        int faces[6];       // ����cube���棺ʵ��idx��洢����1��ƫ��

        int hasTriangle;    // ��¼�������cube�Ƿ���������
        int hasIntersection;// ��¼�������cube�Ƿ�������ཻ(Surface-Edge Intersection)
    };

    /**
     * \brief OctNode�ļ򻯰棬ȥ����face�����Ǳ���ཻ���.
     */
    class EasyOctNode {
    public:
        int key;
        int parent;
        int children[8];
        int neighs[27];

        // (real idx) + 1,
        // idx start from (0 + 1)
        // encode the vertices idx?
        int vertices[8];

        // (real idx) + 1,
        // idx start from (0 + 1)
        int edges[12];

        __device__ EasyOctNode& operator = (const OctNode& n) {
            key = n.key;
            parent = n.parent;
#pragma unroll  // ��ѭ������չ�������Ż�������ͨ����ѭ�����еĴ��븴�ƶ��������ѭ���������Ӷ���߳����ִ���ٶ�
            for (int i = 0; i < 8; ++i) {
                children[i] = n.children[i];
                vertices[i] = n.vertices[i];
            }
#pragma unroll
            for (int i = 0; i < 27; ++i) {
                neighs[i] = n.neighs[i];
            }
#pragma unroll
            for (int i = 0; i < 12; ++i) {
                edges[i] = n.edges[i];
            }
        }
    };

    /**
     * \brief ����vertex�������ͣ���С 56 Bytes.
     */
    class VertexNode {
    public:
        Point3D<float> pos = Point3D<float>(0.0f, 0.0f, 0.0f);  // ��ǰVertex��λ��
        int ownerNodeIdx = 0;                                   // ���Vertex����NodeArray����һ���ڵ�(index)��ͬһ���ڵ����ӵ�ж��vertex�� ��һ��vertexֻ����һ���ڵ�index
        int vertexKind = 0;                                     // �����Ӧ��λ��index
        int depth = 0;                                          // ���Vertex��Ӧ�Ľڵ����
        int nodes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };              // ��¼��ǰ�����Owner�ھ��У��������vertex�Ľڵ�index(һ��Vertex���Ա�8���ڵ㹲��)
    };

    /**
     * \brief ��Edge�������ͣ���С 24 Bytes.
     */
    class EdgeNode {
    public:
        int edgeKind = 0;                                       // �߶�Ӧ��λ��˳��
        int ownerNodeIdx = 0;                                   // ��¼Edge����NodeArray�е���һ���ڵ�(index)��ͬһ���ڵ��ӵ�ж��Edge������һ��Edgeֻ��ӵ��һ���ڵ�index
        int nodes[4] = { 0, 0, 0, 0 };                          // ��¼��ǰ�ߵ�Owner�ھ��У��������Edge�Ľڵ�index(һ��Edge���Ա�4���ڵ㹲��)
    };

    /**
     * \brief ��Face�������ͣ���С 20 Bytes.
     */
    class FaceNode {
    public:
        int faceKind = -1;                                      // ���Ӧ��λ��˳��
        int ownerNodeIdx = -1;                                  // ��¼������NodeArray�е���һ���ڵ�
        int hasParentFace = -1;                                 // ��¼��ǰ���Ƿ�����ڵ㹲һ����
        int nodes[2] = { -1, -1 };                              // ��¼��ǰ���Owner�ھ��У����������������ڵ�index
    };
}

