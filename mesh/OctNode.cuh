/*****************************************************************//**
 * \file   OctNode.cuh
 * \brief  算法的数据结构
 * 
 * \author LUOJIAXUAN
 * \date   May 2nd 2024
 *********************************************************************/
#pragma once
#include "Geometry.h"

namespace SparseSurfelFusion{
    /**
     * \brief 记录八叉树节点的数据类型，大小 276 Bytes.
     */
    class OctNode {
    public:
        int key;            // 节点的键key
        int pidx;           // 节点在SortedArray中的第一个元素的index
        int pnum;           // 与当前节点key相同的稠密点的数量
        int parent;         // 1个父节点
        int children[8];    // 8个孩子节点
        int neighs[27];     // 27个邻居节点
        // record the start in maxDepth NodeArray the first node at maxDepth is index 0
        int didx;           // 当前节点下属的深度为D的子节点第一个点，idx最小的点
        int dnum;           // 当前节点下属的深度为D的子节点的数量，包括有效点和无效点

        int vertices[8];    // 【记录的idx偏移一位】记录节点的顶点信息，这个信息存在VertexArray数组中，这里只是记录着节点在VertexArray中的index

        // (real idx) + 1,
        // idx start from (0 + 1)
        int edges[12];      // 顶点cube的边：实际idx与存储存在1个偏移

        // (real idx) + 1
        // idx start from (0 + 1)
        int faces[6];       // 顶点cube的面：实际idx与存储存在1个偏移

        int hasTriangle;    // 记录这个顶点cube是否有三角形
        int hasIntersection;// 记录这个顶点cube是否有面边相交(Surface-Edge Intersection)
    };

    /**
     * \brief OctNode的简化版，去除了face和三角标记相交标记.
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
#pragma unroll  // 对循环进行展开：种优化技术，通过将循环体中的代码复制多次来减少循环开销，从而提高程序的执行速度
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
     * \brief 顶点vertex数据类型，大小 56 Bytes.
     */
    class VertexNode {
    public:
        Point3D<float> pos = Point3D<float>(0.0f, 0.0f, 0.0f);  // 当前Vertex的位置
        int ownerNodeIdx = 0;                                   // 这个Vertex属于NodeArray中哪一个节点(index)，同一个节点可以拥有多个vertex， 但一个vertex只能有一个节点index
        int vertexKind = 0;                                     // 顶点对应的位置index
        int depth = 0;                                          // 这个Vertex对应的节点深度
        int nodes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };              // 记录当前顶点的Owner邻居中，共享这个vertex的节点index(一个Vertex可以被8个节点共享)
    };

    /**
     * \brief 边Edge数据类型，大小 24 Bytes.
     */
    class EdgeNode {
    public:
        int edgeKind = 0;                                       // 边对应的位置顺序
        int ownerNodeIdx = 0;                                   // 记录Edge属于NodeArray中的哪一个节点(index)，同一个节点可拥有多个Edge，但是一个Edge只能拥有一个节点index
        int nodes[4] = { 0, 0, 0, 0 };                          // 记录当前边的Owner邻居中，共享这个Edge的节点index(一个Edge可以被4个节点共享)
    };

    /**
     * \brief 面Face数据类型，大小 20 Bytes.
     */
    class FaceNode {
    public:
        int faceKind = -1;                                      // 面对应的位置顺序
        int ownerNodeIdx = -1;                                  // 记录面属于NodeArray中的哪一个节点
        int hasParentFace = -1;                                 // 记录当前面是否跟父节点共一个面
        int nodes[2] = { -1, -1 };                              // 记录当前面的Owner邻居中，共享这个面的两个节点index
    };
}

