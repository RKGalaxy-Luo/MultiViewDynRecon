/*****************************************************************//**
 * \file   GLSurfelGeometryVAO.h
 * \brief  顶点数组对象的管理
 * 
 * \author LUO
 * \date   February 22nd 2024
 *********************************************************************/
#pragma once
#include "GLSurfelGeometryVBO.h"

namespace SparseSurfelFusion {
	/**
	 * \brief 注册并绑定FusionMap(Live域实时显示的渲染图)的VAO.
	 * 
	 * \param geometryVBO 传入需要绑定的面元参数(这些参数已经被绑定到了OpenGL的VBO中)
	 * \param fusionMapVAO 绑定后分配的VAO的标识符
	 */
	void buildFusionMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& fusionMapVAO);
	/**
	 * \brief 注册并绑定SolverMap(Canonical域和Live域实时的渲染图)的VAO.
	 * 
	 * \param geometryVBO 传入需要绑定的面元参数(这些参数已经被绑定到了OpenGL的VBO中)
	 * \param solverMapVAO 绑定后分配的VAO的标识符
	 */
	void buildSolverMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& solverMapVAO);
	/**
	 * \brief 注册并绑定SolverMap(Canonical域和Live域实时的渲染图)的VAO.
	 * 
	 * \param geometryVBO 传入需要绑定的面元参数(这些参数已经被绑定到了OpenGL的VBO中)
	 * \param canonicalGeometryVAO 绑定后分配的VAO的标识符
	 */
	void buildCanonicalGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& canonicalGeometryVAO);
	/**
	 * \brief 注册并绑定Live域实时显示的渲染图的VAO.
	 *
	 * \param geometryVBO 传入需要绑定的面元参数(这些参数已经被绑定到了OpenGL的VBO中)
	 * \param liveGeometryVAO 绑定后分配的VAO的标识符
	 */
	void buildLiveGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& liveGeometryVAO);

	//用于每帧的深度面元融合
	void buildFusionDepthGeometryVAO(const GLfusionDepthSurfelVBO& geometryVBO, GLuint& fusionDepthGeometryVAO);


}
