/*****************************************************************//**
 * \file   GLSurfelGeometryVAO.h
 * \brief  �����������Ĺ���
 * 
 * \author LUO
 * \date   February 22nd 2024
 *********************************************************************/
#pragma once
#include "GLSurfelGeometryVBO.h"

namespace SparseSurfelFusion {
	/**
	 * \brief ע�Ტ��FusionMap(Live��ʵʱ��ʾ����Ⱦͼ)��VAO.
	 * 
	 * \param geometryVBO ������Ҫ�󶨵���Ԫ����(��Щ�����Ѿ����󶨵���OpenGL��VBO��)
	 * \param fusionMapVAO �󶨺�����VAO�ı�ʶ��
	 */
	void buildFusionMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& fusionMapVAO);
	/**
	 * \brief ע�Ტ��SolverMap(Canonical���Live��ʵʱ����Ⱦͼ)��VAO.
	 * 
	 * \param geometryVBO ������Ҫ�󶨵���Ԫ����(��Щ�����Ѿ����󶨵���OpenGL��VBO��)
	 * \param solverMapVAO �󶨺�����VAO�ı�ʶ��
	 */
	void buildSolverMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& solverMapVAO);
	/**
	 * \brief ע�Ტ��SolverMap(Canonical���Live��ʵʱ����Ⱦͼ)��VAO.
	 * 
	 * \param geometryVBO ������Ҫ�󶨵���Ԫ����(��Щ�����Ѿ����󶨵���OpenGL��VBO��)
	 * \param canonicalGeometryVAO �󶨺�����VAO�ı�ʶ��
	 */
	void buildCanonicalGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& canonicalGeometryVAO);
	/**
	 * \brief ע�Ტ��Live��ʵʱ��ʾ����Ⱦͼ��VAO.
	 *
	 * \param geometryVBO ������Ҫ�󶨵���Ԫ����(��Щ�����Ѿ����󶨵���OpenGL��VBO��)
	 * \param liveGeometryVAO �󶨺�����VAO�ı�ʶ��
	 */
	void buildLiveGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& liveGeometryVAO);

	//����ÿ֡�������Ԫ�ں�
	void buildFusionDepthGeometryVAO(const GLfusionDepthSurfelVBO& geometryVBO, GLuint& fusionDepthGeometryVAO);


}
