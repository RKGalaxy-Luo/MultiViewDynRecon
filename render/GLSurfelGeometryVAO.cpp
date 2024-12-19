/*****************************************************************//**
 * \file   GLSurfelGeometryVAO.h
 * \brief  �����������Ĺ���
 *
 * \author LUO
 * \date   February 22nd 2024
 *********************************************************************/
#include "GLSurfelGeometryVAO.h"

void SparseSurfelFusion::buildFusionMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& fusionMapVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// �󶨵���OpenGL��������

	//��������VAO
	glGenVertexArrays(1, &fusionMapVAO);
	glBindVertexArray(fusionMapVAO);

/************************* ʹ��VBO��ʶ����VAO���󣬰Ѷ�VBO�����ݵ������浽��һ��VAO�� *************************/
	// ������������geometryVBO.LiveVertexConfidence���뻺����Ŀ��GL_ARRAY_BUFFER �󶨣��������������������������������
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveVertexConfidence);
	// ָ���������Եĸ�ʽ��������Դ
	// ����1��ָ��Ҫ���õĶ������Ե��������ڶ�����ɫ���У�����ͨ��ʹ�� layout(location = index) ��ָ���������Ե�λ��
	// ����2��ָ��ÿ���������Ե������������������ 4����ʾÿ������������ 4 �����������
	// ����3��ָ��ÿ����������������������͡��������� GL_FLOAT����ʾÿ�������һ��������
	// ����4��ָ���Ƿ�ԷǸ������͵����ݽ��й�һ�������������� GL_FALSE����ʾ�����й�һ������
	// ����5��ָ��������������֮����ֽ�ƫ�������������� 0����ʾ��������֮��û�м��
	// ����6��ָ�������������ݵ���ʼ��ַ���������� (void*)0����ʾ�ӻ������������ʼλ�ÿ�ʼ��ȡ������������
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// ���ö����������飬�� 0 ��ʾҪ��������Ϊ 0 �Ķ����������飬���ö������������OpenGL ����ʹ�����������Ӧ�Ķ���������������Ⱦͼ�Ρ�
	// ����ζ���ں�������Ⱦ�����У�������ɫ�����ܹ����ʺ�ʹ��������Ϊ 0 �Ķ������Թ���������
	glEnableVertexAttribArray(0);

	// �󶨲���Ⱦ������������LiveNormalRadius(Live���еķ��ߺͰ뾶)
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// �󶨺���Ⱦ������������Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// ��󻺳������󣬲��� GL_ARRAY_BUFFER ��ʾ�����Ƕ������Ի��������������0 ��ʾ���ǰ�󶨵Ļ���������
	// ͨ���� 0 ��Ϊ�������ݸ� glBindBuffer ���������Խ���ǰ�Ļ����������붥�����Ի�����Ŀ���󣬲��ٽ������Ķ�����������д�뵽�κλ�����������
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// ��󶥵�������󡣲��� 0 ��ʾ���ǰ�󶨵Ķ����������ͨ���� 0 ��Ϊ�������ݸ� glBindVertexArray ������
	// ���Խ���ǰ�Ķ�����������󣬲���ʹ���κζ���������������Ⱦ
	glBindVertexArray(0);
}

void SparseSurfelFusion::buildSolverMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& solverMapVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// �󶨵���OpenGL��������

	// ��������һ��VAO
	glGenVertexArrays(1, &solverMapVAO);
	glBindVertexArray(solverMapVAO);

/****************************************** VAO���������� ******************************************/
	// Canonical��Ķ�������Ŷ�
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalVertexConfidence);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// Canonical��ķ��ߺ���Ԫ�뾶
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// Live��Ķ�������Ŷ�
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveVertexConfidence);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// Live��ķ��ߺ���Ԫ�뾶
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveNormalRadius);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(3);

	// Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(4);

	// ���VAO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SparseSurfelFusion::buildCanonicalGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& canonicalGeometryVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// �󶨵���OpenGL��������

	// ��������һ��VAO
	glGenVertexArrays(1, &canonicalGeometryVAO);
	glBindVertexArray(canonicalGeometryVAO);

/****************************************** VAO������Canonical������� ******************************************/
	// Canonical��Ķ�������Ŷ�
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalVertexConfidence);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// Canonical��ķ��ߺ���Ԫ�뾶
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// ���VAO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SparseSurfelFusion::buildLiveGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& liveGeometryVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// �󶨵���OpenGL��������

	// FusionMap��VAO���ö��㻺��
	buildFusionMapVAO(geometryVBO, liveGeometryVAO);
}

void SparseSurfelFusion::buildFusionDepthGeometryVAO(const GLfusionDepthSurfelVBO& geometryVBO, GLuint& fusionDepthGeometryVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// �󶨵���OpenGL��������

	//��������VAO
	glGenVertexArrays(1, &fusionDepthGeometryVAO);
	glBindVertexArray(fusionDepthGeometryVAO);

	/************************* ʹ��VBO��ʶ����VAO���󣬰Ѷ�VBO�����ݵ������浽��һ��VAO�� *************************/
		// ������������geometryVBO.LiveVertexConfidence���뻺����Ŀ��GL_ARRAY_BUFFER �󶨣��������������������������������
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalVertexConfidence);
	// ָ���������Եĸ�ʽ��������Դ
	// ����1��ָ��Ҫ���õĶ������Ե��������ڶ�����ɫ���У�����ͨ��ʹ�� layout(location = index) ��ָ���������Ե�λ��
	// ����2��ָ��ÿ���������Ե������������������ 4����ʾÿ������������ 4 �����������
	// ����3��ָ��ÿ����������������������͡��������� GL_FLOAT����ʾÿ�������һ��������
	// ����4��ָ���Ƿ�ԷǸ������͵����ݽ��й�һ�������������� GL_FALSE����ʾ�����й�һ������
	// ����5��ָ��������������֮����ֽ�ƫ�������������� 0����ʾ��������֮��û�м��
	// ����6��ָ�������������ݵ���ʼ��ַ���������� (void*)0����ʾ�ӻ������������ʼλ�ÿ�ʼ��ȡ������������
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// ���ö����������飬�� 0 ��ʾҪ��������Ϊ 0 �Ķ����������飬���ö������������OpenGL ����ʹ�����������Ӧ�Ķ���������������Ⱦͼ�Ρ�
	// ����ζ���ں�������Ⱦ�����У�������ɫ�����ܹ����ʺ�ʹ��������Ϊ 0 �Ķ������Թ���������
	glEnableVertexAttribArray(0);

	// �󶨲���Ⱦ������������LiveNormalRadius(Live���еķ��ߺͰ뾶)
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// �󶨺���Ⱦ������������Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// ��󻺳������󣬲��� GL_ARRAY_BUFFER ��ʾ�����Ƕ������Ի��������������0 ��ʾ���ǰ�󶨵Ļ���������
	// ͨ���� 0 ��Ϊ�������ݸ� glBindBuffer ���������Խ���ǰ�Ļ����������붥�����Ի�����Ŀ���󣬲��ٽ������Ķ�����������д�뵽�κλ�����������
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// ��󶥵�������󡣲��� 0 ��ʾ���ǰ�󶨵Ķ����������ͨ���� 0 ��Ϊ�������ݸ� glBindVertexArray ������
	// ���Խ���ǰ�Ķ�����������󣬲���ʹ���κζ���������������Ⱦ
	glBindVertexArray(0);
}



