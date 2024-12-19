/*****************************************************************//**
 * \file   GLSurfelGeometryVAO.h
 * \brief  顶点数组对象的管理
 *
 * \author LUO
 * \date   February 22nd 2024
 *********************************************************************/
#include "GLSurfelGeometryVAO.h"

void SparseSurfelFusion::buildFusionMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& fusionMapVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	//创建并绑定VAO
	glGenVertexArrays(1, &fusionMapVAO);
	glBindVertexArray(fusionMapVAO);

/************************* 使用VBO标识符绑定VAO对象，把对VBO中数据的描述存到了一个VAO中 *************************/
	// 将缓冲区对象geometryVBO.LiveVertexConfidence，与缓冲区目标GL_ARRAY_BUFFER 绑定，后续操作将作用在这个缓冲区对象上
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveVertexConfidence);
	// 指定顶点属性的格式和数据来源
	// 参数1：指定要设置的顶点属性的索引。在顶点着色器中，可以通过使用 layout(location = index) 来指定顶点属性的位置
	// 参数2：指定每个顶点属性的组件数量。在这里是 4，表示每个顶点属性由 4 个浮点数组成
	// 参数3：指定每个顶点属性组件的数据类型。在这里是 GL_FLOAT，表示每个组件是一个浮点数
	// 参数4：指定是否对非浮点类型的数据进行归一化处理。在这里是 GL_FALSE，表示不进行归一化处理
	// 参数5：指定连续顶点属性之间的字节偏移量。在这里是 0，表示顶点属性之间没有间隔
	// 参数6：指定顶点属性数据的起始地址。在这里是 (void*)0，表示从缓冲区对象的起始位置开始读取顶点属性数据
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 启用顶点属性数组，数 0 表示要启用索引为 0 的顶点属性数组，启用顶点属性数组后，OpenGL 将会使用与该索引对应的顶点属性数据来渲染图形。
	// 这意味着在后续的渲染过程中，顶点着色器将能够访问和使用与索引为 0 的顶点属性关联的数据
	glEnableVertexAttribArray(0);

	// 绑定并渲染顶点属性数组LiveNormalRadius(Live域中的法线和半径)
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// 绑定和渲染顶点属性数组Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// 解绑缓冲区对象，参数 GL_ARRAY_BUFFER 表示解绑的是顶点属性缓冲区对象。在这里，0 表示解绑当前绑定的缓冲区对象
	// 通过将 0 作为参数传递给 glBindBuffer 函数，可以将当前的缓冲区对象与顶点属性缓冲区目标解绑，不再将后续的顶点属性数据写入到任何缓冲区对象中
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// 解绑顶点数组对象。参数 0 表示解绑当前绑定的顶点数组对象。通过将 0 作为参数传递给 glBindVertexArray 函数，
	// 可以将当前的顶点数组对象解绑，不再使用任何顶点数组对象进行渲染
	glBindVertexArray(0);
}

void SparseSurfelFusion::buildSolverMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& solverMapVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	// 创建并绑定一个VAO
	glGenVertexArrays(1, &solverMapVAO);
	glBindVertexArray(solverMapVAO);

/****************************************** VAO绑定所有属性 ******************************************/
	// Canonical域的顶点和置信度
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalVertexConfidence);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// Canonical域的法线和面元半径
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// Live域的顶点和置信度
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveVertexConfidence);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// Live域的法线和面元半径
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.LiveNormalRadius);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(3);

	// Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(4);

	// 解绑VAO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SparseSurfelFusion::buildCanonicalGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& canonicalGeometryVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	// 创建并绑定一个VAO
	glGenVertexArrays(1, &canonicalGeometryVAO);
	glBindVertexArray(canonicalGeometryVAO);

/****************************************** VAO绑定所有Canonical域的属性 ******************************************/
	// Canonical域的顶点和置信度
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalVertexConfidence);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// Canonical域的法线和面元半径
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// 解绑VAO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SparseSurfelFusion::buildLiveGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& liveGeometryVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	// FusionMap和VAO共用顶点缓冲
	buildFusionMapVAO(geometryVBO, liveGeometryVAO);
}

void SparseSurfelFusion::buildFusionDepthGeometryVAO(const GLfusionDepthSurfelVBO& geometryVBO, GLuint& fusionDepthGeometryVAO)
{
	glfwMakeContextCurrent(geometryVBO.GLFW_Window);	// 绑定调用OpenGL的上下文

	//创建并绑定VAO
	glGenVertexArrays(1, &fusionDepthGeometryVAO);
	glBindVertexArray(fusionDepthGeometryVAO);

	/************************* 使用VBO标识符绑定VAO对象，把对VBO中数据的描述存到了一个VAO中 *************************/
		// 将缓冲区对象geometryVBO.LiveVertexConfidence，与缓冲区目标GL_ARRAY_BUFFER 绑定，后续操作将作用在这个缓冲区对象上
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalVertexConfidence);
	// 指定顶点属性的格式和数据来源
	// 参数1：指定要设置的顶点属性的索引。在顶点着色器中，可以通过使用 layout(location = index) 来指定顶点属性的位置
	// 参数2：指定每个顶点属性的组件数量。在这里是 4，表示每个顶点属性由 4 个浮点数组成
	// 参数3：指定每个顶点属性组件的数据类型。在这里是 GL_FLOAT，表示每个组件是一个浮点数
	// 参数4：指定是否对非浮点类型的数据进行归一化处理。在这里是 GL_FALSE，表示不进行归一化处理
	// 参数5：指定连续顶点属性之间的字节偏移量。在这里是 0，表示顶点属性之间没有间隔
	// 参数6：指定顶点属性数据的起始地址。在这里是 (void*)0，表示从缓冲区对象的起始位置开始读取顶点属性数据
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 启用顶点属性数组，数 0 表示要启用索引为 0 的顶点属性数组，启用顶点属性数组后，OpenGL 将会使用与该索引对应的顶点属性数据来渲染图形。
	// 这意味着在后续的渲染过程中，顶点着色器将能够访问和使用与索引为 0 的顶点属性关联的数据
	glEnableVertexAttribArray(0);

	// 绑定并渲染顶点属性数组LiveNormalRadius(Live域中的法线和半径)
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.CanonicalNormalRadius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// 绑定和渲染顶点属性数组Color-Time Map
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.ColorTime);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	// 解绑缓冲区对象，参数 GL_ARRAY_BUFFER 表示解绑的是顶点属性缓冲区对象。在这里，0 表示解绑当前绑定的缓冲区对象
	// 通过将 0 作为参数传递给 glBindBuffer 函数，可以将当前的缓冲区对象与顶点属性缓冲区目标解绑，不再将后续的顶点属性数据写入到任何缓冲区对象中
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// 解绑顶点数组对象。参数 0 表示解绑当前绑定的顶点数组对象。通过将 0 作为参数传递给 glBindVertexArray 函数，
	// 可以将当前的顶点数组对象解绑，不再使用任何顶点数组对象进行渲染
	glBindVertexArray(0);
}



