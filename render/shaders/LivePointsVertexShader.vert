#version 460 core

layout (location = 0) in vec3 vertexPos;    // 输入原始点三维坐标
layout (location = 1) in vec3 vertexColor;  // 输入原始点颜色属性

out vec3 color;

uniform mat4 model;         // 模型矩阵
uniform mat4 view;          // 视角矩阵
uniform mat4 projection;    // 投影矩阵


void main()
{
    color = vertexColor;
    highp vec4 normalizedVertexPos = projection * view * model * vec4(vertexPos.xyz, 1.0);        // 将其转换到相机坐标系内，并归一化到[-1,1]范围内，以便OpenGL显示
    gl_Position = normalizedVertexPos;      // 使顶点旋转起来
    gl_PointSize = 2.0f;
}