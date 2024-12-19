#version 460 core

layout (location = 0) in vec3 vertexPos;    // ����ԭʼ����ά����
layout (location = 1) in vec3 vertexColor;  // ����ԭʼ����ɫ����

out vec3 color;

uniform mat4 model;         // ģ�;���
uniform mat4 view;          // �ӽǾ���
uniform mat4 projection;    // ͶӰ����


void main()
{
    color = vertexColor;
    highp vec4 normalizedVertexPos = projection * view * model * vec4(vertexPos.xyz, 1.0);        // ����ת�����������ϵ�ڣ�����һ����[-1,1]��Χ�ڣ��Ա�OpenGL��ʾ
    gl_Position = normalizedVertexPos;      // ʹ������ת����
    gl_PointSize = 2.0f;
}