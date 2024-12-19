#version 460 core

layout (location = 0) in vec3 pos;      // λ�ñ���������λ��ֵΪ 0 
layout (location = 1) in vec3 normal;   // ����������������λ��ֵΪ 1
layout (location = 2) in vec3 color;    // ��ɫ����������λ��ֵΪ 2

uniform mat4 model;         // ģ�;���
uniform mat4 view;          // �ӽǾ���
uniform mat4 projection;    // ͶӰ����

out vec3 Normal;            // ����
out vec3 FragPos;           // ������ռ��еĶ���λ��
out vec3 Color;             // ������ɫ

void main()
{
    FragPos = vec3(model * vec4(pos, 1.0));
    Normal = normal;
    Color = color;
    gl_Position = projection * view * vec4(FragPos, 1.0);        // ���������(ǰ���Ѿ�����˵�FragPos, �˴��������)
}