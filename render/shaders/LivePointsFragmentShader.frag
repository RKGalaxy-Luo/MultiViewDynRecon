#version 460 core

in vec3 color;

out vec4 FragColor; // �����ɫ

void main()
{
    highp vec4 highPrecisionColor = vec4(color.z, color.y, color.x, 1.0f); // ʹ�ø߾������η������߾�����ɫ����
    FragColor = highPrecisionColor;   // Ƭ����ɫ�����ö�����ɫ
} 

