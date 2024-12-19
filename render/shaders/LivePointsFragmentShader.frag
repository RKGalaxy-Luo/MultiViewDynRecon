#version 460 core

in vec3 color;

out vec4 FragColor; // 输出颜色

void main()
{
    highp vec4 highPrecisionColor = vec4(color.z, color.y, color.x, 1.0f); // 使用高精度修饰符声明高精度颜色变量
    FragColor = highPrecisionColor;   // 片段着色器设置顶点颜色
} 

