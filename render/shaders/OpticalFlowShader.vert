#version 460

layout (location = 0) in vec3 Pos;    // 输入坐标系几个关键点的三维坐标

out vec3 color;

uniform mat4 model;         // 模型矩阵
uniform mat4 view;          // 视角矩阵
uniform mat4 projection;    // 投影矩阵

void main()
{
    color = vec3(1.0f, 1.0f, 1.0f);
    highp vec4 normalizedPos = projection * view * model * vec4(Pos.xyz, 1.0);        // 将其转换到相机坐标系内，并归一化到[-1,1]范围内，以便OpenGL显示
    gl_Position = normalizedPos;      // 使顶点旋转起来
}