#version 460

layout (location = 0) in vec3 Pos;    // ��������ϵ�����ؼ������ά����

out vec3 color;

uniform mat4 model;         // ģ�;���
uniform mat4 view;          // �ӽǾ���
uniform mat4 projection;    // ͶӰ����

void main()
{
    color = vec3(1.0f, 1.0f, 1.0f);
    highp vec4 normalizedPos = projection * view * model * vec4(Pos.xyz, 1.0);        // ����ת�����������ϵ�ڣ�����һ����[-1,1]��Χ�ڣ��Ա�OpenGL��ʾ
    gl_Position = normalizedPos;      // ʹ������ת����
}