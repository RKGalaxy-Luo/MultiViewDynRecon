#version 460 core

layout (location = 0) in vec4 vertexConfidence;     // 输入原始点三维坐标
layout (location = 1) in vec4 normalRadius;         // 输入原始点颜色属性
layout (location = 2) in vec4 colorTime;            // 输入原始点颜色属性

//The output used by normal map, phong shading and albedo color
out VertexOut {
    vec4 camera_vertex;
    vec4 camera_normal;
    vec4 normalized_rgb;
} vertexShaderOut;

uniform mat4 model;                     // 模型矩阵
uniform mat4 view;                      // 视角矩阵
uniform mat4 projection;                // 投影矩阵
uniform vec4 intrinsic;                 // cx, cy, fx, fy
uniform float currentTime;              // 当前时间
uniform vec2 confidenceTimeThreshold;   // x是置信度阈值, y是被观测时间阈值


vec3 ProjectPointImage(vec3 p)
{
    return vec3(((intrinsic.z * p.x) / p.z) + intrinsic.x, ((intrinsic.w * p.y) / p.z) + intrinsic.y, p.z);
}

void main()
{
    if (vertexConfidence.w < confidenceTimeThreshold.x && (abs(colorTime.w - currentTime) >= confidenceTimeThreshold.y)) {
        //Make it outside the screes of [-1 1]
        gl_Position = vec4(1000.0f, 1000.0f, 1000.0f, 1000.0f);
    }
    else {
        gl_Position = projection * view * model * vec4(vertexConfidence.xyz, 1.0);
        // //Note that radius is in mm
        // vec3 x1 = normalize(vec3((normalRadius.y - normalRadius.z), -normalRadius.x, normalRadius.x)) * normalRadius.w * 1.41421356 * 0.001;
	    // vec3 y1 = cross(normalRadius.xyz, x1);
        // vec4 proj1 = vec4(ProjectPointImage(vertexConfidence.xyz + x1), 1.0);
	    // vec4 proj2 = vec4(ProjectPointImage(vertexConfidence.xyz + y1), 1.0);
	    // vec4 proj3 = vec4(ProjectPointImage(vertexConfidence.xyz - y1), 1.0);
	    // vec4 proj4 = vec4(ProjectPointImage(vertexConfidence.xyz - x1), 1.0);
        // vec2 xs = vec2(min(proj1.x, min(proj2.x, min(proj3.x, proj4.x))), max(proj1.x, max(proj2.x, max(proj3.x, proj4.x))));
	    // vec2 ys = vec2(min(proj1.y, min(proj2.y, min(proj3.y, proj4.y))), max(proj1.y, max(proj2.y, max(proj3.y, proj4.y))));
        // 
        // //Please refer to Elastic Fusion for these codes
        // float xDiff = abs(xs.y - xs.x);
	    // float yDiff = abs(ys.y - ys.x);
	    // float pointSize = max(0.0f, min(xDiff, yDiff));
        // gl_PointSize = min(5.0f, min(xDiff, yDiff));    // limit surfel radius < 5.0f
        gl_PointSize = min(10.0f, normalRadius.w);
        // gl_PointSize = normalRadius.w;
        //Fill in the output
        vertexShaderOut.camera_vertex = vec4(vertexConfidence.xyz, 1.0);
        vertexShaderOut.camera_normal = vec4(normalRadius.xyz, 1.0);
        vertexShaderOut.normalized_rgb = vec4(colorTime.xyz, 1.0);
    }
}