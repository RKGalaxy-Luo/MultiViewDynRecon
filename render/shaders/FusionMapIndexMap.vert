#version 460 core

layout(location = 0) in vec4 warp_vertex;
layout(location = 1) in vec4 warp_normal;
layout(location = 2) in vec4 color_time;


out vec4 vs_out_warp_vertex;
out vec4 vs_out_warp_normal;
flat out int vs_out_vertex_id;
flat out vec4 vs_out_color_time;


uniform mat4 world2camera;
uniform mat4 initialCameraSE3Inverse;//0号相机转到1号相机的SE3
uniform vec4 intrinsic; //cx, cy, fx, fy
//The width and height are in pixels, the maxdepth is in [m]
//The last element is not used
uniform vec4 width_height_maxdepth; 

vec3 project_point(vec3 p)
{
    return vec3(( (((intrinsic.z * p.x) / p.z) + intrinsic.x) - (width_height_maxdepth.x * 0.5)) / (width_height_maxdepth.x * 0.5),
                ((((intrinsic.w * p.y) / p.z) + intrinsic.y) - (width_height_maxdepth.y * 0.5)) / (width_height_maxdepth.y * 0.5),
                p.z / (width_height_maxdepth.z + 0.05));
}

void main() {
    vec4 warp_vertex_temp = world2camera * vec4(warp_vertex.xyz, 1.0);//1号视角下的live
    vec4 warp_vertex_camera = initialCameraSE3Inverse * warp_vertex_temp;
    if (warp_vertex_camera.z > (width_height_maxdepth.z + 0.05) || warp_vertex_camera.z < 0) {
        gl_Position = vec4(1000.0f, 1000.0f, 1000.0f, 1000.0f); // Make it outside the screes of [-1 1]
    }
    else {
        gl_Position = vec4(project_point(warp_vertex_camera.xyz), 1.0);

        // Collect information for fragment shader input
        float confidence = warp_vertex.w;
        float radius = warp_normal.w;
        vs_out_warp_vertex = initialCameraSE3Inverse * vec4(warp_vertex.xyz, 1.0f);
        vs_out_warp_vertex = vec4(vs_out_warp_vertex.xyz, confidence);
        vs_out_warp_normal = initialCameraSE3Inverse * vec4(warp_normal.xyz, 0.0f);
        vs_out_warp_normal = vec4(vs_out_warp_normal.xyz, radius);
        vs_out_vertex_id = gl_VertexID;
        vs_out_color_time = color_time;
    }
}