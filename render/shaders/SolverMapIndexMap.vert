#version 460 core

layout(location = 0) in vec4 reference_vertex_confid;
layout(location = 1) in vec4 reference_normal_radius;
layout(location = 2) in vec4 warp_vertex;
layout(location = 3) in vec4 warp_normal;
layout(location = 4) in vec4 color_time;

out vec4 vs_out_reference_vertex;
out vec4 vs_out_reference_normal;
out vec4 vs_out_warp_vertex;
out vec4 vs_out_warp_normal;
flat out int vs_out_vertex_id;
out vec4 vs_out_normalized_rgb;

uniform mat4 world2camera;//1号相机的
uniform mat4 initialCameraSE3Inverse;//0号相机转到1号相机的SE3
uniform vec4 intrinsic; //cx, cy, fx, fy 1号相机的内参
//The width and height are in pixels, the maxdepth is in [m]
//The last element is current time
uniform vec4 width_height_maxdepth_currtime;
uniform vec2 confid_time_threshold; // x is confidence threshold, y is time threshold

vec3 project_point(vec3 p)
{
    return vec3(( (((intrinsic.z * p.x) / p.z) + intrinsic.x) - (width_height_maxdepth_currtime.x * 0.5)) / (width_height_maxdepth_currtime.x * 0.5),
                ((((intrinsic.w * p.y) / p.z) + intrinsic.y) - (width_height_maxdepth_currtime.y * 0.5)) / (width_height_maxdepth_currtime.y * 0.5),
                p.z / (width_height_maxdepth_currtime.z + 0.05));
}


void main() {
//这个warp_vertex_camera1是将0号相机空间的动作模型转换回1号相机空间
    vec4 warp_vertex_camera1 = initialCameraSE3Inverse * vec4(warp_vertex.xyz, 1.0);
    vec4 warp_vertex_camera = world2camera * warp_vertex_camera1;
    if(warp_vertex_camera.z > (width_height_maxdepth_currtime.z + 0.05)
    || warp_vertex_camera.z < 0
    || (reference_vertex_confid.w < confid_time_threshold.x && (abs(color_time.z - width_height_maxdepth_currtime.w) >= confid_time_threshold.y))) {
        //Make it outside the screes of [-1 1]
        gl_Position = vec4(1000.0f, 1000.0f, 1000.0f, 1000.0f);
    }
    else {
        gl_Position = vec4(project_point(warp_vertex_camera.xyz), 1.0);

        // Collect information for fragment shader input
        //注意这里ref域的输出是将canical域中的点转到了1号相机空间
        vs_out_reference_vertex = initialCameraSE3Inverse * vec4(reference_vertex_confid.xyz, 1.0);
        vs_out_reference_normal = initialCameraSE3Inverse * vec4(reference_normal_radius.xyz, 0.0);
        vs_out_warp_vertex =  initialCameraSE3Inverse * vec4(warp_vertex.xyz, 1.0);
        vs_out_warp_normal = initialCameraSE3Inverse * vec4(warp_normal.xyz, 0.0);
        vs_out_vertex_id = gl_VertexID;
        //下边的没动
        // The normalized color from color_time
        int encoded_color = floatBitsToInt(color_time.x);
        vec3 color;
        color.x = float(((encoded_color & 0x00FF0000) >> 16)) / 255.f;
        color.y = float(((encoded_color & 0x0000FF00) >>  8)) / 255.f;
        color.z = float(  encoded_color & 0x000000FF) / 255.f;
        vs_out_normalized_rgb = vec4(color.xyz, color_time.y);
    }
}