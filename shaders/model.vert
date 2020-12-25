#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in int texture_index;
layout(location = 4) in vec3 emissive_colour;

layout(location = 5) in vec4 transform_1;
layout(location = 6) in vec4 transform_2;
layout(location = 7) in vec4 transform_3;
layout(location = 8) in vec4 transform_4;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out flat int out_texture_index;
layout(location = 2) out vec3 out_pos;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_emissive_colour;

layout(set = 0, binding = 0) uniform Perspective {
    mat4 perspective;
};

layout(set = 0, binding = 1) uniform View {
    mat4 view;
};

void main() {
    mat4 transform = mat4(transform_1, transform_2, transform_3, transform_4);

    out_uv = uv;
    out_texture_index = texture_index;
    out_pos = vec3(transform * vec4(pos, 1.0));
    out_normal = mat3(transpose(inverse(transform))) * normal;
    out_emissive_colour = emissive_colour;

    gl_Position = perspective * view * transform * vec4(pos, 1.0);
}