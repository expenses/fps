#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in int texture_index;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out flat int out_texture_index;
layout(location = 2) out vec3 out_pos;
layout(location = 3) out vec3 out_normal;

layout(set = 0, binding = 0) uniform Perspective {
    mat4 perspective;
};

layout(set = 0, binding = 1) uniform View {
    mat4 view;
};

void main() {
    out_uv = uv;
    out_texture_index = texture_index;
    out_pos = pos;
    out_normal = normal;

    gl_Position = perspective * view * vec4(pos, 1.0);
}
