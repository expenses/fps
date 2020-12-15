#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) flat in int texture_index;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 2) uniform texture2D u_texture;

layout(set = 1, binding = 0) uniform sampler u_sampler;

void main() {
    colour = vec4(uv, 0.0, 1.0);
    //colour = texture(sampler2D(u_texture, u_sampler), vec2(0.0, 0.5));
}
