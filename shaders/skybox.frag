#version 450

layout(location = 0) in vec3 uv;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(set = 1, binding = 0) uniform textureCube u_texture_cube;

void main() {
    colour = texture(samplerCube(u_texture_cube, u_sampler), uv);
}
