#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uint texture_index;
layout(location = 4) in float emission;
layout(location = 5) in vec2 lightmap_uv;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec2 out_lightmap_uv;
layout(location = 2) out flat uint out_texture_index;
layout(location = 3) out float out_emission;

layout(push_constant) uniform ProjectionView {
    mat4 projection_view;
};

void main() {
    out_uv = uv;
    out_lightmap_uv = lightmap_uv;
    out_texture_index = texture_index;
    out_emission = emission;

    gl_Position = projection_view * vec4(pos, 1.0);
}
