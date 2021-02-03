#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uint texture_index;
layout(location = 4) in float emission;
layout(location = 5) in vec2 lightmap_uv;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out flat uint out_texture_index;
layout(location = 2) out vec3 out_pos;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out float out_emission;
layout(location = 5) out vec2 out_lightmap_uv;

layout(push_constant) uniform ProjectionView {
    mat4 projection_view;
};

void main() {
    out_uv = uv;
    out_texture_index = texture_index;
    out_pos = pos;
    out_normal = normal;
    out_emission = emission;
    out_lightmap_uv = lightmap_uv;

    gl_Position = projection_view * vec4(pos, 1.0);
}
