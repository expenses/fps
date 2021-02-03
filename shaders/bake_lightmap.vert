#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in uint texture_index;
layout(location = 4) in float emission;
layout(location = 5) in vec2 lightmap_uv;

layout(location = 0) out vec3 out_pos;
layout(location = 1) out vec3 out_normal;

void main() {
    out_pos = pos;
    out_normal = normal;

    vec2 position = (lightmap_uv * 2.0 - vec2(1.0)) * vec2(1.0, -1.0);

    gl_Position = vec4(position, 0.0, 1.0);
}
