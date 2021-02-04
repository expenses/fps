#version 450
#extension GL_EXT_nonuniform_qualifier: enable

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 lightmap_uv;
layout(location = 2) flat in uint texture_index;
layout(location = 3) in float emission;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform sampler u_nearest_sampler;
layout(set = 0, binding = 1) uniform sampler u_linear_sampler;

layout(set = 1, binding = 0) uniform texture2D u_texture[];

layout(set = 2, binding = 0) uniform texture2D lightmap_texture;

const vec3 AMBIENT = vec3(0.05);

void main() {
    vec4 sampled = texture(sampler2D(u_texture[texture_index], u_nearest_sampler), uv);

    if (sampled.a < 0.5) {
        discard;
    }

    vec3 total_lighting = AMBIENT + emission;

    total_lighting += texture(sampler2D(lightmap_texture, u_linear_sampler), lightmap_uv).rgb;

    colour = vec4(sampled.rgb * total_lighting, sampled.a);
}
