#version 450

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

const vec2 OFFSETS[8] = {
    vec2(-1,0), vec2(1,0), vec2(0,1), vec2(0,-1), vec2(-1,1), vec2(1,1), vec2(1,-1), vec2(-1,-1),
};

// Basically copied from https://shaderbits.com/blog/uv-dilation
vec4 dilate(uint max_steps) {
    vec4 current_sample = texture(sampler2D(u_texture, u_sampler), uv);

    if (current_sample.a != 0.0) {
        return current_sample;
    }

    vec2 texel_size = 1.0 / textureSize(sampler2D(u_texture, u_sampler), 0);

    for (uint i = 1; i <= max_steps; i++) {
        vec2 step = texel_size * i;

        for (uint j = 0; j < 8; j++) {
            vec2 current_uv = uv + OFFSETS[j] * step;
            vec4 offset_sample = texture(sampler2D(u_texture, u_sampler), current_uv);

            if (offset_sample.a != 0.0) {
                return offset_sample;
            }
        }
    }

    return current_sample;
}

void main() {
    colour = dilate(5);
}
