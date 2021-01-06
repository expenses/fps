#version 450

/* sum(rgb * a, a) */
layout(set = 0, binding = 0) uniform texture2D accumTexture;

/* prod(1 - a) */
layout(set = 0, binding = 1) uniform texture2D revealageTexture;

layout(set = 0, binding = 2) uniform sampler u_sampler;

layout(location = 0) out vec4 result;

float maxComponent(vec4 vec) {
    return max(max(vec.r, vec.g), max(vec.b, vec.a));
}

void main() {
    ivec2 C = ivec2(gl_FragCoord.xy);

    float revealage = texelFetch(sampler2D(revealageTexture, u_sampler), C, 0).r;
    if (revealage == 1.0) {
        // Save the blending and color texture fetch cost
        discard;
    }

    vec4 accum = texelFetch(sampler2D(accumTexture, u_sampler), C, 0);

    // Suppress overflow
    if (isinf(maxComponent(abs(accum)))) {
        accum.rgb = vec3(accum.a);
    }

    // dst' =  (accum.rgb / accum.a) * (1 - revealage) + dst
    // [dst has already been modulated by the transmission colors and coverage and the blend mode
    // inverts revealage for us]
    result = vec4(accum.rgb / max(accum.a, 0.00001), revealage);
}
