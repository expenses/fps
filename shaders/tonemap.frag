#version 450

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

float tonemap(float x) {
    // see https://www.desmos.com/calculator/0eo9pzo1at
    // toe = 1;
    // shoulder = 0.987
    // max luminance = 20;
    // grey in = 0.75;
    // grey out = 0.5;

    const float A = 1;
    const float B = 1.00090507637;
    const float C = 0.746508497591;
    const float D = 0.987;

    float z = pow(x, A);

    return z / (pow(z, D) * B + C);
}

vec3 lerp(vec3 a, vec3 b, float factor) {
    return (1.0 - factor) * a + factor * b;
}

void main() {
    colour = texture(sampler2D(u_texture, u_sampler), uv);
    vec3 rgb = colour.rgb;

    float peak = max(max(rgb.r, rgb.g), rgb.b);
    vec3 ratio = rgb / peak;
    peak = tonemap(peak);

    // Apply channel crosstalk

    float saturation = 1.5;
    float crossSaturation = 2.0;
    float crosstalk = 1.0 / 10.0;

    ratio = pow(ratio, vec3(saturation / crossSaturation));
    ratio = lerp(ratio, vec3(1.0), pow(peak, 1.0 / crosstalk));
    ratio = pow(ratio, vec3(crossSaturation));

    colour.rgb = peak * ratio;

    // compute luma
    colour.a = sqrt(dot(colour.rgb, vec3(0.299, 0.587, 0.114)));
}

