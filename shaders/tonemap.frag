#version 450

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

const float invGamma = 1.0 / 2.0;

float luminance(vec3 color) {
    const float RAmount = 0.2126;
    const float GAmount = 0.7152;
    const float BAmount = 0.0722;

    return dot(color, vec3(RAmount, GAmount, BAmount));
}

vec3 applyLuminance(vec3 color, float lum) {
    float originalLuminance = luminance(color);
    float scale = lum / originalLuminance;

    return color * scale;
}

float aces(float lum) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    lum = (lum * (a * lum + b)) / (lum * (c * lum + d) + e);
    lum = pow(lum, invGamma);

    return clamp(lum, 0, 1);
}

vec3 aces(vec3 color) {
    float lum = aces(luminance(color));

    return applyLuminance(color, lum);
}

vec3 tonemap(vec3 col) {
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

    vec3 z = pow(col, vec3(A));

    return z / (pow(z, vec3(D)) * B + vec3(C));
}

void main() {
    colour = texture(sampler2D(u_texture, u_sampler), uv);
    colour.rgb = tonemap(colour.rgb); // linear color output
    colour.a = sqrt(dot(colour.rgb, vec3(0.299, 0.587, 0.114))); // compute luma
}

