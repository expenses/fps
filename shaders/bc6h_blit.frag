#version 450

// Adapted from:
// https://github.com/knarkowicz/GPURealTimeBC6H/blob/ba0c5df59f5b073fc349cb23cdb1c4152315308f/bin/blit.hlsl

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D texture_a;
layout(set = 0, binding = 1) uniform texture2D texture_b;
layout(set = 0, binding = 2) uniform sampler u_sampler;

layout(push_constant) uniform PushConstants {
	float Exposure;
	uint BlitMode;
};

float Luminance(vec3 x)
{
	vec3 luminanceWeights = vec3(0.299f, 0.587f, 0.114f);
	return dot(x, luminanceWeights);
}

void main() {
	vec3 a = textureLod(sampler2D(texture_a, u_sampler), uv, 0.0).rgb * Exposure;
	vec3 b = textureLod(sampler2D(texture_b, u_sampler), uv, 0.0).rgb * Exposure;
	vec3 delta = log(a + 1.0f) - log(b + 1.0f);
	vec3 deltaSq = delta * delta * 16.0f;

	if (BlitMode == 0)
	{
		colour = vec4(a, 1.0);
        return;
	}

	if (BlitMode == 1)
	{
        colour = vec4(b, 1.0);
        return;
	}

	if (BlitMode == 2)
	{
        colour = vec4(deltaSq, 1.0);
        return;
	}

	colour = vec4(vec3(Luminance(deltaSq)), 1.0);
}
