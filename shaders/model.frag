#version 450
#extension GL_EXT_nonuniform_qualifier: enable

layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in float emission;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform sampler u_nearest_sampler;

layout(set = 1, binding = 0) uniform texture2D u_texture[];

struct Light {
    vec3 colour_output;
    float range;
    vec3 position;
};

layout(set = 2, binding = 0) readonly buffer Lights {
	Light lights[];
};

const vec3 AMBIENT = vec3(0.05);
const float MIN_LIGHT_DISTANCE = 0.5;

void main() {
    vec3 norm = normalize(normal);

    vec3 total = AMBIENT;

    for (int i = 0; i < lights.length(); i++) {
        Light light = lights[i];

        vec3 vector = light.position - pos;

        float distance = length(vector);
        // This uses the following equation except without raising 'distance / light.range' to a
        // power in order to match what blender does.
        // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
        float attenuation = clamp(1.0 - (distance / light.range), 0.0, 1.0) / pow(distance, 2);

        vec3 light_dir = normalize(vector);
        float facing = max(dot(norm, light_dir), 0.0);

        // Multiplying the `floats` first results in one less `OpVectorTimesScalar` in the spirv
        total += (facing * attenuation) * light.colour_output;
    }

    vec4 sampled = texture(sampler2D(u_texture[texture_index], u_nearest_sampler), uv);

    colour = vec4(sampled.rgb * (total + emission), sampled.a);
}
