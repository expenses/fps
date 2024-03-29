#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec4 colour;

struct Light {
    vec3 colour_output;
    float range;
    vec3 position;
};

layout(set = 0, binding = 0) readonly buffer Lights {
	Light lights[];
};

void main() {
    vec3 norm = normalize(normal);

    vec3 total = vec3(0.0);

    for (int i = 0; i < lights.length(); i++) {
        Light light = lights[i];

        vec3 vector = light.position - pos;

        float distance = length(vector);
        // This uses the following equation except without raising 'distance / light.range' to a
        // power in order to match what blender does.
        // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
        float attenuation = clamp(1.0 - (distance / light.range), 0.0, 1.0) / (distance * distance);

        vec3 light_dir = vector / distance;
        float facing = max(dot(norm, light_dir), 0.0);

        // Multiplying the `floats` first results in one less `OpVectorTimesScalar` in the spirv
        total += (facing * attenuation) * light.colour_output;
    }

    colour = vec4(total, 1.0);
}
