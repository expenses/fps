#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) flat in int texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in vec3 emissive_colour;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 2) uniform sampler u_sampler;

layout(set = 1, binding = 0) uniform texture2DArray u_texture;

struct Light {
    vec3 colour_output;
    vec3 position;
};

layout(set = 2, binding = 0) readonly buffer Lights {
	Light lights[];
};


void main() {
    vec3 ambient = vec3(0.05);

    vec3 total = ambient;

    vec3 norm = normalize(normal);

    for (int i = 0; i < lights.length(); i++) {
        Light light = lights[i];

        vec3 vector = light.position - pos;

        float distance = length(vector);
        float intensity_at_point = pow(distance, -2.0);

        vec3 light_dir = normalize(vector);
        float facing = max(dot(norm, light_dir), 0.0);

        // Multiplying the `floats` first results in one less `OpVectorTimesScalar` in the spirv
        total += (facing * intensity_at_point) * light.colour_output;
    }

    vec4 sampled = texture(sampler2DArray(u_texture, u_sampler), vec3(uv, texture_index));

    colour = vec4(total * sampled.rgb + emissive_colour, sampled.a);
}
