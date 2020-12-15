#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) flat in int texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 2) uniform sampler u_sampler;

layout(set = 1, binding = 0) uniform texture2DArray u_texture;

struct Light {
    vec3 colour;
    float intensity;
    vec3 position;
};

layout(set = 1, binding = 1) readonly buffer Lights {
	Light lights[];
};


void main() {
    vec3 ambient = vec3(0.05);

    vec3 total = ambient;

    vec3 norm = normalize(normal);

    for (int i = 0; i < lights.length(); i++) {
        Light light = lights[i];

        float distance = length(light.position - pos);
        float intensity_at_point = pow(distance, -2.0);

        vec3 light_dir = normalize(light.position - pos);
        float diffuse = max(dot(norm, light_dir), 0.0);

        total += diffuse * light.colour * intensity_at_point * light.intensity;
    }

    vec4 sampled = texture(sampler2DArray(u_texture, u_sampler), vec3(uv, texture_index));

    colour = vec4(total * sampled.rgb, sampled.a);
}
