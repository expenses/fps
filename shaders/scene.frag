#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) flat in int texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in vec3 emissive_colour;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 2) uniform sampler u_nearest_sampler;
layout(set = 0, binding = 3) uniform sampler u_anisotrophic_sampler;

layout(set = 1, binding = 0) uniform texture2DArray u_texture;

struct Light {
    vec3 colour_output;
    vec3 position;
};

layout(set = 2, binding = 0) readonly buffer Lights {
	Light lights[];
};

// https://community.khronos.org/t/mipmap-level-calculation-using-dfdx-dfdy/67480/2
float mip_map_level(in vec2 unnormalised_texture_coordinates) {
    vec2  dx_vtc        = dFdx(unnormalised_texture_coordinates);
    vec2  dy_vtc        = dFdy(unnormalised_texture_coordinates);
    float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));

    return 0.5 * log2(delta_max_sqr);
}

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

    // Here we calculate the mip level that would be used, and based on that we choose either a
    // nearest neighbour sample (looks sharp close up) or a anisotrophic sample
    // (blurs nicely but not too much).

    ivec2 texture_size = textureSize(sampler2DArray(u_texture, u_nearest_sampler), 0).xy;

    float mip_level = clamp((mip_map_level(uv * texture_size) - 0.9) * 10.0, 0.0, 1.0);

    vec4 nearest_sample = texture(sampler2DArray(u_texture, u_nearest_sampler), vec3(uv, texture_index));
    vec4 anisotrophic_sample = texture(sampler2DArray(u_texture, u_anisotrophic_sampler), vec3(uv, texture_index));

    vec4 sampled = mix(
        nearest_sample, anisotrophic_sample,
        mip_level
    );

    colour = vec4(sampled.rgb * total + emissive_colour, sampled.a);
}
