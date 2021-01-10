#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) flat in int texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in float emission;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 1) uniform sampler u_nearest_sampler;

layout(set = 1, binding = 0) uniform texture2D u_texture;

struct TextureLocation {
    vec2 offset;
    vec2 size;
};

layout(set = 1, binding = 1) readonly buffer TextureLocations {
    TextureLocation texture_locations[];
};

struct Light {
    vec3 colour_output;
    float range;
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

    TextureLocation location = texture_locations[texture_index];

    float mip_map_level = mip_map_level(uv * location.size * textureSize(sampler2D(u_texture, u_nearest_sampler), 0));

    vec2 new_uv = location.offset + location.size * fract(uv);

    vec4 sampled = textureLod(sampler2D(u_texture, u_nearest_sampler), new_uv, mip_map_level);

    colour = vec4(sampled.rgb * (total + emission), sampled.a);
    //colour = vec4(new_uv, 0.0, 1.0);
}
