#version 450
#extension GL_EXT_nonuniform_qualifier: enable

layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in float emission;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform sampler u_nearest_sampler;
layout(set = 0, binding = 1) uniform sampler u_linear_sampler;

layout(set = 1, binding = 0) uniform texture2D u_texture[];

struct Light {
    vec3 colour_output;
    float range;
    vec3 position;
};

layout(set = 2, binding = 0) readonly buffer Lights {
	Light lights[];
};

layout(set = 2, binding = 1) uniform texture3D irradience_textures[6];

layout(set = 2, binding = 2) uniform IrradienceVolumeUniforms {
    vec3 position;
    vec3 scale;
} irradience_volume;

const uint X = 0;
const uint NEG_X = 1;
const uint Y = 2;
const uint NEG_Y = 3;
const uint Z = 4;
const uint NEG_Z = 5;

const vec3 X_NORMAL = vec3(1.0, 0.0, 0.0);
const vec3 Y_NORMAL = vec3(0.0, 1.0, 0.0);
const vec3 Z_NORMAL = vec3(0.0, 0.0, 1.0);

const vec3 AMBIENT = vec3(0.05);
const float MIN_LIGHT_DISTANCE = 0.5;

void main() {
    vec4 sampled = texture(sampler2D(u_texture[texture_index], u_nearest_sampler), uv);
    vec3 norm = normalize(normal);

    vec3 total_lighting = AMBIENT + emission;

    vec3 sample_index = vec3(0.5) + ((pos - irradience_volume.position) / irradience_volume.scale);

    float x_dot = dot(X_NORMAL, norm);

    if (x_dot > 0.0) {
        total_lighting += texture(sampler3D(irradience_textures[X], u_linear_sampler), sample_index).rgb * x_dot;
    } else {
        total_lighting += texture(sampler3D(irradience_textures[NEG_X], u_linear_sampler), sample_index).rgb * -x_dot;
    }

    float y_dot = dot(Y_NORMAL, norm);

    if (y_dot > 0.0) {
        total_lighting += texture(sampler3D(irradience_textures[Y], u_linear_sampler), sample_index).rgb * y_dot;
    } else {
        total_lighting += texture(sampler3D(irradience_textures[NEG_Y], u_linear_sampler), sample_index).rgb * -y_dot;
    }

    float z_dot = dot(Z_NORMAL, norm);

    if (z_dot > 0.0) {
        total_lighting += texture(sampler3D(irradience_textures[Z], u_linear_sampler), sample_index).rgb * z_dot;
    } else {
        total_lighting += texture(sampler3D(irradience_textures[NEG_Z], u_linear_sampler), sample_index).rgb * -z_dot;
    }

    colour = vec4(sampled.rgb * total_lighting, sampled.a);
}
