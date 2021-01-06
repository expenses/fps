#version 450

layout(location = 0) in vec2 uv;
layout(location = 1) flat in int texture_index;
layout(location = 2) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in float emission;

layout(location = 0) out vec4 accum;
layout(location = 1) out float revealage;
layout(location = 2) out vec4 modulate;

layout(set = 0, binding = 1) uniform sampler u_nearest_sampler;

layout(set = 1, binding = 0) uniform texture2DArray u_texture;

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

    vec4 sampled = texture(sampler2DArray(u_texture, u_nearest_sampler), vec3(uv, texture_index));

    vec4 colour = vec4(sampled.rgb * (total + emission), sampled.a);

    vec4 premultipliedReflect = vec4(colour.rgb * colour.a, colour.a);
    vec3 transmit = vec3(0.0);
    float csZ = 0.0;

    /* NEW: Perform this operation before modifying the coverage to account for transmission. */
    modulate = vec4(premultipliedReflect.a * (vec3(1.0) - transmit), 1.0);

    /* Modulate the net coverage for composition by the transmission. This does not affect the color channels of the
       transparent surface because the caller's BSDF model should have already taken into account if transmission modulates
       reflection. See

       McGuire and Enderton, Colored Stochastic Shadow Maps, ACM I3D, February 2011
       http://graphics.cs.williams.edu/papers/CSSM/

       for a full explanation and derivation.*/
     premultipliedReflect.a *= 1.0 - (transmit.r + transmit.g + transmit.b) * (1.0 / 3.0);

     // Intermediate terms to be cubed
     float tmp = (premultipliedReflect.a * 8.0 + 0.01) *
                 (-gl_FragCoord.z * 0.95 + 1.0);

     /* If a lot of the scene is close to the far plane, then gl_FragCoord.z does not
        provide enough discrimination. Add this term to compensate:

        tmp /= sqrt(abs(csZ)); */

     float w    = clamp(tmp * tmp * tmp * 1e3, 1e-2, 3e2);
     accum     = premultipliedReflect * w;
     revealage = premultipliedReflect.a;
}
