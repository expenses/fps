#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in int texture_index;
layout(location = 4) in float emission;
layout(location = 5) in uvec4 joint_indices;
layout(location = 6) in vec4 joint_weights;

layout(location = 7) in vec4 transform_1;
layout(location = 8) in vec4 transform_2;
layout(location = 9) in vec4 transform_3;
layout(location = 10) in vec4 transform_4;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out flat int out_texture_index;
layout(location = 2) out vec3 out_pos;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out float out_emission;

layout(set = 0, binding = 0) uniform Perspective {
    mat4 perspective;
};

layout(set = 0, binding = 1) uniform View {
    mat4 view;
};

layout(set = 1, binding = 1) uniform AnimatedModelUniforms {
    uint num_joints;
};

layout(set = 3, binding = 0) readonly buffer JointTransforms {
	mat4 joint_transforms[];
};

void main() {
    mat4 transform = mat4(transform_1, transform_2, transform_3, transform_4);

    uint joint_offset = gl_InstanceIndex * num_joints;

    // Calculate skinned matrix from weights and joint indices of the current vertex
	mat4 skin = 
		joint_weights.x * joint_transforms[joint_indices.x + joint_offset] +
		joint_weights.y * joint_transforms[joint_indices.y + joint_offset] +
		joint_weights.z * joint_transforms[joint_indices.z + joint_offset] +
		joint_weights.w * joint_transforms[joint_indices.w + joint_offset];

    mat4 skinned_transform = transform * skin;

    out_uv = uv;
    out_texture_index = texture_index;
    out_pos = vec3(skinned_transform * vec4(pos, 1.0));
    out_normal = mat3(transpose(inverse(skinned_transform))) * normal;
    out_emission = emission;

    gl_Position = perspective * view * skinned_transform * vec4(pos, 1.0);
}
