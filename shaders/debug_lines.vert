#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;

layout(location = 0) out vec4 out_colour;

layout(set = 0, binding = 0) uniform Perspective {
    mat4 perspective;
};

layout(set = 0, binding = 1) uniform View {
    mat4 view;
};

void main() {
    out_colour = colour;
    gl_Position = perspective * view * vec4(position, 1.0);
    // Put the line closer to the camera in the z buffer so that it can be drawn as an overlay on
    // the debug geometry.
    gl_Position.z = gl_Position.z * 0.9999;
}
