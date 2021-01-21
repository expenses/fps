#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;

layout(location = 0) out vec4 out_colour;

layout(push_constant) uniform ProjectionView {
    mat4 projection_view;
};

void main() {
    out_colour = colour;
    gl_Position = projection_view * vec4(position, 1.0);
    gl_Position.z = 0.0;
}
