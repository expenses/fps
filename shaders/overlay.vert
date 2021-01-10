#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 colour;

layout(location = 0) out vec4 out_colour;

layout(push_constant) uniform ScreenDimensions {
    vec2 screen_dimensions;
};

void main() {
    out_colour = colour;

    vec2 adjusted_position = vec2(
        (position.x / screen_dimensions.x * 2.0) - 1.0,
        1.0 - (position.y / screen_dimensions.y * 2.0)
    );
    gl_Position = vec4(adjusted_position, 0.0, 1.0);
}
