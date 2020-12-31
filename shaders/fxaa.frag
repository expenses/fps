#version 450

layout(location = 0) in vec2 pos;

layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(push_constant) uniform ScreenDimensions {
    vec2 screen_dimensions;
};

#define FXAA_PC 1
#define FXAA_GLSL_130 1
// This is same as the file linked below, except I have commented where 'FxaaTex' is used and
// replaced it with 'sampler2D(u_texture, u_sampler)' instead.
// https://gist.github.com/kosua20/0c506b81b3812ac900048059d2383126
#include "fxaa3_11.h"

void main() {
    colour = FxaaPixelShader(
        // pos
        pos,
        // fxaaConsolePosPos
        vec4(0.0),
        // tex
        // ...
        // fxaaConsole360TexExpBiasNegOne
        // ...
        // fxaaConsole360TexExpBiasNegTwo
        // ...
        // fxaaQualityRcpFrame
        1.0 / screen_dimensions,
        // fxaaConsoleRcpFrameOpt
        vec4(0.0),
        // fxaaConsoleRcpFrameOpt2
        vec4(0.0),
        // fxaaConsole360RcpFrameOpt2
        vec4(0.0),
        // fxaaQualitySubpix
        0.75,
        // fxaaQualityEdgeThreshold
        0.166,
        // fxaaQualityEdgeThresholdMin
        0.0,
        // fxaaConsoleEdgeSharpness
        0.0,
        // fxaaConsoleEdgeThreshold
        0.0,
        // fxaaConsoleEdgeThresholdMin
        0.0,
        // fxaaConsole360ConstDir
        vec4(0.0)
    );
}
