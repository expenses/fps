#version 450

layout(location = 0) out vec3 out_uv;

layout(set = 0, binding = 0) uniform Perspective {
    mat4 perspective;
};

layout(set = 0, binding = 1) uniform View {
    mat4 view;
};

void main() {
    // https://github.com/gfx-rs/wgpu-rs/blob/master/examples/skybox/shader.vert

    vec4 pos = vec4(0.0);

    switch(gl_VertexIndex) {
        case 0: pos = vec4(-1.0, -1.0, 0.0, 1.0); break;
        case 1: pos = vec4( 3.0, -1.0, 0.0, 1.0); break;
        case 2: pos = vec4(-1.0,  3.0, 0.0, 1.0); break;
    }

    mat3 inv_view = transpose(mat3(view));
    vec3 unprojected = (inverse(perspective) * pos).xyz;
    out_uv = inv_view * unprojected;

    // https://learnopengl.com/Advanced-OpenGL/Cubemaps 'An optimisation'
    gl_Position = pos.xyww;
}
