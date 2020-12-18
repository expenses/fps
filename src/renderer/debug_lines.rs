use super::{alpha_blend_colour_descriptor, Renderer, DEPTH_FORMAT, INDEX_FORMAT, SAMPLE_COUNT};
use ultraviolet::{Vec3, Vec4};

pub fn debug_lines_pipeline(renderer: &Renderer) -> wgpu::RenderPipeline {
    let debug_lines_pipeline_layout =
        renderer
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("debug lines pipeline layout"),
                bind_group_layouts: &[&renderer.main_bind_group_layout],
                push_constant_ranges: &[],
            });

    let vs = wgpu::include_spirv!("../../shaders/compiled/debug_lines.vert.spv");
    let vs_module = renderer.device.create_shader_module(vs);
    let fs = wgpu::include_spirv!("../../shaders/compiled/debug_lines.frag.spv");
    let fs_module = renderer.device.create_shader_module(fs);

    renderer
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("debug lines pipeline"),
            layout: Some(&debug_lines_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                cull_mode: wgpu::CullMode::Back,
                ..Default::default()
            }),
            primitive_topology: wgpu::PrimitiveTopology::LineList,
            color_states: &[alpha_blend_colour_descriptor()],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilStateDescriptor::default(),
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: INDEX_FORMAT,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float4],
                }],
            },
            sample_count: SAMPLE_COUNT,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        })
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Vertex {
    position: Vec3,
    colour: Vec4,
}

pub fn draw_line(a: Vec3, b: Vec3, colour: Vec4, mut func: impl FnMut(Vertex)) {
    func(Vertex {
        position: a,
        colour,
    });
    func(Vertex {
        position: b,
        colour,
    });
}

pub fn draw_tri(a: Vec3, b: Vec3, c: Vec3, colour: Vec4, mut func: impl FnMut(Vertex)) {
    draw_line(a, b, colour, &mut func);
    draw_line(b, c, colour, &mut func);
    draw_line(c, a, colour, &mut func);
}
