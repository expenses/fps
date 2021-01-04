use super::{alpha_blend_colour_descriptor, Renderer, DEPTH_FORMAT, INDEX_FORMAT};
use crate::Settings;
use ultraviolet::{Mat4, Vec3, Vec4};

pub fn debug_lines_pipeline(
    renderer: &Renderer,
    _settings: &Settings,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    let debug_lines_pipeline_layout =
        renderer
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("debug lines pipeline layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStage::VERTEX,
                    range: 0..std::mem::size_of::<Mat4>() as u32,
                }],
            });

    let shader = wgpu::include_spirv!("../../shaders/debug_lines/target/spirv-unknown-unknown/release/debug_lines_shader.spv");
    let shader_module = renderer.device.create_shader_module(&shader);

    renderer
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("debug lines pipeline"),
            layout: Some(&debug_lines_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader_module,
                entry_point: "main_vs",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &shader_module,
                entry_point: "main_fs",
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
                depth_compare,
                stencil: wgpu::StencilStateDescriptor::default(),
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: Some(INDEX_FORMAT),
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float4],
                }],
            },
            sample_count: 1,
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

pub fn draw_marker(at: Vec3, size: f32, mut func: impl FnMut(Vertex)) {
    draw_line(
        at - Vec3::unit_x() * size / 2.0,
        at + Vec3::unit_x() * size / 2.0,
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        &mut func,
    );
    draw_line(
        at - Vec3::unit_y() * size / 2.0,
        at + Vec3::unit_y() * size / 2.0,
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        &mut func,
    );
    draw_line(
        at - Vec3::unit_z() * size / 2.0,
        at + Vec3::unit_z() * size / 2.0,
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        &mut func,
    );
}
