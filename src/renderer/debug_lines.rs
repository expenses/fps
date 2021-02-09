use super::{alpha_blend_colour_target_state, load_shader, Renderer, DEPTH_FORMAT};
use ultraviolet::{Mat4, Vec3, Vec4};

pub struct DebugLinesPipelines {
    pub always: wgpu::RenderPipeline,
    pub less: wgpu::RenderPipeline,
}

impl DebugLinesPipelines {
    pub fn new(renderer: &Renderer) -> anyhow::Result<Self> {
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

        let vs_less = load_shader(
            "shaders/compiled/debug_lines_less.vert.spv",
            &renderer.device,
        )?;
        let vs_always = load_shader(
            "shaders/compiled/debug_lines_always.vert.spv",
            &renderer.device,
        )?;
        let fs = load_shader("shaders/compiled/debug_lines.frag.spv", &renderer.device)?;

        Ok(Self {
            always: renderer
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("debug lines always pipeline"),
                    layout: Some(&debug_lines_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vs_always,
                        entry_point: "main",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as u64,
                            step_mode: wgpu::InputStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float4],
                        }],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs,
                        entry_point: "main",
                        targets: &[alpha_blend_colour_target_state()],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::LineList,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                        clamp_depth: false,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                }),
            less: renderer
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("debug lines less pipeline"),
                    layout: Some(&debug_lines_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vs_less,
                        entry_point: "main",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as u64,
                            step_mode: wgpu::InputStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float4],
                        }],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs,
                        entry_point: "main",
                        targets: &[alpha_blend_colour_target_state()],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::LineList,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                        clamp_depth: false,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                }),
        })
    }
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
