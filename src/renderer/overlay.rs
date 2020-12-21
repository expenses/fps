use super::{alpha_blend_colour_descriptor, DynamicBuffer, Renderer, DEPTH_FORMAT, INDEX_FORMAT};
use ultraviolet::{Vec2, Vec4};
use wgpu::util::DeviceExt;

pub struct OverlayPipeline {
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    screen_dimension_uniform_buffer: wgpu::Buffer,
}

impl OverlayPipeline {
    pub fn new(renderer: &Renderer) -> Self {
        let window_size = renderer.window.inner_size();

        let screen_dimension_uniform_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("screen_dimension_uniform_buffer"),
                    contents: bytemuck::bytes_of(&Vec2::new(
                        window_size.width as f32,
                        window_size.height as f32,
                    )),
                    usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
                });

        let bind_group_layout =
            renderer
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("overlay bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("overlay bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &screen_dimension_uniform_buffer,
                        offset: 0,
                        size: None,
                    },
                }],
            });

        let pipeline_layout =
            renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("overlay pipeline layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let vs = wgpu::include_spirv!("../../shaders/compiled/overlay.vert.spv");
        let vs_module = renderer.device.create_shader_module(vs);
        let fs = wgpu::include_spirv!("../../shaders/compiled/overlay.frag.spv");
        let fs_module = renderer.device.create_shader_module(fs);

        let pipeline = renderer
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay pipeline"),
                layout: Some(&pipeline_layout),
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_module,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[alpha_blend_colour_descriptor()],
                depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilStateDescriptor::default(),
                }),
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: INDEX_FORMAT,
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float4],
                    }],
                },
                sample_count: 4,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

        Self {
            bind_group,
            pipeline,
            screen_dimension_uniform_buffer,
        }
    }

    pub fn resize(&self, renderer: &Renderer, width: u32, height: u32) {
        renderer.queue.write_buffer(
            &self.screen_dimension_uniform_buffer,
            0,
            bytemuck::bytes_of(&Vec2::new(width as f32, height as f32)),
        );
    }
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Vertex {
    position: Vec2,
    colour: Vec4,
}

use lyon_tessellation::{
    basic_shapes::stroke_circle,
    math::{rect, Point},
    BasicVertexConstructor, BuffersBuilder, FillOptions, StrokeAttributes, StrokeOptions,
    StrokeVertexConstructor, VertexBuffers,
};

struct Constructor {
    colour: Vec4,
}

impl StrokeVertexConstructor<Vertex> for Constructor {
    fn new_vertex(&mut self, point: Point, _: StrokeAttributes) -> Vertex {
        Vertex {
            position: Vec2::new(point.x, point.y),
            colour: self.colour,
        }
    }
}

impl BasicVertexConstructor<Vertex> for Constructor {
    fn new_vertex(&mut self, point: Point) -> Vertex {
        Vertex {
            position: Vec2::new(point.x, point.y),
            colour: self.colour,
        }
    }
}

pub struct OverlayBuffers {
    vertices: DynamicBuffer<Vertex>,
    indices: DynamicBuffer<u32>,
    lyon_buffers: VertexBuffers<Vertex, u16>,
}

impl OverlayBuffers {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            vertices: DynamicBuffer::new(
                device,
                1,
                "overlay vertex buffer",
                wgpu::BufferUsage::VERTEX,
            ),
            indices: DynamicBuffer::new(
                device,
                1,
                "overlay index buffer",
                wgpu::BufferUsage::INDEX,
            ),
            lyon_buffers: VertexBuffers::new(),
        }
    }

    pub fn draw_circle_outline(&mut self, position: Vec2, radius: f32) {
        stroke_circle(
            [position.x, position.y].into(),
            radius,
            &StrokeOptions::default(),
            &mut BuffersBuilder::new(
                &mut self.lyon_buffers,
                Constructor {
                    colour: Vec4::one(),
                },
            ),
        )
        .unwrap();

        self.buffer()
    }

    pub fn upload(&mut self, renderer: &Renderer) {
        self.vertices.upload(renderer);
        self.indices.upload(renderer);
    }

    pub fn get(&self) -> Option<(wgpu::BufferSlice, wgpu::BufferSlice, u32)> {
        match (self.vertices.get(), self.indices.get()) {
            (Some((vertices_slice, _)), Some((indices_slice, num_indices))) => {
                Some((vertices_slice, indices_slice, num_indices))
            }
            _ => None,
        }
    }

    fn buffer(&mut self) {
        let num_vertices = self.vertices.len_waiting();

        for vertex in self.lyon_buffers.vertices.drain(..) {
            self.vertices.push(vertex);
        }

        for index in self.lyon_buffers.indices.drain(..) {
            self.indices.push(index as u32 + num_vertices as u32);
        }
    }
}
