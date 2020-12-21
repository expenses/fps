use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

pub mod debug_lines;
pub mod overlay;

const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;
const SAMPLE_COUNT: u32 = 4;

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub texture_index: i32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Instance {
    pub transform: Mat4,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub main_bind_group_layout: wgpu::BindGroupLayout,
    pub lights_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_array_bind_group_layout: wgpu::BindGroupLayout,
    pub skybox_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub window: winit::window::Window,
    pub swap_chain: wgpu::SwapChain,
    pub depth_texture: wgpu::TextureView,
    surface: wgpu::Surface,
    view_buffer: wgpu::Buffer,
    perspective_buffer: wgpu::Buffer,
    pub main_bind_group: wgpu::BindGroup,
    pub multisampled_framebuffer_texture: wgpu::TextureView,
    pub identity_instance_buffer: wgpu::Buffer,
    pub opaque_render_pipeline: wgpu::RenderPipeline,
    pub transparent_render_pipeline: wgpu::RenderPipeline,
    pub skybox_render_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    pub async fn new(event_loop: &winit::event_loop::EventLoop<()>) -> anyhow::Result<Self> {
        let window = winit::window::WindowBuilder::new().build(event_loop)?;

        window.set_cursor_visible(false);

        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };

        window.set_cursor_grab(true)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or_else(|| anyhow::anyhow!(
                "'request_adapter' failed. If you get this on linux, try installing the vulkan drivers for your gpu. \
                You can check that they're working properly by running `vulkaninfo` or `vkcube`."
            ))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await?;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            label: Some("nearest sampler"),
            ..Default::default()
        });

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view buffer"),
            contents: bytemuck::bytes_of(&Mat4::identity()),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let window_size = window.inner_size();

        let perspective_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perspective buffer"),
            contents: bytemuck::bytes_of(&perspective_matrix(
                window_size.width,
                window_size.height,
            )),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("main bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
                        },
                        count: None,
                    },
                ],
            });

        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("main bind group"),
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: perspective_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: view_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let swap_chain = device.create_swap_chain(
            &surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                format: DISPLAY_FORMAT,
                width: window_size.width,
                height: window_size.height,
                present_mode: wgpu::PresentMode::Fifo,
            },
        );

        let depth_texture = create_texture(
            &device,
            "depth texture",
            window_size.width,
            window_size.height,
            DEPTH_FORMAT,
            wgpu::TextureUsage::RENDER_ATTACHMENT,
            SAMPLE_COUNT,
        );

        let multisampled_framebuffer_texture = create_texture(
            &device,
            "multisampled framebuffer texture",
            window_size.width,
            window_size.height,
            DISPLAY_FORMAT,
            wgpu::TextureUsage::RENDER_ATTACHMENT,
            SAMPLE_COUNT,
        );

        let texture_array_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture array bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let lights_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lights bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let skybox_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox texture bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let vs_model = wgpu::include_spirv!("../shaders/compiled/scene.vert.spv");
        let vs_model_module = device.create_shader_module(vs_model);
        let fs_model = wgpu::include_spirv!("../shaders/compiled/scene.frag.spv");
        let fs_model_module = device.create_shader_module(fs_model);

        let vs_skybox = wgpu::include_spirv!("../shaders/compiled/skybox.vert.spv");
        let vs_skybox_module = device.create_shader_module(vs_skybox);
        let fs_skybox = wgpu::include_spirv!("../shaders/compiled/skybox.frag.spv");
        let fs_skybox_module = device.create_shader_module(fs_skybox);

        let model_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[
                    &main_bind_group_layout,
                    &texture_array_bind_group_layout,
                    &lights_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let skybox_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox pipeline layout"),
                bind_group_layouts: &[&main_bind_group_layout, &skybox_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let opaque_render_pipeline = create_render_pipeline(
            &device,
            "opaque render pipeline",
            &model_render_pipeline_layout,
            &vs_model_module,
            &fs_model_module,
            replace_colour_descriptor(),
            true,
            wgpu::CompareFunction::Less,
        );

        let transparent_render_pipeline = create_render_pipeline(
            &device,
            "transparent render pipeline",
            &model_render_pipeline_layout,
            &vs_model_module,
            &fs_model_module,
            alpha_blend_colour_descriptor(),
            // Can't remember if this is a good idea or not.
            false,
            wgpu::CompareFunction::Less,
        );

        let skybox_render_pipeline = create_render_pipeline(
            &device,
            "skybox render pipeline",
            &skybox_render_pipeline_layout,
            &vs_skybox_module,
            &fs_skybox_module,
            replace_colour_descriptor(),
            true,
            wgpu::CompareFunction::Equal,
        );

        let identity_instance_buffer = single_instance_buffer(
            &device,
            Instance {
                transform: Mat4::identity(),
            },
            "identity instance buffer",
        );

        Ok(Self {
            device,
            queue,
            window,
            view_buffer,
            perspective_buffer,
            main_bind_group,
            surface,
            swap_chain,
            depth_texture,
            opaque_render_pipeline,
            transparent_render_pipeline,
            multisampled_framebuffer_texture,
            identity_instance_buffer,
            texture_array_bind_group_layout,
            skybox_texture_bind_group_layout,
            lights_bind_group_layout,
            skybox_render_pipeline,
            main_bind_group_layout,
        })
    }

    pub fn set_camera_view(&self, view: Mat4) {
        self.queue
            .write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&view));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.swap_chain = self.device.create_swap_chain(
            &self.surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                format: DISPLAY_FORMAT,
                width: width,
                height: height,
                present_mode: wgpu::PresentMode::Fifo,
            },
        );

        self.depth_texture = create_texture(
            &self.device,
            "depth texture",
            width,
            height,
            DEPTH_FORMAT,
            wgpu::TextureUsage::RENDER_ATTACHMENT,
            SAMPLE_COUNT,
        );

        self.multisampled_framebuffer_texture = create_texture(
            &self.device,
            "multisampled framebuffer texture",
            width,
            height,
            DISPLAY_FORMAT,
            wgpu::TextureUsage::RENDER_ATTACHMENT,
            SAMPLE_COUNT,
        );

        self.queue.write_buffer(
            &self.perspective_buffer,
            0,
            bytemuck::bytes_of(&perspective_matrix(width, height)),
        );
    }

    pub fn screen_center(&self) -> winit::dpi::LogicalPosition<f64> {
        let window_size = self.window.inner_size();
        winit::dpi::LogicalPosition::new(
            window_size.width as f64 / 2.0,
            window_size.height as f64 / 2.0,
        )
    }
}

fn perspective_matrix(width: u32, height: u32) -> Mat4 {
    ultraviolet::projection::perspective_wgpu_dx(
        // http://themetalmuncher.github.io/fov-calc/
        59.0_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        2500.0,
    )
}

fn create_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsage,
    sample_count: u32,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn single_instance_buffer(
    device: &wgpu::Device,
    instance: Instance,
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsage::VERTEX,
    })
}

fn replace_colour_descriptor() -> wgpu::ColorStateDescriptor {
    wgpu::ColorStateDescriptor {
        format: DISPLAY_FORMAT,
        color_blend: wgpu::BlendDescriptor::REPLACE,
        alpha_blend: wgpu::BlendDescriptor::REPLACE,
        write_mask: wgpu::ColorWrite::ALL,
    }
}

pub fn alpha_blend_colour_descriptor() -> wgpu::ColorStateDescriptor {
    wgpu::ColorStateDescriptor {
        format: DISPLAY_FORMAT,
        color_blend: wgpu::BlendDescriptor {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha_blend: wgpu::BlendDescriptor {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
        write_mask: wgpu::ColorWrite::ALL,
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    vs_module: &wgpu::ShaderModule,
    fs_module: &wgpu::ShaderModule,
    colour_descriptor: wgpu::ColorStateDescriptor,
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            cull_mode: wgpu::CullMode::Back,
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[colour_descriptor],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: DEPTH_FORMAT,
            depth_write_enabled,
            depth_compare,
            stencil: wgpu::StencilStateDescriptor::default(),
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: INDEX_FORMAT,
            vertex_buffers: &[
                wgpu::VertexBufferDescriptor {
                    stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float2, 3 => Int],
                },
                wgpu::VertexBufferDescriptor {
                    stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::InputStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![4 => Float4, 5 => Float4, 6 => Float4, 7 => Float4],
                },
            ],
        },
        sample_count: SAMPLE_COUNT,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}

pub enum Decal {
    Shadow,
    BulletImpact,
}

impl Decal {
    fn uvs(&self) -> (Vec2, Vec2) {
        let (uv_offset, uv_size) = match self {
            Self::Shadow => (Vec2::zero(), Vec2::broadcast(0.5)),
            Self::BulletImpact => (Vec2::new(0.5, 0.0), Vec2::broadcast(0.25)),
        };

        (uv_offset, uv_size)
    }
}

pub fn decal_square(position: Vec3, normal: Vec3, size: Vec2, decal: Decal) -> [Vertex; 6] {
    let offset = size / 2.0;

    let rotation = if normal == Vec3::new(0.0, -1.0, 0.0) {
        // The above case NaN's the rotor so we just use a matrix instead.
        ultraviolet::Mat3::from_rotation_x(180.0_f32.to_radians())
    } else {
        ultraviolet::Rotor3::from_rotation_between(Vec3::unit_y(), normal).into_matrix()
    };

    let mut offsets = [
        Vec3::new(-offset.x, 0.0, -offset.y), // top left
        Vec3::new(offset.x, 0.0, -offset.y),  // top right
        Vec3::new(-offset.x, 0.0, offset.y),  // bottom left
        Vec3::new(offset.x, 0.0, offset.y),   // bottom right
    ];

    for i in 0..4 {
        offsets[i] = rotation * offsets[i];
    }

    let (uv_offset, uv_size) = decal.uvs();

    let uvs = [
        Vec2::zero() + uv_offset,
        Vec2::new(uv_size.x, 0.0) + uv_offset,
        Vec2::new(0.0, uv_size.y) + uv_offset,
        uv_size + uv_offset,
    ];

    let vertex = |index| Vertex {
        position: position + offsets[index],
        normal,
        uv: uvs[index],
        texture_index: 0,
    };

    [
        vertex(1),
        vertex(0),
        vertex(2),
        vertex(1),
        vertex(2),
        vertex(3),
    ]
}

pub struct DynamicBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    capacity: usize,
    len: usize,
    label: &'static str,
    waiting: Vec<T>,
    usage: wgpu::BufferUsage,
}

impl<T: bytemuck::Pod> DynamicBuffer<T> {
    fn new(
        device: &wgpu::Device,
        base_capacity: usize,
        label: &'static str,
        usage: wgpu::BufferUsage,
    ) -> Self {
        Self {
            capacity: base_capacity,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (base_capacity * std::mem::size_of::<T>()) as u64,
                usage: usage | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }),
            len: 0,
            label,
            waiting: Vec::with_capacity(base_capacity),
            usage,
        }
    }

    pub fn push(&mut self, item: T) {
        self.waiting.push(item)
    }

    // Upload the waiting buffer to the gpu. Returns whether the gpu buffer was resized.
    fn upload(&mut self, renderer: &Renderer) -> bool {
        if self.waiting.is_empty() {
            self.len = 0;
            return false;
        }

        self.len = self.waiting.len();
        let bytes = bytemuck::cast_slice(&self.waiting);

        if self.waiting.len() <= self.capacity {
            renderer.queue.write_buffer(&self.buffer, 0, bytes);
            self.waiting.clear();
            false
        } else {
            self.capacity = (self.capacity * 2).max(self.waiting.len());
            self.buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: (self.capacity * std::mem::size_of::<T>()) as u64,
                usage: self.usage | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: true,
            });
            self.buffer
                .slice(..bytes.len() as u64)
                .get_mapped_range_mut()
                .copy_from_slice(bytes);
            self.buffer.unmap();
            self.waiting.clear();
            true
        }
    }

    fn get(&self) -> Option<(wgpu::BufferSlice, u32)> {
        if self.len > 0 {
            let byte_len = (self.len * std::mem::size_of::<T>()) as u64;

            Some((self.buffer.slice(..byte_len), self.len as u32))
        } else {
            None
        }
    }

    pub fn len_waiting(&self) -> usize {
        self.waiting.len()
    }
}
