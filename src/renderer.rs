use crate::Settings;
use ultraviolet::{Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt;

pub mod debug_lines;
pub mod overlay;

const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;
const PRE_TONEMAP_FRAMEBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub texture_index: i32,
    pub emission_strength: f32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct AnimatedVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub texture_index: i32,
    pub emission_strength: f32,
    pub joints: [u16; 4],
    pub joint_weights: Vec4,
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
    pub identity_instance_buffer: wgpu::Buffer,
    pub linear_sampler: wgpu::Sampler,
    pub mipmap_generation_pipeline: wgpu::RenderPipeline,
    pub mipmap_generation_bind_group_layout: wgpu::BindGroupLayout,
    screen_dimension_uniform_buffer: wgpu::Buffer,
    pub animated_model_bind_group_layout: wgpu::BindGroupLayout,

    pub static_opaque_render_pipeline: wgpu::RenderPipeline,
    pub animated_opaque_render_pipeline: wgpu::RenderPipeline,
    pub static_transparent_render_pipeline: wgpu::RenderPipeline,
    pub animated_transparent_render_pipeline: wgpu::RenderPipeline,
    pub skybox_render_pipeline: wgpu::RenderPipeline,

    pub pre_fxaa_framebuffer: wgpu::TextureView,
    pub fxaa_bind_group: wgpu::BindGroup,
    pub pre_tonemap_framebuffer: wgpu::TextureView,
    pub tonemap_bind_group: wgpu::BindGroup,
    post_processing_bind_group_layout: wgpu::BindGroupLayout,
    pub fxaa_pipeline: wgpu::RenderPipeline,
    pub tonemap_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    pub async fn new(
        event_loop: &winit::event_loop::EventLoop<()>,
        _settings: &Settings,
    ) -> anyhow::Result<Self> {
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
                },
                None,
            )
            .await?;

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            label: Some("nearest sampler"),
            ..Default::default()
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            label: Some("linear sampler"),
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
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
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

        let animated_model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("animated model bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
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
                ],
            });

        let vs_static_model = wgpu::include_spirv!("../shaders/compiled/static_model.vert.spv");
        let vs_static_model_module = device.create_shader_module(&vs_static_model);

        let vs_animated_model = wgpu::include_spirv!("../shaders/compiled/animated_model.vert.spv");
        let vs_animated_model_module = device.create_shader_module(&vs_animated_model);

        let fs_model = wgpu::include_spirv!("../shaders/compiled/model.frag.spv");
        let fs_model_module = device.create_shader_module(&fs_model);

        let vs_skybox = wgpu::include_spirv!("../shaders/compiled/skybox.vert.spv");
        let vs_skybox_module = device.create_shader_module(&vs_skybox);
        let fs_skybox = wgpu::include_spirv!("../shaders/compiled/skybox.frag.spv");
        let fs_skybox_module = device.create_shader_module(&fs_skybox);

        let static_model_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("static model pipeline layout"),
                bind_group_layouts: &[
                    &main_bind_group_layout,
                    &texture_array_bind_group_layout,
                    &lights_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let animated_model_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("static model pipeline layout"),
                bind_group_layouts: &[
                    &main_bind_group_layout,
                    &animated_model_bind_group_layout,
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

        let static_opaque_render_pipeline = create_render_pipeline(
            &device,
            "static opaque render pipeline",
            &static_model_pipeline_layout,
            &vs_static_model_module,
            &fs_model_module,
            PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            true,
            false,
        );

        let animated_opaque_render_pipeline = create_render_pipeline(
            &device,
            "animated opaque render pipeline",
            &animated_model_pipeline_layout,
            &vs_animated_model_module,
            &fs_model_module,
            PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            true,
            true,
        );

        let static_transparent_render_pipeline = create_render_pipeline(
            &device,
            "static transparent render pipeline",
            &static_model_pipeline_layout,
            &vs_static_model_module,
            &fs_model_module,
            alpha_blend_colour_descriptor(),
            // Can't remember if this is a good idea or not.
            false,
            false,
        );

        let animated_transparent_render_pipeline = create_render_pipeline(
            &device,
            "animated transparent render pipeline",
            &animated_model_pipeline_layout,
            &vs_animated_model_module,
            &fs_model_module,
            alpha_blend_colour_descriptor(),
            // Can't remember if this is a good idea or not.
            false,
            true,
        );

        let skybox_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skybox render pipeline"),
                layout: Some(&skybox_render_pipeline_layout),
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_skybox_module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_skybox_module,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[PRE_TONEMAP_FRAMEBUFFER_FORMAT.into()],
                depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Equal,
                    stencil: wgpu::StencilStateDescriptor::default(),
                }),
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: None,
                    vertex_buffers: &[],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

        let identity_instance_buffer = single_instance_buffer(
            &device,
            Instance {
                transform: Mat4::identity(),
            },
            "identity instance buffer",
        );

        let mipmap_generation_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox texture bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
                        },
                        count: None,
                    },
                ],
            });

        let mipmap_generation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mipmap generation pipeline layout"),
                bind_group_layouts: &[&mipmap_generation_bind_group_layout],
                push_constant_ranges: &[],
            });

        let vs_full_screen_tri =
            wgpu::include_spirv!("../shaders/compiled/full_screen_tri.vert.spv");
        let vs_full_screen_tri_module = device.create_shader_module(&vs_full_screen_tri);

        let fs_blit_mipmap = wgpu::include_spirv!("../shaders/compiled/blit_mipmap.frag.spv");
        let fs_blit_mipmap_module = device.create_shader_module(&fs_blit_mipmap);

        let mipmap_generation_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mipmap generation pipeline"),
                layout: Some(&mipmap_generation_pipeline_layout),
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_full_screen_tri_module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_blit_mipmap_module,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[TEXTURE_FORMAT.into()],
                depth_stencil_state: None,
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: Some(INDEX_FORMAT),
                    vertex_buffers: &[],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

        let screen_dimension_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("screen_dimension_uniform_buffer"),
                contents: bytemuck::bytes_of(&Vec2::new(
                    window_size.width as f32,
                    window_size.height as f32,
                )),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

        let post_processing_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post processing bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let (pre_tonemap_framebuffer, tonemap_bind_group) =
            post_processing_framebuffer_and_bind_group(
                &device,
                window_size.width,
                window_size.height,
                "tonemap",
                &post_processing_bind_group_layout,
                &linear_sampler,
                &screen_dimension_uniform_buffer,
                DISPLAY_FORMAT,
            );

        let (pre_fxaa_framebuffer, fxaa_bind_group) = post_processing_framebuffer_and_bind_group(
            &device,
            window_size.width,
            window_size.height,
            "fxaa",
            &post_processing_bind_group_layout,
            &linear_sampler,
            &screen_dimension_uniform_buffer,
            PRE_TONEMAP_FRAMEBUFFER_FORMAT,
        );

        let post_processing_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post processing pipeline layout"),
                bind_group_layouts: &[&post_processing_bind_group_layout],
                push_constant_ranges: &[],
            });

        let fs_fxaa = wgpu::include_spirv!("../shaders/compiled/fxaa.frag.spv");
        let fs_fxaa_module = device.create_shader_module(&fs_fxaa);

        let fxaa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fxaa pipeline"),
            layout: Some(&post_processing_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_full_screen_tri_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_fxaa_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[DISPLAY_FORMAT.into()],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: Some(INDEX_FORMAT),
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let fs_tonemap = wgpu::include_spirv!("../shaders/compiled/tonemap.frag.spv");
        let fs_tonemap_module = device.create_shader_module(&fs_tonemap);

        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap pipeline"),
            layout: Some(&post_processing_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_full_screen_tri_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_tonemap_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[DISPLAY_FORMAT.into()],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: Some(INDEX_FORMAT),
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

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
            static_opaque_render_pipeline,
            static_transparent_render_pipeline,
            identity_instance_buffer,
            texture_array_bind_group_layout,
            skybox_texture_bind_group_layout,
            lights_bind_group_layout,
            skybox_render_pipeline,
            main_bind_group_layout,
            mipmap_generation_bind_group_layout,
            mipmap_generation_pipeline,
            linear_sampler,
            screen_dimension_uniform_buffer,
            animated_opaque_render_pipeline,
            animated_transparent_render_pipeline,
            animated_model_bind_group_layout,

            post_processing_bind_group_layout,
            pre_fxaa_framebuffer,
            fxaa_bind_group,
            fxaa_pipeline,
            pre_tonemap_framebuffer,
            tonemap_bind_group,
            tonemap_pipeline,
        })
    }

    pub fn set_camera_view(&self, view: Mat4) {
        self.queue
            .write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&view));
    }

    pub fn resize(&mut self, width: u32, height: u32, _settings: &Settings) {
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
        );

        self.queue.write_buffer(
            &self.perspective_buffer,
            0,
            bytemuck::bytes_of(&perspective_matrix(width, height)),
        );

        self.queue.write_buffer(
            &self.screen_dimension_uniform_buffer,
            0,
            bytemuck::bytes_of(&Vec2::new(width as f32, height as f32)),
        );

        let (pre_fxaa_framebuffer, fxaa_bind_group) = post_processing_framebuffer_and_bind_group(
            &self.device,
            width,
            height,
            "fxaa",
            &self.post_processing_bind_group_layout,
            &self.linear_sampler,
            &self.screen_dimension_uniform_buffer,
            DISPLAY_FORMAT,
        );
        self.pre_fxaa_framebuffer = pre_fxaa_framebuffer;
        self.fxaa_bind_group = fxaa_bind_group;

        let (pre_tonemap_framebuffer, tonemap_bind_group) =
            post_processing_framebuffer_and_bind_group(
                &self.device,
                width,
                height,
                "tonemap",
                &self.post_processing_bind_group_layout,
                &self.linear_sampler,
                &self.screen_dimension_uniform_buffer,
                PRE_TONEMAP_FRAMEBUFFER_FORMAT,
            );
        self.pre_tonemap_framebuffer = pre_tonemap_framebuffer;
        self.tonemap_bind_group = tonemap_bind_group;
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
            sample_count: 1,
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

pub fn alpha_blend_colour_descriptor() -> wgpu::ColorStateDescriptor {
    wgpu::ColorStateDescriptor {
        format: PRE_TONEMAP_FRAMEBUFFER_FORMAT,
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

const STATIC_VERTEX_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 5] =
    wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float2, 3 => Int, 4 => Float];
const STATIC_INSTANCE_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 4] =
    wgpu::vertex_attr_array![5 => Float4, 6 => Float4, 7 => Float4, 8 => Float4];

const ANIMATED_VERTEX_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 7] = wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float2, 3 => Int, 4 => Float, 5 => Ushort4, 6 => Float4];
const ANIMATED_INSTANCE_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 4] =
    wgpu::vertex_attr_array![7 => Float4, 8 => Float4, 9 => Float4, 10 => Float4];

fn create_render_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    vs_module: &wgpu::ShaderModule,
    fs_module: &wgpu::ShaderModule,
    colour_descriptor: wgpu::ColorStateDescriptor,
    depth_write_enabled: bool,
    animated: bool,
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
            depth_compare:  wgpu::CompareFunction::Less,
            stencil: wgpu::StencilStateDescriptor::default(),
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: Some(INDEX_FORMAT),
            vertex_buffers: &[
                wgpu::VertexBufferDescriptor {
                    stride: if animated {
                        std::mem::size_of::<AnimatedVertex>()
                    } else {
                        std::mem::size_of::<Vertex>()
                    } as u64,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &if animated {
                        &ANIMATED_VERTEX_ATTR_ARRAY[..]
                    } else {
                        &STATIC_VERTEX_ATTR_ARRAY[..]
                    },
                },
                wgpu::VertexBufferDescriptor {
                    stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::InputStepMode::Instance,
                    attributes: &if animated {
                        ANIMATED_INSTANCE_ATTR_ARRAY
                    } else {
                        STATIC_INSTANCE_ATTR_ARRAY
                    },
                },
            ],
        },
        sample_count: 1,
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
        emission_strength: 0.0,
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
    // the offset of `waiting` to upload.
    upload_offset: usize,
}

impl<T: bytemuck::Pod> DynamicBuffer<T> {
    pub fn new(
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
            upload_offset: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        self.waiting.push(item)
    }

    // Upload the waiting buffer to the gpu. Returns whether the gpu buffer was resized.
    pub fn upload(&mut self, renderer: &Renderer) -> bool {
        if self.waiting.is_empty() {
            self.len = 0;
            return false;
        }

        if self.waiting.len() == self.upload_offset {
            return false;
        }

        let size_of_t = std::mem::size_of::<T>() as u64;

        self.len = self.waiting.len();

        let return_bool = if self.waiting.len() <= self.capacity {
            let bytes = bytemuck::cast_slice(&self.waiting[self.upload_offset..]);
            renderer
                .queue
                .write_buffer(&self.buffer, size_of_t * self.upload_offset as u64, bytes);
            false
        } else {
            let bytes = bytemuck::cast_slice(&self.waiting);
            self.capacity = (self.capacity * 2).max(self.waiting.len());
            log::debug!(
                "Resizing '{}' to {} items to fit {} items",
                self.label,
                self.capacity,
                self.len
            );
            self.buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.capacity as u64 * size_of_t,
                usage: self.usage | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: true,
            });
            self.buffer
                .slice(..bytes.len() as u64)
                .get_mapped_range_mut()
                .copy_from_slice(bytes);
            self.buffer.unmap();
            true
        };

        self.upload_offset = self.waiting.len();

        return_bool
    }

    pub fn pop(&mut self) {
        if self.waiting.pop().is_some() {
            self.upload_offset -= 1;
        }
    }

    pub fn clear(&mut self) {
        self.waiting.clear();
        self.upload_offset = 0;
    }

    pub fn get(&self) -> Option<(wgpu::BufferSlice, u32)> {
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

fn post_processing_framebuffer_and_bind_group(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    name: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    screen_dimension_uniform_buffer: &wgpu::Buffer,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    let framebuffer = create_texture(
        &device,
        &format!("pre-{} framebuffer", name),
        width,
        height,
        texture_format,
        wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{} bind group", name)),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&framebuffer),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer {
                    buffer: screen_dimension_uniform_buffer,
                    offset: 0,
                    size: None,
                },
            },
        ],
    });

    (framebuffer, bind_group)
}
