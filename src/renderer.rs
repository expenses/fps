use crate::Settings;
use ultraviolet::{Mat3, Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt;

pub mod debug_lines;
pub mod overlay;

pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;
const PRE_TONEMAP_FRAMEBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
pub const UNCOMPRESSED_LIGHTMAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
pub const LIGHTVOL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub texture_index: u32,
    pub emission_strength: f32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct LevelVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub texture_index: u32,
    pub emission_strength: f32,
    pub lightmap_uv: Vec2,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct AnimatedVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub texture_index: u32,
    pub emission_strength: f32,
    pub joints: [u16; 4],
    pub joint_weights: Vec4,
}

const STATIC_VERTEX_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 5] =
    wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float2, 3 => Uint, 4 => Float];
const STATIC_INSTANCE_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 7] = wgpu::vertex_attr_array![5 => Float4, 6 => Float4, 7 => Float4, 8 => Float4, 9 => Float3, 10 => Float3, 11 => Float3];

const ANIMATED_VERTEX_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 7] = wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float2, 3 => Uint, 4 => Float, 5 => Ushort4, 6 => Float4];
const ANIMATED_INSTANCE_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 6] = wgpu::vertex_attr_array![7 => Float4, 8 => Float4, 9 => Float4, 10 => Float4, 11 => Uint, 12 => Uint];

const LEVEL_VERTEX_ATTR_ARRAY: [wgpu::VertexAttributeDescriptor; 6] = wgpu::vertex_attr_array![0 => Float3, 1 => Float3, 2 => Float2, 3 => Uint, 4 => Float, 5 => Float2];

pub fn normal_matrix(transform: Mat4) -> Mat3 {
    let inverse_transpose = transform.inversed().transposed();
    let array = inverse_transpose.as_component_array();
    Mat3::new(array[0].xyz(), array[1].xyz(), array[2].xyz())
}

// An instance with a cached normal transform
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Instance {
    transform: Mat4,
    normal_transform: Mat3,
}

impl Instance {
    pub fn new(transform: Mat4) -> Self {
        Self {
            transform,
            normal_transform: normal_matrix(transform),
        }
    }
}

// We can't cache normal transforms for animated instances.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct AnimatedInstance {
    transform: Mat4,
    model_index: u32,
    num_joints: u32,
}

impl AnimatedInstance {
    pub fn new(transform: Mat4, model_index: usize, num_joints: u32) -> Self {
        Self {
            transform,
            model_index: model_index as u32,
            num_joints,
        }
    }
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct LightVolUniforms {
    position: Vec3,
    padding: u32,
    scale: Vec3,
}

impl LightVolUniforms {
    pub fn new(position: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            scale,
            padding: 0,
        }
    }
}

fn load_shader(filename: &str, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
    let bytes = std::fs::read(filename)?;

    Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some(filename),
        source: wgpu::util::make_spirv(&bytes),
        flags: wgpu::ShaderFlags::VALIDATION,
    }))
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    main_bind_group_layout: wgpu::BindGroupLayout,
    pub lights_bind_group_layout: wgpu::BindGroupLayout,
    pub array_of_textures_bind_group_layout: wgpu::BindGroupLayout,
    pub skybox_texture_bind_group_layout: wgpu::BindGroupLayout,
    pub window: winit::window::Window,
    pub swap_chain: wgpu::SwapChain,
    pub depth_texture: wgpu::TextureView,
    surface: wgpu::Surface,

    pub projection: Mat4,
    pub projection_view: Mat4,
    pub view: Mat4,
    pub screen_dimensions: Vec2,

    pub main_bind_group: wgpu::BindGroup,
    pub identity_instance_buffer: wgpu::Buffer,
    pub linear_sampler: wgpu::Sampler,
    pub mipmap_generation_pipeline: wgpu::RenderPipeline,
    pub mipmap_generation_bind_group_layout: wgpu::BindGroupLayout,

    pub animated_models_bind_group_layout: wgpu::BindGroupLayout,

    vs_model_static: wgpu::ShaderModule,
    vs_model_static_gun: wgpu::ShaderModule,
    vs_model_animated: wgpu::ShaderModule,

    fs_model: wgpu::ShaderModule,
    fs_model_alpha_clip: wgpu::ShaderModule,

    vs_level: wgpu::ShaderModule,

    fs_level: wgpu::ShaderModule,
    fs_level_alpha_clip: wgpu::ShaderModule,

    pub render_pipelines: RenderPipelines,

    pub skybox_render_pipeline: wgpu::RenderPipeline,

    pub pre_fxaa_framebuffer: wgpu::TextureView,
    pub fxaa_bind_group: wgpu::BindGroup,
    pub pre_tonemap_framebuffer: wgpu::TextureView,
    pub tonemap_bind_group: wgpu::BindGroup,
    pub post_processing_bind_group_layout: wgpu::BindGroupLayout,
    pub fxaa_pipeline: wgpu::RenderPipeline,
    pub tonemap_pipeline: wgpu::RenderPipeline,
    pub texture_dilation_pipeline: wgpu::RenderPipeline,

    pub bake_lightmap_pipeline: wgpu::RenderPipeline,
    pub lightmap_bind_group_layout: wgpu::BindGroupLayout,

    pub bake_lightvol_bind_group_layout: wgpu::BindGroupLayout,
    pub bake_lightvol_pipeline: wgpu::ComputePipeline,

    pub bc6h_compressor_2d: wgpu_bc6h_compression::Compressor2D,
    pub bc6h_compressor_3d: wgpu_bc6h_compression::Compressor3D,

    display_format: wgpu::TextureFormat,
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
                    features: wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY
                        | wgpu::Features::UNSIZED_BINDING_ARRAY
                        | wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::MULTI_DRAW_INDIRECT
                        | wgpu::Features::TEXTURE_COMPRESSION_BC,
                    limits: wgpu::Limits {
                        max_push_constant_size: std::mem::size_of::<[Mat4; 2]>() as u32,
                        // todo: set this to something crazy high.
                        max_sampled_textures_per_shader_stage: 32,
                        max_storage_textures_per_shader_stage: 6,
                        ..Default::default()
                    },
                },
                None,
            )
            .await?;

        let display_format = device.get_swap_chain_preferred_format();

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

        let window_size = window.inner_size();
        let width = window_size.width;
        let height = window_size.height;

        let projection = perspective_matrix(width, height);
        let projection_view = Mat4::identity();
        let view = Mat4::identity();
        let screen_dimensions = Vec2::new(width as f32, height as f32);

        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("main bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
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

        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("main bind group"),
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });

        let swap_chain = device.create_swap_chain(
            &surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                format: display_format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
            },
        );

        let depth_texture = create_texture(
            &device,
            "depth texture",
            width,
            height,
            DEPTH_FORMAT,
            wgpu::TextureUsage::RENDER_ATTACHMENT,
        );

        let lights_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lights bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // light vol consisting of 6 textures
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: Some(std::num::NonZeroU32::new(6).unwrap()),
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

        let animated_models_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("animated models bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let lightmap_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lightmap bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let vs_model_static = load_shader("shaders/compiled/model_static.vert.spv", &device)?;
        let vs_model_animated = load_shader("shaders/compiled/model_animated.vert.spv", &device)?;

        let fs_model = load_shader("shaders/compiled/model.frag.spv", &device)?;

        let fs_model_alpha_clip =
            load_shader("shaders/compiled/model_alpha_clip.frag.spv", &device)?;

        let vs_skybox = load_shader("shaders/compiled/skybox.vert.spv", &device)?;
        let fs_skybox = load_shader("shaders/compiled/skybox.frag.spv", &device)?;

        let vs_model_static_gun =
            load_shader("shaders/compiled/model_static_gun.vert.spv", &device)?;

        let vs_level = load_shader("shaders/compiled/level.vert.spv", &device)?;

        let fs_level = load_shader("shaders/compiled/level.frag.spv", &device)?;
        let fs_level_alpha_clip =
            load_shader("shaders/compiled/level_alpha_clip.frag.spv", &device)?;

        let (array_of_textures_bind_group_layout, render_pipelines) =
            render_pipelines_for_num_textures(
                1,
                &device,
                &vs_model_static,
                &vs_model_animated,
                &fs_model,
                &fs_model_alpha_clip,
                &vs_model_static_gun,
                &vs_level,
                &fs_level,
                &fs_level_alpha_clip,
                &main_bind_group_layout,
                &lights_bind_group_layout,
                &animated_models_bind_group_layout,
                &lightmap_bind_group_layout,
            );

        let skybox_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox pipeline layout"),
                bind_group_layouts: &[&main_bind_group_layout, &skybox_texture_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStage::VERTEX,
                    range: 0..std::mem::size_of::<[Mat4; 2]>() as u32,
                }],
            });

        let skybox_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skybox render pipeline"),
                layout: Some(&skybox_render_pipeline_layout),
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_skybox,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_skybox,
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
            Instance::new(Mat4::identity()),
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

        let vs_full_screen_tri_module =
            load_shader("shaders/compiled/full_screen_tri.vert.spv", &device)?;

        let fs_blit_mipmap_module = load_shader("shaders/compiled/blit_mipmap.frag.spv", &device)?;

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
                ],
            });

        let (pre_tonemap_framebuffer, tonemap_bind_group) =
            post_processing_framebuffer_and_bind_group(
                &device,
                width,
                height,
                "tonemap",
                &post_processing_bind_group_layout,
                &linear_sampler,
                display_format,
            );

        let (pre_fxaa_framebuffer, fxaa_bind_group) = post_processing_framebuffer_and_bind_group(
            &device,
            width,
            height,
            "fxaa",
            &post_processing_bind_group_layout,
            &linear_sampler,
            PRE_TONEMAP_FRAMEBUFFER_FORMAT,
        );

        let post_processing_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post processing pipeline layout"),
                bind_group_layouts: &[&post_processing_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStage::FRAGMENT,
                    range: 0..std::mem::size_of::<Vec2>() as u32,
                }],
            });

        let fs_fxaa_module = load_shader("shaders/compiled/fxaa.frag.spv", &device)?;

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
            color_states: &[display_format.into()],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: Some(INDEX_FORMAT),
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let fs_tonemap_module = load_shader("shaders/compiled/tonemap.frag.spv", &device)?;

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
            color_states: &[display_format.into()],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: Some(INDEX_FORMAT),
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let fs_texture_dilation =
            load_shader("shaders/compiled/texture_dilation.frag.spv", &device)?;

        let texture_dilation_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("texture dilation pipeline"),
                layout: Some(&post_processing_pipeline_layout),
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_full_screen_tri_module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_texture_dilation,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[UNCOMPRESSED_LIGHTMAP_FORMAT.into()],
                depth_stencil_state: None,
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: Some(INDEX_FORMAT),
                    vertex_buffers: &[],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

        let vs_bake_lightmap = load_shader("shaders/compiled/bake_lightmap.vert.spv", &device)?;
        let fs_bake_lightmap = load_shader("shaders/compiled/bake_lightmap.frag.spv", &device)?;

        let bake_lightmap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bake lightmap pipeline layout"),
                bind_group_layouts: &[&lights_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bake_lightmap_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bake lightmap pipeline"),
                layout: Some(&bake_lightmap_pipeline_layout),
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs_bake_lightmap,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs_bake_lightmap,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[UNCOMPRESSED_LIGHTMAP_FORMAT.into()],
                depth_stencil_state: None,
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: Some(INDEX_FORMAT),
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<LevelVertex>() as u64,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &LEVEL_VERTEX_ATTR_ARRAY[..],
                    }],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

        let comp_bake_lightvol = load_shader("shaders/compiled/bake_lightvol.comp.spv", &device)?;

        let lightvol_texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: LIGHTVOL_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };

        let bake_lightvol_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bake light vol bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6 lightvol storage textures
                    lightvol_texture(1),
                    lightvol_texture(2),
                    lightvol_texture(3),
                    lightvol_texture(4),
                    lightvol_texture(5),
                    lightvol_texture(6),
                ],
            });

        let bake_lightvol_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bake lightmap pipeline layout"),
                bind_group_layouts: &[&bake_lightvol_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStage::COMPUTE,
                    range: 0..std::mem::size_of::<LightVolUniforms>() as u32,
                }],
            });

        let bake_lightvol_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bake lightvol compute pipeline"),
                layout: Some(&bake_lightvol_pipeline_layout),
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &comp_bake_lightvol,
                    entry_point: "main",
                },
            });

        Ok(Self {
            bc6h_compressor_2d: wgpu_bc6h_compression::Compressor2D::new(&device),
            bc6h_compressor_3d: wgpu_bc6h_compression::Compressor3D::new(&device),

            device,
            queue,
            window,

            projection,
            projection_view,
            view,
            screen_dimensions,

            main_bind_group,
            surface,
            swap_chain,
            depth_texture,
            identity_instance_buffer,
            array_of_textures_bind_group_layout,
            skybox_texture_bind_group_layout,
            lights_bind_group_layout,
            skybox_render_pipeline,
            main_bind_group_layout,
            mipmap_generation_bind_group_layout,
            mipmap_generation_pipeline,
            linear_sampler,

            render_pipelines,

            animated_models_bind_group_layout,

            post_processing_bind_group_layout,
            pre_fxaa_framebuffer,
            fxaa_bind_group,
            fxaa_pipeline,
            pre_tonemap_framebuffer,
            tonemap_bind_group,
            tonemap_pipeline,

            fs_model,
            vs_model_static,
            vs_model_animated,
            fs_model_alpha_clip,
            vs_model_static_gun,
            vs_level,
            fs_level,
            fs_level_alpha_clip,

            display_format,

            bake_lightmap_pipeline,
            lightmap_bind_group_layout,
            texture_dilation_pipeline,

            bake_lightvol_bind_group_layout,
            bake_lightvol_pipeline,
        })
    }

    // Must be called after camera movement or window resizing.
    pub fn set_camera_view(&mut self, view: Mat4) {
        self.view = view;
        self.projection_view = self.projection * self.view;
    }

    pub fn resize(&mut self, width: u32, height: u32, _settings: &Settings) {
        self.swap_chain = self.device.create_swap_chain(
            &self.surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                format: self.display_format,
                width,
                height,
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

        self.projection = perspective_matrix(width, height);
        self.screen_dimensions = Vec2::new(width as f32, height as f32);

        let (pre_fxaa_framebuffer, fxaa_bind_group) = post_processing_framebuffer_and_bind_group(
            &self.device,
            width,
            height,
            "fxaa",
            &self.post_processing_bind_group_layout,
            &self.linear_sampler,
            self.display_format,
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

    pub fn rebuild_pipelines_for_textures(&mut self, num_textures: u32) {
        let (array_of_textures_bind_group_layout, render_pipelines) =
            render_pipelines_for_num_textures(
                num_textures,
                &self.device,
                &self.vs_model_static,
                &self.vs_model_animated,
                &self.fs_model,
                &self.fs_model_alpha_clip,
                &self.vs_model_static_gun,
                &self.vs_level,
                &self.fs_level,
                &self.fs_level_alpha_clip,
                &self.main_bind_group_layout,
                &self.lights_bind_group_layout,
                &self.animated_models_bind_group_layout,
                &self.lightmap_bind_group_layout,
            );

        self.array_of_textures_bind_group_layout = array_of_textures_bind_group_layout;
        self.render_pipelines = render_pipelines;
    }
}

fn perspective_matrix(width: u32, height: u32) -> Mat4 {
    ultraviolet::projection::perspective_infinite_z_wgpu_dx(
        // http://themetalmuncher.github.io/fov-calc/
        59.0_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
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

struct PipelineParams<'a> {
    device: &'a wgpu::Device,
    label: &'a str,
    layout: &'a wgpu::PipelineLayout,
    vs_module: &'a wgpu::ShaderModule,
    fs_module: &'a wgpu::ShaderModule,
    colour_descriptor: wgpu::ColorStateDescriptor,
    animated: bool,
    backface_culling: bool,
    depth_write: bool,
}

fn create_render_pipeline(params: PipelineParams) -> wgpu::RenderPipeline {
    let PipelineParams {
        device,
        label,
        layout,
        vs_module,
        fs_module,
        colour_descriptor,
        animated,
        backface_culling,
        depth_write,
    } = params;

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
            cull_mode: if backface_culling {
                wgpu::CullMode::Back
            } else {
                wgpu::CullMode::None
            },
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[colour_descriptor],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: DEPTH_FORMAT,
            depth_write_enabled: depth_write,
            depth_compare: wgpu::CompareFunction::Less,
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
                    attributes: if animated {
                        &ANIMATED_VERTEX_ATTR_ARRAY[..]
                    } else {
                        &STATIC_VERTEX_ATTR_ARRAY[..]
                    },
                },
                wgpu::VertexBufferDescriptor {
                    stride: if animated {
                        std::mem::size_of::<AnimatedInstance>()
                    } else {
                        std::mem::size_of::<Instance>()
                    } as u64,
                    step_mode: wgpu::InputStepMode::Instance,
                    attributes: if animated {
                        &ANIMATED_INSTANCE_ATTR_ARRAY[..]
                    } else {
                        &STATIC_INSTANCE_ATTR_ARRAY[..]
                    },
                },
            ],
        },
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}

struct LevelPipelineParams<'a> {
    device: &'a wgpu::Device,
    label: &'a str,
    layout: &'a wgpu::PipelineLayout,
    vs_module: &'a wgpu::ShaderModule,
    fs_module: &'a wgpu::ShaderModule,
    colour_descriptor: wgpu::ColorStateDescriptor,
    backface_culling: bool,
    depth_write: bool,
}

fn create_level_render_pipeline(params: LevelPipelineParams) -> wgpu::RenderPipeline {
    let LevelPipelineParams {
        device,
        label,
        layout,
        vs_module,
        fs_module,
        colour_descriptor,
        backface_culling,
        depth_write,
    } = params;

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
            cull_mode: if backface_culling {
                wgpu::CullMode::Back
            } else {
                wgpu::CullMode::None
            },
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[colour_descriptor],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: DEPTH_FORMAT,
            depth_write_enabled: depth_write,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilStateDescriptor::default(),
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: Some(INDEX_FORMAT),
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<LevelVertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &LEVEL_VERTEX_ATTR_ARRAY[..],
            }],
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

pub fn decal_square(
    position: Vec3,
    normal: Vec3,
    size: Vec2,
    decal: Decal,
    decals_texture_atlas_index: usize,
) -> [Vertex; 6] {
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

    for offset in &mut offsets {
        *offset = rotation * *offset;
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
        texture_index: decals_texture_atlas_index as u32,
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
    label: String,
    waiting: Vec<T>,
    usage: wgpu::BufferUsage,
    // the offset of `waiting` to upload.
    upload_offset: usize,
}

impl<T: bytemuck::Pod> DynamicBuffer<T> {
    pub fn new(
        device: &wgpu::Device,
        base_capacity: usize,
        label: &str,
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
            label: label.to_string(),
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
                label: Some(&self.label),
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
        ],
    });

    (framebuffer, bind_group)
}

pub struct RenderPipelines {
    pub static_opaque: wgpu::RenderPipeline,
    pub static_alpha_blend: wgpu::RenderPipeline,
    pub static_alpha_clip: wgpu::RenderPipeline,

    pub animated_opaque: wgpu::RenderPipeline,
    pub animated_alpha_blend: wgpu::RenderPipeline,
    pub animated_alpha_clip: wgpu::RenderPipeline,

    pub static_opaque_gun: wgpu::RenderPipeline,
    pub static_alpha_clip_gun: wgpu::RenderPipeline,
    pub static_alpha_blend_gun: wgpu::RenderPipeline,

    pub level_opaque: wgpu::RenderPipeline,
    pub level_alpha_clip: wgpu::RenderPipeline,
    pub level_alpha_blend: wgpu::RenderPipeline,
}

fn render_pipelines_for_num_textures(
    num_textures: u32,
    device: &wgpu::Device,

    vs_model_static: &wgpu::ShaderModule,
    vs_model_animated: &wgpu::ShaderModule,
    fs_model: &wgpu::ShaderModule,
    fs_model_alpha_clip: &wgpu::ShaderModule,

    vs_model_static_gun: &wgpu::ShaderModule,

    vs_level: &wgpu::ShaderModule,
    fs_level: &wgpu::ShaderModule,
    fs_level_alpha_clip: &wgpu::ShaderModule,

    main_bind_group_layout: &wgpu::BindGroupLayout,
    lights_bind_group_layout: &wgpu::BindGroupLayout,
    animated_models_bind_group_layout: &wgpu::BindGroupLayout,
    lightmap_bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroupLayout, RenderPipelines) {
    let array_of_textures_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("array of textures bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(std::num::NonZeroU32::new(num_textures).unwrap()),
            }],
        });

    let static_model_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("static model pipeline layout"),
            bind_group_layouts: &[
                main_bind_group_layout,
                &array_of_textures_bind_group_layout,
                lights_bind_group_layout,
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::VERTEX,
                range: 0..std::mem::size_of::<Mat4>() as u32,
            }],
        });

    let animated_model_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("static model pipeline layout"),
            bind_group_layouts: &[
                main_bind_group_layout,
                &array_of_textures_bind_group_layout,
                lights_bind_group_layout,
                animated_models_bind_group_layout,
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::VERTEX,
                range: 0..std::mem::size_of::<Mat4>() as u32,
            }],
        });

    let level_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("level pipeline layout"),
        bind_group_layouts: &[
            main_bind_group_layout,
            &array_of_textures_bind_group_layout,
            lightmap_bind_group_layout,
        ],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::VERTEX,
            range: 0..std::mem::size_of::<Mat4>() as u32,
        }],
    });

    // Decisions:
    //
    // Alpha clipping and backface culling:
    //
    // It might be a good idea to turn off backface culling for alpha clipping.
    // Most things that you want alpha clipping for (bushes, barbed fences etc) are going to be
    // double-sided. On the otherhand, this breaks certain objects such as the question-mark in the
    // mario-cube. You can always enumate the behaviour of having backface-culling off by duplicating
    // the meshes and flipping the uvs, but you can't emulate the behaviour of backface culling with
    // it off.
    //
    // Alpha blending and depth writing:
    //
    // For decals like bullet holes, we want depth writes with alpha blending off, so that they
    // accumulate on a surface. However for larger meshes like club mate bottles, we want depth
    // writes on so that the worst case scenario is a bottle erasing another one, not a bottle in
    // the background being drawn over one in the foreground. I think the solution here would be to
    // have a seperate pipeline with depth writes off for decals.

    let pipelines = RenderPipelines {
        static_opaque: create_render_pipeline(PipelineParams {
            device,
            label: "static opaque render pipeline",
            layout: &static_model_pipeline_layout,
            vs_module: vs_model_static,
            fs_module: fs_model,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            animated: false,
            backface_culling: true,
            depth_write: true,
        }),
        static_alpha_clip: create_render_pipeline(PipelineParams {
            device,
            label: "static alpha clip render pipeline",
            layout: &static_model_pipeline_layout,
            vs_module: vs_model_static,
            fs_module: fs_model_alpha_clip,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            animated: false,
            backface_culling: false,
            depth_write: true,
        }),
        static_alpha_blend: create_render_pipeline(PipelineParams {
            device,
            label: "static alpha blend render pipeline",
            layout: &static_model_pipeline_layout,
            vs_module: vs_model_static,
            fs_module: fs_model,
            colour_descriptor: alpha_blend_colour_descriptor(),
            animated: false,
            backface_culling: true,
            depth_write: false,
        }),

        animated_opaque: create_render_pipeline(PipelineParams {
            device,
            label: "animated opaque render pipeline",
            layout: &animated_model_pipeline_layout,
            vs_module: vs_model_animated,
            fs_module: fs_model,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            animated: true,
            backface_culling: true,
            depth_write: true,
        }),
        animated_alpha_clip: create_render_pipeline(PipelineParams {
            device,
            label: "animated alpha clip render pipeline",
            layout: &animated_model_pipeline_layout,
            vs_module: vs_model_animated,
            fs_module: fs_model_alpha_clip,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            animated: true,
            backface_culling: false,
            depth_write: true,
        }),
        animated_alpha_blend: create_render_pipeline(PipelineParams {
            device,
            label: "animated alpha blend render pipeline",
            layout: &animated_model_pipeline_layout,
            vs_module: vs_model_animated,
            fs_module: fs_model,
            colour_descriptor: alpha_blend_colour_descriptor(),
            animated: true,
            backface_culling: true,
            depth_write: false,
        }),

        static_opaque_gun: create_render_pipeline(PipelineParams {
            device,
            label: "static opaque gun render pipeline",
            layout: &static_model_pipeline_layout,
            vs_module: vs_model_static_gun,
            fs_module: fs_model,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            animated: false,
            backface_culling: true,
            depth_write: true,
        }),
        static_alpha_clip_gun: create_render_pipeline(PipelineParams {
            device,
            label: "static alpha clip gun render pipeline",
            layout: &static_model_pipeline_layout,
            vs_module: vs_model_static_gun,
            fs_module: fs_model_alpha_clip,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            animated: false,
            backface_culling: false,
            depth_write: true,
        }),
        static_alpha_blend_gun: create_render_pipeline(PipelineParams {
            device,
            label: "static alpha blend gun render pipeline",
            layout: &static_model_pipeline_layout,
            vs_module: vs_model_static_gun,
            fs_module: fs_model,
            colour_descriptor: alpha_blend_colour_descriptor(),
            animated: false,
            backface_culling: true,
            depth_write: false,
        }),

        level_opaque: create_level_render_pipeline(LevelPipelineParams {
            device,
            label: "level opaque render pipeline",
            layout: &level_pipeline_layout,
            vs_module: vs_level,
            fs_module: fs_level,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            backface_culling: true,
            depth_write: true,
        }),
        level_alpha_clip: create_level_render_pipeline(LevelPipelineParams {
            device,
            label: "level alpha clip render pipeline",
            layout: &level_pipeline_layout,
            vs_module: vs_level,
            fs_module: fs_level_alpha_clip,
            colour_descriptor: PRE_TONEMAP_FRAMEBUFFER_FORMAT.into(),
            backface_culling: false,
            depth_write: true,
        }),
        level_alpha_blend: create_level_render_pipeline(LevelPipelineParams {
            device,
            label: "level alpha blend render pipeline",
            layout: &level_pipeline_layout,
            vs_module: vs_level,
            fs_module: fs_level,
            colour_descriptor: alpha_blend_colour_descriptor(),
            backface_culling: true,
            depth_write: false,
        }),
    };

    (array_of_textures_bind_group_layout, pipelines)
}
