
use wgpu::util::DeviceExt;
use std::collections::HashMap;
use ultraviolet::{Vec3, Vec2, Mat4};

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    texture_index: i32,
}

pub fn level_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("level bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture {
                    dimension: wgpu::TextureViewDimension::D2,
                    component_type: wgpu::TextureComponentType::Float,
                    multisampled: false,
                },
                count: None,
            }
        ]
    })
}

pub struct Level {
    pub geometry_vertices: wgpu::Buffer,
    pub geometry_indices: wgpu::Buffer,
    pub num_indices: u32,
    pub bind_group: wgpu::BindGroup,
    pub camera_transform: Mat4,
}

impl Level {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;

        let buffers = load_buffers(&gltf)?;

        let mut geometry_vertices = Vec::new();
        let mut geometry_indices = Vec::new();

        let num_textures = gltf.textures().count() as u32;

        //let mut textures = Vec::new();
        let mut gltf_texture_index_to_texture_array_index = HashMap::new();

        let texture_array_extent = wgpu::Extent3d {
            width: 128,
            height: 128,
            depth: num_textures,
        };

        let texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("level texture array"),
            size: texture_array_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        for (i, texture) in gltf.textures().enumerate() {
            let view = match texture.source().source() {
                gltf::image::Source::View { view, .. } => view,
                gltf::image::Source::Uri { .. } => return Err(anyhow::anyhow!("Level textures must be packed."))
            };

            let start = view.offset();
            let end = start + view.length();
            let bytes = &buffers[view.buffer().index()][start..end];
            let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.into_rgba8();
            let (width, height) = image.dimensions();

            assert_eq!(width, 128);
            assert_eq!(height, 128);

            let temp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("texture staging buffer"),
                contents: &*image,
                usage: wgpu::BufferUsage::COPY_SRC,
            });

            encoder.copy_buffer_to_texture(
                wgpu::BufferCopyView {
                    buffer: &temp_buf,
                    layout: wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: 4 * 128,
                        rows_per_image: 0,
                    },
                },
                wgpu::TextureCopyView {
                    texture: &texture_array,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0, y: 0, z: i as u32,
                    },
                },
                texture_array_extent,
            );

            gltf_texture_index_to_texture_array_index.insert(texture.index(), i);
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("level bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_array.create_view(&wgpu::TextureViewDescriptor::default()))
                }
            ]
        });

        for mesh in gltf.meshes() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives with {:?} are not allowed. Triangles only.",
                        primitive.mode()
                    ));
                }

                let texture_index = primitive.material().pbr_metallic_roughness().base_color_texture()
                    .ok_or_else(|| anyhow::anyhow!("All level primitives need textures."))?
                    .texture().index();

                let texture_array_index = gltf_texture_index_to_texture_array_index[&texture_index];

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap();
                let tex_coordinates = reader.read_tex_coords(0).unwrap().into_f32();
                let normals = reader.read_normals().unwrap();

                let num_vertices = geometry_vertices.len() as u32;

                geometry_indices.extend(
                    reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .map(|i| i + num_vertices),
                );

                positions
                    .zip(tex_coordinates)
                    .zip(normals)
                    .for_each(|((p, uv), n)| {
                        geometry_vertices.push(Vertex {
                            position: p.into(),
                            normal: n.into(),
                            uv: uv.into(),
                            texture_index: texture_array_index as i32,
                        });
                    });
            }
        }

        let camera_transform = gltf.nodes()
            .find(|node| node.camera().is_some())
            .ok_or_else(|| anyhow::anyhow!("All levels need a camera."))?
            .transform().matrix().into();

        println!(
            "Level loaded. Vertices: {}. Indices: {}. Textures: {}",
            geometry_vertices.len(),
            geometry_indices.len(),
            num_textures,
        );

        Ok(Self {
            geometry_vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("level geometry vertices"),
                contents: bytemuck::cast_slice(&geometry_vertices),
                usage: wgpu::BufferUsage::VERTEX,
            }),
            geometry_indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("level geometry indices"),
                contents: bytemuck::cast_slice(&geometry_indices),
                usage: wgpu::BufferUsage::INDEX,
            }),
            num_indices: geometry_indices.len() as u32,
            bind_group,
            camera_transform,
        })
    }
}

// Load the buffers from a gltf document into a vector of byte vectors.
// I mostly copied what bevy does for this because it's a little confusing at first.
// https://github.com/bevyengine/bevy/blob/master/crates/bevy_gltf/src/loader.rs
fn load_buffers(gltf: &gltf::Gltf) -> anyhow::Result<Vec<Vec<u8>>> {
    const OCTET_STREAM_URI: &str = "data:application/octet-stream;base64,";

    let mut buffers = Vec::new();

    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                if uri.starts_with(OCTET_STREAM_URI) {
                    buffers.push(base64::decode(&uri[OCTET_STREAM_URI.len()..])?);
                } else {
                    return Err(anyhow::anyhow!(
                        "Only octet streams are supported with data:"
                    ));
                }
            }
            gltf::buffer::Source::Bin => {
                if let Some(blob) = gltf.blob.as_deref() {
                    buffers.push(blob.into());
                } else {
                    return Err(anyhow::anyhow!("Missing blob"));
                }
            }
        }
    }

    Ok(buffers)
}

/*
fn load_texture(
    bytes: &[u8],
    label: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
) -> anyhow::Result<wgpu::BindGroup> {
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.into_rgba();

    let temp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cheese texture staging buffer"),
        contents: &*image,
        usage: wgpu::BufferUsage::COPY_SRC,
    });

    let texture_extent = wgpu::Extent3d {
        width: image.width(),
        height: image.height(),
        depth: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        label: Some(label),
    });

    encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &temp_buf,
            layout: wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * image.width(),
                rows_per_image: 0,
            },
        },
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        texture_extent,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Cheese texture bind group"),
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    }))
}
*/
