use crate::renderer::{Vertex, TEXTURE_FORMAT};
use ultraviolet::{Mat3, Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt;

pub fn level_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("level bind group layout"),
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
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Light {
    output: Vec3,
    padding_0: i32,
    position: Vec3,
    padding_1: i32,
}

pub struct Level {
    pub geometry_vertices: wgpu::Buffer,
    pub geometry_indices: wgpu::Buffer,
    pub num_indices: u32,
    pub bind_group: wgpu::BindGroup,
}

pub struct LevelPhysics {
    pub collider: rapier3d::geometry::Collider,
    pub rigid_body: rapier3d::dynamics::RigidBody,
}

impl Level {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<(Self, LevelPhysics)> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;

        let mut node_tree: Vec<(Mat4, usize)> =
            vec![(Mat4::identity(), usize::max_value()); gltf.nodes().count()];

        for node in gltf.nodes() {
            node_tree[node.index()].0 = node.transform().matrix().into();
            for child in node.children() {
                node_tree[child.index()].1 = node.index();
            }
        }

        fn transform_of(mut node_index: usize, node_tree: &[(Mat4, usize)]) -> Mat4 {
            let mut sum_transform = Mat4::identity();

            while node_index != usize::max_value() {
                let (transform, parent_index) = node_tree[node_index];
                sum_transform = transform * sum_transform;
                node_index = parent_index;
            }

            sum_transform
        }

        let buffers = load_buffers(&gltf)?;

        let mut geometry_vertices = Vec::new();
        let mut geometry_indices = Vec::new();

        let num_textures = gltf.textures().count() as u32;

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
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        for (i, texture) in gltf.textures().enumerate() {
            let view = match texture.source().source() {
                gltf::image::Source::View { view, .. } => view,
                gltf::image::Source::Uri { .. } => {
                    return Err(anyhow::anyhow!("Level textures must be packed."))
                }
            };

            let start = view.offset();
            let end = start + view.length();
            let bytes = &buffers[view.buffer().index()][start..end];
            let image =
                image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.into_rgba8();
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
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                },
                texture_array_extent,
            );
        }

        let texture_view = texture_array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("level texture array view"),
            format: Some(TEXTURE_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let lights: Vec<_> = gltf
            .nodes()
            .filter_map(|node| node.light().map(|light| (node, light)))
            .map(|(node, light)| {
                assert!(matches!(
                    light.kind(),
                    gltf::khr_lights_punctual::Kind::Point
                ));

                let transform = transform_of(node.index(), &node_tree);

                // We reduce the intensity by 4PI because of
                // https://github.com/KhronosGroup/glTF-Blender-IO/issues/564.
                // Reducing by 2 again seems to bring it in line with blender but idk why
                let intensity = light.intensity() / (2.0 * 4.0 * std::f32::consts::PI);
                let colour: Vec3 = light.color().into();

                Light {
                    output: colour * intensity,
                    padding_0: 0,
                    position: transform.extract_translation(),
                    padding_1: 0,
                }
            })
            .collect();

        let lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("level lights"),
            contents: bytemuck::cast_slice(&lights),
            usage: wgpu::BufferUsage::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("level bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
            ],
        });

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            assert_eq!(mesh.primitives().count(), 1);

            let transform: Mat4 = transform_of(node.index(), &node_tree);
            let normal_matrix = normal_matrix(transform);

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives with {:?} are not allowed. Triangles only.",
                        primitive.mode()
                    ));
                }

                let texture_index = primitive
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .ok_or_else(|| anyhow::anyhow!("All level primitives need textures."))?
                    .texture()
                    .index();

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
                        let position = transform * Vec4::new(p[0], p[1], p[2], 1.0);
                        assert_eq!(position.w, 1.0);
                        let position = position.xyz();

                        let normal: Vec3 = n.into();
                        let normal = (normal_matrix * normal).normalized();

                        geometry_vertices.push(Vertex {
                            position,
                            normal,
                            uv: uv.into(),
                            texture_index: texture_index as i32,
                        });
                    });
            }
        }

        let physics_collider = rapier3d::geometry::ColliderBuilder::trimesh(
            geometry_vertices
                .iter()
                .map(|vertex| {
                    let position: [f32; 3] = vertex.position.into();
                    position.into()
                })
                .collect(),
            geometry_indices
                .chunks(3)
                .map(|slice| [slice[0], slice[1], slice[2]].into())
                .collect(),
        )
        .build();

        let physics_rigid_body = rapier3d::dynamics::RigidBodyBuilder::new_static().build();

        let level_physics = LevelPhysics {
            collider: physics_collider,
            rigid_body: physics_rigid_body,
        };

        println!(
            "Level loaded. Vertices: {}. Indices: {}. Textures: {}. Lights: {}",
            geometry_vertices.len(),
            geometry_indices.len(),
            num_textures,
            lights.len(),
        );

        Ok((
            Self {
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
            },
            level_physics,
        ))
    }
}

fn normal_matrix(transform: Mat4) -> Mat3 {
    let inverse_transpose = transform.inversed().transposed();
    let array = inverse_transpose.as_component_array();
    Mat3::new(array[0].xyz(), array[1].xyz(), array[2].xyz())
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
