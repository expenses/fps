use crate::renderer::{Renderer, Vertex, TEXTURE_FORMAT};
use std::collections::HashMap;
use ultraviolet::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Light {
    output: Vec3,
    padding_0: i32,
    position: Vec3,
    padding_1: i32,
}

struct StagingModelBuffers<T> {
    vertices: Vec<T>,
    indices: Vec<u32>,
}

impl<T> Default for StagingModelBuffers<T> {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}

impl StagingModelBuffers<Vertex> {
    fn upload(&self, device: &wgpu::Device) -> ModelBuffers {
        ModelBuffers {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("level geometry vertices"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsage::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("level geometry indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsage::INDEX,
            }),
            num_indices: self.indices.len() as u32,
        }
    }
}

pub struct ModelBuffers {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub num_indices: u32,
}

#[derive(Debug)]
pub enum Extras {
    Spawn(Character),
}

impl Extras {
    fn parse(string: &str) -> anyhow::Result<Self> {
        if string.starts_with("spawn/") {
            let remainder = &string["spawn/".len()..];
            let character = Character::parse(remainder)?;
            Ok(Self::Spawn(character))
        } else {
            Err(anyhow::anyhow!("Unrecognised string '{}'", string))
        }
    }
}

#[derive(Debug)]
pub enum Character {
    Robot,
}

impl Character {
    fn parse(string: &str) -> anyhow::Result<Self> {
        match string {
            "robot" => Ok(Self::Robot),
            _ => Err(anyhow::anyhow!("Unrecognised string '{}'", string)),
        }
    }
}

pub struct Level {
    pub opaque_geometry: ModelBuffers,
    pub transparent_geometry: ModelBuffers,
    pub texture_array_bind_group: wgpu::BindGroup,
    pub lights_bind_group: wgpu::BindGroup,
    pub extras: HashMap<usize, Extras>,
    pub node_tree: NodeTree,
    pub collision_mesh: ncollide3d::shape::TriMesh<f32>,
}

impl Level {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;

        let node_tree = NodeTree::new(&gltf);

        let extras = gltf
            .nodes()
            .filter_map(|node| {
                node.extras()
                    .as_ref()
                    .map(|json_value| (node.index(), json_value))
            })
            .map(|(node_index, json_value)| {
                let map: HashMap<String, serde_json::Value> =
                    serde_json::from_str(json_value.get())?;
                assert_eq!(map.len(), 1);
                let key = map.keys().next().unwrap();
                Ok((node_index, (Extras::parse(&key)?)))
            })
            .collect::<anyhow::Result<HashMap<usize, Extras>>>()?;

        let buffers = load_buffers(&gltf)?;

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut transparent_geometry = StagingModelBuffers::default();
        let mut collision_geometry = StagingModelBuffers::default();

        let texture_array_bind_group = load_texture_array(&gltf, &buffers, renderer, encoder)?;

        let lights: Vec<_> = gltf
            .nodes()
            .filter_map(|node| node.light().map(|light| (node, light)))
            .map(|(node, light)| {
                assert!(matches!(
                    light.kind(),
                    gltf::khr_lights_punctual::Kind::Point
                ));

                let transform = node_tree.transform_of(node.index());

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

        let lights_buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("level lights"),
                contents: bytemuck::cast_slice(&lights),
                usage: wgpu::BufferUsage::STORAGE,
            });

        let lights_bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lights bind group"),
                layout: &renderer.lights_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lights_buffer.as_entire_binding(),
                }],
            });

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            if node.extras().is_some() {
                continue;
            }

            assert_eq!(mesh.primitives().count(), 1);

            let transform = node_tree.transform_of(node.index());
            let normal_matrix = normal_matrix(transform);

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives mode {:?} is not implemented.",
                        primitive.mode()
                    ));
                }

                if primitive.material().alpha_mode() == gltf::material::AlphaMode::Blend {
                    add_primitive_geometry_to_buffers(
                        &primitive,
                        transform,
                        normal_matrix,
                        &buffers,
                        &mut transparent_geometry,
                        Some(&mut collision_geometry),
                    )?;
                } else {
                    add_primitive_geometry_to_buffers(
                        &primitive,
                        transform,
                        normal_matrix,
                        &buffers,
                        &mut opaque_geometry,
                        Some(&mut collision_geometry),
                    )?;
                }
            }
        }

        let collision_mesh = ncollide3d::shape::TriMesh::new(
            collision_geometry
                .vertices
                .iter()
                .map(|&vertex| {
                    let position: [f32; 3] = vertex.into();
                    position.into()
                })
                .collect(),
            collision_geometry
                .indices
                .chunks(3)
                .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize].into())
                .collect(),
            None,
        );

        println!(
            "Level loaded. Vertices: {}. Indices: {}. Textures: {}. Lights: {}",
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.textures().count() as u32,
            lights.len(),
        );

        Ok(Self {
            opaque_geometry: opaque_geometry.upload(&renderer.device),
            transparent_geometry: transparent_geometry.upload(&renderer.device),
            texture_array_bind_group,
            lights_bind_group,
            extras,
            node_tree,
            collision_mesh,
        })
    }
}

pub struct Model {
    pub buffers: ModelBuffers,
    pub textures: wgpu::BindGroup,
}

impl Model {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;
        let node_tree = NodeTree::new(&gltf);
        let buffers = load_buffers(&gltf)?;

        let textures = load_texture_array(&gltf, &buffers, renderer, encoder)?;

        let mut staging_buffers = StagingModelBuffers::default();

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            assert_eq!(mesh.primitives().count(), 1);

            let transform = if node.skin().is_some() {
                // We can't use node transforms for animated models.
                Mat4::identity()
            } else {
                node_tree.transform_of(node.index())
            };
            let normal_matrix = normal_matrix(transform);

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives mode {:?} is not implemented.",
                        primitive.mode()
                    ));
                }

                add_primitive_geometry_to_buffers(
                    &primitive,
                    transform,
                    normal_matrix,
                    &buffers,
                    &mut staging_buffers,
                    None,
                )?;
            }
        }

        println!(
            "Model loaded. Vertices: {}. Indices: {}. Textures: {}",
            staging_buffers.vertices.len(),
            staging_buffers.indices.len(),
            gltf.textures().count() as u32,
        );

        Ok(Self {
            buffers: staging_buffers.upload(&renderer.device),
            textures,
        })
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

pub struct NodeTree {
    inner: Vec<(Mat4, usize)>,
}

impl NodeTree {
    fn new(gltf: &gltf::Gltf) -> Self {
        let mut inner = vec![(Mat4::identity(), usize::max_value()); gltf.nodes().count()];

        for node in gltf.nodes() {
            inner[node.index()].0 = node.transform().matrix().into();
            for child in node.children() {
                inner[child.index()].1 = node.index();
            }
        }

        Self { inner }
    }

    pub fn transform_of(&self, mut index: usize) -> Mat4 {
        let mut transform_sum = Mat4::identity();

        while index != usize::max_value() {
            let (transform, parent_index) = self.inner[index];
            transform_sum = transform * transform_sum;
            index = parent_index;
        }

        transform_sum
    }
}

fn add_primitive_geometry_to_buffers(
    primitive: &gltf::Primitive,
    transform: Mat4,
    normal_matrix: Mat3,
    gltf_buffers: &Vec<Vec<u8>>,
    staging_buffers: &mut StagingModelBuffers<Vertex>,
    mut collision_buffers: Option<&mut StagingModelBuffers<Vec3>>,
) -> anyhow::Result<()> {
    let texture_index = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
        .ok_or_else(|| anyhow::anyhow!("All level primitives need textures."))?
        .texture()
        .index();

    let reader = primitive.reader(|buffer| Some(&gltf_buffers[buffer.index()]));

    let positions = reader.read_positions().unwrap();
    let tex_coordinates = reader.read_tex_coords(0).unwrap().into_f32();
    let normals = reader.read_normals().unwrap();

    let num_vertices = staging_buffers.vertices.len() as u32;

    staging_buffers.indices.extend(
        reader
            .read_indices()
            .unwrap()
            .into_u32()
            .map(|i| i + num_vertices),
    );

    if let Some(collision_buffers) = collision_buffers.as_mut() {
        let num_vertices_collision = collision_buffers.vertices.len() as u32;

        collision_buffers.indices.extend(
            reader
                .read_indices()
                .unwrap()
                .into_u32()
                .map(|i| i + num_vertices_collision),
        );
    }

    positions
        .zip(tex_coordinates)
        .zip(normals)
        .for_each(|((p, uv), n)| {
            let position = transform * Vec4::new(p[0], p[1], p[2], 1.0);
            assert_eq!(position.w, 1.0);
            let position = position.xyz();

            let normal: Vec3 = n.into();
            let normal = (normal_matrix * normal).normalized();

            staging_buffers.vertices.push(Vertex {
                position,
                normal,
                uv: uv.into(),
                texture_index: texture_index as i32,
            });

            if let Some(collision_buffers) = collision_buffers.as_mut() {
                collision_buffers.vertices.push(position);
            }
        });

    Ok(())
}

pub fn load_skybox(
    png_bytes: &[u8],
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
) -> anyhow::Result<wgpu::BindGroup> {
    let image =
        image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)?.into_rgba8();

    let staging_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skybox staging buffer"),
            contents: &*image,
            usage: wgpu::BufferUsage::COPY_SRC,
        });

    let skybox_texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skybox texture"),
        size: wgpu::Extent3d {
            width: 128,
            height: 128,
            depth: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    });

    for i in 0..6 {
        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &staging_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: 4 * 128,
                    rows_per_image: 0,
                },
            },
            wgpu::TextureCopyView {
                texture: &skybox_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: i },
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth: 1,
            },
        );
    }

    let skybox_texture_view = skybox_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox texture view"),
        format: Some(TEXTURE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    let skybox_bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox bind group"),
            layout: &renderer.skybox_texture_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&skybox_texture_view),
            }],
        });

    Ok(skybox_bind_group)
}

fn load_texture_array(
    gltf: &gltf::Gltf,
    gltf_buffers: &Vec<Vec<u8>>,
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
) -> anyhow::Result<wgpu::BindGroup> {
    let num_textures = gltf.textures().count() as u32;

    if num_textures == 0 {
        return Err(anyhow::anyhow!("No textures in gltf file."));
    }

    let mut texture_array: Option<(wgpu::Texture, u32, u32)> = None;

    for (i, texture) in gltf.textures().enumerate() {
        let view = match texture.source().source() {
            gltf::image::Source::View { view, mime_type } => {
                assert_eq!(mime_type, "image/png");
                view
            }
            gltf::image::Source::Uri { .. } => {
                return Err(anyhow::anyhow!("Textures must be packed."))
            }
        };

        let start = view.offset();
        let end = start + view.length();
        let bytes = &gltf_buffers[view.buffer().index()][start..end];
        let image =
            image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.into_rgba8();

        let (image_width, image_height) = image.dimensions();

        let (texture_array, texture_width, texture_height) =
            texture_array.get_or_insert_with(|| {
                assert_eq!(image_width % 64, 0);
                assert_eq!(image_height % 64, 0);

                let texture_array = renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("level texture array"),
                    size: wgpu::Extent3d {
                        width: image_width,
                        height: image_height,
                        depth: num_textures,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: TEXTURE_FORMAT,
                    usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                });

                (texture_array, image_width, image_height)
            });

        assert_eq!(image_width, *texture_width);
        assert_eq!(image_height, *texture_height);

        let staging_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("texture staging buffer"),
                    contents: &*image,
                    usage: wgpu::BufferUsage::COPY_SRC,
                });

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &staging_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: 4 * *texture_width,
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
            wgpu::Extent3d {
                width: *texture_width,
                height: *texture_width,
                depth: 1,
            },
        );
    }

    let (texture_array, ..) = texture_array.unwrap();

    let texture_view = texture_array.create_view(&wgpu::TextureViewDescriptor {
        label: Some("level texture array view"),
        format: Some(TEXTURE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let texture_array_bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture array bind group"),
            layout: &renderer.texture_array_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            }],
        });

    Ok(texture_array_bind_group)
}
