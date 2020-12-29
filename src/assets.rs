use crate::renderer::{Renderer, Vertex, TEXTURE_FORMAT};
use crate::vec3_into;
use std::collections::HashMap;
use ultraviolet::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

mod animated_model;

pub use animated_model::{AnimatedModel, AnimationJoints};

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Light {
    output: Vec3,
    range: f32,
    position: Vec3,
    padding: i32,
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

impl<T: bytemuck::Pod> StagingModelBuffers<T> {
    fn upload(&self, device: &wgpu::Device, name: &str) -> ModelBuffers {
        ModelBuffers {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} geometry vertices", name)),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsage::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} geometry indices", name)),
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
pub enum Property {
    Spawn(Character),
    NoCollide,
}

impl Property {
    fn parse(string: &str) -> anyhow::Result<Self> {
        if string.starts_with("spawn/") {
            let remainder = &string["spawn/".len()..];
            let character = Character::parse(remainder)?;
            Ok(Self::Spawn(character))
        } else {
            match string {
                "nocollide" => Ok(Self::NoCollide),
                _ => Err(anyhow::anyhow!("Unrecognised string '{}'", string)),
            }
        }
    }
}

#[derive(Debug)]
pub enum Character {
    Robot,
    Mouse,
}

impl Character {
    fn parse(string: &str) -> anyhow::Result<Self> {
        match string {
            "robot" => Ok(Self::Robot),
            "mouse" => Ok(Self::Mouse),
            _ => Err(anyhow::anyhow!("Unrecognised string '{}'", string)),
        }
    }
}

#[derive(Debug)]
pub enum MaterialProperty {
    // This is to work around this issue:
    // https://github.com/KhronosGroup/glTF-Blender-IO/pull/1159#issuecomment-678563107
    EmissionStrength(f32),
}

impl MaterialProperty {
    fn parse(string: &str, value: f32) -> anyhow::Result<Self> {
        match string {
            "emission_strength" => Ok(Self::EmissionStrength(value)),
            _ => Err(anyhow::anyhow!("Unrecognised string '{}'", string)),
        }
    }
}

pub struct Level {
    pub opaque_geometry: ModelBuffers,
    pub transparent_geometry: ModelBuffers,
    pub texture_array_bind_group: wgpu::BindGroup,
    pub lights_bind_group: wgpu::BindGroup,
    pub properties: HashMap<usize, Property>,
    pub node_tree: NodeTree,
    pub collision_mesh: ncollide3d::shape::TriMesh<f32>,
    pub nav_mesh: (Vec<Vec3>, Vec<u32>),
}

impl Level {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;

        let node_tree = NodeTree::new(&gltf);

        let properties = gltf
            .nodes()
            .filter_map(|node| {
                node.extras()
                    .as_ref()
                    .map(|json_value| (node.index(), json_value))
            })
            .map(|(node_index, json_value)| {
                let map: HashMap<String, f32> = serde_json::from_str(json_value.get())?;
                assert_eq!(map.len(), 1);
                let key = map.keys().next().unwrap();
                Ok((node_index, (Property::parse(&key)?)))
            })
            .collect::<anyhow::Result<HashMap<usize, Property>>>()?;

        let material_properties = gltf
            .materials()
            .filter_map(|material| {
                material
                    .extras()
                    .as_ref()
                    .map(|json_value| (material.index(), json_value))
            })
            .map(|(material_index, json_value)| {
                let map: HashMap<String, f32> = serde_json::from_str(json_value.get())?;
                assert_eq!(map.len(), 1);
                let key = map.keys().next().unwrap();
                Ok((
                    material_index,
                    MaterialProperty::parse(&key[..], map[&key[..]])?,
                ))
            })
            .collect::<anyhow::Result<HashMap<Option<usize>, MaterialProperty>>>()?;

        let buffer_blob = gltf.blob.as_ref().unwrap();

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut transparent_geometry = StagingModelBuffers::default();
        let mut collision_geometry = StagingModelBuffers::default();

        let texture_array_view = load_texture_array(
            &gltf,
            buffer_blob,
            renderer,
            encoder,
            &format!("{} level", name),
        )?;

        let texture_array_bind_group =
            renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{} level texture bind group", name)),
                    layout: &renderer.texture_array_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_array_view),
                    }],
                });

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
                    range: light.range().unwrap_or(std::f32::INFINITY),
                    position: transform.extract_translation(),
                    padding: 0,
                }
            })
            .collect();

        let lights_buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} level lights", name)),
                contents: bytemuck::cast_slice(&lights),
                usage: wgpu::BufferUsage::STORAGE,
            });

        let lights_bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} level bind group", name)),
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
            let collide = match properties.get(&node.index()) {
                Some(Property::NoCollide) => false,
                Some(_) => continue,
                None => true,
            };

            let transform = node_tree.transform_of(node.index());
            let normal_matrix = normal_matrix(transform);

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives mode {:?} is not implemented.",
                        primitive.mode()
                    ));
                }

                let collision_geometry = if collide {
                    Some(&mut collision_geometry)
                } else {
                    None
                };

                let staging_buffers =
                    if primitive.material().alpha_mode() == gltf::material::AlphaMode::Blend {
                        &mut transparent_geometry
                    } else {
                        &mut opaque_geometry
                    };

                add_primitive_geometry_to_buffers(
                    &primitive,
                    &node,
                    transform,
                    normal_matrix,
                    buffer_blob,
                    &material_properties,
                    staging_buffers,
                    collision_geometry,
                )?;
            }
        }

        let collision_mesh = ncollide3d::shape::TriMesh::new(
            collision_geometry
                .vertices
                .iter()
                .map(|&(position, _)| vec3_into(position))
                .collect(),
            collision_geometry
                .indices
                .chunks(3)
                .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize].into())
                .collect(),
            None,
        );

        let nav_mesh = create_navmesh(&collision_geometry);

        println!(
            "'{}' level loaded. Vertices: {}. Indices: {}. Textures: {}. Lights: {}. Nav mesh vertices: {}. Nav mesh indices: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.textures().count() as u32,
            lights.len(),
            nav_mesh.vertices.len(), nav_mesh.indices.len(),
        );

        Ok(Self {
            opaque_geometry: opaque_geometry
                .upload(&renderer.device, &format!("{} level opaque", name)),
            transparent_geometry: transparent_geometry
                .upload(&renderer.device, &format!("{} level transparent", name)),
            texture_array_bind_group,
            lights_bind_group,
            properties,
            node_tree,
            collision_mesh,
            nav_mesh: (nav_mesh.vertices, nav_mesh.indices),
        })
    }
}

fn create_navmesh(
    collision_geometry: &StagingModelBuffers<(Vec3, Vec3)>,
) -> StagingModelBuffers<Vec3> {
    let mut nav_mesh_geometry = StagingModelBuffers::default();
    let mut collision_indices_to_nav_mesh_indices =
        vec![usize::max_value(); collision_geometry.vertices.len()];

    for chunk in collision_geometry.indices.chunks(3) {
        let shallow_enough = chunk.iter().all(|index| {
            let normal = collision_geometry.vertices[*index as usize].1;
            normal.dot(Vec3::unit_y()).acos() < 45.0_f32.to_radians()
        });

        if shallow_enough {
            for index in chunk {
                let new_index = collision_indices_to_nav_mesh_indices[*index as usize];

                if new_index == usize::max_value() {
                    let len = nav_mesh_geometry.vertices.len();
                    nav_mesh_geometry
                        .vertices
                        .push(collision_geometry.vertices[*index as usize].0);
                    nav_mesh_geometry.indices.push(len as u32);
                    collision_indices_to_nav_mesh_indices[*index as usize] = len;
                } else {
                    nav_mesh_geometry.indices.push(new_index as u32);
                }
            }
        }
    }

    nav_mesh_geometry
}

pub struct Model {
    pub opaque_geometry: ModelBuffers,
    pub transparent_geometry: ModelBuffers,
    pub textures: wgpu::BindGroup,
}

impl Model {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;
        let node_tree = NodeTree::new(&gltf);

        let buffer_blob = gltf.blob.as_ref().unwrap();

        let textures = load_texture_array(&gltf, buffer_blob, renderer, encoder, name)?;

        let textures = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} texture bind group", name)),
                layout: &renderer.texture_array_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&textures),
                }],
            });

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut transparent_geometry = StagingModelBuffers::default();

        assert_eq!(gltf.skins().count(), 0);

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            let transform = node_tree.transform_of(node.index());
            let normal_matrix = normal_matrix(transform);

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives mode {:?} is not implemented.",
                        primitive.mode()
                    ));
                }

                let staging_buffers =
                    if primitive.material().alpha_mode() == gltf::material::AlphaMode::Blend {
                        &mut transparent_geometry
                    } else {
                        &mut opaque_geometry
                    };

                add_primitive_geometry_to_buffers(
                    &primitive,
                    &node,
                    transform,
                    normal_matrix,
                    buffer_blob,
                    &HashMap::new(),
                    staging_buffers,
                    None,
                )?;
            }
        }

        println!(
            "'{}' model loaded. Vertices: {}. Indices: {}. Textures: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.textures().count() as u32,
        );

        Ok(Self {
            opaque_geometry: opaque_geometry.upload(&renderer.device, &format!("{} opaque", name)),
            transparent_geometry: transparent_geometry
                .upload(&renderer.device, &format!("{} transparent", name)),
            textures,
        })
    }
}

fn normal_matrix(transform: Mat4) -> Mat3 {
    let inverse_transpose = transform.inversed().transposed();
    let array = inverse_transpose.as_component_array();
    Mat3::new(array[0].xyz(), array[1].xyz(), array[2].xyz())
}

#[derive(Clone)]
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
    node: &gltf::Node,
    transform: Mat4,
    normal_matrix: Mat3,
    buffer_blob: &[u8],
    material_properties: &HashMap<Option<usize>, MaterialProperty>,
    staging_buffers: &mut StagingModelBuffers<Vertex>,
    mut collision_buffers: Option<&mut StagingModelBuffers<(Vec3, Vec3)>>,
) -> anyhow::Result<()> {
    let emission_strength = match material_properties.get(&primitive.material().index()) {
        Some(MaterialProperty::EmissionStrength(strength)) => *strength,
        _ => 0.0,
    };

    let texture_index = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No textures found for the mesh on node {:?}; primitive {}",
                node.name(),
                primitive.index()
            )
        })?
        .texture()
        .index();

    let reader = primitive.reader(|buffer| {
        assert_eq!(buffer.index(), 0);
        Some(buffer_blob)
    });

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

    let positions = reader.read_positions().unwrap();
    let tex_coordinates = reader.read_tex_coords(0).unwrap().into_f32();
    let normals = reader.read_normals().unwrap();

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
                emission_strength,
            });

            if let Some(collision_buffers) = collision_buffers.as_mut() {
                collision_buffers.vertices.push((position, normal));
            }
        });

    Ok(())
}

pub fn load_skybox(
    png_bytes: &[u8],
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
    name: &str,
) -> anyhow::Result<wgpu::BindGroup> {
    let image =
        image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)?.into_rgba8();

    let staging_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} staging buffer", name)),
            contents: &*image,
            usage: wgpu::BufferUsage::COPY_SRC,
        });

    let skybox_texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("{} texture", name)),
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
        label: Some(&format!("{} texture view", name)),
        format: Some(TEXTURE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    let skybox_bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} bind group", name)),
            layout: &renderer.skybox_texture_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&skybox_texture_view),
            }],
        });

    Ok(skybox_bind_group)
}

const MIPMAP_LEVELS: u32 = 7;

fn load_texture_array(
    gltf: &gltf::Gltf,
    buffer_blob: &[u8],
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
    name: &str,
) -> anyhow::Result<wgpu::TextureView> {
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

        assert_eq!(view.buffer().index(), 0);

        let start = view.offset();
        let end = start + view.length();
        let bytes = &buffer_blob[start..end];
        let image =
            image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.into_rgba8();

        let (image_width, image_height) = image.dimensions();

        let (texture_array, texture_width, texture_height) =
            texture_array.get_or_insert_with(|| {
                assert_eq!(image_width % 64, 0);
                assert_eq!(image_height % 64, 0);

                let texture_array = renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("{} texture array", name)),
                    size: wgpu::Extent3d {
                        width: image_width,
                        height: image_height,
                        depth: num_textures,
                    },
                    mip_level_count: MIPMAP_LEVELS,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: TEXTURE_FORMAT,
                    usage: wgpu::TextureUsage::SAMPLED
                        | wgpu::TextureUsage::COPY_DST
                        | wgpu::TextureUsage::RENDER_ATTACHMENT,
                });

                (texture_array, image_width, image_height)
            });

        assert_eq!(image_width, *texture_width);
        assert_eq!(image_height, *texture_height);

        let staging_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} texture staging buffer", name)),
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

        // Mipmap generation
        let mipmap_views: Vec<_> = (0..MIPMAP_LEVELS)
            .map(|level| {
                texture_array.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("{} texture mipmap view for lod {}", name, level)),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: level,
                    level_count: Some(std::num::NonZeroU32::new(1).unwrap()),
                    base_array_layer: i as u32,
                    array_layer_count: Some(std::num::NonZeroU32::new(1).unwrap()),
                    ..Default::default()
                })
            })
            .collect();

        for level in 1..MIPMAP_LEVELS as usize {
            let bind_group = renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "{} texture mipmap generation bind group for lod {}",
                        name, level
                    )),
                    layout: &renderer.mipmap_generation_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&mipmap_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&renderer.linear_sampler),
                        },
                    ],
                });

            let label = format!(
                "{} texture mipmap generation render pass for lod {}",
                name, level
            );

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&label),
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &mipmap_views[level],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&renderer.mipmap_generation_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }

    let (texture_array, ..) = texture_array.unwrap();

    Ok(texture_array.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{} texture view", name)),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    }))
}

pub fn load_single_texture(
    png_bytes: &[u8],
    renderer: &Renderer,
    name: &str,
) -> anyhow::Result<wgpu::BindGroup> {
    let image =
        image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)?.into_rgba8();

    assert_eq!(image.width() % 64, 0);
    assert_eq!(image.height() % 64, 0);

    let texture = renderer.device.create_texture_with_data(
        &renderer.queue,
        &wgpu::TextureDescriptor {
            label: Some(&format!("{} texture", name)),
            size: wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        },
        &*image,
    );

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{} texture view", name)),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} texture bind group", name)),
            layout: &renderer.texture_array_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            }],
        });

    Ok(bind_group)
}
