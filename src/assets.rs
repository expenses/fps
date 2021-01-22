use crate::array_of_textures::ArrayOfTextures;
use crate::intersection_maths::IntersectionTriangle;
use crate::renderer::{normal_matrix, Renderer, Vertex, TEXTURE_FORMAT};
use crate::vec3_into;
use std::collections::HashMap;
use ultraviolet::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

mod animated_model;

pub use animated_model::AnimatedModel;
pub use animation::AnimationJoints;

#[derive(Debug)]
pub struct IndexBufferView {
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Light {
    output: Vec3,
    range: f32,
    position: Vec3,
    padding: i32,
}

#[derive(Clone)]
pub struct StagingModelBuffers<T> {
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
    pub fn upload(&self, device: &wgpu::Device, name: &str) -> (wgpu::Buffer, wgpu::Buffer) {
        (
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} vertices", name)),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsage::VERTEX,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} indices", name)),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsage::INDEX,
            }),
        )
    }

    fn merge(&mut self, other: StagingModelBuffers<T>) -> IndexBufferView {
        let num_vertices = self.vertices.len() as u32;

        let view = IndexBufferView {
            offset: self.indices.len() as u32,
            size: other.indices.len() as u32,
        };

        self.vertices.extend(other.vertices.into_iter());
        self.indices
            .extend(other.indices.into_iter().map(|index| num_vertices + index));

        view
    }
}

#[derive(Debug)]
pub enum Property {
    Spawn(String),
    NoCollide,
    RenderOrder(u8),
}

impl Property {
    fn parse(string: &str) -> anyhow::Result<Self> {
        if let Some(remainder) = string.strip_prefix("spawn/") {
            Ok(Self::Spawn(remainder.to_string()))
        } else if let Some(remainder) = string.strip_prefix("render_order/") {
            let order = remainder.parse()?;
            Ok(Self::RenderOrder(order))
        } else {
            match string {
                "nocollide" => Ok(Self::NoCollide),
                _ => Err(anyhow::anyhow!("Unrecognised string '{}'", string)),
            }
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

fn ordered_mesh_nodes<'a>(
    gltf: &'a gltf::Document,
    properties: &HashMap<usize, Property>,
) -> impl Iterator<Item = (gltf::Node<'a>, gltf::Mesh<'a>)> + 'a {
    let mut nodes: Vec<_> = gltf
        .nodes()
        .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        .map(|(node, mesh)| {
            let order = match properties.get(&node.index()) {
                Some(Property::RenderOrder(order)) => *order,
                _ => 0,
            };

            (node, mesh, order)
        })
        .collect();

    nodes.sort_by_key(|&(.., order)| order);

    nodes.into_iter().map(|(node, mesh, _)| (node, mesh))
}

fn load_properties(gltf: &gltf::Document) -> anyhow::Result<HashMap<usize, Property>> {
    gltf.nodes()
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
        .collect()
}

fn load_material_properties(
    gltf: &gltf::Document,
) -> anyhow::Result<HashMap<Option<usize>, MaterialProperty>> {
    gltf.materials()
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
        .collect()
}

pub struct Triangle {
    pub triangle: ncollide3d::shape::Triangle<f32>,
    pub intersection_triangle: IntersectionTriangle,
    bounding_box: collision_octree::BoundingBox,
}

impl Triangle {
    fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let edge_b_a = b - a;
        let edge_c_a = c - a;
        let crossed_normal = edge_b_a.cross(edge_c_a);
        let normal = crossed_normal.normalized();

        Self {
            bounding_box: collision_octree::BoundingBox::from_triangle(a, b, c),
            triangle: ncollide3d::shape::Triangle::new(vec3_into(a), vec3_into(b), vec3_into(c)),
            intersection_triangle: IntersectionTriangle {
                a,
                b,
                c,
                edge_b_a,
                edge_c_a,
                crossed_normal,
                normal,
            },
        }
    }
}

impl collision_octree::HasBoundingBox for Triangle {
    fn bounding_box(&self) -> collision_octree::BoundingBox {
        self.bounding_box
    }
}

pub struct Level {
    pub model: Model,
    pub lights_bind_group: wgpu::BindGroup,
    pub properties: HashMap<usize, Property>,
    pub node_tree: NodeTree,
    pub collision_mesh: ncollide3d::shape::TriMesh<f32>,
    pub nav_mesh: (Vec<Vec3>, Vec<u32>),
    pub collision_octree: collision_octree::Octree<Triangle>,
}

impl Level {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        array_of_textures: &mut ArrayOfTextures,
        staging_buffers: &mut StagingModelBuffers<Vertex>,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;

        let node_tree = NodeTree::new(&gltf);

        let properties = load_properties(&gltf)?;

        let material_properties = load_material_properties(&gltf)?;

        let buffer_blob = gltf.blob.as_ref().unwrap();

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut alpha_blend_geometry = StagingModelBuffers::default();
        let mut alpha_clip_geometry = StagingModelBuffers::default();
        let mut collision_geometry = StagingModelBuffers::default();

        let image_index_to_array_index = load_images_into_array(
            &gltf,
            buffer_blob,
            renderer,
            encoder,
            array_of_textures,
            name,
        )?;

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

        for (node, mesh) in ordered_mesh_nodes(&gltf, &properties) {
            let collide = match properties.get(&node.index()) {
                Some(Property::NoCollide) => false,
                Some(Property::Spawn(_)) => continue,
                Some(Property::RenderOrder(_)) => true,
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

                let staging_buffers = match primitive.material().alpha_mode() {
                    gltf::material::AlphaMode::Blend => &mut alpha_blend_geometry,
                    gltf::material::AlphaMode::Opaque => &mut opaque_geometry,
                    gltf::material::AlphaMode::Mask => &mut alpha_clip_geometry,
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
                    &image_index_to_array_index,
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

        let collision_triangles: Vec<_> = collision_geometry
            .indices
            .chunks(3)
            .map(|chunk| {
                Triangle::new(
                    collision_geometry.vertices[chunk[0] as usize].0,
                    collision_geometry.vertices[chunk[1] as usize].0,
                    collision_geometry.vertices[chunk[2] as usize].0,
                )
            })
            .collect();

        let collision_octree = collision_octree::Octree::construct(collision_triangles);

        collision_octree.debug_print_sizes();

        let nav_mesh = create_navmesh(&collision_geometry);

        println!(
            "'{}' level loaded. Vertices: {}. Indices: {}. Images: {}. Lights: {}. Nav mesh vertices: {}. Nav mesh indices: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.images().count() as u32,
            lights.len(),
            nav_mesh.vertices.len(), nav_mesh.indices.len(),
        );

        Ok(Self {
            model: Model {
                opaque_geometry: staging_buffers.merge(opaque_geometry),
                alpha_clip_geometry: staging_buffers.merge(alpha_clip_geometry),
                alpha_blend_geometry: staging_buffers.merge(alpha_blend_geometry),
            },
            lights_bind_group,
            properties,
            node_tree,
            collision_mesh,
            nav_mesh: (nav_mesh.vertices, nav_mesh.indices),
            collision_octree,
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
    pub opaque_geometry: IndexBufferView,
    pub alpha_clip_geometry: IndexBufferView,
    pub alpha_blend_geometry: IndexBufferView,
}

impl Model {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        array_of_textures: &mut ArrayOfTextures,
        staging_buffers: &mut StagingModelBuffers<Vertex>,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;
        let node_tree = NodeTree::new(&gltf);

        let properties = load_properties(&gltf)?;
        let material_properties = load_material_properties(&gltf)?;

        let buffer_blob = gltf.blob.as_ref().unwrap();

        let image_index_to_array_index = load_images_into_array(
            &gltf,
            buffer_blob,
            renderer,
            encoder,
            array_of_textures,
            name,
        )?;

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut alpha_blend_geometry = StagingModelBuffers::default();
        let mut alpha_clip_geometry = StagingModelBuffers::default();

        assert_eq!(gltf.skins().count(), 0);

        for (node, mesh) in ordered_mesh_nodes(&gltf, &properties) {
            let transform = node_tree.transform_of(node.index());
            let normal_matrix = normal_matrix(transform);

            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(anyhow::anyhow!(
                        "Primitives mode {:?} is not implemented.",
                        primitive.mode()
                    ));
                }

                let staging_buffers = match primitive.material().alpha_mode() {
                    gltf::material::AlphaMode::Blend => &mut alpha_blend_geometry,
                    gltf::material::AlphaMode::Opaque => &mut opaque_geometry,
                    gltf::material::AlphaMode::Mask => &mut alpha_clip_geometry,
                };

                add_primitive_geometry_to_buffers(
                    &primitive,
                    &node,
                    transform,
                    normal_matrix,
                    buffer_blob,
                    &material_properties,
                    staging_buffers,
                    None,
                    &image_index_to_array_index,
                )?;
            }
        }

        println!(
            "'{}' model loaded. Vertices: {}. Indices: {}. Images: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.images().count() as u32,
        );

        Ok(Self {
            opaque_geometry: staging_buffers.merge(opaque_geometry),
            alpha_clip_geometry: staging_buffers.merge(alpha_clip_geometry),
            alpha_blend_geometry: staging_buffers.merge(alpha_blend_geometry),
        })
    }
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

    // It turns out that we can just reverse the array to iter through nodes depth first! Useful for applying animations.
    fn iter_depth_first(&self) -> impl Iterator<Item = (usize, Option<usize>)> + '_ {
        self.inner
            .iter()
            .enumerate()
            .rev()
            .map(|(index, &(_, parent))| {
                (
                    index,
                    if parent != usize::max_value() {
                        Some(parent)
                    } else {
                        None
                    },
                )
            })
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
    image_index_to_array_index: &[usize],
) -> anyhow::Result<()> {
    let emission_strength = match material_properties.get(&primitive.material().index()) {
        Some(MaterialProperty::EmissionStrength(strength)) => *strength,
        _ => 0.0,
    };

    let image_index = primitive
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
        .source()
        .index();

    let array_index = image_index_to_array_index[image_index];

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
            let position = position.xyz();

            let normal: Vec3 = n.into();
            let normal = (normal_matrix * normal).normalized();

            staging_buffers.vertices.push(Vertex {
                position,
                normal,
                uv: uv.into(),
                texture_index: array_index as u32,
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

fn load_images_into_array(
    gltf: &gltf::Gltf,
    buffer_blob: &[u8],
    renderer: &Renderer,
    encoder: &mut wgpu::CommandEncoder,
    array_of_textures: &mut ArrayOfTextures,
    name: &str,
) -> anyhow::Result<Vec<usize>> {
    gltf.images()
        .map(|image| {
            let view = match image.source() {
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
            let image_name = image.name().unwrap();

            let image =
                image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.into_rgba8();

            let index = array_of_textures.add(
                &image,
                &format!("{} - {}", name, image_name),
                &renderer,
                encoder,
            );

            Ok(index)
        })
        .collect()
}

pub fn load_single_texture(
    png_bytes: &[u8],
    renderer: &Renderer,
    name: &str,
    encoder: &mut wgpu::CommandEncoder,
    array_of_textures: &mut ArrayOfTextures,
) -> anyhow::Result<usize> {
    let image =
        image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)?.into_rgba8();

    Ok(array_of_textures.add(&image, name, &renderer, encoder))
}
