use super::{ModelBuffers, StagingModelBuffers, NodeTree, MaterialProperty, normal_matrix, load_texture_array};
use ultraviolet::{Vec3, Vec4, Mat3, Mat4};
use std::collections::HashMap;
use crate::renderer::{Renderer, AnimatedVertex};
use wgpu::util::DeviceExt;

pub struct AnimatedModel {
    pub opaque_geometry: ModelBuffers,
    pub transparent_geometry: ModelBuffers,
    pub bind_group: wgpu::BindGroup,
}

impl AnimatedModel {
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

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut transparent_geometry = StagingModelBuffers::default();

        assert_eq!(gltf.skins().count(), 1);

        let skin = gltf.skins().next().unwrap();
        let num_joints = skin.joints().count() as u32;

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            assert!(node.skin().is_some());

            // Not sure if we can do transforms on animated models.
            let transform = Mat4::identity(); //node_tree.transform_of(node.index());
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

                add_animated_primitive_geometry_to_buffers(
                    &primitive,
                    &node,
                    transform,
                    normal_matrix,
                    buffer_blob,
                    &HashMap::new(),
                    staging_buffers,
                )?;
            }
        }

        let animated_model_uniform_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} animated model uniform buffer", name)),
                    contents: bytemuck::bytes_of(&num_joints),
                    usage: wgpu::BufferUsage::UNIFORM,
                });

        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} bind group", name)),
                layout: &renderer.animated_model_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&textures),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &animated_model_uniform_buffer,
                            offset: 0,
                            size: None,
                        },
                    },
                ],
            });

        println!(
            "'{}' animated model loaded. Vertices: {}. Indices: {}. Textures: {} Transparent indices: {}. Transparent vertices: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.textures().count() as u32,
            transparent_geometry.indices.len(),
            transparent_geometry.vertices.len(),
        );

        Ok(Self {
            opaque_geometry: opaque_geometry.upload(&renderer.device, &format!("{} opaque", name)),
            transparent_geometry: transparent_geometry
                .upload(&renderer.device, &format!("{} transparent", name)),
            bind_group,
        })
    }
}

fn add_animated_primitive_geometry_to_buffers(
    primitive: &gltf::Primitive,
    node: &gltf::Node,
    transform: Mat4,
    normal_matrix: Mat3,
    buffer_blob: &[u8],
    material_properties: &HashMap<Option<usize>, MaterialProperty>,
    staging_buffers: &mut StagingModelBuffers<AnimatedVertex>,
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

    let positions = reader.read_positions().unwrap();
    let tex_coordinates = reader.read_tex_coords(0).unwrap().into_f32();
    let normals = reader.read_normals().unwrap();
    let joints = reader.read_joints(0).unwrap().into_u16();
    let joint_weights = reader.read_weights(0).unwrap().into_f32();

    positions
        .zip(tex_coordinates)
        .zip(normals)
        .zip(joints)
        .zip(joint_weights)
        .for_each(|((((p, uv), n), j), jw)| {
            let position = transform * Vec4::new(p[0], p[1], p[2], 1.0);
            assert_eq!(position.w, 1.0);
            let position = position.xyz();

            let normal: Vec3 = n.into();
            let normal = (normal_matrix * normal).normalized();

            staging_buffers.vertices.push(AnimatedVertex {
                position,
                normal,
                uv: uv.into(),
                texture_index: texture_index as i32,
                emission_strength,
                joints: j,
                joint_weights: jw.into(),
            });
        });

    Ok(())
}
