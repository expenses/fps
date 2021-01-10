use super::{
    load_images_into_array, normal_matrix, MaterialProperty, ModelBuffers, NodeTree,
    StagingModelBuffers,
};
use crate::array_of_textures::ArrayOfTextures;
use crate::renderer::{AnimatedVertex, Renderer};
use animation::{Animation, AnimationJoints};
use std::collections::HashMap;
use ultraviolet::{Isometry3, Mat3, Mat4, Rotor3, Vec3, Vec4};
use wgpu::util::DeviceExt;

pub struct AnimatedModel {
    pub opaque_geometry: Option<ModelBuffers>,
    pub alpha_clip_geometry: Option<ModelBuffers>,
    pub alpha_blend_geometry: Option<ModelBuffers>,

    pub bind_group: wgpu::BindGroup,
    pub animations: Vec<Animation>,
    pub num_joints: usize,
    pub animation_joints: AnimationJoints,

    pub joint_indices_to_node_indices: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
    pub depth_first_nodes: Vec<(usize, Option<usize>)>,
}

impl AnimatedModel {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        getter: impl FnOnce(gltf::iter::Animations, gltf::skin::iter::Joints),
        array_of_textures: &mut ArrayOfTextures,
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;
        let node_tree = NodeTree::new(&gltf);

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

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            assert!(node.skin().is_some());

            // We can't apply transformations on animated models, but we also don't need to..
            let transform = Mat4::identity();
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

                add_animated_primitive_geometry_to_buffers(
                    &primitive,
                    &node,
                    transform,
                    normal_matrix,
                    buffer_blob,
                    &HashMap::new(),
                    staging_buffers,
                    &image_index_to_array_index,
                )?;
            }
        }

        assert_eq!(gltf.skins().count(), 1);
        let skin = gltf.skins().next().unwrap();
        assert!(skin.skeleton().is_none());

        assert_eq!(gltf.scenes().count(), 1);
        let scene = gltf.scenes().next().unwrap();
        let root_nodes = scene.nodes().count();

        if root_nodes > 1 {
            log::warn!("[{}] There are {} root nodes in the scene instead of the expected 1. Are you exporting lights/cameras?", name, root_nodes);
            for (i, node) in scene.nodes().enumerate() {
                log::warn!("root node {}: {:?}", i, node.name().unwrap());
            }
        }

        let animations = animation::read_animations(&gltf, buffer_blob, name);

        getter(gltf.animations(), skin.joints());

        let num_joints = skin.joints().count();

        let animated_model_uniform_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} animated model uniform buffer", name)),
                    contents: bytemuck::bytes_of(&(num_joints as u32)),
                    usage: wgpu::BufferUsage::UNIFORM,
                });

        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} bind group", name)),
                layout: &renderer.animated_model_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &animated_model_uniform_buffer,
                        offset: 0,
                        size: None,
                    },
                }],
            });

        println!(
            "'{}' animated model loaded. Vertices: {}. Indices: {}. Textures: {}. Animations: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.textures().count() as u32,
            animations.len(),
        );

        log::info!("Joints: {}, Nodes: {}", num_joints, gltf.nodes().count());

        let joint_isometries: Vec<_> = gltf
            .nodes()
            .map(|node| {
                let (translation, rotation, _) = node.transform().decomposed();
                let translation = Vec3::from(translation);
                let rotation = Rotor3::from_quaternion_array(rotation);
                Isometry3::new(translation, rotation)
            })
            .collect();

        let depth_first_nodes: Vec<_> = node_tree.iter_depth_first().collect();
        let animation_joints = AnimationJoints::new(joint_isometries, &depth_first_nodes[..]);

        Ok(Self {
            opaque_geometry: opaque_geometry
                .upload(&renderer.device, &format!("{} level opaque", name)),
            alpha_blend_geometry: alpha_blend_geometry
                .upload(&renderer.device, &format!("{} level alpha blend", name)),
            alpha_clip_geometry: alpha_clip_geometry
                .upload(&renderer.device, &format!("{} level alpha clip", name)),
            bind_group,
            animations,
            num_joints,

            animation_joints,

            joint_indices_to_node_indices: skin.joints().map(|node| node.index()).collect(),
            inverse_bind_matrices: skin
                .reader(|buffer| {
                    assert_eq!(buffer.index(), 0);
                    Some(buffer_blob)
                })
                .read_inverse_bind_matrices()
                .ok_or_else(|| anyhow::anyhow!("Missing inverse bind matrices"))?
                .map(|mat| mat.into())
                .collect(),
            depth_first_nodes,
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
                texture_index: array_index as i32,
                emission_strength,
                joints: j,
                joint_weights: jw.into(),
            });
        });

    Ok(())
}
