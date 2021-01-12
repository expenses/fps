use super::{
    load_images_into_array, load_material_properties, normal_matrix, IndexBufferView,
    MaterialProperty, NodeTree, StagingModelBuffers, ordered_mesh_nodes,
    load_properties,
};
use crate::array_of_textures::ArrayOfTextures;
use crate::renderer::{AnimatedVertex, Renderer};
use animation::{Animation, AnimationJoints};
use std::collections::HashMap;
use ultraviolet::{Mat3, Mat4, Vec3, Vec4};

pub struct AnimatedModel {
    pub opaque_geometry: IndexBufferView,
    pub alpha_clip_geometry: IndexBufferView,
    pub alpha_blend_geometry: IndexBufferView,

    pub animations: Vec<Animation>,
    pub num_joints: u32,
    pub animation_joints: AnimationJoints,

    pub joint_indices_to_node_indices: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
    pub depth_first_nodes: Vec<(usize, Option<usize>)>,

    pub name: String,
}

impl AnimatedModel {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        getter: impl FnOnce(gltf::iter::Animations, Option<gltf::skin::iter::Joints>),
        array_of_textures: &mut ArrayOfTextures,
        staging_buffers: &mut StagingModelBuffers<AnimatedVertex>,
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

        assert!(gltf.skins().count() <= 1);
        let skin = gltf.skins().next();
        if let Some(skin) = skin.as_ref() {
            assert!(skin.skeleton().is_none());
        }

        for (node, mesh) in ordered_mesh_nodes(&gltf, &properties) {
            let transform = if node.skin().is_some() {
                // We can't apply transformations on animated models, but we also don't need to..
                Mat4::identity()
            } else {
                // We allow combining animated and non animated meshes
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
                    &material_properties,
                    staging_buffers,
                    &image_index_to_array_index,
                    skin.is_some(),
                )?;
            }
        }

        assert_eq!(gltf.scenes().count(), 1);
        let scene = gltf.scenes().next().unwrap();
        let root_nodes = scene.nodes().count();

        if root_nodes > 1 {
            log::warn!("[{}] There are {} root nodes in the scene instead of the expected 1. Are you exporting lights/cameras?", name, root_nodes);
            for (i, node) in scene.nodes().enumerate() {
                log::warn!("root node {}: {:?}", i, node.name().unwrap());
            }
        }

        let animations = animation::read_animations(gltf.animations(), buffer_blob, name);

        getter(gltf.animations(), skin.as_ref().map(|skin| skin.joints()));

        let (joint_indices_to_node_indices, num_joints) = if let Some(skin) = skin.as_ref() {
            (
                skin.joints().map(|node| node.index()).collect(),
                skin.joints().count() as u32,
            )
        } else {
            (
                gltf.nodes().map(|node| node.index()).collect(),
                gltf.nodes().count() as u32,
            )
        };

        println!(
            "'{}' animated model loaded. Vertices: {}. Indices: {}. Images: {}. Joints: {}. Animations: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.images().count() as u32,
            num_joints,
            animations.len(),
        );

        log::info!("Joints: {}, Nodes: {}", num_joints, gltf.nodes().count());

        let depth_first_nodes: Vec<_> = node_tree.iter_depth_first().collect();
        let animation_joints = AnimationJoints::new(gltf.nodes(), &depth_first_nodes[..]);

        let inverse_bind_matrices = if let Some(skin) = skin.as_ref() {
            skin.reader(|buffer| {
                assert_eq!(buffer.index(), 0);
                Some(buffer_blob)
            })
            .read_inverse_bind_matrices()
            .ok_or_else(|| anyhow::anyhow!("Missing inverse bind matrices"))?
            .map(|mat| mat.into())
            .collect()
        } else {
            gltf.nodes()
                .map(|node| node_tree.transform_of(node.index()).inversed())
                .collect()
        };

        Ok(Self {
            opaque_geometry: staging_buffers.merge(opaque_geometry.clone()),
            alpha_clip_geometry: staging_buffers.merge(alpha_clip_geometry.clone()),
            alpha_blend_geometry: staging_buffers.merge(alpha_blend_geometry.clone()),

            animations,
            num_joints,

            animation_joints,

            joint_indices_to_node_indices,
            inverse_bind_matrices,
            depth_first_nodes,
            name: name.to_string(),
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
    is_skinned: bool,
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

    let mut joints = reader.read_joints(0).map(|iter| iter.into_u16());
    let mut joint_weights = reader.read_weights(0).map(|iter| iter.into_f32());

    positions
        .zip(tex_coordinates)
        .zip(normals)
        .for_each(|((p, uv), n)| {
            let j = match joints {
                Some(ref mut iter) => iter.next().unwrap(),
                // If the mesh is skinned we use the root joint (presumed to be 0), otherwise we're
                // just using the nodes as joints and should use the node index instead.
                None => {
                    let joint = if is_skinned { 0 } else { node.index() as u16 };

                    [joint; 4]
                }
            };
            let jw = match joint_weights {
                Some(ref mut iter) => iter.next().unwrap(),
                None => [1.0, 0.0, 0.0, 0.0],
            };

            let position = transform * Vec4::new(p[0], p[1], p[2], 1.0);
            assert_eq!(position.w, 1.0);
            let position = position.xyz();

            let normal: Vec3 = n.into();
            let normal = (normal_matrix * normal).normalized();

            staging_buffers.vertices.push(AnimatedVertex {
                position,
                normal,
                uv: uv.into(),
                texture_index: array_index as u32,
                emission_strength,
                joints: j,
                joint_weights: jw.into(),
            });
        });

    Ok(())
}
