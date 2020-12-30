use super::{
    load_texture_array, normal_matrix, MaterialProperty, ModelBuffers, NodeTree,
    StagingModelBuffers,
};
use crate::renderer::{AnimatedVertex, Renderer};
use gltf::animation::Interpolation;
use std::collections::HashMap;
use ultraviolet::{Isometry3, Lerp, Mat3, Mat4, Rotor3, Slerp, Vec3, Vec4};
use wgpu::util::DeviceExt;

pub struct AnimatedModel {
    pub opaque_geometry: ModelBuffers,
    pub transparent_geometry: ModelBuffers,
    pub bind_group: wgpu::BindGroup,
    pub animations: Vec<Animation>,
    pub num_joints: usize,
    pub animation_joints: AnimationJoints,

    joint_indices_to_node_indices: Vec<usize>,
    inverse_bind_matrices: Vec<Mat4>,
    depth_first_nodes: Vec<(usize, Option<usize>)>,
}

impl AnimatedModel {
    pub fn load_gltf(
        gltf_bytes: &[u8],
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        getter: impl FnOnce(gltf::iter::Animations, gltf::skin::iter::Joints),
    ) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(gltf_bytes)?;
        let node_tree = NodeTree::new(&gltf);

        let buffer_blob = gltf.blob.as_ref().unwrap();

        let textures = load_texture_array(&gltf, buffer_blob, renderer, encoder, name)?;

        let mut opaque_geometry = StagingModelBuffers::default();
        let mut transparent_geometry = StagingModelBuffers::default();

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

        assert_eq!(gltf.skins().count(), 1);
        let skin = gltf.skins().next().unwrap();
        assert!(skin.skeleton().is_none());

        assert_eq!(gltf.scenes().count(), 1);
        let scene = gltf.scenes().next().unwrap();
        assert_eq!(scene.nodes().count(), 1);

        let mut animations = Vec::new();

        for animation in gltf.animations() {
            let mut translation_channels = Vec::new();
            let mut rotation_channels = Vec::new();

            for channel in animation.channels() {
                let reader = channel.reader(|buffer| {
                    assert_eq!(buffer.index(), 0);
                    Some(buffer_blob)
                });

                let inputs = reader.read_inputs().unwrap().collect();

                match channel.target().property() {
                    gltf::animation::Property::Translation => {
                        let outputs = match reader.read_outputs().unwrap() {
                            gltf::animation::util::ReadOutputs::Translations(translations) => {
                                translations.map(|translation| translation.into()).collect()
                            }
                            _ => unreachable!(),
                        };

                        translation_channels.push(Channel {
                            interpolation: channel.sampler().interpolation(),
                            inputs,
                            outputs,
                            node_index: channel.target().node().index(),
                        });
                    }
                    gltf::animation::Property::Rotation => {
                        let outputs = match reader.read_outputs().unwrap() {
                            gltf::animation::util::ReadOutputs::Rotations(rotations) => rotations
                                .into_f32()
                                .map(|rotation| Rotor3::from_quaternion_array(rotation))
                                .collect(),
                            _ => unreachable!(),
                        };

                        rotation_channels.push(Channel {
                            interpolation: channel.sampler().interpolation(),
                            inputs,
                            outputs,
                            node_index: channel.target().node().index(),
                        });
                    }
                    property => {
                        return Err(anyhow::anyhow!(
                            "Animation type {:?} is not supported",
                            property
                        ))
                    }
                }
            }

            let total_time = translation_channels
                .iter()
                .map(|channel| channel.inputs[channel.inputs.len() - 1])
                .chain(
                    rotation_channels
                        .iter()
                        .map(|channel| channel.inputs[channel.inputs.len() - 1]),
                )
                .max_by_key(|&time| ordered_float::OrderedFloat(time))
                .unwrap();

            rotation_channels[0].sample(0.4999);

            animations.push(Animation {
                total_time,
                translation_channels,
                rotation_channels,
            });
        }

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
            "'{}' animated model loaded. Vertices: {}. Indices: {}. Textures: {}. Animations: {}",
            name,
            opaque_geometry.vertices.len(),
            opaque_geometry.indices.len(),
            gltf.textures().count() as u32,
            animations.len(),
        );

        let joint_isometries: Vec<_> = gltf
            .nodes()
            .map(|node| {
                let (translation, rotation, _) = node.transform().decomposed();
                let translation = Vec3::from(translation);
                let rotation = Rotor3::from_quaternion_array(rotation);
                Isometry3::new(translation, rotation)
            })
            .collect();

        let mut model = Self {
            opaque_geometry: opaque_geometry.upload(&renderer.device, &format!("{} opaque", name)),
            transparent_geometry: transparent_geometry
                .upload(&renderer.device, &format!("{} transparent", name)),
            bind_group,
            animations,
            num_joints,

            animation_joints: AnimationJoints {
                global_transforms: joint_isometries.clone(),
                local_transforms: joint_isometries,
            },

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
            depth_first_nodes: node_tree.iter_depth_first().collect(),
        };

        model.animation_joints.update(&model.depth_first_nodes);

        Ok(model)
    }
}

#[derive(Clone)]
pub struct AnimationJoints {
    global_transforms: Vec<Isometry3>,
    local_transforms: Vec<Isometry3>,
}

impl AnimationJoints {
    pub fn iter<'a>(&'a self, model: &'a AnimatedModel) -> impl Iterator<Item = Mat4> + 'a {
        model.joint_indices_to_node_indices.iter().enumerate().map(
            move |(joint_index, &node_index)| {
                self.global_transforms[node_index].into_homogeneous_matrix()
                    * model.inverse_bind_matrices[joint_index]
            },
        )
    }

    fn update(&mut self, depth_first_nodes: &[(usize, Option<usize>)]) {
        for &(index, parent) in depth_first_nodes.iter() {
            if let Some(parent) = parent {
                let parent_transform = self.global_transforms[parent];
                self.global_transforms[index] = parent_transform * self.local_transforms[index];
            }
        }
    }

    pub fn get_global_transform(&self, joint_index: usize, model: &AnimatedModel) -> Isometry3 {
        let node_index = model.joint_indices_to_node_indices[joint_index];
        self.global_transforms[node_index]
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

#[derive(Debug)]
struct Channel<T> {
    interpolation: gltf::animation::Interpolation,
    inputs: Vec<f32>,
    outputs: Vec<T>,
    node_index: usize,
}

impl<T: Interpolate> Channel<T> {
    fn sample(&self, t: f32) -> (usize, T) {
        let index = self
            .inputs
            .binary_search_by_key(&ordered_float::OrderedFloat(t), |t| {
                ordered_float::OrderedFloat(*t)
            });
        let i = match index {
            Ok(exact) => exact,
            Err(would_be_inserted_at) => would_be_inserted_at - 1,
        };

        let previous_time = self.inputs[i];
        let next_time = self.inputs[i + 1];
        let delta = next_time - previous_time;
        let from_start = t - previous_time;
        let factor = from_start / delta;

        let i = match self.interpolation {
            Interpolation::Step => self.outputs[i],
            Interpolation::Linear => {
                let previous_value = self.outputs[i];
                let next_value = self.outputs[i + 1];

                previous_value.linear(next_value, factor)
            }
            Interpolation::CubicSpline => {
                let previous_values = [
                    self.outputs[i * 3],
                    self.outputs[i * 3 + 1],
                    self.outputs[i * 3 + 2],
                ];
                let next_values = [
                    self.outputs[i * 3 + 3],
                    self.outputs[i * 3 + 4],
                    self.outputs[i * 3 + 5],
                ];
                Interpolate::cubic_spline(
                    previous_values,
                    previous_time,
                    next_values,
                    next_time,
                    factor,
                )
            }
        };

        (self.node_index, i)
    }
}

#[derive(Debug)]
pub struct Animation {
    pub total_time: f32,
    translation_channels: Vec<Channel<Vec3>>,
    rotation_channels: Vec<Channel<Rotor3>>,
}

impl Animation {
    pub fn animate(
        &self,
        animation_joints: &mut AnimationJoints,
        time: f32,
        model: &AnimatedModel,
    ) {
        self.translation_channels
            .iter()
            .map(move |channel| channel.sample(time))
            .for_each(|(node_index, translation)| {
                animation_joints.local_transforms[node_index].translation = translation;
            });

        self.rotation_channels
            .iter()
            .map(move |channel| channel.sample(time))
            .for_each(|(node_index, rotation)| {
                animation_joints.local_transforms[node_index].rotation = rotation;
            });

        animation_joints.update(&model.depth_first_nodes);
    }
}

trait Interpolate: Copy {
    fn linear(self, other: Self, t: f32) -> Self;

    fn cubic_spline(
        source: [Self; 3],
        source_time: f32,
        target: [Self; 3],
        target_time: f32,
        t: f32,
    ) -> Self;
}

impl Interpolate for Vec3 {
    fn linear(self, other: Self, t: f32) -> Self {
        self.lerp(other, t)
    }

    fn cubic_spline(
        source: [Self; 3],
        source_time: f32,
        target: [Self; 3],
        target_time: f32,
        t: f32,
    ) -> Self {
        let p0 = source[1];
        let m0 = (target_time - source_time) * source[2];
        let p1 = target[1];
        let m1 = (target_time - source_time) * target[0];

        (2.0 * t * t * t - 3.0 * t * t + 1.0) * p0
            + (t * t * t - 2.0 * t * t + t) * m0
            + (-2.0 * t * t * t + 3.0 * t * t) * p1
            + (t * t * t - t * t) * m1
    }
}

impl Interpolate for Rotor3 {
    fn linear(self, other: Self, t: f32) -> Self {
        self.slerp(other, t)
    }

    fn cubic_spline(
        source: [Self; 3],
        source_time: f32,
        target: [Self; 3],
        target_time: f32,
        t: f32,
    ) -> Self {
        let p0 = source[1];
        let m0 = (target_time - source_time) * source[2];
        let p1 = target[1];
        let m1 = (target_time - source_time) * target[0];

        let result = (2.0 * t * t * t - 3.0 * t * t + 1.0) * p0
            + (t * t * t - 2.0 * t * t + t) * m0
            + (-2.0 * t * t * t + 3.0 * t * t) * p1
            + (t * t * t - t * t) * m1;

        result.normalized()
    }
}
