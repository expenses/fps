use gltf::animation::Interpolation;
use ultraviolet::{Isometry3, Lerp, Mat4, Rotor3, Slerp, Vec3};

pub fn read_animations(
    gltf: &gltf::Document,
    gltf_binary_buffer_blob: &[u8],
    model_name: &str,
) -> Vec<Animation> {
    gltf.animations()
        .map(|animation| {
            let mut translation_channels = Vec::new();
            let mut rotation_channels = Vec::new();

            for (channel_index, channel) in animation.channels().enumerate() {
                let reader = channel.reader(|buffer| {
                    assert_eq!(buffer.index(), 0);
                    Some(gltf_binary_buffer_blob)
                });

                let inputs = reader.read_inputs().unwrap().collect();

                log::trace!(
                    "[{}] animation {:?}, channel {} ({:?}) uses {:?} interpolation.",
                    model_name,
                    animation.name(),
                    channel_index,
                    channel.target().property(),
                    channel.sampler().interpolation()
                );

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
                        log::warn!(
                            "[{}] Animation type {:?} is not supported, ignoring.",
                            model_name,
                            property
                        );
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

            Animation {
                total_time,
                translation_channels,
                rotation_channels,
            }
        })
        .collect()
}

#[derive(Clone)]
pub struct AnimationJoints {
    global_transforms: Vec<Isometry3>,
    local_transforms: Vec<Isometry3>,
}

impl AnimationJoints {
    pub fn new(
        joint_isometries: Vec<Isometry3>,
        depth_first_nodes: &[(usize, Option<usize>)],
    ) -> Self {
        let mut joints = Self {
            global_transforms: joint_isometries.clone(),
            local_transforms: joint_isometries,
        };

        joints.update(depth_first_nodes);

        joints
    }

    pub fn iter<'a>(
        &'a self,
        joint_indices_to_node_indices: &'a [usize],
        inverse_bind_matrices: &'a [Mat4],
    ) -> impl Iterator<Item = Mat4> + 'a {
        joint_indices_to_node_indices
            .iter()
            .enumerate()
            .map(move |(joint_index, &node_index)| {
                self.global_transforms[node_index].into_homogeneous_matrix()
                    * inverse_bind_matrices[joint_index]
            })
    }

    fn update(&mut self, depth_first_nodes: &[(usize, Option<usize>)]) {
        for &(index, parent) in depth_first_nodes.iter() {
            if let Some(parent) = parent {
                let parent_transform = self.global_transforms[parent];
                self.global_transforms[index] = parent_transform * self.local_transforms[index];
            }
        }
    }

    pub fn get_global_transform(&self, node_index: usize) -> Isometry3 {
        self.global_transforms[node_index]
    }
}

#[derive(Debug)]
struct Channel<T> {
    interpolation: Interpolation,
    inputs: Vec<f32>,
    outputs: Vec<T>,
    node_index: usize,
}

impl<T: Interpolate> Channel<T> {
    fn sample(&self, t: f32) -> Option<(usize, T)> {
        if t < self.inputs[0] || t > self.inputs[self.inputs.len() - 1] {
            return None;
        }

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

        Some((self.node_index, i))
    }
}

#[derive(Debug)]
pub struct Animation {
    total_time: f32,
    translation_channels: Vec<Channel<Vec3>>,
    rotation_channels: Vec<Channel<Rotor3>>,
}

impl Animation {
    pub fn total_time(&self) -> f32 {
        self.total_time
    }

    pub fn animate(
        &self,
        animation_joints: &mut AnimationJoints,
        time: f32,
        depth_first_nodes: &[(usize, Option<usize>)],
    ) {
        self.translation_channels
            .iter()
            .filter_map(move |channel| channel.sample(time))
            .for_each(|(node_index, translation)| {
                animation_joints.local_transforms[node_index].translation = translation;
            });

        self.rotation_channels
            .iter()
            .filter_map(move |channel| channel.sample(time))
            .for_each(|(node_index, rotation)| {
                animation_joints.local_transforms[node_index].rotation = rotation;
            });

        animation_joints.update(depth_first_nodes);
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