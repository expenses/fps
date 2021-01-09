use crate::assets::AnimationJoints;
use crate::{renderer, vec3_into, Model, ModelBuffers};
use ncollide3d::query::PointQuery;
use ncollide3d::transformation::ToTriMesh;
use ultraviolet::{Rotor3, Vec3, Vec4};

pub use ultraviolet::transform::Isometry3;

pub struct VisionCone {
    cone: ncollide3d::shape::Cone<f32>,
    trimesh: ncollide3d::shape::TriMesh<f32>,
}

pub struct AnimationState {
    pub animation: usize,
    pub time: f32,
    pub joints: AnimationJoints,
}

impl VisionCone {
    pub fn new(cone: ncollide3d::shape::Cone<f32>) -> Self {
        Self {
            trimesh: cone.to_trimesh(8).into(),
            cone,
        }
    }
}

pub struct DebugVisionCones(pub renderer::DynamicBuffer<renderer::debug_lines::Vertex>);
pub struct PlayerPosition(pub Vec3);

#[legion::system(for_each)]
#[filter(!legion::component::<AnimationState>())]
fn render_models(
    #[resource] model_buffers: &mut ModelBuffers,
    model: &Model,
    isometry: &Isometry3,
) {
    let (mut instances, _) = model_buffers.get_buffer(model);
    instances.push(isometry.into_homogeneous_matrix());
}

#[legion::system(for_each)]
fn render_animated_models(
    #[resource] model_buffers: &mut ModelBuffers,
    model: &Model,
    isometry: &Isometry3,
    animation_state: &mut AnimationState,
) {
    let (mut instances, animation_info) = model_buffers.get_buffer(model);
    let (model, joint_buffer) = animation_info.unwrap();
    instances.push(isometry.into_homogeneous_matrix());

    let animation = &model.animations[animation_state.animation];
    animation_state.time = (animation_state.time + 1.0 / 60.0) % animation.total_time();
    animation.animate(
        &mut animation_state.joints, animation_state.time,
        &model.depth_first_nodes,
    );

    for joint_transform in animation_state.joints.iter(&model.joint_indices_to_node_indices, &model.inverse_bind_matrices) {
        joint_buffer.push(joint_transform);
    }
}

#[legion::system(for_each)]
fn debug_render_vision_cones(
    #[resource] buffer: &mut DebugVisionCones,
    #[resource] player_position: &PlayerPosition,
    #[resource] model_buffers: &ModelBuffers,
    isometry: &Isometry3,
    vision_cone: &VisionCone,
    animation_state: &AnimationState,
) {
    let base_node = model_buffers.animation_info.robot_base_node;
    let base_joint = animation_state.joints.get_global_transform(base_node);
    let joint_rotation = base_joint.rotation;

    let mut isometry = *isometry;
    isometry.rotation =
        isometry.rotation * joint_rotation * Rotor3::from_rotation_yz(-90_f32.to_radians());
    isometry.prepend_translation(Vec3::new(0.0, -vision_cone.cone.half_height, 0.0));

    let na_isometry = isometry3_to_na_isometry(isometry);

    let player_is_visible = vision_cone
        .cone
        .contains_point(&na_isometry, &vec3_into(player_position.0));

    let colour = if player_is_visible {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        Vec4::new(0.0, 1.0, 0.0, 1.0)
    };

    crate::render_debug_mesh(&vision_cone.trimesh, &isometry, &mut buffer.0, colour);
}

pub fn render_schedule() -> legion::Schedule {
    legion::Schedule::builder()
        .add_system(render_models_system())
        .add_system(render_animated_models_system())
        .add_system(debug_render_vision_cones_system())
        .build()
}

fn isometry3_to_na_isometry(isometry: Isometry3) -> ncollide3d::math::Isometry<f32> {
    let translation: ncollide3d::math::Vector<f32> = vec3_into(isometry.translation);
    let quaternion: ncollide3d::na::Vector4<f32> = isometry.rotation.into_quaternion_array().into();
    ncollide3d::math::Isometry::from_parts(
        ncollide3d::na::geometry::Translation::from(translation),
        ncollide3d::na::geometry::UnitQuaternion::from_quaternion(
            ncollide3d::na::geometry::Quaternion::from(quaternion),
        ),
    )
}
