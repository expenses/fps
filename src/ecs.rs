use crate::assets::{AnimationJoints, Level};
use crate::intersection_maths::{ray_bounding_box_intersection, ray_triangle_intersection};
use crate::{renderer, ModelBuffers};
use collision_octree::HasBoundingBox;
use renderer::{AnimatedInstance, Instance};
use ultraviolet::{Rotor3, Vec3, Vec4};

pub struct StaticModel(pub usize);
pub struct AnimatedModel(pub usize);

pub use ultraviolet::transform::Isometry3;

pub struct VisionCone {
    section: VisionSphereSection,
    node_index: usize,
    vision_distance: f32,
    field_of_view: f32,
}

impl VisionCone {
    pub fn new(vision_distance: f32, field_of_view: f32, node_index: usize) -> Self {
        Self {
            section: VisionSphereSection::new(vision_distance, field_of_view),
            node_index,
            vision_distance,
            field_of_view,
        }
    }
}

pub struct AnimationState {
    pub animation: usize,
    pub time: f32,
    pub joints: AnimationJoints,
}

pub struct DebugVisionCones(pub renderer::DynamicBuffer<renderer::debug_lines::Vertex>);
pub struct PlayerPosition(pub Vec3);

#[legion::system(for_each)]
fn render_static_models(
    #[resource] model_buffers: &mut ModelBuffers,
    model: &StaticModel,
    isometry: &Isometry3,
) {
    let instances = model_buffers.get_static_buffer(model.0);
    instances.push(Instance::new(isometry.into_homogeneous_matrix()));
}

#[legion::system(for_each)]
fn render_animated_models(
    #[resource] model_buffers: &mut ModelBuffers,
    model: &AnimatedModel,
    isometry: &Isometry3,
    animation_state: &mut AnimationState,
) {
    let &AnimatedModel(index) = model;

    let (instances, model, staging_buffer) = model_buffers.get_animated_buffer(index);
    instances.push(AnimatedInstance::new(
        isometry.into_homogeneous_matrix(),
        index,
        model.num_joints,
    ));

    let animation = &model.animations[animation_state.animation];
    animation_state.time = (animation_state.time + 1.0 / 60.0) % animation.total_time();

    animation.animate(
        &mut animation_state.joints,
        animation_state.time,
        &model.depth_first_nodes,
    );

    for joint_transform in animation_state.joints.iter(
        &model.joint_indices_to_node_indices,
        &model.inverse_bind_matrices,
    ) {
        staging_buffer.push(joint_transform);
    }
}

#[legion::system(for_each)]
fn debug_render_vision_cones(
    #[resource] buffer: &mut DebugVisionCones,
    #[resource] player_position: &PlayerPosition,
    #[resource] level: &Level,
    isometry: &Isometry3,
    vision_cone: &VisionCone,
    animation_state: &AnimationState,
) {
    let joint_transform = animation_state
        .joints
        .get_global_transform(vision_cone.node_index);

    let mut origin_isometry = *isometry;
    origin_isometry.prepend_translation(joint_transform.translation);
    origin_isometry.rotation = origin_isometry.rotation
        * joint_transform.rotation
        * Rotor3::from_rotation_yz(-180_f32.to_radians());

    let mut player_is_visible = vision_cone
        .section
        .intersects(origin_isometry, player_position.0);

    player_is_visible &= {
        let origin = origin_isometry.translation;
        let direction = player_position.0 - origin;
        let magnitude = direction.mag();
        let direction = direction / magnitude;

        let line_bounding_box = collision_octree::BoundingBox::from_line(origin, player_position.0);

        let mut stack = Vec::with_capacity(8);

        !level.collision_octree.intersects(
            |bounding_box| {
                if !line_bounding_box.intersects(bounding_box) {
                    return false;
                }

                ray_bounding_box_intersection(origin, direction, magnitude, bounding_box)
            },
            |triangle| {
                if !line_bounding_box.intersects(triangle.bounding_box()) {
                    return false;
                }

                ray_triangle_intersection(
                    origin,
                    direction,
                    magnitude,
                    triangle.intersection_triangle,
                )
                .is_some()
            },
            &mut stack,
        )
    };

    let colour = if player_is_visible {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        Vec4::new(0.0, 1.0, 0.0, 1.0)
    };

    crate::mesh_generation::vision_sphere_section(
        vision_cone.vision_distance,
        vision_cone.field_of_view,
        origin_isometry,
        &mut buffer.0,
        colour,
    );
}

struct VisionSphereSection {
    vision_distance_sq: f32,
    field_of_view_cosine: f32,
}

impl VisionSphereSection {
    fn new(vision_distance: f32, field_of_view: f32) -> Self {
        Self {
            vision_distance_sq: vision_distance * vision_distance,
            field_of_view_cosine: field_of_view.cos(),
        }
    }

    fn intersects(&self, isometry: Isometry3, point: Vec3) -> bool {
        let direction_vector = point - isometry.translation;

        if direction_vector.mag_sq() > self.vision_distance_sq {
            return false;
        }

        let rotated_direction = isometry.rotation.reversed() * direction_vector;

        let rotated_normal = rotated_direction.normalized();

        let down = -Vec3::unit_y();

        let angle_cosine = down.dot(rotated_normal);

        angle_cosine > self.field_of_view_cosine
    }
}

pub fn render_schedule() -> legion::Schedule {
    legion::Schedule::builder()
        .add_system(render_static_models_system())
        .add_system(render_animated_models_system())
        .add_system(debug_render_vision_cones_system())
        .build()
}
