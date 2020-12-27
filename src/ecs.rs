use crate::{Model, ModelBuffers, renderer};
use ultraviolet::{Vec3, Vec4, Rotor3};
use ncollide3d::transformation::ToTriMesh;
use ncollide3d::query::PointQuery;

pub use ultraviolet::transform::Isometry3;
pub struct VisionCone(pub ncollide3d::shape::Cone<f32>);
pub struct DebugVisionCones(pub renderer::DynamicBuffer<renderer::debug_lines::Vertex>);
pub struct PlayerPosition(pub Vec3);

#[legion::system(for_each)]
fn render_models(
    #[resource] model_buffers: &mut ModelBuffers,
    model: &Model,
    isometry: &Isometry3,
) {
    model_buffers.push(model, isometry.into_homogeneous_matrix())
}

#[legion::system(for_each)]
pub fn debug_render_vision_cones(
    #[resource] buffer: &mut DebugVisionCones,
    #[resource] player_position: &PlayerPosition,
    isometry: &Isometry3,
    vision_cone: &VisionCone,
) {
    let mut isometry = *isometry;
    isometry.rotation = isometry.rotation * Rotor3::from_rotation_yz(-90_f32.to_radians());
    isometry.prepend_translation(Vec3::new(0.0, -vision_cone.0.half_height, 0.0));

    let na_isometry = isometry3_to_na_isometry(isometry);

    let player_position: [f32; 3] = player_position.0.into();
    let player_is_visible = vision_cone.0.contains_point(&na_isometry, &player_position.into());

    let colour = if player_is_visible {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        Vec4::new(0.0, 1.0, 0.0, 1.0)
    };

    crate::render_debug_mesh(&vision_cone.0.to_trimesh(8).into(), &isometry, &mut buffer.0, colour);
}

pub fn render_schedule() -> legion::Schedule {
    legion::Schedule::builder()
        .add_system(render_models_system())
        .add_system(debug_render_vision_cones_system())
        .build()
}

fn isometry3_to_na_isometry(isometry: Isometry3) -> ncollide3d::math::Isometry<f32> {
    let rotation_arr = *isometry.rotation.into_matrix().as_array();
    let rotation_mat = ncollide3d::na::base::Matrix3::<f32>::from_column_slice(&rotation_arr[..]);
    let translation: [f32; 3] = isometry.translation.into();
    let translation = ncollide3d::math::Vector::from(translation);
    ncollide3d::math::Isometry::from_parts(
        ncollide3d::na::geometry::Translation::from(translation),
        ncollide3d::na::geometry::UnitQuaternion::from_matrix(&rotation_mat),
    )
}
