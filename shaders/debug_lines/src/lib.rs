#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
#[macro_use]
pub extern crate spirv_std_macros;
use spirv_std::glam::{vec2, Vec2, Vec3, vec4, Vec4, Mat4};
use spirv_std::storage_class::{Input, Output, PushConstant};

#[allow(unused_attributes)]
#[derive(Copy, Clone)]
#[spirv(block)]
pub struct ProjectionView {
    projection_view: Mat4,
    //_unused: u32,
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn main_vs(
    position: Input<Vec3>,
    colour: Input<Vec4>,
    mut out_colour: Output<Vec4>,
    constants: PushConstant<ProjectionView>,
    #[spirv(position)] mut out_pos: Output<Vec4>,
) {
    let ProjectionView { projection_view, .. } = constants.load();

    out_colour.store(colour.load());

    let mut out_position: Vec4 = projection_view * position.load().extend(1.0);
    out_position.z *= 0.9999;

    out_pos.store(out_position);
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn main_fs(
    colour: Input<Vec4>,
    mut out_colour: Output<Vec4>,
) {
    out_colour.store(colour.load());
}
