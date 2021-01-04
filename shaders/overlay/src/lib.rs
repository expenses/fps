#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
#[macro_use]
pub extern crate spirv_std_macros;
use spirv_std::glam::{vec2, Vec2, vec4, Vec4};
use spirv_std::storage_class::{Input, Output, PushConstant};

#[allow(unused_attributes)]
#[derive(Copy, Clone)]
#[spirv(block)]
pub struct ScreenDimensions {
    screen_dimensions: Vec2,
    _unused: u32,
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn main_vs(
    position: Input<Vec2>,
    colour: Input<Vec4>,
    mut out_colour: Output<Vec4>,
    constants: PushConstant<ScreenDimensions>,
    #[spirv(position)] mut out_pos: Output<Vec4>,
) {
    let ScreenDimensions { screen_dimensions, .. } = constants.load();

    out_colour.store(colour.load());

    let position = position.load();
    let adjusted_position = vec2(
        (position.x / screen_dimensions.x * 2.0) - 1.0,
        1.0 - (position.y / screen_dimensions.y * 2.0)
    );

    out_pos.store(vec4(adjusted_position.x, adjusted_position.y, 0.0, 1.0));
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn main_fs(
    colour: Input<Vec4>,
    mut out_colour: Output<Vec4>,
) {
    out_colour.store(colour.load());
}
