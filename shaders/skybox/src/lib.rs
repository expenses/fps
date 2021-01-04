#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
#[macro_use]
pub extern crate spirv_std_macros;
use spirv_std::glam::{vec4, Vec3, Vec4, Mat3, Mat4};
use spirv_std::storage_class::{Input, Output, PushConstant, UniformConstant};
use spirv_std::{Sampler, ImageCube, ImageFormat};

#[allow(unused_attributes)]
#[derive(Copy, Clone)]
#[spirv(block)]
pub struct ProjectionAndView {
    projection: Mat4,
    view: Mat4,
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(vertex_index)] vertex_index: Input<i32>,
    constants: PushConstant<ProjectionAndView>,
    mut out_uv: Output<Vec3>,
    #[spirv(position)] mut out_pos: Output<Vec4>,
) {
    let ProjectionAndView { projection, view } = constants.load();

    let pos = match vertex_index.load() {
        0 => vec4(-1.0, -1.0, 0.0, 1.0),
        1 => vec4(3.0, -1.0, 0.0, 1.0),
        2 => vec4(-1.0, 3.0, 0.0, 1.0),
        _ => Vec4::zero()
    };

    let inv_view = mat4_to_mat3(view).transpose();
    let unprojected = (projection.inverse() * pos).truncate();
    out_uv.store(inv_view * unprojected);

    out_pos.store(vec4(pos.x, pos.y, pos.w, pos.w));
}

fn mat4_to_mat3(mat4: Mat4) -> Mat3 {
    Mat3::from_cols(mat4.x_axis.truncate(), mat4.y_axis.truncate(), mat4.z_axis.truncate())
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn main_fs(
    uv: Input<Vec3>,
    mut colour: Output<Vec4>,
    #[spirv(descriptor_set = 0, binding = 0)] sampler: UniformConstant<Sampler>,
    #[spirv(descriptor_set = 1, binding = 0)] image_cube: UniformConstant<ImageCube<{ ImageFormat::Unknown }>>,
) {
    colour.store(image_cube.load().sample(sampler.load(), uv.load()));
}
