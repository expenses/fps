#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
#[macro_use]
pub extern crate spirv_std_macros;
use spirv_std::glam::{vec4, vec3, Vec2, Vec3, Vec4, Mat3, Mat4};
use spirv_std::storage_class::{Input, Output, PushConstant, UniformConstant, StorageBuffer};
use spirv_std::{Sampler, Image2DArray, ImageFormat};
use spirv_std::num_traits::{clamp, Pow};

#[allow(unused_attributes)]
#[derive(Copy, Clone)]
#[spirv(block)]
pub struct ProjectionView {
    projection_view: Mat4,
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn main_vs(
    pos: Input<Vec3>,
    normal: Input<Vec3>,
    uv: Input<Vec2>,
    texture_index: Input<i32>,
    emission: Input<f32>,

    transform_1: Input<Vec4>,
    transform_2: Input<Vec4>,
    transform_3: Input<Vec4>,
    transform_4: Input<Vec4>,

    normal_transform_1: Input<Vec3>,
    normal_transform_2: Input<Vec3>,
    normal_transform_3: Input<Vec3>,

    mut out_uv: Output<Vec2>,
    #[spirv(flat)] mut out_texture_index: Output<i32>,
    mut out_pos: Output<Vec3>,
    mut out_normal: Output<Vec3>,
    mut out_emission: Output<f32>,

    constants: PushConstant<ProjectionView>,
    #[spirv(position)] mut spirv_position: Output<Vec4>,
) {
    let ProjectionView { projection_view } = constants.load();

    let transform = Mat4::from_cols(
        transform_1.load(), transform_2.load(), transform_3.load(), transform_4.load(),
    );

    let normal_transform = Mat3::from_cols(
        normal_transform_1.load(), normal_transform_2.load(), normal_transform_3.load(),
    );

    let transformed_pos = transform * pos.load().extend(1.0);

    out_uv.store(uv.load());
    out_texture_index.store(texture_index.load());
    out_pos.store(transformed_pos.truncate());
    out_normal.store(normal_transform * normal.load());
    out_emission.store(emission.load());

    spirv_position.store(projection_view * transformed_pos);
}

#[allow(unused_attributes)]
#[derive(Copy, Clone)]
#[spirv(block)]
pub struct Light {
    colour_output: Vec3,
    range: f32,
    position: Vec3,
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn main_fs(
    uv: Input<Vec2>,
    #[spirv(flat)] texture_index: Input<i32>,
    pos: Input<Vec3>,
    normal: Input<Vec3>,
    emission: Input<f32>,

    mut colour: Output<Vec4>,

    #[spirv(descriptor_set = 0, binding = 0)] sampler: UniformConstant<Sampler>,
    #[spirv(descriptor_set = 1, binding = 0)] image: UniformConstant<Image2DArray>,
    #[spirv(descriptor_set = 2, binding = 0)] lights: StorageBuffer<Light>,
) {
    let normal = normal.load().normalize();

    let ambient = Vec3::new(0.05, 0.05, 0.05);
    let mut total = ambient;

    let lights = lights.load();

    {
        let light = lights;

        let vector = light.position - pos.load();
        let distance = vector.length();
        let attenuation = clamp(1.0 - (distance / light.range), 0.0, 1.0) / (distance * distance);

        let light_dir = vector.normalize();
        let facing = normal.dot(light_dir).max(0.0);

        total += (facing * attenuation) * light.colour_output;
    }

    let uv = uv.load();
    let uv = vec3(uv.x, uv.y, texture_index.load() as f32);

    let sampled = image.load().sample(sampler.load(), uv);
    let colour_rbg = sampled.truncate() * (total + Vec3::splat(emission.load()));

    colour.store(colour_rbg.extend(sampled.w));
}
