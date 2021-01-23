use ultraviolet::{Isometry3, Vec3, Vec4};
use crate::renderer::{DynamicBuffer, debug_lines::{self, Vertex as LineVertex}};

pub fn vision_sphere_section(
    distance: f32, field_of_view: f32,
    isometry: Isometry3, buffer: &mut DynamicBuffer<LineVertex>, colour: Vec4,
) {
    let end_points = [
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -field_of_view.cos(), field_of_view.sin()),
        Vec3::new(field_of_view.sin(), -field_of_view.cos(), 0.0),
        Vec3::new(0.0, -field_of_view.cos(), -field_of_view.sin()),
        Vec3::new(-field_of_view.sin(), -field_of_view.cos(), 0.0)
    ];

    for &end_point in &end_points {
        debug_lines::draw_line(isometry.translation, isometry * (end_point * distance), colour, |vertex| {
            buffer.push(vertex);
        });
    }

    let cap_pairs = [
        [end_points[1], end_points[2]],
        [end_points[2], end_points[3]],
        [end_points[3], end_points[4]],
        [end_points[4], end_points[1]],

        [end_points[0], end_points[1]],
        [end_points[0], end_points[2]],
        [end_points[0], end_points[3]],
        [end_points[0], end_points[4]],
    ];

    for &[a, b] in &cap_pairs {
        let a = isometry * (a * distance);
        let b = isometry * (b * distance);

        debug_lines::draw_line(a, b, colour, |vertex| {
            buffer.push(vertex);
        });
    }
}
