use ultraviolet::Vec3;

// Adapted from https://github.com/rustgd/collision-rs/blob/122169f7f804667525dab74678d91012c3a56956/src/volume/aabb/aabb3.rs#L165-L194
// itself copied from... who knows.
pub fn ray_bounding_box_intersection(
    ray_origin: Vec3,
    ray_direction: Vec3,
    max_time_of_impact: f32,
    bounding_box: collision_octree::BoundingBox,
) -> bool {
    let inv_dir = Vec3::one() / ray_direction;

    let mut t1 = (bounding_box.min.x - ray_origin.x) * inv_dir.x;
    let mut t2 = (bounding_box.max.x - ray_origin.x) * inv_dir.x;

    let mut tmin = t1.min(t2);
    let mut tmax = t1.max(t2);

    for i in 1..3 {
        t1 = (bounding_box.min[i] - ray_origin[i]) * inv_dir[i];
        t2 = (bounding_box.max[i] - ray_origin[i]) * inv_dir[i];

        tmin = tmin.max(t1.min(t2));
        tmax = tmax.min(t1.max(t2));
    }

    if (tmin < 0.0 && tmax < 0.0) || tmax < tmin {
        false
    } else {
        tmin <= max_time_of_impact
    }
}

#[derive(Copy, Clone)]
pub struct IntersectionTriangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub edge_b_a: Vec3,
    pub edge_c_a: Vec3,
    pub crossed_normal: Vec3,
    pub normal: Vec3,
}

// Adapted from https://www.iquilezles.org/www/articles/intersectors/intersectors.htm.
pub fn ray_triangle_intersection(
    ray_origin: Vec3,
    ray_direction: Vec3,
    max_time_of_impact: f32,
    triangle: IntersectionTriangle,
) -> Option<f32> {
    let IntersectionTriangle {
        a,
        b: _,
        c: _,
        edge_b_a,
        edge_c_a,
        crossed_normal,
        normal: _,
    } = triangle;

    let ro_a = ray_origin - a;

    let q = ro_a.cross(ray_direction);
    let d = 1.0 / ray_direction.dot(crossed_normal);
    let u = d * (-q).dot(edge_c_a);
    let v = d * q.dot(edge_b_a);
    let t = d * (-crossed_normal).dot(ro_a);

    if u < 0.0 || u > 1.0 || v < 0.0 || (u + v) > 1.0 {
        return None;
    }

    if t <= max_time_of_impact {
        Some(t)
    } else {
        None
    }
}

// Adapted from https://wickedengine.net/2020/04/26/capsule-collision-detection/
pub fn sphere_triangle_intersection(
    sphere_center: Vec3,
    radius: f32,
    triangle: IntersectionTriangle,
) -> Option<(Vec3, Vec3, f32)> {
    let IntersectionTriangle {
        a,
        b,
        c,
        normal,
        crossed_normal: _,
        edge_b_a,
        edge_c_a,
    } = triangle;

    let distance = (sphere_center - a).dot(normal);

    if distance < -radius || distance > radius {
        return None;
    }

    let radius_sq = radius * radius;

    let projected_center = sphere_center - normal * distance;

    let edge_c_b = c - b;
    let edge_a_c = -edge_c_a;

    let c0 = (projected_center - a).cross(edge_b_a);
    let c1 = (projected_center - b).cross(edge_c_b);
    let c2 = (projected_center - c).cross(edge_a_c);

    let inside = c0.dot(normal) <= 0.0 && c1.dot(normal) <= 0.0 && c2.dot(normal) <= 0.0;

    let intersection_vec = if inside {
        sphere_center - projected_center
    } else {
        let vector_1 = sphere_center - closest_point_on_line_segment(a, b, sphere_center);
        let vector_2 = sphere_center - closest_point_on_line_segment(b, c, sphere_center);
        let vector_3 = sphere_center - closest_point_on_line_segment(c, a, sphere_center);

        let distance_1_sq = vector_1.mag_sq();
        let distance_2_sq = vector_2.mag_sq();
        let distance_3_sq = vector_3.mag_sq();

        let intersects =
            distance_1_sq < radius_sq || distance_2_sq < radius_sq || distance_3_sq < radius_sq;

        if !intersects {
            return None;
        }

        let mut best_vector = vector_1;
        let mut best_distance_sq = distance_1_sq;

        if distance_2_sq < best_distance_sq {
            best_vector = vector_2;
            best_distance_sq = distance_2_sq;
        }

        if distance_3_sq < best_distance_sq {
            best_vector = vector_2;
        }

        best_vector
    };

    let length = intersection_vec.mag();
    let normal = -intersection_vec / length;
    let depth = radius - length;
    let position_on_triangle = sphere_center - intersection_vec;
    Some((normal, position_on_triangle, depth))
}

// Adapted from https://wickedengine.net/2020/04/26/capsule-collision-detection/
pub fn capsule_triangle_intersection(
    capsule_center: Vec3,
    radius: f32,
    capsule_length: f32,
    triangle: IntersectionTriangle,
) -> Option<(Vec3, Vec3, f32)> {
    let IntersectionTriangle {
        a,
        b,
        c,
        normal: triangle_normal,
        edge_b_a,
        edge_c_a,
        crossed_normal: _,
    } = triangle;

    // we need to flip this because uuuhhh....
    let triangle_normal = -triangle_normal;

    let capsule_normal = Vec3::unit_y();

    let offset = capsule_normal * capsule_length / 2.0;

    // These are the base and tips of the inner capsule line, not the base and tip of the capsule itself.
    // So if you have a capsule of length 0 (which is just a sphere) then these will just be the capsule center.
    let base = capsule_center - offset;
    let tip = capsule_center + offset;

    let reference_point = {
        let abs_normal_dot_product = triangle_normal.dot(capsule_normal).abs();

        if abs_normal_dot_product == 0.0 {
            a
        } else {
            let t = triangle_normal.dot(a - base) / abs_normal_dot_product;
            let line_plane_intersection = base + capsule_normal * t;

            let edge_c_b = c - b;
            let edge_a_c = -edge_c_a;

            let c0 = (line_plane_intersection - a).cross(edge_b_a);
            let c1 = (line_plane_intersection - b).cross(edge_c_b);
            let c2 = (line_plane_intersection - c).cross(edge_a_c);

            let inside = c0.dot(triangle_normal) <= 0.0
                && c1.dot(triangle_normal) <= 0.0
                && c2.dot(triangle_normal) <= 0.0;

            if inside {
                line_plane_intersection
            } else {
                let point_1 = closest_point_on_line_segment(a, b, line_plane_intersection);
                let point_2 = closest_point_on_line_segment(b, c, line_plane_intersection);
                let point_3 = closest_point_on_line_segment(c, a, line_plane_intersection);

                let distance_1_sq = (line_plane_intersection - point_1).mag_sq();
                let distance_2_sq = (line_plane_intersection - point_2).mag_sq();
                let distance_3_sq = (line_plane_intersection - point_3).mag_sq();

                let mut reference_point = point_1;
                let mut best_distance_sq = distance_1_sq;

                if distance_2_sq < best_distance_sq {
                    reference_point = point_2;
                    best_distance_sq = distance_2_sq;
                }

                if distance_3_sq < best_distance_sq {
                    reference_point = point_3;
                }

                reference_point
            }
        }
    };

    //println!("!!");
    let center = closest_point_on_line_segment(base, tip, reference_point);
    //println!("displacement: {:?}", center - capsule_center);

    sphere_triangle_intersection(center, radius, triangle)
}

fn closest_point_on_line_segment(a: Vec3, b: Vec3, point: Vec3) -> Vec3 {
    let ab = b - a;
    let t = (point - a).dot(ab);
    //println!("{:?}", saturate(t));
    a + saturate(t) * ab
}

fn saturate(value: f32) -> f32 {
    value.max(0.0).min(1.0)
}
