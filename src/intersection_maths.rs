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
    pub b_neg_a: Vec3,
    pub c_neg_a: Vec3,
    pub crossed_normal: Vec3,
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
        b_neg_a,
        c_neg_a,
        crossed_normal,
    } = triangle;

    let ro_a = ray_origin - a;

    let q = ro_a.cross(ray_direction);
    let d = 1.0 / ray_direction.dot(crossed_normal);
    let u = d * (-q).dot(c_neg_a);
    let v = d * q.dot(b_neg_a);
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
