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

// Adapted from https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/aabb-triangle.html
fn triangle_bounding_box_intersection(mut a: Vec3, mut b: Vec3, mut c: Vec3, bounding_box: collision_octree::BoundingBox) -> bool {
    // Convert AABB to center-extents form
    let center = (bounding_box.min + bounding_box.max) / 2.0;
    let extents = (bounding_box.max - bounding_box.min) / 2.0;

    // Translate the triangle as conceptually moving the AABB to origin
    // This is the same as we did with the point in triangle test
    a -= center;
    b -= center;
    c -= center;

    // Compute the edge vectors of the triangle  (ABC)
    // That is, get the lines between the points as vectors
    let edge_b_a = b - a;
    let edge_c_b = c - b;
    let edge_a_c = a - c;

    // We first test against 9 axis, these axis are given by
    // cross product combinations of the edges of the triangle
    // and the edges of the AABB. You need to get an axis testing
    // each of the 3 sides of the AABB against each of the 3 sides
    // of the triangle. The result is 9 axis of seperation

    // Compute the 9 axis
    let axises = [
        Vec3::unit_x().cross(edge_b_a),
        Vec3::unit_x().cross(edge_c_b),
        Vec3::unit_x().cross(edge_a_c),

        Vec3::unit_y().cross(edge_b_a),
        Vec3::unit_y().cross(edge_c_b),
        Vec3::unit_y().cross(edge_a_c),

        Vec3::unit_z().cross(edge_b_a),
        Vec3::unit_z().cross(edge_c_b),
        Vec3::unit_z().cross(edge_a_c),
    ];

    for i in 0 .. axises.len() {
        if seperating_axis_test(a, b, c, axises[i], extents) {
            return false;
        }
    }

    // We can skip this because we already have the bounding box of the triangle.
    //
    // Next, we have 3 face normals from the AABB
    // for these tests we are conceptually checking if the bounding box
    // of the triangle intersects the bounding box of the AABB
    // that is to say, the seperating axis for all tests are axis aligned:
    // axis1: (1, 0, 0), axis2: (0, 1, 0), axis3 (0, 0, 1)

    // Finally, we have one last axis to test, the face normal of the triangle
    // We can get the normal of the triangle by crossing the first two line segments
    let triangle_normal = edge_b_a.cross(edge_c_b);

    if seperating_axis_test(a, b, c, triangle_normal, extents) {
        return false;
    }

    return true;
}

fn seperating_axis_test(a: Vec3, b: Vec3, c: Vec3, axis: Vec3, extents: Vec3) -> bool {
    // Project all 3 vertices of the triangle onto the Seperating axis
    let a_projected = a.dot(axis);
    let b_projected = b.dot(axis);
    let c_projected = c.dot(axis);

    // Project the AABB onto the seperating axis
    // We don't care about the end points of the prjection
    // just the length of the half-size of the AABB
    // That is, we're only casting the extents onto the
    // seperating axis, not the AABB center. We don't
    // need to cast the center, because we know that the
    // aabb is at origin compared to the triangle!
    let r =
        extents.x * Vec3::unit_x().dot(axis).abs() +
        extents.y * Vec3::unit_y().dot(axis).abs() +
        extents.z * Vec3::unit_z().dot(axis).abs();

    // Now do the actual test, basically see if either of
    // the most extreme of the triangle points intersects r
    // You might need to write Min & Max functions that take 3 arguments
    //
    // `true` means BOTH of the points of the projected triangle
    // are outside the projected half-length of the AABB
    // Therefore the axis is seperating and we can exit

    let max = a_projected.max(b_projected).max(c_projected);
    let min = a_projected.min(b_projected).min(c_projected);

    (-max).max(min) > r
}
