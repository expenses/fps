use arrayvec::ArrayVec;
use ultraviolet::Vec3;

const MIN_NODES_BEFORE_SPLITTING: usize = 4;

#[derive(Debug)]
pub struct Octree<T> {
    nodes: Vec<Node<T>>,
}

impl<T: HasBoundingBox> Octree<T> {
    pub fn construct(objects: Vec<T>) -> Self {
        let bounding_box = objects
            .iter()
            .map(|object| object.bounding_box())
            .fold(BoundingBox::default(), |a, b| a.union(b))
            .into_cube();

        let mut octree = Self {
            nodes: vec![Node {
                children_offset: None,
                objects: Vec::new(),
                bounding_box,
            }],
        };

        let mut stack = Vec::with_capacity(8);

        for object in objects.into_iter() {
            let best_box = octree.best_box(object.bounding_box(), &mut stack);

            let num_nodes = octree.nodes.len();

            let node = &mut octree.nodes[best_box];

            if node.objects.len() >= MIN_NODES_BEFORE_SPLITTING {
                let mut index = None;

                for (i, sub_box) in node.bounding_box.subdivide().enumerate() {
                    if sub_box.contains(object.bounding_box()) {
                        index = Some(num_nodes + i);
                        break;
                    }
                }

                if let Some(index) = index {
                    node.children_offset = Some(num_nodes);
                    let bounding_box = node.bounding_box;
                    drop(node);

                    for sub_box in bounding_box.subdivide() {
                        octree.nodes.push(Node {
                            children_offset: None,
                            objects: Vec::new(),
                            bounding_box: sub_box,
                        });
                    }

                    octree.nodes[index].objects.push(object);
                } else {
                    node.objects.push(object);
                }
            } else {
                node.objects.push(object);
            }
        }

        octree
    }

    fn best_box(&self, bounding_box: BoundingBox, stack: &mut Vec<usize>) -> usize {
        stack.clear();
        stack.push(0);

        let mut best_index = 0;

        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];

            if node.bounding_box.contains(bounding_box) {
                best_index = index;

                // Only this node or it's children can fit the box.
                stack.clear();

                if let Some(&children_offset) = node.children_offset.as_ref() {
                    for child in children_offset..children_offset + 8 {
                        stack.push(child);
                    }
                }
            }
        }

        best_index
    }

    pub fn intersects(
        &self,
        // You are recommended to check bounded boxes first in these checks.
        bounding_box_intersection_check: impl Fn(BoundingBox) -> bool,
        object_intersection_check: impl Fn(&T) -> bool,
        stack: &mut Vec<usize>,
    ) -> bool {
        stack.clear();
        stack.push(0);

        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];

            if bounding_box_intersection_check(node.bounding_box) {
                for object in &node.objects {
                    if object_intersection_check(object) {
                        return true;
                    }
                }

                if let Some(&children_offset) = node.children_offset.as_ref() {
                    for child in children_offset..children_offset + 8 {
                        let child_node = &self.nodes[child];

                        if !child_node.objects.is_empty() || child_node.children_offset.is_some() {
                            stack.push(child);
                        }
                    }
                }
            }
        }

        false
    }

    pub fn bounding_boxes(&self) -> impl Iterator<Item = (bool, BoundingBox)> + '_ {
        self.nodes.iter().flat_map(|node| {
            std::iter::once((false, node.bounding_box)).chain(
                node.objects
                    .iter()
                    .map(|object| (true, object.bounding_box())),
            )
        })
    }

    pub fn debug_print_sizes(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            if !node.objects.is_empty() {
                println!("Node {}: {} objects.", i, self.nodes[i].objects.len());
            }
        }
    }
}

#[derive(Debug)]
struct Node<T> {
    children_offset: Option<usize>,
    objects: Vec<T>,
    bounding_box: BoundingBox,
}

pub trait HasBoundingBox {
    fn bounding_box(&self) -> BoundingBox;
}

#[derive(Debug, Copy, Clone)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            min: Vec3::broadcast(f32::MAX),
            max: Vec3::broadcast(f32::MIN),
        }
    }
}

impl BoundingBox {
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min_by_component(other.min),
            max: self.max.max_by_component(other.max),
        }
    }

    fn subdivide(self) -> impl Iterator<Item = Self> {
        let center = (self.min + self.max) / 2.0;

        let min = std::iter::once(Self {
            min: self.min,
            max: center,
        });

        let max = std::iter::once(Self {
            min: center,
            max: self.max,
        });

        let mins = ArrayVec::from([0, 1, 2]).into_iter().map(move |i| {
            let mut min = self.min;
            min[i] = center[i];
            let mut max = center;
            max[i] = self.max[i];
            Self { min, max }
        });

        let maxs = ArrayVec::from([0, 1, 2]).into_iter().map(move |i| {
            let mut min = center;
            min[i] = self.min[i];
            let mut max = self.max;
            max[i] = center[i];
            Self { min, max }
        });

        min.chain(max).chain(mins).chain(maxs)
    }

    fn into_cube(self) -> Self {
        let dimensions = self.max - self.min;
        let half_cube_size =
            Vec3::broadcast(dimensions.x.max(dimensions.y).max(dimensions.z) / 2.0);
        let center = (self.min + self.max) / 2.0;
        Self {
            min: center - half_cube_size,
            max: center + half_cube_size,
        }
    }

    pub fn contains(self, other: Self) -> bool {
        self.min.min_by_component(other.min) == self.min
            && self.max.max_by_component(other.max) == self.max
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        !(self.min.x > other.max.x
            || self.min.y > other.max.y
            || self.min.z > other.max.z
            || self.max.x < other.min.x
            || self.max.y < other.min.y
            || self.max.z < other.min.z)
    }

    pub fn lines(self) -> impl Iterator<Item = (Vec3, Vec3)> {
        let xs = ArrayVec::from([
            self.min,
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
        ]);

        let xs = xs.into_iter().map(move |base| {
            let mut opp = base;
            opp.x = self.max.x;
            (base, opp)
        });

        let ys = ArrayVec::from([
            self.min,
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
        ]);

        let ys = ys.into_iter().map(move |base| {
            let mut opp = base;
            opp.y = self.max.y;
            (base, opp)
        });

        let zs = ArrayVec::from([
            self.min,
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
        ]);

        let zs = zs.into_iter().map(move |base| {
            let mut opp = base;
            opp.z = self.max.z;
            (base, opp)
        });

        xs.chain(ys).chain(zs)
    }

    pub fn from_triangle(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self {
            min: a.min_by_component(b).min_by_component(c),
            max: a.max_by_component(b).max_by_component(c),
        }
    }

    pub fn from_line(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min_by_component(b),
            max: a.max_by_component(b),
        }
    }
}
