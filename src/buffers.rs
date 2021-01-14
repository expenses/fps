use crate::array_of_textures::ArrayOfTextures;
use crate::assets::{AnimatedModel, AnimationJoints, Level, Model, StagingModelBuffers};
use crate::renderer::{AnimatedInstance, Instance, Renderer, Vertex, INDEX_FORMAT};
use ultraviolet::Mat4;
use wgpu::util::DeviceExt;

struct StaticModelBuffer {
    model: Model,
    instances: Vec<Instance>,
}

impl StaticModelBuffer {
    fn load(model: Model) -> Self {
        Self {
            instances: Vec::new(),
            model,
        }
    }
}

struct AnimatedModelBuffer {
    model: AnimatedModel,
    instances: Vec<AnimatedInstance>,
    joints: Vec<Mat4>,
}

impl AnimatedModelBuffer {
    fn load(model: AnimatedModel) -> Self {
        Self {
            instances: Vec::new(),
            joints: Vec::new(),
            model,
        }
    }
}

#[derive(Default, Debug)]
pub struct AnimationInfo {
    pub mouse_walk_animation: usize,
    pub mouse_idle_animation: usize,

    pub robot_base_node: usize,

    pub tentacle_poke_animation: usize,
}

fn create_animated_models_bind_group(
    animated_joints: &MergedBuffer<Mat4>,
    renderer: &Renderer,
    animated_model_offsets_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("animated models bind group"),
            layout: &renderer.animated_models_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &animated_joints.buffer,
                        offset: 0,
                        size: None,
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: animated_model_offsets_buffer,
                        offset: 0,
                        size: None,
                    },
                },
            ],
        })
}

// A buffer that different slices are uploaded and merged into
struct MergedBuffer<T> {
    capacity: usize,
    buffer: wgpu::Buffer,
    name: String,
    usage: wgpu::BufferUsage,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> MergedBuffer<T> {
    fn new(capacity: usize, renderer: &Renderer, usage: wgpu::BufferUsage, name: &str) -> Self {
        Self {
            capacity,
            buffer: renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (capacity * std::mem::size_of::<T>()) as u64,
                usage: usage | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }),
            name: name.to_string(),
            usage,
            _phantom: std::marker::PhantomData,
        }
    }

    fn upload<'a>(
        &mut self,
        staging_buffers: impl Iterator<Item = &'a [T]> + Clone,
        renderer: &Renderer,
    ) -> bool {
        let num_items = staging_buffers.clone().map(|buffer| buffer.len()).sum();
        let mut resized = false;

        if num_items > self.capacity {
            let new_capacity = (self.capacity * 2).max(num_items);

            self.buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&self.name),
                size: (new_capacity * std::mem::size_of::<T>()) as u64,
                usage: self.usage | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            resized = true;
        }

        let mut offset = 0;

        for buffer in staging_buffers {
            renderer.queue.write_buffer(
                &self.buffer,
                (offset * std::mem::size_of::<T>()) as u64,
                bytemuck::cast_slice(buffer),
            );

            offset += buffer.len();
        }

        resized
    }

    fn slice(&self) -> wgpu::BufferSlice {
        self.buffer.slice(..)
    }
}

struct DrawBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    to_be_uploaded: Vec<DrawIndexedIndirect>,
    uploaded: u32,
    name: String,
}

impl DrawBuffer {
    fn new(renderer: &Renderer, name: &str) -> Self {
        let capacity = 1;

        Self {
            buffer: renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (capacity * std::mem::size_of::<DrawIndexedIndirect>()) as u64,
                usage: wgpu::BufferUsage::INDIRECT | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity,
            to_be_uploaded: Vec::new(),
            uploaded: 0,
            name: name.to_string(),
        }
    }

    fn push(&mut self, draw: DrawIndexedIndirect) {
        self.to_be_uploaded.push(draw);
    }

    fn upload(&mut self, renderer: &Renderer, reverse: bool) {
        if reverse {
            self.to_be_uploaded.reverse();
        }

        if self.to_be_uploaded.len() > self.capacity {
            let new_capacity = (self.capacity * 2).max(self.to_be_uploaded.len());

            self.buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&self.name),
                size: (new_capacity * std::mem::size_of::<DrawIndexedIndirect>()) as u64,
                usage: wgpu::BufferUsage::INDIRECT | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });
        }

        renderer
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.to_be_uploaded));

        self.uploaded = self.to_be_uploaded.len() as u32;
        self.to_be_uploaded.clear();
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    vertex_count: u32,   // The number of vertices to draw.
    instance_count: u32, // The number of instances to draw.
    base_index: u32,     // The base index within the index buffer.
    vertex_offset: i32, // The value added to the vertex index before indexing into the vertex buffer.
    base_instance: u32, // The instance ID of the first instance to draw.
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Offset {
    joint_offset: u32,
    instance_offset: u32,
}

#[derive(Copy, Clone)]
pub enum StaticModelType {
    MateBottle = 0,
    Bush = 1,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AnimatedModelType {
    Robot = 0,
    Mouse = 1,
    Tentacle = 2,
    MarioCube = 3,
    JugglingBalls = 4,
    Explosion = 5,
}

pub struct ModelBuffers {
    static_hand_models: Vec<Model>,

    animated_models: Vec<AnimatedModelBuffer>,
    static_models: Vec<StaticModelBuffer>,
    pub animation_info: AnimationInfo,
    pub array_of_textures_bind_group: wgpu::BindGroup,

    animated_model_offsets_buffer: wgpu::Buffer,
    animated_joints: MergedBuffer<Mat4>,
    animated_models_bind_group: wgpu::BindGroup,
    animated_instances: MergedBuffer<AnimatedInstance>,

    animated_model_vertices: wgpu::Buffer,
    animated_model_indices: wgpu::Buffer,

    static_model_vertices: wgpu::Buffer,
    static_model_indices: wgpu::Buffer,
    static_instances: MergedBuffer<Instance>,

    static_model_opaque_draws: DrawBuffer,
    static_model_alpha_clip_draws: DrawBuffer,
    static_model_alpha_blend_draws: DrawBuffer,

    animated_model_opaque_draws: DrawBuffer,
    animated_model_alpha_clip_draws: DrawBuffer,
    animated_model_alpha_blend_draws: DrawBuffer,
}

impl ModelBuffers {
    pub fn new(
        renderer: &mut Renderer,
        mut init_encoder: &mut wgpu::CommandEncoder,
        mut array_of_textures: ArrayOfTextures,
        mut static_staging_buffers: StagingModelBuffers<Vertex>,
    ) -> anyhow::Result<Self> {
        let mut animation_info = AnimationInfo::default();

        let static_hand_models = vec![Model::load_gltf(
            include_bytes!("../models/mate_gun.glb"),
            &renderer,
            &mut init_encoder,
            "mate gun",
            &mut array_of_textures,
            &mut static_staging_buffers,
        )?];

        let static_models = vec![
            StaticModelBuffer::load(Model::load_gltf(
                include_bytes!("../models/mate.glb"),
                &renderer,
                &mut init_encoder,
                "mate bottle",
                &mut array_of_textures,
                &mut static_staging_buffers,
            )?),
            StaticModelBuffer::load(Model::load_gltf(
                include_bytes!("../models/bush.glb"),
                &renderer,
                &mut init_encoder,
                "bush",
                &mut array_of_textures,
                &mut static_staging_buffers,
            )?),
        ];

        let mut animated_staging_buffers = StagingModelBuffers::default();

        let animated_models = vec![
            AnimatedModelBuffer::load(AnimatedModel::load_gltf(
                include_bytes!("../models/warehouse_robot.glb"),
                &renderer,
                &mut init_encoder,
                "warehouse robot",
                |_, joints| {
                    animation_info.robot_base_node = joints
                        .unwrap()
                        .find(|node| node.name() == Some("Base"))
                        .map(|node| node.index())
                        .unwrap();
                },
                &mut array_of_textures,
                &mut animated_staging_buffers,
            )?),
            AnimatedModelBuffer::load(AnimatedModel::load_gltf(
                include_bytes!("../models/mouse.glb"),
                &renderer,
                &mut init_encoder,
                "mouse",
                |animations, _| {
                    let mut animations: std::collections::HashMap<_, _> = animations
                        .map(|animation| (animation.name().unwrap(), animation.index()))
                        .collect();

                    animation_info.mouse_idle_animation = animations.remove("Idle").unwrap();
                    animation_info.mouse_walk_animation = animations.remove("Walk").unwrap();

                    if !animations.is_empty() {
                        log::debug!("Unhandled animations:");
                        animations.keys().for_each(|name| log::debug!("{}", name));
                    }
                },
                &mut array_of_textures,
                &mut animated_staging_buffers,
            )?),
            AnimatedModelBuffer::load(AnimatedModel::load_gltf(
                include_bytes!("../models/tentacle.glb"),
                &renderer,
                &mut init_encoder,
                "tentacle",
                |mut animations, _| {
                    assert_eq!(animations.clone().count(), 2);

                    animation_info.tentacle_poke_animation = animations
                        .find(|animation| animation.name() == Some("poke_baked"))
                        .map(|animation| animation.index())
                        .unwrap();
                },
                &mut array_of_textures,
                &mut animated_staging_buffers,
            )?),
            AnimatedModelBuffer::load(AnimatedModel::load_gltf(
                include_bytes!("../models/mario_kart_square.glb"),
                &renderer,
                &mut init_encoder,
                "mario cube",
                |_, _| {},
                &mut array_of_textures,
                &mut animated_staging_buffers,
            )?),
            AnimatedModelBuffer::load(AnimatedModel::load_gltf(
                include_bytes!("../models/juggling_balls.glb"),
                &renderer,
                &mut init_encoder,
                "jugging balls",
                |_, _| {},
                &mut array_of_textures,
                &mut animated_staging_buffers,
            )?),
            AnimatedModelBuffer::load(AnimatedModel::load_gltf(
                include_bytes!("../models/explosion.glb"),
                &renderer,
                &mut init_encoder,
                "explosion",
                |_, _| {},
                &mut array_of_textures,
                &mut animated_staging_buffers,
            )?),
        ];

        let animated_model_offsets = vec![
            Offset {
                joint_offset: 0,
                instance_offset: 0
            };
            animated_models.len()
        ];

        let animated_model_offsets_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("joint offsets buffer"),
                    contents: bytemuck::cast_slice(&animated_model_offsets),
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                });

        let animated_joints = MergedBuffer::new(
            1,
            &renderer,
            wgpu::BufferUsage::STORAGE,
            "animated joints buffer",
        );
        let static_instances = MergedBuffer::new(
            1,
            &renderer,
            wgpu::BufferUsage::VERTEX,
            "static instances buffer",
        );
        let animated_instances = MergedBuffer::new(
            1,
            &renderer,
            wgpu::BufferUsage::VERTEX,
            "animated instances buffer",
        );

        let animated_models_bind_group = create_animated_models_bind_group(
            &animated_joints,
            renderer,
            &animated_model_offsets_buffer,
        );

        log::info!("{:?}", animation_info);

        let array_of_textures_bind_group = array_of_textures.bind(renderer);

        let (static_model_vertices, static_model_indices) =
            static_staging_buffers.upload(&renderer.device, "static model");

        let (animated_model_vertices, animated_model_indices) =
            animated_staging_buffers.upload(&renderer.device, "animated model");

        Ok(Self {
            static_hand_models,
            static_models,
            animated_models,
            animation_info,
            array_of_textures_bind_group,

            animated_model_offsets_buffer,
            animated_joints,
            animated_models_bind_group,
            animated_instances,

            animated_model_vertices,
            animated_model_indices,

            static_model_vertices,
            static_model_indices,
            static_instances,

            animated_model_opaque_draws: DrawBuffer::new(renderer, "animated model opaque draws"),
            animated_model_alpha_clip_draws: DrawBuffer::new(
                renderer,
                "animated model alpha clip draws",
            ),
            animated_model_alpha_blend_draws: DrawBuffer::new(
                renderer,
                "animated model alpha blend draws",
            ),
            static_model_opaque_draws: DrawBuffer::new(renderer, "static model opaque draws"),
            static_model_alpha_clip_draws: DrawBuffer::new(
                renderer,
                "static model alpha clip draws",
            ),
            static_model_alpha_blend_draws: DrawBuffer::new(
                renderer,
                "static model alpha blend draws",
            ),
        })
    }

    pub fn get_static_buffer(&mut self, model: &StaticModelType) -> &mut Vec<Instance> {
        &mut self.static_models[*model as usize].instances
    }

    pub fn get_animated_buffer(
        &mut self,
        model: &AnimatedModelType,
    ) -> (&mut Vec<AnimatedInstance>, &AnimatedModel, &mut Vec<Mat4>) {
        let index = *model as usize;

        let AnimatedModelBuffer {
            instances,
            model,
            joints,
            ..
        } = &mut self.animated_models[index];

        (instances, model, joints)
    }

    pub fn clone_animation_joints(&self, model: &AnimatedModelType) -> AnimationJoints {
        self.animated_models[*model as usize]
            .model
            .animation_joints
            .clone()
    }

    pub fn upload(
        &mut self,
        renderer: &Renderer,
        camera_view: Mat4,
        draw_gun: bool,
        level: &Level,
    ) {
        // Joints

        let resized = self.animated_joints.upload(
            self.animated_models.iter().map(|model| &model.joints[..]),
            renderer,
        );

        if resized {
            self.animated_models_bind_group = create_animated_models_bind_group(
                &self.animated_joints,
                renderer,
                &self.animated_model_offsets_buffer,
            );
        }

        let animated_model_offsets: Vec<_> = {
            let mut current_joint_offset = 0;
            let mut current_instance_offset = 0;

            self.animated_models
                .iter()
                .map(|model| {
                    let joint_offset = current_joint_offset;
                    let instance_offset = current_instance_offset;
                    current_joint_offset += model.joints.len() as u32;
                    current_instance_offset += model.instances.len() as u32;
                    Offset {
                        joint_offset,
                        instance_offset,
                    }
                })
                .collect()
        };

        renderer.queue.write_buffer(
            &self.animated_model_offsets_buffer,
            0,
            bytemuck::cast_slice(&animated_model_offsets),
        );

        // Instances

        self.static_instances.upload(
            [&[
                Instance::new(camera_view.inversed()),
                Instance::new(Mat4::identity()),
            ][..]]
            .iter()
            .cloned()
            .chain(self.static_models.iter().map(|model| &model.instances[..])),
            renderer,
        );

        self.animated_instances.upload(
            self.animated_models
                .iter()
                .map(|model| &model.instances[..]),
            renderer,
        );

        // Upload draw commands to buffers.

        // Static models
        {
            let gun_instances = draw_gun as u32;
            let mut instance_offset = 1 - gun_instances;

            let extra_models = [
                (&self.static_hand_models[0], gun_instances),
                (&level.model, 1),
            ];

            let models = extra_models.iter().cloned().chain(
                self.static_models
                    .iter()
                    .map(|model| (&model.model, model.instances.len() as u32)),
            );

            for (model, instance_count) in models {
                if model.opaque_geometry.size > 0 {
                    self.static_model_opaque_draws.push(DrawIndexedIndirect {
                        vertex_count: model.opaque_geometry.size,
                        instance_count,
                        base_index: model.opaque_geometry.offset,
                        vertex_offset: 0,
                        base_instance: instance_offset,
                    });
                }

                if model.alpha_clip_geometry.size > 0 {
                    self.static_model_alpha_clip_draws
                        .push(DrawIndexedIndirect {
                            vertex_count: model.alpha_clip_geometry.size,
                            instance_count,
                            base_index: model.alpha_clip_geometry.offset,
                            vertex_offset: 0,
                            base_instance: instance_offset,
                        });
                }

                if model.alpha_blend_geometry.size > 0 {
                    self.static_model_alpha_blend_draws
                        .push(DrawIndexedIndirect {
                            vertex_count: model.alpha_blend_geometry.size,
                            instance_count,
                            base_index: model.alpha_blend_geometry.offset,
                            vertex_offset: 0,
                            base_instance: instance_offset,
                        });
                }

                instance_offset += instance_count;
            }
        }

        // Animated models
        {
            let mut instance_offset = 0;

            for model in &self.animated_models {
                let instance_count = model.instances.len() as u32;

                if model.model.opaque_geometry.size > 0 {
                    self.animated_model_opaque_draws.push(DrawIndexedIndirect {
                        vertex_count: model.model.opaque_geometry.size,
                        instance_count,
                        base_index: model.model.opaque_geometry.offset,
                        vertex_offset: 0,
                        base_instance: instance_offset,
                    });
                }

                if model.model.alpha_clip_geometry.size > 0 {
                    self.animated_model_alpha_clip_draws
                        .push(DrawIndexedIndirect {
                            vertex_count: model.model.alpha_clip_geometry.size,
                            instance_count,
                            base_index: model.model.alpha_clip_geometry.offset,
                            vertex_offset: 0,
                            base_instance: instance_offset,
                        });
                }

                if model.model.alpha_blend_geometry.size > 0 {
                    self.animated_model_alpha_blend_draws
                        .push(DrawIndexedIndirect {
                            vertex_count: model.model.alpha_blend_geometry.size,
                            instance_count,
                            base_index: model.model.alpha_blend_geometry.offset,
                            vertex_offset: 0,
                            base_instance: instance_offset,
                        });
                }

                instance_offset += instance_count;
            }
        }

        self.static_model_opaque_draws.upload(renderer, false);
        self.static_model_alpha_clip_draws.upload(renderer, false);
        self.static_model_alpha_blend_draws.upload(renderer, true);

        self.animated_model_opaque_draws.upload(renderer, false);
        self.animated_model_alpha_clip_draws.upload(renderer, false);
        self.animated_model_alpha_blend_draws.upload(renderer, true);

        // Clears.

        for model in &mut self.animated_models {
            model.joints.clear();
            model.instances.clear();
        }

        for model in &mut self.static_models {
            model.instances.clear();
        }
    }

    fn render_static<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &'a Renderer,
        pipeline: &'a wgpu::RenderPipeline,
        draw_buffer: &'a DrawBuffer,
    ) {
        render_pass.set_pipeline(pipeline);
        render_pass.set_push_constants(
            wgpu::ShaderStage::VERTEX,
            0,
            bytemuck::bytes_of(&renderer.projection_view),
        );
        render_pass.set_vertex_buffer(0, self.static_model_vertices.slice(..));
        render_pass.set_vertex_buffer(1, self.static_instances.slice());
        render_pass.set_index_buffer(self.static_model_indices.slice(..), INDEX_FORMAT);
        render_pass.multi_draw_indexed_indirect(&draw_buffer.buffer, 0, draw_buffer.uploaded);
    }

    fn render_animated<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &'a Renderer,
        pipeline: &'a wgpu::RenderPipeline,
        draw_buffer: &'a DrawBuffer,
    ) {
        render_pass.set_pipeline(pipeline);
        render_pass.set_push_constants(
            wgpu::ShaderStage::VERTEX,
            0,
            bytemuck::bytes_of(&renderer.projection_view),
        );
        render_pass.set_bind_group(3, &self.animated_models_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.animated_model_vertices.slice(..));
        render_pass.set_vertex_buffer(1, self.animated_instances.slice());
        render_pass.set_index_buffer(self.animated_model_indices.slice(..), INDEX_FORMAT);
        render_pass.multi_draw_indexed_indirect(&draw_buffer.buffer, 0, draw_buffer.uploaded);
    }

    pub fn render_opaque<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &'a Renderer,
    ) {
        self.render_static(
            render_pass,
            renderer,
            &renderer.static_opaque_render_pipeline,
            &self.static_model_opaque_draws,
        );

        self.render_animated(
            render_pass,
            renderer,
            &renderer.animated_opaque_render_pipeline,
            &self.animated_model_opaque_draws,
        );
    }

    pub fn render_alpha_clip<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &'a Renderer,
    ) {
        self.render_static(
            render_pass,
            renderer,
            &renderer.static_alpha_clip_render_pipeline,
            &self.static_model_alpha_clip_draws,
        );

        self.render_animated(
            render_pass,
            renderer,
            &renderer.animated_alpha_clip_render_pipeline,
            &self.animated_model_alpha_clip_draws,
        );
    }

    pub fn render_alpha_blend<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &'a Renderer,
    ) {
        self.render_animated(
            render_pass,
            renderer,
            &renderer.animated_alpha_blend_render_pipeline,
            &self.animated_model_alpha_blend_draws,
        );

        self.render_static(
            render_pass,
            renderer,
            &renderer.static_alpha_blend_render_pipeline,
            &self.static_model_alpha_blend_draws,
        );
    }
}
