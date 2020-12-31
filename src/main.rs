//mod animation;
mod assets;
mod ecs;
mod renderer;

use crate::assets::AnimationJoints;
use crate::renderer::{AnimatedInstance, DynamicBuffer, Instance, Renderer, INDEX_FORMAT};
use ncollide3d::query::RayCast;
use ncollide3d::transformation::ToTriMesh;
use std::f32::consts::PI;
use ultraviolet::{transform::Isometry3, Mat3, Mat4, Rotor3, Vec2, Vec3, Vec4};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

pub struct Settings {
    draw_gun: bool,
    cursor_grab: bool,
    draw_collision_geometry: bool,
    draw_contact_points: bool,
    draw_player_collider: bool,
    noclip: bool,
    fxaa_enabled: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    futures::executor::block_on(run())
}

#[derive(Default)]
struct ControlStates {
    forwards_pressed: bool,
    backwards_pressed: bool,
    left_pressed: bool,
    right_pressed: bool,
    jump_pressed: bool,
    mouse_held: bool,
}

struct Player {
    position: Vec3,
    facing: PlayerFacing,
    on_ground: bool,
    velocity: Vec3,
    gun_cooldown: f32,
    jump_primed: bool,
    body_shape: ncollide3d::shape::Capsule<f32>,
    feet_shape: ncollide3d::shape::Cuboid<f32>,
}

impl Player {
    const RADIUS: f32 = 0.5;
    const BODY_BOTTOM_DISTANCE_FROM_GROUND: f32 = 0.1;
    const BODY_BASE_DEPTH: f32 = 2.5;
    const FEET_HALF_DEPTH: f32 =
        (Self::BODY_BASE_DEPTH + Self::RADIUS + Self::BODY_BOTTOM_DISTANCE_FROM_GROUND) / 2.0;
    const FEET_HALF_WIDTH: f32 = 0.7 / 2.0;

    const FEET_RELATIVE: Vec3 = Vec3::new(0.0, Self::FEET_HALF_DEPTH, 0.0);
    const BODY_RELATIVE: Vec3 = Vec3::new(
        0.0,
        Self::BODY_BOTTOM_DISTANCE_FROM_GROUND + Self::BODY_BASE_DEPTH / 2.0 + Self::RADIUS,
        0.0,
    );
    const HEAD_RELATIVE: Vec3 = Vec3::new(
        0.0,
        Self::BODY_BOTTOM_DISTANCE_FROM_GROUND + Self::BODY_BASE_DEPTH + Self::RADIUS,
        0.0,
    );
}

#[derive(Debug)]
struct PlayerFacing {
    horizontal: f32,
    vertical: f32,
}

impl PlayerFacing {
    fn normal(&self) -> Vec3 {
        let up_amount = self.vertical.sin();
        let forwards_amount = self.vertical.cos();

        let x_amount = -self.horizontal.sin();
        let z_amount = -self.horizontal.cos();

        Vec3::new(
            forwards_amount * x_amount,
            up_amount,
            forwards_amount * z_amount,
        )
    }
}

fn vec3_to_ncollide_iso(vec: Vec3) -> ncollide3d::math::Isometry<f32> {
    ncollide3d::math::Isometry::translation(vec.x, vec.y, vec.z)
}

fn vec3_from(t: impl Into<[f32; 3]>) -> Vec3 {
    let arr = t.into();
    arr.into()
}

fn vec3_into<T: From<[f32; 3]>>(vec3: Vec3) -> T {
    let arr: [f32; 3] = vec3.into();
    arr.into()
}

fn create_joint_transforms_bind_group(
    buffer: &DynamicBuffer<Mat4>,
    renderer: &Renderer,
    name: &str,
) -> wgpu::BindGroup {
    renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} joint transforms bind group", name)),
            layout: &renderer.joint_transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: buffer.buffer(),
                    offset: 0,
                    size: None,
                },
            }],
        })
}

enum ModelBuffer {
    Static {
        model: assets::Model,
        instances: DynamicBuffer<Instance>,
    },
    Animated {
        model: assets::AnimatedModel,
        instances: DynamicBuffer<AnimatedInstance>,
        joint_transforms: DynamicBuffer<Mat4>,
        joint_transforms_bind_group: wgpu::BindGroup,
        name: String,
    },
}

impl ModelBuffer {
    fn load_animated(model: assets::AnimatedModel, name: &str, renderer: &Renderer) -> Self {
        let joint_transforms = DynamicBuffer::new(
            &renderer.device,
            model.num_joints as usize,
            &format!("{} joint transforms", name),
            wgpu::BufferUsage::STORAGE,
        );

        let joint_transforms_bind_group =
            create_joint_transforms_bind_group(&joint_transforms, renderer, name);

        Self::Animated {
            model,
            instances: DynamicBuffer::new(
                &renderer.device,
                0,
                &format!("{} instances", name),
                wgpu::BufferUsage::VERTEX,
            ),
            joint_transforms,
            joint_transforms_bind_group,
            name: name.to_string(),
        }
    }
}

enum InstancePusher<'a> {
    Static(&'a mut DynamicBuffer<Instance>),
    Animated(&'a mut DynamicBuffer<AnimatedInstance>),
}

impl<'a> InstancePusher<'a> {
    fn push(&mut self, transform: Mat4) {
        match self {
            Self::Static(buffer) => buffer.push(Instance::new(transform)),
            Self::Animated(buffer) => buffer.push(AnimatedInstance::new(transform)),
        }
    }
}

#[derive(Default, Debug)]
struct AnimationInfo {
    mouse_idle_animation: usize,
    mouse_walk_animation: usize,

    robot_base_node: usize,

    tentacle_poke_animation: usize,
}

struct ModelBuffers {
    inner: Vec<ModelBuffer>,
    animation_info: AnimationInfo,
}

impl ModelBuffers {
    fn new(
        renderer: &Renderer,
        mut init_encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<Self> {
        let mut animation_info = AnimationInfo::default();

        let buffers = vec![
            ModelBuffer::load_animated(
                assets::AnimatedModel::load_gltf(
                    include_bytes!("../models/warehouse_robot.glb"),
                    &renderer,
                    &mut init_encoder,
                    "warehouse robot",
                    |_, mut joints| {
                        animation_info.robot_base_node = joints
                            .find(|node| node.name() == Some("Base"))
                            .map(|node| node.index())
                            .unwrap();
                    },
                )?,
                "robot",
                renderer,
            ),
            ModelBuffer::load_animated(
                assets::AnimatedModel::load_gltf(
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
                )?,
                "mouse",
                renderer,
            ),
            ModelBuffer::load_animated(
                assets::AnimatedModel::load_gltf(
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
                )?,
                "tentacle",
                renderer,
            ),
        ];

        log::info!("{:?}", animation_info);

        Ok(Self {
            inner: buffers,
            animation_info,
        })
    }

    fn get_buffer(
        &mut self,
        model: &Model,
    ) -> (
        InstancePusher,
        Option<(&assets::AnimatedModel, &mut DynamicBuffer<Mat4>)>,
    ) {
        match &mut self.inner[*model as usize] {
            ModelBuffer::Static { instances, .. } => (InstancePusher::Static(instances), None),
            ModelBuffer::Animated {
                instances,
                joint_transforms,
                model,
                ..
            } => (
                InstancePusher::Animated(instances),
                Some((model, joint_transforms)),
            ),
        }
    }

    fn clone_animation_joints(&self, model: &Model) -> Option<AnimationJoints> {
        match &self.inner[*model as usize] {
            ModelBuffer::Static { .. } => None,
            ModelBuffer::Animated { model, .. } => Some(model.animation_joints.clone()),
        }
    }

    fn upload(&mut self, renderer: &Renderer) {
        for model_buffer in &mut self.inner {
            match model_buffer {
                ModelBuffer::Static { instances, .. } => {
                    instances.upload(renderer);
                    instances.clear();
                }
                ModelBuffer::Animated {
                    instances,
                    joint_transforms,
                    joint_transforms_bind_group,
                    name,
                    ..
                } => {
                    instances.upload(renderer);

                    let resized = joint_transforms.upload(renderer);

                    if resized {
                        *joint_transforms_bind_group =
                            create_joint_transforms_bind_group(&joint_transforms, renderer, &name);
                    }

                    instances.clear();
                    joint_transforms.clear();
                }
            }
        }
    }

    fn render_opaque<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, renderer: &'a Renderer) {
        for model_buffer in &self.inner {
            match model_buffer {
                ModelBuffer::Static { instances, model } => {
                    if let Some((slice, num)) = instances.get() {
                        render_model_opaque(&model, render_pass, renderer, slice, num);
                    }
                }
                ModelBuffer::Animated {
                    instances,
                    model,
                    joint_transforms_bind_group,
                    ..
                } => {
                    if let Some((slice, num)) = instances.get() {
                        render_animated_model_opaque(
                            &model,
                            render_pass,
                            renderer,
                            slice,
                            num,
                            joint_transforms_bind_group,
                        );
                    }
                }
            }
        }
    }

    fn render_transparent<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &'a Renderer,
    ) {
        for model_buffer in &self.inner {
            match model_buffer {
                ModelBuffer::Static { instances, model } => {
                    if model.transparent_geometry.num_indices > 0 {
                        if let Some((slice, num)) = instances.get() {
                            render_model_transparent(&model, render_pass, renderer, slice, num);
                        }
                    }
                }
                ModelBuffer::Animated {
                    instances,
                    model,
                    joint_transforms_bind_group,
                    ..
                } => {
                    if model.transparent_geometry.num_indices > 0 {
                        if let Some((slice, num)) = instances.get() {
                            render_animated_model_transparent(
                                &model,
                                render_pass,
                                renderer,
                                slice,
                                num,
                                joint_transforms_bind_group,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
enum Model {
    Robot = 0,
    Mouse = 1,
    Tentacle = 2,
}

async fn run() -> anyhow::Result<()> {
    let level_filename = std::env::args().nth(1).unwrap();
    let level_bytes = std::fs::read(&level_filename)?;

    let event_loop = winit::event_loop::EventLoop::new();

    let mut settings = Settings {
        draw_gun: true,
        cursor_grab: true,
        draw_collision_geometry: false,
        draw_contact_points: false,
        draw_player_collider: false,
        noclip: false,
        fxaa_enabled: true,
    };

    let mut renderer = Renderer::new(&event_loop, &settings).await?;

    let mut init_encoder =
        renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("init encoder"),
            });

    let level =
        assets::Level::load_gltf(&level_bytes, &renderer, &mut init_encoder, &level_filename)?;

    let model_buffers = ModelBuffers::new(&renderer, &mut init_encoder)?;

    let monkey_gun = assets::Model::load_gltf(
        include_bytes!("../models/monkey_test_gun.glb"),
        &renderer,
        &mut init_encoder,
        "monkey gun",
    )?;

    let skybox_texture = assets::load_skybox(
        include_bytes!("../textures/skybox.png"),
        &renderer,
        &mut init_encoder,
        "star skybox",
    )?;

    let decals_texture = assets::load_single_texture(
        include_bytes!("../textures/decals.png"),
        &renderer,
        "decals",
    )?;

    renderer.queue.submit(Some(init_encoder.finish()));

    let mut world = legion::World::default();
    let mut resources = legion::Resources::default();
    let mut render_schedule = ecs::render_schedule();

    for (node_index, property) in level.properties.iter() {
        if let assets::Property::Spawn(character) = property {
            let model = match character {
                assets::Character::Robot => Model::Robot,
                assets::Character::Mouse => Model::Mouse,
                assets::Character::Tentacle => Model::Tentacle,
            };

            let entity = world.push((
                model,
                level.node_tree.transform_of(*node_index).into_isometry(),
            ));

            let mut entry = world.entry(entity).unwrap();

            match model {
                Model::Robot => {
                    entry.add_component(ecs::VisionCone::new(ncollide3d::shape::Cone::new(
                        10.0, 10.0,
                    )));
                }
                Model::Mouse | Model::Tentacle => {}
            }

            if let Some(joints) = model_buffers.clone_animation_joints(&model) {
                let animation = match model {
                    Model::Robot => 0,
                    Model::Tentacle => model_buffers.animation_info.tentacle_poke_animation,
                    Model::Mouse => model_buffers.animation_info.mouse_idle_animation,
                };

                entry.add_component(ecs::AnimationState {
                    animation,
                    time: 0.0,
                    joints,
                });
            }
        }
    }

    let overlay_pipeline = renderer::overlay::overlay_pipeline(&renderer, &settings);
    let mut overlay_buffers = renderer::overlay::OverlayBuffers::new(&renderer.device);

    let mut debug_contact_points_buffer = DynamicBuffer::new(
        &renderer.device,
        40,
        "debug contact points buffer",
        wgpu::BufferUsage::VERTEX,
    );

    let mut debug_collision_geometry_buffer = DynamicBuffer::new(
        &renderer.device,
        40,
        "debug collsion geometry buffer",
        wgpu::BufferUsage::VERTEX,
    );

    let mut debug_player_collider_buffer = DynamicBuffer::new(
        &renderer.device,
        40,
        "debug player collider buffer",
        wgpu::BufferUsage::VERTEX,
    );

    let debug_vision_cones_buffer = DynamicBuffer::new(
        &renderer.device,
        40,
        "debug vision cones buffer",
        wgpu::BufferUsage::VERTEX,
    );

    resources.insert(model_buffers);
    resources.insert(ecs::DebugVisionCones(debug_vision_cones_buffer));

    render_debug_mesh(
        &level.collision_mesh,
        &Isometry3::identity(),
        &mut debug_collision_geometry_buffer,
        Vec4::new(1.0, 0.5, 0.25, 1.0),
    );

    /*
    for chunk in level.nav_mesh.1.chunks(3) {
        let a = level.nav_mesh.0[chunk[0] as usize];
        let b = level.nav_mesh.0[chunk[1] as usize];
        let c = level.nav_mesh.0[chunk[2] as usize];
        let colour = Vec4::new(1.0, 0.0, 0.0, 1.0);
        renderer::debug_lines::draw_tri(a, b, c, colour, |vertex| {
            debug_collision_geometry_buffer.push(vertex);
        })
    }
    */

    debug_collision_geometry_buffer.upload(&renderer);

    let mut decal_buffer = DynamicBuffer::new(
        &renderer.device,
        10,
        "decal buffer",
        wgpu::BufferUsage::VERTEX,
    );

    let debug_lines_always_pipeline = renderer::debug_lines::debug_lines_pipeline(
        &renderer,
        &settings,
        wgpu::CompareFunction::Always,
    );
    let debug_lines_less_pipeline = renderer::debug_lines::debug_lines_pipeline(
        &renderer,
        &settings,
        wgpu::CompareFunction::Less,
    );

    let mut control_states = ControlStates::default();

    let mut player = Player {
        position: Vec3::new(0.0, 0.0, 0.0),
        facing: PlayerFacing {
            horizontal: 0.0,
            vertical: 0.0,
        },
        on_ground: false,
        velocity: Vec3::zero(),
        gun_cooldown: 0.0,
        jump_primed: true,
        body_shape: ncollide3d::shape::Capsule::new(Player::BODY_BASE_DEPTH / 2.0, Player::RADIUS),
        feet_shape: ncollide3d::shape::Cuboid::new(
            [
                Player::FEET_HALF_WIDTH,
                Player::FEET_HALF_DEPTH,
                Player::FEET_HALF_WIDTH,
            ]
            .into(),
        ),
    };

    let player_body_debug_mesh: ncollide3d::shape::TriMesh<f32> =
        player.body_shape.to_trimesh((8, 8)).into();
    let player_feet_debug_mesh: ncollide3d::shape::TriMesh<f32> =
        player.feet_shape.to_trimesh(()).into();

    let mut screen_center = renderer.screen_center();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { ref event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                renderer.resize(size.width as u32, size.height as u32, &settings);
                screen_center = renderer.screen_center();
                renderer.window.set_cursor_position(screen_center).unwrap();
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let pressed = state == &ElementState::Pressed;

                match keycode {
                    VirtualKeyCode::W => control_states.forwards_pressed = pressed,
                    VirtualKeyCode::A => control_states.left_pressed = pressed,
                    VirtualKeyCode::S => control_states.backwards_pressed = pressed,
                    VirtualKeyCode::D => control_states.right_pressed = pressed,
                    VirtualKeyCode::Space => control_states.jump_pressed = pressed,
                    VirtualKeyCode::P if pressed => {
                        settings.cursor_grab = !settings.cursor_grab;
                        renderer.window.set_cursor_visible(!settings.cursor_grab);
                        renderer
                            .window
                            .set_cursor_grab(settings.cursor_grab)
                            .unwrap();
                    }
                    VirtualKeyCode::G if pressed => settings.draw_gun = !settings.draw_gun,
                    VirtualKeyCode::C if pressed => {
                        settings.draw_collision_geometry = !settings.draw_collision_geometry
                    }
                    VirtualKeyCode::X if pressed => {
                        settings.draw_contact_points = !settings.draw_contact_points
                    }
                    VirtualKeyCode::Z if pressed => {
                        settings.draw_player_collider = !settings.draw_player_collider
                    }
                    VirtualKeyCode::V if pressed => settings.noclip = !settings.noclip,
                    VirtualKeyCode::F if pressed => settings.fxaa_enabled = !settings.fxaa_enabled,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if settings.cursor_grab {
                    let position = position.to_logical::<f64>(renderer.window.scale_factor());

                    let delta = Vec2::new(
                        (position.x - screen_center.x) as f32,
                        (position.y - screen_center.y) as f32,
                    );

                    renderer.window.set_cursor_position(screen_center).unwrap();

                    player.facing.horizontal -= delta.x.to_radians() * 0.05;
                    player.facing.vertical = (player.facing.vertical - delta.y.to_radians() * 0.05)
                        .min(PI / 2.0)
                        .max(-PI / 2.0);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                let pressed = state == &ElementState::Pressed;

                control_states.mouse_held = pressed;
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            player.gun_cooldown -= 1.0 / 60.0;

            let moved = move_player(&mut player, &control_states, &settings);

            if control_states.mouse_held && player.gun_cooldown < 0.0 {
                player.gun_cooldown = 0.05;

                let normal = player.facing.normal();
                let origin = vec3_into(player.position + Player::HEAD_RELATIVE);
                let ray = ncollide3d::query::Ray::new(origin, vec3_into(normal));

                let intersection = level.collision_mesh.toi_and_normal_with_ray(
                    &vec3_to_ncollide_iso(Vec3::zero()),
                    &ray,
                    1000.0,
                    false,
                );

                if let Some(intersection) = intersection {
                    let hit_position =
                        player.position + Player::HEAD_RELATIVE + normal * intersection.toi;
                    let normal = vec3_from(intersection.normal);

                    renderer::debug_lines::draw_line(
                        hit_position,
                        hit_position + normal,
                        Vec4::new(1.0, 0.0, 0.0, 1.0),
                        |vertex| debug_contact_points_buffer.push(vertex),
                    );

                    for vertex in renderer::decal_square(
                        hit_position + normal * 0.01,
                        normal,
                        Vec2::broadcast(0.5),
                        renderer::Decal::BulletImpact,
                    )
                    .iter()
                    {
                        decal_buffer.push(*vertex);
                    }
                }
            }

            let collisions_need_updating = (moved || !player.on_ground) && !settings.noclip;

            if collisions_need_updating {
                collision_handling(&mut player, &level, &mut debug_contact_points_buffer);
            }

            resources.insert(ecs::PlayerPosition(player.position + Player::HEAD_RELATIVE));

            renderer.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            render_schedule.execute(&mut world, &mut resources);

            let mut model_buffers = resources.get_mut::<ModelBuffers>().unwrap();
            let mut debug_vision_cones = resources.get_mut::<ecs::DebugVisionCones>().unwrap();

            /*
            let static_view = fps_view_rh(
                Player::HEAD_RELATIVE,
                player.facing.vertical,
                player.facing.horizontal,
            );
            */

            let camera_view = fps_view_rh(
                player.position + Player::HEAD_RELATIVE,
                player.facing.vertical,
                player.facing.horizontal,
            );

            renderer.set_camera_view(camera_view);

            if settings.draw_gun {
                overlay_buffers.draw_circle_outline(
                    Vec2::new(screen_center.x as f32, screen_center.y as f32),
                    100.0,
                );
            }
            overlay_buffers.upload(&renderer);

            debug_player_collider_buffer.clear();

            renderer::debug_lines::draw_marker(player.position, 0.5, |vertex| {
                debug_player_collider_buffer.push(vertex);
            });

            renderer::debug_lines::draw_marker(
                player.position + Player::FEET_RELATIVE,
                0.5,
                |vertex| {
                    debug_player_collider_buffer.push(vertex);
                },
            );

            renderer::debug_lines::draw_marker(
                player.position + Player::BODY_RELATIVE,
                0.5,
                |vertex| {
                    debug_player_collider_buffer.push(vertex);
                },
            );

            renderer::debug_lines::draw_marker(
                player.position + Player::HEAD_RELATIVE,
                0.5,
                |vertex| {
                    debug_player_collider_buffer.push(vertex);
                },
            );

            render_debug_mesh(
                &player_body_debug_mesh,
                &Isometry3::new(player.position + Player::BODY_RELATIVE, Rotor3::identity()),
                &mut debug_player_collider_buffer,
                Vec4::new(1.0, 0.5, 0.25, 1.0),
            );

            render_debug_mesh(
                &player_feet_debug_mesh,
                &Isometry3::new(player.position + Player::FEET_RELATIVE, Rotor3::identity()),
                &mut debug_player_collider_buffer,
                Vec4::new(1.0, 0.0, 0.25, 1.0),
            );

            let gun_instance = renderer::single_instance_buffer(
                &renderer.device,
                renderer::Instance::new(camera_view.inversed()),
                "gun instance",
            );

            decal_buffer.upload(&renderer);
            debug_contact_points_buffer.upload(&renderer);
            debug_player_collider_buffer.upload(&renderer);

            model_buffers.upload(&renderer);
            debug_vision_cones.0.upload(&renderer);
            debug_vision_cones.0.clear();

            match renderer.swap_chain.get_current_frame() {
                Ok(frame) => {
                    let mut encoder =
                        renderer
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("render encoder"),
                            });

                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main render pass"),
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &renderer.pre_tonemap_framebuffer,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: Some(
                            wgpu::RenderPassDepthStencilAttachmentDescriptor {
                                attachment: &renderer.depth_texture,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: true,
                                }),
                                stencil_ops: None,
                            },
                        ),
                    });

                    // Render opaque things

                    render_pass.set_pipeline(&renderer.static_opaque_render_pipeline);
                    set_vertex_push_constant(&mut render_pass, renderer.projection_view_matrix);
                    render_pass.set_bind_group(0, &renderer.sampler_bind_group, &[]);
                    render_pass.set_bind_group(2, &level.lights_bind_group, &[]);

                    // Render gun

                    if settings.draw_gun {
                        render_model_opaque(
                            &monkey_gun,
                            &mut render_pass,
                            &renderer,
                            gun_instance.slice(..),
                            1,
                        );
                    }

                    // Render level

                    render_pass.set_bind_group(1, &level.texture_array_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, level.opaque_geometry.vertices.slice(..));
                    render_pass.set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                    render_pass
                        .set_index_buffer(level.opaque_geometry.indices.slice(..), INDEX_FORMAT);
                    render_pass.draw_indexed(0..level.opaque_geometry.num_indices, 0, 0..1);

                    model_buffers.render_opaque(&mut render_pass, &renderer);

                    // Render the skybox

                    render_pass.set_pipeline(&renderer.skybox_render_pipeline);
                    set_vertex_push_constant(
                        &mut render_pass,
                        [renderer.projection_matrix, renderer.view_matrix],
                    );
                    render_pass.set_bind_group(1, &skybox_texture, &[]);
                    render_pass.draw(0..3, 0..1);

                    // Render transparent things

                    render_pass.set_pipeline(&renderer.static_transparent_render_pipeline);
                    set_vertex_push_constant(&mut render_pass, renderer.projection_view_matrix);
                    render_pass.set_bind_group(1, &level.texture_array_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, level.transparent_geometry.vertices.slice(..));
                    render_pass.set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                    render_pass.set_index_buffer(
                        level.transparent_geometry.indices.slice(..),
                        INDEX_FORMAT,
                    );
                    render_pass.draw_indexed(0..level.transparent_geometry.num_indices, 0, 0..1);

                    // Render decals

                    if let Some((slice, len)) = decal_buffer.get() {
                        render_pass.set_bind_group(1, &decals_texture, &[]);
                        render_pass.set_vertex_buffer(0, slice);
                        render_pass
                            .set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                        render_pass.draw(0..len, 0..1);
                    }

                    model_buffers.render_transparent(&mut render_pass, &renderer);

                    // Render debug lines

                    if settings.draw_collision_geometry {
                        render_debug_lines(
                            &debug_collision_geometry_buffer,
                            &mut render_pass,
                            &debug_lines_less_pipeline,
                            &renderer,
                        );
                    }

                    if settings.draw_contact_points {
                        render_debug_lines(
                            &debug_contact_points_buffer,
                            &mut render_pass,
                            &debug_lines_always_pipeline,
                            &renderer,
                        );
                    }

                    if settings.draw_player_collider {
                        render_debug_lines(
                            &debug_player_collider_buffer,
                            &mut render_pass,
                            &debug_lines_always_pipeline,
                            &renderer,
                        );
                    }

                    render_debug_lines(
                        &debug_vision_cones.0,
                        &mut render_pass,
                        &debug_lines_less_pipeline,
                        &renderer,
                    );

                    // Render overlay

                    if let Some((vertices, indices, num_indices)) = overlay_buffers.get() {
                        render_pass.set_pipeline(&overlay_pipeline);
                        set_vertex_push_constant(&mut render_pass, renderer.screen_dimensions_vec);
                        render_pass.set_vertex_buffer(0, vertices);
                        render_pass.set_index_buffer(indices, INDEX_FORMAT);
                        render_pass.draw_indexed(0..num_indices, 0, 0..1);
                    }

                    drop(render_pass);

                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tonemap render pass"),
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: if settings.fxaa_enabled {
                                &renderer.pre_fxaa_framebuffer
                            } else {
                                &frame.output.view
                            },
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });

                    render_pass.set_pipeline(&renderer.tonemap_pipeline);
                    render_pass.set_bind_group(0, &renderer.tonemap_bind_group, &[]);
                    render_pass.draw(0..3, 0..1);

                    drop(render_pass);

                    if settings.fxaa_enabled {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("fxaa render pass"),
                                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                    attachment: &frame.output.view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: true,
                                    },
                                }],
                                depth_stencil_attachment: None,
                            });

                        render_pass.set_pipeline(&renderer.fxaa_pipeline);
                        set_fragment_push_constant(
                            &mut render_pass,
                            renderer.screen_dimensions_vec,
                        );
                        render_pass.set_bind_group(0, &renderer.fxaa_bind_group, &[]);
                        render_pass.draw(0..3, 0..1);

                        drop(render_pass);
                    }

                    renderer.queue.submit(Some(encoder.finish()));
                }
                Err(error) => println!("Frame error: {:?}", error),
            }
        }
        _ => {}
    });
}

// https://www.3dgep.com/understanding-the-view-matrix/#FPS_Camera
fn fps_view_rh(eye: Vec3, pitch: f32, yaw: f32) -> Mat4 {
    let x_axis = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
    let y_axis = Vec3::new(
        yaw.sin() * pitch.sin(),
        pitch.cos(),
        yaw.cos() * pitch.sin(),
    );
    let z_axis = Vec3::new(
        yaw.sin() * pitch.cos(),
        -pitch.sin(),
        pitch.cos() * yaw.cos(),
    );

    // Create a 4x4 view matrix from the right, up, forward and eye position vectors
    Mat4::new(
        Vec4::new(x_axis.x, y_axis.x, z_axis.x, 0.0),
        Vec4::new(x_axis.y, y_axis.y, z_axis.y, 0.0),
        Vec4::new(x_axis.z, y_axis.z, z_axis.z, 0.0),
        Vec4::new(-x_axis.dot(eye), -y_axis.dot(eye), -z_axis.dot(eye), 1.0),
    )
}

fn render_debug_mesh(
    mesh: &ncollide3d::shape::TriMesh<f32>,
    isometry: &Isometry3,
    buffer: &mut DynamicBuffer<renderer::debug_lines::Vertex>,
    colour: Vec4,
) {
    for face in mesh.faces() {
        let points = mesh.points();
        let a = isometry.transform_vec(vec3_from(points[face.indices.x].coords));
        let b = isometry.transform_vec(vec3_from(points[face.indices.y].coords));
        let c = isometry.transform_vec(vec3_from(points[face.indices.z].coords));
        renderer::debug_lines::draw_tri(a, b, c, colour, |vertex| {
            buffer.push(vertex);
        })
    }
}

fn move_player(player: &mut Player, control_states: &ControlStates, settings: &Settings) -> bool {
    const NOCLIP_SPEED: f32 = 0.5;
    const MAX_SPEED: f32 = 0.2;
    const JUMP_SPEED: f32 = 0.6;
    const GRAVITY: f32 = -0.025;
    const ACCELERATION: f32 = 0.04;

    if !control_states.jump_pressed {
        player.jump_primed = true;
    }

    if settings.noclip {
        player.velocity = Vec3::zero();

        if control_states.forwards_pressed {
            player.velocity.z -= NOCLIP_SPEED * player.facing.vertical.cos();
            player.velocity.y += NOCLIP_SPEED * player.facing.vertical.sin();
        }

        if control_states.backwards_pressed {
            player.velocity.z += NOCLIP_SPEED * player.facing.vertical.cos();
            player.velocity.y -= NOCLIP_SPEED * player.facing.vertical.sin();
        }

        if control_states.left_pressed {
            player.velocity.x -= NOCLIP_SPEED;
        }

        if control_states.right_pressed {
            player.velocity.x += NOCLIP_SPEED;
        }

        player.position += Mat3::from_rotation_y(-player.facing.horizontal)
            * Vec3::new(player.velocity.x, 0.0, player.velocity.z);
        player.position += Vec3::new(0.0, player.velocity.y, 0.0);
    } else {
        let mut movement_direction = Vec3::zero();

        if control_states.forwards_pressed {
            movement_direction.z -= 1.0;
        }

        if control_states.backwards_pressed {
            movement_direction.z += 1.0;
        }

        if control_states.left_pressed {
            movement_direction.x -= 1.0;
        }

        if control_states.right_pressed {
            movement_direction.x += 1.0;
        }

        if movement_direction != Vec3::zero() {
            movement_direction = movement_direction.normalized();

            use ultraviolet::Lerp;
            let horizontal_velocity = Vec3::new(player.velocity.x, 0.0, player.velocity.z)
                .lerp(movement_direction * MAX_SPEED, ACCELERATION / MAX_SPEED);
            player.velocity.x = horizontal_velocity.x;
            player.velocity.z = horizontal_velocity.z;
        } else {
            player.velocity.x = approach_zero(player.velocity.x, ACCELERATION);
            player.velocity.z = approach_zero(player.velocity.z, ACCELERATION);
        }

        if player.on_ground {
            player.velocity.y = 0.0;

            if control_states.jump_pressed && player.jump_primed {
                player.velocity.y += JUMP_SPEED;
                player.on_ground = false;
                player.jump_primed = false;
            }

            player.position += Mat3::from_rotation_y(-player.facing.horizontal) * player.velocity;
        } else {
            player.velocity.y += GRAVITY;
            player.position += Mat3::from_rotation_y(-player.facing.horizontal) * player.velocity;
        }
    }

    player.velocity != Vec3::zero()
}

fn approach_zero(value: f32, change: f32) -> f32 {
    if change > value.abs() {
        0.0
    } else {
        value - change * value.signum()
    }
}

fn collision_handling(
    player: &mut Player,
    level: &assets::Level,
    debug_contact_points_buffer: &mut DynamicBuffer<renderer::debug_lines::Vertex>,
) {
    const HORIZONTAL_COLOUR: Vec4 = Vec4::new(0.75, 0.75, 0.0, 1.0);
    const HEAD_COLOUR: Vec4 = Vec4::new(0.25, 0.25, 0.5, 1.0);

    // Falling/jumping clipping prevention
    if !player.on_ground {
        if player.velocity.y < -Player::RADIUS {
            let body_contact = ncollide3d::query::contact(
                &vec3_to_ncollide_iso(player.position + Player::BODY_RELATIVE),
                &player.body_shape,
                &vec3_to_ncollide_iso(Vec3::zero()),
                &level.collision_mesh,
                0.0,
            );

            if let Some(contact) = body_contact {
                let contact_point = vec3_from(contact.world2.coords);

                let relative = contact_point.y - player.position.y;

                if relative > Player::RADIUS {
                    println!(
                        "Adjusting vertical position by {} to prevent clipping into the floor",
                        contact.depth
                    );
                    player.position.y += contact.depth;
                }
            }
        } else if player.velocity.y > Player::RADIUS {
            let body_contact = ncollide3d::query::contact(
                &vec3_to_ncollide_iso(player.position + Player::BODY_RELATIVE),
                &player.body_shape,
                &vec3_to_ncollide_iso(Vec3::zero()),
                &level.collision_mesh,
                0.0,
            );

            if let Some(contact) = body_contact {
                let contact_point = vec3_from(contact.world2.coords);

                let relative = player.position.y + Player::HEAD_RELATIVE.y - contact_point.y;

                if relative > 0.0 {
                    println!(
                        "Adjusting vertical position by {} to prevent clipping through the ceiling",
                        -contact.depth
                    );
                    player.position.y -= contact.depth;
                }
            }
        }
    }

    let mut body_resolution_iterations = 0;

    for _ in 0..10 {
        body_resolution_iterations += 1;

        let body_contact = ncollide3d::query::contact(
            &vec3_to_ncollide_iso(player.position + Player::BODY_RELATIVE),
            &player.body_shape,
            &vec3_to_ncollide_iso(Vec3::zero()),
            &level.collision_mesh,
            0.0,
        );

        match body_contact {
            Some(contact) => {
                let contact_point = vec3_from(contact.world2.coords);
                let epsilon = 0.001;

                if contact_point.y - player.position.y - Player::BODY_BOTTOM_DISTANCE_FROM_GROUND
                    < Player::RADIUS
                {
                    let normal = vec3_from(contact.normal.into_inner());
                    let slope = normal.dot(-Vec3::unit_y()).acos();

                    if slope < 45.0_f32.to_radians() {
                        // If the deepest(?) contact point is the bottom hemisphere
                        // contacting the ground at an acceptable angle, we stop
                        // doing body contacts and 'break' to do the feet contacts.
                        break;
                    } else {
                        renderer::debug_lines::draw_line(
                            contact_point,
                            player.position + Vec3::new(0.0, Player::RADIUS, 0.0),
                            Vec4::new(1.0, 0.0, 0.0, 1.0),
                            |vertex| debug_contact_points_buffer.push(vertex),
                        );

                        let mut vector_away_from_wall = player.position - contact_point;
                        vector_away_from_wall.y = 0.0;
                        let push_strength =
                            (Player::RADIUS - vector_away_from_wall.mag()) + epsilon;
                        let push = vector_away_from_wall.normalized() * push_strength;

                        renderer::debug_lines::draw_line(
                            contact_point,
                            Vec3::new(player.position.x, contact_point.y, player.position.z),
                            Vec4::new(0.0, 1.0, 0.0, 1.0),
                            |vertex| debug_contact_points_buffer.push(vertex),
                        );
                        player.position += push;
                    }
                } else if contact_point.y - player.position.y > Player::HEAD_RELATIVE.y {
                    renderer::debug_lines::draw_line(
                        contact_point,
                        player.position + Player::HEAD_RELATIVE,
                        HEAD_COLOUR,
                        |vertex| debug_contact_points_buffer.push(vertex),
                    );

                    // Handle hitting the top hemisphere on the ceiling.
                    let vector_away_from_ceiling =
                        (player.position + Player::HEAD_RELATIVE) - contact_point;

                    let push_strength = (Player::RADIUS - vector_away_from_ceiling.mag()) + epsilon;

                    player.position += vector_away_from_ceiling.normalized() * push_strength;

                    let normal = vec3_from(contact.normal.into_inner());
                    let slope = normal.dot(Vec3::unit_y()).acos();

                    // Kill the velocity if jumping, but not if falling.
                    // Only do this if the slope of the ceiling is shallow, not if it's a wall.
                    if slope < 45.0_f32.to_radians() {
                        player.velocity.y = player.velocity.y.min(0.0);
                    }
                } else {
                    // Handle horizontal contacts.

                    renderer::debug_lines::draw_line(
                        contact_point,
                        Vec3::new(player.position.x, contact_point.y, player.position.z),
                        HORIZONTAL_COLOUR,
                        |vertex| debug_contact_points_buffer.push(vertex),
                    );

                    /*
                    render_debug_mesh(
                        &player.body_shape_debug_mesh,
                        player.position + Player::BODY_RELATIVE,
                        &mut debug_contact_points_buffer,
                        Vec4::new(1.0, 0.0, 0.0, 1.0),
                    );*/

                    let mut vector_away_from_wall = player.position - contact_point;

                    vector_away_from_wall.y = 0.0;
                    let push_strength = (Player::RADIUS - vector_away_from_wall.mag()) + epsilon;
                    let push = vector_away_from_wall.normalized() * push_strength;

                    /*
                    println!(
                        "player: {:?}\nvector_away_from_wall: {:?}\npush: {:?}\ncp {:?} ps {}\nnew: {:?}\nxxx",
                        player.position, vector_away_from_wall, push, contact_point, push_strength, player.position + push
                    );

                    println!("{:?}", push.y);
                    */

                    player.position += push;
                }
            }
            None => break,
        }
    }

    if body_resolution_iterations > 3 {
        println!(
            "Ran too many body collision resolutions: {}",
            body_resolution_iterations
        );
    }

    let feet_contact = ncollide3d::query::contact(
        &vec3_to_ncollide_iso(player.position + Player::FEET_RELATIVE),
        &player.feet_shape,
        &vec3_to_ncollide_iso(Vec3::zero()),
        &level.collision_mesh,
        0.0,
    );

    match feet_contact {
        Some(contact) => {
            player.position.y += contact.depth;
            player.on_ground = true;
        }
        None => {
            player.on_ground = false;
        }
    }
}

fn render_model_opaque<'a>(
    model: &'a assets::Model,
    render_pass: &mut wgpu::RenderPass<'a>,
    renderer: &'a Renderer,
    instances: wgpu::BufferSlice<'a>,
    num_instances: u32,
) {
    render_pass.set_pipeline(&renderer.static_opaque_render_pipeline);
    set_vertex_push_constant(render_pass, renderer.projection_view_matrix);
    render_pass.set_bind_group(1, &model.textures, &[]);
    render_pass.set_vertex_buffer(0, model.opaque_geometry.vertices.slice(..));
    render_pass.set_vertex_buffer(1, instances);
    render_pass.set_index_buffer(model.opaque_geometry.indices.slice(..), INDEX_FORMAT);
    render_pass.draw_indexed(0..model.opaque_geometry.num_indices, 0, 0..num_instances)
}

fn render_animated_model_opaque<'a>(
    model: &'a assets::AnimatedModel,
    render_pass: &mut wgpu::RenderPass<'a>,
    renderer: &'a Renderer,
    instances: wgpu::BufferSlice<'a>,
    num_instances: u32,
    joint_transforms_bind_group: &'a wgpu::BindGroup,
) {
    render_pass.set_pipeline(&renderer.animated_opaque_render_pipeline);
    set_vertex_push_constant(
        render_pass,
        renderer::AnimatedPushConstants {
            projection_view: renderer.projection_view_matrix,
            num_joints: model.num_joints,
        },
    );
    render_pass.set_bind_group(1, &model.textures, &[]);
    render_pass.set_bind_group(3, joint_transforms_bind_group, &[]);
    render_pass.set_vertex_buffer(0, model.opaque_geometry.vertices.slice(..));
    render_pass.set_vertex_buffer(1, instances);
    render_pass.set_index_buffer(model.opaque_geometry.indices.slice(..), INDEX_FORMAT);
    render_pass.draw_indexed(0..model.opaque_geometry.num_indices, 0, 0..num_instances)
}

fn render_model_transparent<'a>(
    model: &'a assets::Model,
    render_pass: &mut wgpu::RenderPass<'a>,
    renderer: &'a Renderer,
    instances: wgpu::BufferSlice<'a>,
    num_instances: u32,
) {
    render_pass.set_pipeline(&renderer.static_transparent_render_pipeline);
    set_vertex_push_constant(render_pass, renderer.projection_view_matrix);
    render_pass.set_bind_group(1, &model.textures, &[]);
    render_pass.set_vertex_buffer(0, model.transparent_geometry.vertices.slice(..));
    render_pass.set_vertex_buffer(1, instances);
    render_pass.set_index_buffer(model.transparent_geometry.indices.slice(..), INDEX_FORMAT);
    render_pass.draw_indexed(
        0..model.transparent_geometry.num_indices,
        0,
        0..num_instances,
    )
}

fn render_animated_model_transparent<'a>(
    model: &'a assets::AnimatedModel,
    render_pass: &mut wgpu::RenderPass<'a>,
    renderer: &'a Renderer,
    instances: wgpu::BufferSlice<'a>,
    num_instances: u32,
    joint_transforms_bind_group: &'a wgpu::BindGroup,
) {
    render_pass.set_pipeline(&renderer.animated_transparent_render_pipeline);
    set_vertex_push_constant(
        render_pass,
        renderer::AnimatedPushConstants {
            projection_view: renderer.projection_view_matrix,
            num_joints: model.num_joints,
        },
    );
    render_pass.set_bind_group(1, &model.textures, &[]);
    render_pass.set_bind_group(3, joint_transforms_bind_group, &[]);
    render_pass.set_vertex_buffer(0, model.transparent_geometry.vertices.slice(..));
    render_pass.set_vertex_buffer(1, instances);
    render_pass.set_index_buffer(model.transparent_geometry.indices.slice(..), INDEX_FORMAT);
    render_pass.draw_indexed(
        0..model.transparent_geometry.num_indices,
        0,
        0..num_instances,
    )
}

fn render_debug_lines<'a>(
    buffer: &'a DynamicBuffer<renderer::debug_lines::Vertex>,
    render_pass: &mut wgpu::RenderPass<'a>,
    pipeline: &'a wgpu::RenderPipeline,
    renderer: &'a Renderer,
) {
    if let Some((slice, len)) = buffer.get() {
        render_pass.set_pipeline(pipeline);
        set_vertex_push_constant(render_pass, renderer.projection_view_matrix);
        render_pass.set_vertex_buffer(0, slice);
        render_pass.draw(0..len, 0..1);
    }
}

fn set_vertex_push_constant<T: bytemuck::Pod>(render_pass: &mut wgpu::RenderPass, value: T) {
    render_pass.set_push_constants(wgpu::ShaderStage::VERTEX, 0, bytemuck::bytes_of(&value));
}

fn set_fragment_push_constant<T: bytemuck::Pod>(render_pass: &mut wgpu::RenderPass, value: T) {
    render_pass.set_push_constants(wgpu::ShaderStage::FRAGMENT, 0, bytemuck::bytes_of(&value));
}
