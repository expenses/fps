//mod animation;
mod assets;
mod renderer;

use ncollide3d::query::RayCast;
use std::f32::consts::PI;
use ultraviolet::{Mat3, Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

fn main() -> anyhow::Result<()> {
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
    velocity: f32,
    gun_cooldown: f32,
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

fn vec3_from_arr(arr: [f32; 3]) -> Vec3 {
    arr.into()
}

async fn run() -> anyhow::Result<()> {
    let event_loop = winit::event_loop::EventLoop::new();
    let mut renderer = renderer::Renderer::new(&event_loop).await?;

    let mut init_encoder =
        renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("init encoder"),
            });

    let level = assets::Level::load_gltf(
        include_bytes!("../warehouse.glb"),
        &renderer,
        &mut init_encoder,
    )?;

    let robot = assets::Model::load_gltf(
        include_bytes!("../warehouse_robot.glb"),
        &renderer,
        &mut init_encoder,
    )?;

    let monkey_gun = assets::Model::load_gltf(
        include_bytes!("../monkey_test_gun.glb"),
        &renderer,
        &mut init_encoder,
    )?;

    let skybox_texture = assets::load_skybox(
        include_bytes!("../skybox.png"),
        &renderer,
        &mut init_encoder,
    )?;

    let decals_texture = assets::load_single_texture(
        include_bytes!("../decals.png"),
        &renderer,
        &mut init_encoder,
    )?;

    renderer.queue.submit(Some(init_encoder.finish()));

    let mut robot_instances = Vec::new();

    for (node_index, property) in level.properties.iter() {
        if let assets::Property::Spawn(assets::Character::Robot) = property {
            robot_instances.push(level.node_tree.transform_of(*node_index));
        }
    }

    let robot_instances_buffer =
        renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("robot instances"),
                contents: bytemuck::cast_slice(&robot_instances),
                usage: wgpu::BufferUsage::VERTEX,
            });

    let overlay_pipeline = renderer::overlay::OverlayPipeline::new(&renderer);
    let mut overlay_buffers = renderer::overlay::OverlayBuffers::new(&renderer.device);

    let mut debug_line_vertices: Vec<renderer::debug_lines::Vertex> = Vec::new();

    /*
    for face in level.collision_mesh.faces() {
        let points = level_collider.collision_mesh.points();
        let a: [f32; 3] = points[face.indices.x].coords.into();
        let b: [f32; 3] = points[face.indices.y].coords.into();
        let c: [f32; 3] = points[face.indices.z].coords.into();
        renderer::debug_lines::draw_tri(
            a.into(),
            b.into(),
            c.into(),
            Vec4::new(1.0, 0.5, 0.25, 1.0),
            |vertex| {
                debug_line_vertices.push(vertex);
            },
        )
    }*/

    let mut decals_vertices = Vec::new();

    let debug_lines_pipeline = renderer::debug_lines::debug_lines_pipeline(&renderer);

    let mut control_states = ControlStates::default();
    let mut player = Player {
        position: Vec3::new(0.0, 5.0, 0.0),
        facing: PlayerFacing {
            horizontal: 0.0,
            vertical: 0.0,
        },
        on_ground: false,
        velocity: 0.0,
        gun_cooldown: 0.0,
    };

    let player_body = ncollide3d::shape::Capsule::new(1.25, 0.5);
    let player_feet = ncollide3d::shape::Cuboid::new([0.75 / 2.0, 1.0, 0.75 / 2.0].into());

    let player_feet_relative = Vec3::new(0.0, 1.0, 0.0);
    let player_body_relative = Vec3::new(0.0, 2.6, 0.0);
    let player_head_relative = Vec3::new(0.0, 5.1, 0.0);

    let mut screen_center = renderer.screen_center();
    let mut cursor_grab = true;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { ref event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                renderer.resize(size.width as u32, size.height as u32);
                screen_center = renderer.screen_center();
                renderer.window.set_cursor_position(screen_center).unwrap();
                overlay_pipeline.resize(&renderer, size.width as u32, size.height as u32);
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
                        cursor_grab = !cursor_grab;
                        renderer.window.set_cursor_visible(!cursor_grab);
                        renderer.window.set_cursor_grab(cursor_grab).unwrap();
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if cursor_grab {
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
            let speed = 0.2;

            player.gun_cooldown -= 1.0 / 60.0;

            let mut movement = Vec3::zero();

            if control_states.forwards_pressed {
                movement.z -= speed;
            }

            if control_states.backwards_pressed {
                movement.z += speed;
            }

            if control_states.left_pressed {
                movement.x -= speed;
            }

            if control_states.right_pressed {
                movement.x += speed;
            }

            if player.on_ground {
                player.velocity = 0.0;

                if control_states.jump_pressed {
                    player.velocity -= speed * 3.0;
                    player.on_ground = false;
                }
            }

            if !player.on_ground {
                player.velocity += 0.025;
                player.position.y -= player.velocity;
            }

            player.position += Mat3::from_rotation_y(-player.facing.horizontal) * movement;

            if control_states.mouse_held && player.gun_cooldown < 0.0 {
                player.gun_cooldown = 0.05;

                let normal = player.facing.normal();
                let origin: [f32; 3] = (player.position + player_head_relative).into();
                let normal_arr: [f32; 3] = normal.into();
                let ray = ncollide3d::query::Ray::new(origin.into(), normal_arr.into());

                let intersection = level.collision_mesh.toi_and_normal_with_ray(
                    &vec3_to_ncollide_iso(Vec3::zero()),
                    &ray,
                    1000.0,
                    false,
                );

                if let Some(intersection) = intersection {
                    let hit_position =
                        player.position + player_head_relative + normal * intersection.toi;
                    let normal = vec3_from_arr(intersection.normal.into());

                    /*renderer::debug_lines::draw_line(
                        hit_position,
                        hit_position + normal,
                        Vec4::new(1.0, 0.0, 0.0, 1.0),
                        |vertex| debug_line_vertices.push(vertex),
                    );*/

                    decals_vertices.extend_from_slice(&renderer::decal_square(
                        hit_position + normal * 0.01,
                        normal,
                        Vec2::broadcast(0.5),
                        renderer::Decal::BulletImpact,
                    ));
                }
            }

            let collisions_need_updating = movement != Vec3::zero() || !player.on_ground;

            if collisions_need_updating {
                for _ in 0..10 {
                    let body_contact = ncollide3d::query::contact(
                        &vec3_to_ncollide_iso(player.position + player_body_relative),
                        &player_body,
                        &vec3_to_ncollide_iso(Vec3::zero()),
                        &level.collision_mesh,
                        0.0,
                    );

                    match body_contact {
                        Some(contact) => {
                            let contact_point = vec3_from_arr(contact.world2.coords.into());

                            renderer::debug_lines::draw_line(
                                contact_point,
                                Vec3::new(player.position.x, contact_point.y, player.position.z),
                                Vec4::one(),
                                |vertex| debug_line_vertices.push(vertex),
                            );

                            let epsilon = 0.001;
                            let body_radius = 0.5;

                            let mut vector_away_from_wall = player.position - contact_point;

                            //println!("{:?}", vector_away_from_wall + player_head_relative);

                            if vector_away_from_wall.x == 0.0 && vector_away_from_wall.z == 0.0 {
                                // Need to handle this case
                            } else {
                                vector_away_from_wall.y = 0.0;
                                let push_strength =
                                    (body_radius - vector_away_from_wall.mag()) + epsilon;
                                let push_strength = push_strength.max(0.0);
                                let push = vector_away_from_wall.normalized() * push_strength;

                                /*println!(
                                    "player: {:?}\nvector_away_from_wall: {:?}\npush: {:?}\ncp {:?} ps {}\nnew: {:?}\nxxx",
                                    player.position, vector_away_from_wall, push, contact_point, push_strength, player.position + push
                                );

                                println!("{:?}", push.y);*/

                                player.position += push;
                            }
                        }
                        None => break,
                    }
                }

                let feet_contact = ncollide3d::query::contact(
                    &vec3_to_ncollide_iso(player.position + player_feet_relative),
                    &player_feet,
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

            renderer.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let camera_view = fps_view_rh(
                player.position + player_head_relative,
                player.facing.vertical,
                player.facing.horizontal,
            );

            renderer.set_camera_view(camera_view);

            overlay_buffers.draw_circle_outline(
                Vec2::new(screen_center.x as f32, screen_center.y as f32),
                100.0,
            );
            overlay_buffers.upload(&renderer);

            let gun_instance = renderer::single_instance_buffer(
                &renderer.device,
                renderer::Instance {
                    transform: camera_view.inversed(),
                },
                "gun instance",
            );

            let debug_lines_buffer =
                renderer
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("debug line vertices"),
                        contents: bytemuck::cast_slice(&debug_line_vertices),
                        usage: wgpu::BufferUsage::VERTEX,
                    });

            let decals_buffer =
                renderer
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("decals buffer"),
                        contents: bytemuck::cast_slice(&decals_vertices),
                        usage: wgpu::BufferUsage::VERTEX,
                    });

            match renderer.swap_chain.get_current_frame() {
                Ok(frame) => {
                    let mut encoder =
                        renderer
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("render encoder"),
                            });

                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &renderer.multisampled_framebuffer_texture,
                            resolve_target: Some(&frame.output.view),
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

                    render_pass.set_pipeline(&renderer.opaque_render_pipeline);
                    render_pass.set_bind_group(0, &renderer.main_bind_group, &[]);
                    render_pass.set_bind_group(2, &level.lights_bind_group, &[]);

                    // Render gun

                    render_pass.set_bind_group(1, &monkey_gun.textures, &[]);
                    render_pass.set_vertex_buffer(0, monkey_gun.buffers.vertices.slice(..));
                    render_pass.set_vertex_buffer(1, gun_instance.slice(..));
                    render_pass.set_index_buffer(monkey_gun.buffers.indices.slice(..));
                    render_pass.draw_indexed(0..monkey_gun.buffers.num_indices, 0, 0..1);

                    // Render level

                    render_pass.set_bind_group(1, &level.texture_array_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, level.opaque_geometry.vertices.slice(..));
                    render_pass.set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                    render_pass.set_index_buffer(level.opaque_geometry.indices.slice(..));
                    render_pass.draw_indexed(0..level.opaque_geometry.num_indices, 0, 0..1);

                    render_pass.set_bind_group(1, &robot.textures, &[]);
                    render_pass.set_vertex_buffer(0, robot.buffers.vertices.slice(..));
                    render_pass.set_vertex_buffer(1, robot_instances_buffer.slice(..));
                    render_pass.set_index_buffer(robot.buffers.indices.slice(..));
                    render_pass.draw_indexed(
                        0..robot.buffers.num_indices,
                        0,
                        0..robot_instances.len() as u32,
                    );

                    // Render the skybox

                    render_pass.set_pipeline(&renderer.skybox_render_pipeline);
                    render_pass.set_bind_group(1, &skybox_texture, &[]);
                    render_pass.draw(0..3, 0..1);

                    // Render transparent things

                    render_pass.set_pipeline(&renderer.transparent_render_pipeline);

                    render_pass.set_bind_group(1, &level.texture_array_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, level.transparent_geometry.vertices.slice(..));
                    render_pass.set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                    render_pass.set_index_buffer(level.transparent_geometry.indices.slice(..));
                    render_pass.draw_indexed(0..level.transparent_geometry.num_indices, 0, 0..1);

                    // Render decals

                    render_pass.set_bind_group(1, &decals_texture, &[]);
                    render_pass.set_vertex_buffer(0, decals_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                    render_pass.draw(0..decals_vertices.len() as u32, 0..1);

                    // Render debug lines

                    render_pass.set_pipeline(&debug_lines_pipeline);
                    render_pass.set_vertex_buffer(0, debug_lines_buffer.slice(..));
                    render_pass.draw(0..debug_line_vertices.len() as u32, 0..1);

                    if let Some((vertices, indices, num_indices)) = overlay_buffers.get() {
                        render_pass.set_pipeline(&overlay_pipeline.pipeline);
                        render_pass.set_bind_group(0, &overlay_pipeline.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, vertices);
                        render_pass.set_index_buffer(indices);
                        render_pass.draw_indexed(0..num_indices, 0, 0..1);
                    }

                    drop(render_pass);

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
