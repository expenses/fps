//mod animation;
mod assets;
mod renderer;

use std::f32::consts::PI;
use ultraviolet::{Mat3, Mat4, Vec2, Vec3, Vec4};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

fn main() -> anyhow::Result<()> {
    futures::executor::block_on(run())
}

#[derive(Default)]
struct KeyStates {
    forwards_pressed: bool,
    backwards_pressed: bool,
    left_pressed: bool,
    right_pressed: bool,
    jump_pressed: bool,
}

struct Player {
    position: Vec3,
    facing: PlayerFacing,
    on_ground: bool,
    velocity: f32,
}

#[derive(Debug)]
struct PlayerFacing {
    horizontal: f32,
    vertical: f32,
}

fn vec3_to_ncollide_iso(vec: Vec3) -> ncollide3d::math::Isometry<f32> {
    ncollide3d::math::Isometry::translation(vec.x, vec.y, vec.z)
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

    let (level, level_collider) = assets::Level::load_gltf(
        include_bytes!("../warehouse.glb"),
        &renderer.device,
        &mut init_encoder,
        &renderer.texture_array_bind_group_layout,
        &renderer.lights_bind_group_layout,
    )?;

    let (cylinder, _) = assets::Level::load_gltf(
        include_bytes!("../cylinder.glb"),
        &renderer.device,
        &mut init_encoder,
        &renderer.texture_array_bind_group_layout,
        &renderer.lights_bind_group_layout,
    )?;

    let mut key_states = KeyStates::default();
    let mut player = Player {
        position: Vec3::new(0.0, 5.0, 0.0),
        facing: PlayerFacing {
            horizontal: 0.0,
            vertical: 0.0,
        },
        on_ground: false,
        velocity: 0.0,
    };

    let player_collider = ncollide3d::shape::Capsule::new(1.25, 0.5);
    let player_feet = ncollide3d::shape::Cuboid::new([0.75 / 2.0, 1.0, 0.75 / 2.0].into());

    let player_feet_relative = Vec3::new(0.0, 1.0, 0.0);
    let player_collider_relative = Vec3::new(0.0, 2.6, 0.0);
    let player_head_relative = Vec3::new(0.0, 5.1, 0.0);

    let mut collision_world = ncollide3d::world::CollisionWorld::new(0.2);

    let player_collision_group = ncollide3d::pipeline::object::CollisionGroups::new().with_membership(&[0]).with_whitelist(&[1]);
    let level_collision_group = ncollide3d::pipeline::object::CollisionGroups::new().with_membership(&[1]).with_whitelist(&[0]);

    let (player_feet_handle, _) = collision_world.add(
        vec3_to_ncollide_iso(player.position + player_feet_relative),
        ncollide3d::shape::ShapeHandle::<f32>::new(player_feet),
        player_collision_group,
        ncollide3d::pipeline::object::GeometricQueryType::Contacts(0.0, 0.0),
        ()
    );

    let (player_collider_handle, _) = collision_world.add(
        vec3_to_ncollide_iso(player.position + player_collider_relative),
        ncollide3d::shape::ShapeHandle::<f32>::new(player_collider),
        player_collision_group,
        ncollide3d::pipeline::object::GeometricQueryType::Contacts(0.0, 0.0),
        ()
    );

    collision_world.add(
        ncollide3d::math::Isometry::translation(0.0, 0.0, 0.0),
        ncollide3d::shape::ShapeHandle::<f32>::new(level_collider.collision_mesh),
        level_collision_group,
        ncollide3d::pipeline::object::GeometricQueryType::Contacts(0.0, 0.0),
        ()
    );

    let mut screen_center = renderer.screen_center();

    renderer.queue.submit(Some(init_encoder.finish()));

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { ref event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                renderer.resize(size.width as u32, size.height as u32);
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
                    VirtualKeyCode::W => key_states.forwards_pressed = pressed,
                    VirtualKeyCode::A => key_states.left_pressed = pressed,
                    VirtualKeyCode::S => key_states.backwards_pressed = pressed,
                    VirtualKeyCode::D => key_states.right_pressed = pressed,
                    VirtualKeyCode::Space => key_states.jump_pressed = pressed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
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
            _ => {}
        },
        Event::MainEventsCleared => {
            let speed = 0.2;

            let mut movement = Vec3::zero();

            if key_states.forwards_pressed {
                movement.z -= speed;
            }

            if key_states.backwards_pressed {
                movement.z += speed;
            }

            if key_states.left_pressed {
                movement.x -= speed;
            }

            if key_states.right_pressed {
                movement.x += speed;
            }

            if player.on_ground {
                player.velocity = 0.0;

                if key_states.jump_pressed {
                    player.velocity -= speed * 3.0;
                    player.on_ground = false;
                }
            } else {
                player.velocity += 0.025;
                player.position.y -= player.velocity;
            }

            player.position += Mat3::from_rotation_y(-player.facing.horizontal) * movement;

            let mut collisions_need_updating = movement != Vec3::zero() || !player.on_ground;

            while collisions_need_updating {
                collisions_need_updating = false;

                collision_world.get_mut(player_feet_handle).unwrap().set_position(vec3_to_ncollide_iso(player.position + player_feet_relative));
                collision_world.get_mut(player_collider_handle).unwrap().set_position(vec3_to_ncollide_iso(player.position + player_collider_relative));

                collision_world.update();

                for event in collision_world.contact_events() {
                    let (contact_manifold, handle_1, handle_2) = match event {
                        ncollide3d::pipeline::narrow_phase::ContactEvent::Started(handle_1, handle_2) => {
                            let (.., manifold) = collision_world.contact_pair(*handle_1, *handle_2, true).unwrap();
                            (Some(manifold), handle_1, handle_2)
                        },
                        ncollide3d::pipeline::narrow_phase::ContactEvent::Stopped(handle_1, handle_2) => {
                            (None, handle_1, handle_2)
                        }
                    };

                    let is_feet = *handle_1 == player_feet_handle || *handle_2 == player_feet_handle;
                    let player_first = *handle_1 == player_feet_handle || *handle_1 == player_collider_handle;

                    println!("{} {}", if contact_manifold.is_some() { "Started" } else { "Stopped" }, if is_feet { "feet" } else { "body" });

                    match contact_manifold {
                        Some(manifold) => {
                            if is_feet {
                                let deepest = manifold.deepest_contact().unwrap().contact;

                                println!("{:?} {}", player.position, deepest.depth);

                                player.position.y += deepest.depth;
                                println!("{:?}", player.position);
                                player.on_ground = true;
                            } else {
                                // todo: loop through all contacts.

                                let contact = manifold.deepest_contact().unwrap().contact;

                                let contact_point: [f32; 3] = if player_first { contact.world2.coords.into() } else { contact.world1.coords.into() };
                                let contact_point: Vec3 = contact_point.into();
                                let epsilon = 0.01;

                                println!("{:?} {:?}", player.position, contact_point);
                                
                                let mut wall_direction = player.position - contact_point;

                                // Need to handle this case
                                if wall_direction.x == 0.0 && wall_direction.z == 0.0 {

                                } else {
                                    println!("{} {}", wall_direction.y, player.position.y);
                                    wall_direction.y = 0.0;
                                    let push_strength = 0.5 + epsilon - wall_direction.mag();
                                    let push = wall_direction.normalized() * push_strength;

                                    println!("{:?} {:?} {:?}", wall_direction, push, player.position);

                                    player.position += push;
                                }
                            }

                            collisions_need_updating = true;
                        },
                        None => {
                            if is_feet {
                                player.on_ground = false;
                            }
                        }
                    }
                }
            }

            renderer.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            renderer.set_camera_view(fps_view_rh(
                player.position + player_head_relative,
                player.facing.vertical,
                player.facing.horizontal,
            ));

            let player_position_instance = renderer::single_instance_buffer(&renderer.device, renderer::Instance { transform: Mat4::from_translation(player.position) }, "player position instance");
            
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
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
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

                    render_pass.set_pipeline(&renderer.scene_render_pipeline);
                    render_pass.set_bind_group(0, &renderer.main_bind_group, &[]);
                    render_pass.set_bind_group(2, &level.lights_bind_group, &[]);

                    render_pass.set_bind_group(1, &level.texture_array_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, level.geometry_vertices.slice(..));
                    render_pass.set_vertex_buffer(1, renderer.identity_instance_buffer.slice(..));
                    render_pass.set_index_buffer(level.geometry_indices.slice(..));
                    render_pass.draw_indexed(0..level.num_indices, 0, 0..1);

                    render_pass.set_bind_group(1, &cylinder.texture_array_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, cylinder.geometry_vertices.slice(..));
                    render_pass.set_vertex_buffer(1, player_position_instance.slice(..));
                    render_pass.set_index_buffer(cylinder.geometry_indices.slice(..));
                    render_pass.draw_indexed(0..cylinder.num_indices, 0, 0..1);

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
