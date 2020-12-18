mod animation;
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
}

#[derive(Debug)]
struct PlayerFacing {
    horizontal: f32,
    vertical: f32,
}

struct PhysicsDebugEventPrinter;

impl rapier3d::pipeline::EventHandler for PhysicsDebugEventPrinter {
    fn handle_proximity_event(&self, _: rapier3d::geometry::ProximityEvent) {}

    fn handle_contact_event(&self, e: rapier3d::geometry::ContactEvent) {
        println!("Physics contact event: {:?}", e);
    }
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

    let (level, level_physics) = assets::Level::load_gltf(
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
    };

    let mut screen_center = renderer.screen_center();

    renderer.queue.submit(Some(init_encoder.finish()));

    let mut physics_pipeline = rapier3d::pipeline::PhysicsPipeline::new();
    let gravity = rapier3d::na::Vector3::new(0.0, -1.0 * 60.0, 0.0);
    let integration_parameters = rapier3d::dynamics::IntegrationParameters::default();
    let mut broad_phase = rapier3d::geometry::BroadPhase::new();
    let mut narrow_phase = rapier3d::geometry::NarrowPhase::new();
    let mut bodies = rapier3d::dynamics::RigidBodySet::new();
    let mut colliders = rapier3d::geometry::ColliderSet::new();
    let mut joints = rapier3d::dynamics::JointSet::new();

    let level_rigid_body_handle = bodies.insert(level_physics.rigid_body);
    let level_collider_handle =
        colliders.insert(level_physics.collider, level_rigid_body_handle, &mut bodies);

    let player_rigid_body = rapier3d::dynamics::RigidBodyBuilder::new_dynamic()
        .translation(0.0, 4.0, 0.0)
        .build();
    let player_rigid_body_handle = bodies.insert(player_rigid_body);
    let player_collider = rapier3d::geometry::ColliderBuilder::cylinder(2.0, 1.0)
        //.friction(1000.0)
        .build();
    let player_collider_handle =
        colliders.insert(player_collider, player_rigid_body_handle, &mut bodies);

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
            physics_pipeline.step(
                &gravity,
                &integration_parameters,
                &mut broad_phase,
                &mut narrow_phase,
                &mut bodies,
                &mut colliders,
                &mut joints,
                None,
                None,
                &PhysicsDebugEventPrinter,
            );

            let mut position: [f32; 3] = bodies
                .get(player_rigid_body_handle)
                .unwrap()
                .position()
                .translation
                .vector
                .into();

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

            if key_states.jump_pressed {
                movement.y += speed;
            }

            let movement: [f32; 3] =
                (Mat3::from_rotation_y(-player.facing.horizontal) * movement * 60.0).into();
            //println!("{:?}", movement);

            let player_rigid_body = bodies.get_mut(player_rigid_body_handle).unwrap();
            let mut position = *player_rigid_body.position();
            position.rotation = Default::default();
            player_rigid_body.set_position(position, false);

            player_rigid_body.set_linvel([0.0; 3].into(), true);

            if movement != [0.0; 3] {
                player_rigid_body.set_linvel([movement[0], movement[1], movement[2]].into(), true);
            }

            let position: [f32; 3] = position.translation.vector.into();
            player.position = position.into();

            renderer.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            renderer.set_camera_view(fps_view_rh(
                player.position,
                player.facing.vertical,
                player.facing.horizontal,
            ));

            let cylinder_transform: [[f32; 4]; 4] = bodies
                .get(player_rigid_body_handle)
                .unwrap()
                .position()
                .to_homogeneous()
                .into();
            let cylinder_instance = renderer::single_instance_buffer(
                &renderer.device,
                renderer::Instance {
                    transform: cylinder_transform.into(),
                },
                "cylinder instance",
            );

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
                    render_pass.set_vertex_buffer(1, cylinder_instance.slice(..));
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

    Ok(())
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
