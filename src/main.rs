mod animation;
mod assets;
mod renderer;

use std::f32::consts::PI;
use ultraviolet::{Mat3, Mat4, Vec2, Vec3};
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

struct PlayerFacing {
    horizontal: f32,
    vertical: f32,
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
        &renderer.device,
        &mut init_encoder,
        &renderer.level_bind_group_layout,
    )?;

    let mut key_states = KeyStates::default();
    let mut player = Player {
        position: Vec3::new(0.0, -5.0, 0.0),
        facing: PlayerFacing {
            horizontal: 0.0,
            vertical: 0.0,
        },
    };

    let mut screen_center = renderer.screen_center();

    renderer.queue.submit(Some(init_encoder.finish()));

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { ref event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                renderer.resize(size.width as u32, size.height as u32);
                let mut screen_center = renderer.screen_center();
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
                player.facing.vertical = (player.facing.vertical + delta.y.to_radians() * 0.05)
                    .min(PI / 2.0)
                    .max(-PI / 2.0);
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            let speed = 0.2;

            let mut movement = Vec3::zero();

            if key_states.forwards_pressed {
                movement.z += speed;
            }

            if key_states.backwards_pressed {
                movement.z -= speed;
            }

            if key_states.left_pressed {
                movement.x += speed;
            }

            if key_states.right_pressed {
                movement.x -= speed;
            }

            player.position += Mat3::from_rotation_y(-player.facing.horizontal) * movement;

            renderer.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            renderer.set_camera_view({
                Mat4::from_rotation_x(player.facing.vertical)
                    * Mat4::from_rotation_y(player.facing.horizontal)
                    * Mat4::from_translation(player.position)
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
                    render_pass.set_bind_group(1, &level.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, level.geometry_vertices.slice(..));
                    render_pass.set_index_buffer(level.geometry_indices.slice(..));
                    render_pass.draw_indexed(0..level.num_indices, 0, 0..1);

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
