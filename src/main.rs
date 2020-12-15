mod animation;
mod assets;

fn main() -> anyhow::Result<()> {
    futures::executor::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!(
            "'request_adapter' failed. If you get this on linux, try installing the vulkan drivers for your gpu. \
            You can check that they're working properly by running `vulkaninfo` or `vkcube`."
        ))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await?;

    let level_bind_group_layout = assets::level_bind_group_layout(&device);

    let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("init encoder"),
    });

    let level = assets::Level::load_gltf(
        include_bytes!("../warehouse.glb"),
        &device,
        &mut init_encoder,
        &level_bind_group_layout,
    )?;

    queue.submit(Some(init_encoder.finish()));

    Ok(())
}
