use crate::renderer::{Renderer, TEXTURE_FORMAT};
use wgpu::util::DeviceExt;

const MIPMAP_LEVELS: u32 = 7;

#[derive(Default)]
pub struct ArrayOfTextures {
    texture_views: Vec<wgpu::TextureView>,
}

impl ArrayOfTextures {
    pub fn add(
        &mut self,
        image: &image::RgbaImage,
        label: &str,
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> usize {
        let (image_width, image_height) = image.dimensions();
        assert_eq!(image_width % (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / 4), 0);

        let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth: 1,
            },
            mip_level_count: MIPMAP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::RENDER_ATTACHMENT,
        });

        let staging_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &*image,
                    usage: wgpu::BufferUsage::COPY_SRC,
                });

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &staging_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: 4 * image_width,
                    rows_per_image: 0,
                },
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth: 1,
            },
        );

        // Mipmap generation
        let mipmap_views: Vec<_> = (0..MIPMAP_LEVELS)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: level,
                    level_count: Some(std::num::NonZeroU32::new(1).unwrap()),
                    base_array_layer: 0,
                    array_layer_count: Some(std::num::NonZeroU32::new(1).unwrap()),
                    ..Default::default()
                })
            })
            .collect();

        for level in 1..MIPMAP_LEVELS as usize {
            let bind_group = renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &renderer.mipmap_generation_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&mipmap_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&renderer.linear_sampler),
                        },
                    ],
                });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &mipmap_views[level],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&renderer.mipmap_generation_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        let index = self.texture_views.len();

        self.texture_views
            .push(texture.create_view(&wgpu::TextureViewDescriptor::default()));

        index
    }

    pub fn bind(&self, renderer: &mut Renderer) -> wgpu::BindGroup {
        renderer.rebuild_pipelines_for_textures(self.texture_views.len() as u32);

        let views: Vec<_> = self.texture_views.iter().collect();

        renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("array of textures bind group"),
                layout: &renderer.array_of_textures_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&views),
                }],
            })
    }
}
