use guillotiere::{size2, AtlasAllocator as SimpleAtlasAllocator, Rectangle, Size};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct TextureLocation {
    offset: [f32; 2],
    size: [f32; 2],
}

pub struct TextureAtlas {
    allocator: SimpleAtlasAllocator,
    texture: wgpu::Texture,
    texture_format: wgpu::TextureFormat,
    mipmap_levels: u32,
    rectangles: Vec<Rectangle>,
    texture_view: wgpu::TextureView,
}

impl TextureAtlas {
    pub fn new(
        size: u32,
        mipmap_levels: u32,
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
    ) -> Self {
        let allocator = SimpleAtlasAllocator::with_options(
            size2(size as i32, size as i32),
            &guillotiere::AllocatorOptions {
                //alignment: size2(16, 16),
                ..Default::default()
            },
        );

        let (texture, texture_view) =
            atlas_texture_of_size(allocator.size(), device, mipmap_levels, texture_format);

        Self {
            allocator,
            mipmap_levels,
            texture,
            texture_format,
            texture_view,
            rectangles: Vec::new(),
        }
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.texture_view
    }

    pub fn texture_locations_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let atlas_size = self.allocator.size();
        let (atlas_width, atlas_height) = (atlas_size.width as f32, atlas_size.height as f32);

        let texture_locations: Vec<_> = self
            .rectangles
            .iter()
            .map(move |rectangle| {
                let top_left = rectangle.min;

                TextureLocation {
                    offset: [
                        top_left.x as f32 / atlas_width,
                        top_left.y as f32 / atlas_height,
                    ],
                    size: [
                        rectangle.width() as f32 / atlas_width,
                        rectangle.height() as f32 / atlas_height,
                    ],
                }
            })
            .collect();

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texture atlas texture locations buffer"),
            contents: bytemuck::cast_slice(&texture_locations),
            usage: wgpu::BufferUsage::STORAGE,
        })
    }

    pub fn allocate(
        &mut self,
        image: &image::RgbaImage,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> (usize, bool) {
        let (image_width, image_height) = image.dimensions();
        let border = size2(1, 1);
        let padded_size = size2(image_width as i32, image_height as i32) + border * 2;

        let mut rectangle = self.allocator.allocate(padded_size);
        let mut resized = false;
        let old_size = self.allocator.size();

        while rectangle.is_none() {
            self.allocator.grow(self.allocator.size() + padded_size);
            resized = true;
            rectangle = self.allocator.allocate(padded_size);
        }

        if resized {
            let (new_texture, new_texture_view) = atlas_texture_of_size(
                self.allocator.size(),
                device,
                self.mipmap_levels,
                self.texture_format,
            );

            encoder.copy_texture_to_texture(
                wgpu::TextureCopyView {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::TextureCopyView {
                    texture: &new_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::Extent3d {
                    width: old_size.width as u32,
                    height: old_size.height as u32,
                    depth: 1,
                },
            );

            self.texture = new_texture;
            self.texture_view = new_texture_view;
        }

        let mut rectangle = rectangle.unwrap().rectangle;
        rectangle.min += border;
        rectangle.max -= border;
        let index = self.rectangles.len();
        self.rectangles.push(rectangle);

        let top_left = rectangle.min;
        let top_left_x = top_left.x as u32;
        let top_left_y = top_left.y as u32;

        let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texture atlas staging buffer"),
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
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: top_left_x,
                    y: top_left_y,
                    z: 0,
                },
            },
            wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth: 1,
            },
        );

        /*
        // Copy the texture horizontally

        copy_region_to_location(
            top_left_x,
            top_left_y,
            1,
            image_height,
            top_left_x - 1,
            top_left_y,
            encoder,
            &self.texture,
        );

        copy_region_to_location(
            top_left_x + image_width - 1,
            top_left_y,
            1,
            image_height,
            top_left_x + image_width,
            top_left_y,
            encoder,
            &self.texture,
        );

        // and vertically

        copy_region_to_location(
            top_left_x - 1,
            top_left_y,
            image_width + 2,
            1,
            top_left_x - 1,
            top_left_y - 1,
            encoder,
            &self.texture,
        );

        copy_region_to_location(
            top_left_x - 1,
            top_left_y + image_height - 1,
            image_width + 2,
            1,
            top_left_x - 1,
            top_left_y + image_height,
            encoder,
            &self.texture,
        );
        */

        (index, resized)
    }

    pub fn generate_mipmaps(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: MipmapGenerationParams,
    ) {
        let mipmap_views: Vec<_> = (0..self.mipmap_levels)
            .map(|level| {
                self.texture.create_view(&wgpu::TextureViewDescriptor {
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

        for level in 1..self.mipmap_levels as usize {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: params.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&mipmap_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&params.sampler),
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
            render_pass.set_pipeline(&params.render_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

fn atlas_texture_of_size(
    size: Size,
    device: &wgpu::Device,
    mipmap_levels: u32,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture atlas"),
        size: wgpu::Extent3d {
            width: size.width as u32,
            height: size.height as u32,
            depth: 1,
        },
        mip_level_count: mipmap_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture_format,
        usage: wgpu::TextureUsage::SAMPLED
            | wgpu::TextureUsage::COPY_DST
            | wgpu::TextureUsage::RENDER_ATTACHMENT
            | wgpu::TextureUsage::COPY_SRC,
    });

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("texture atlas view"),
        format: Some(texture_format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        level_count: Some(std::num::NonZeroU32::new(mipmap_levels).unwrap()),
        ..Default::default()
    });

    (texture, texture_view)
}

pub struct MipmapGenerationParams<'a> {
    pub bind_group_layout: &'a wgpu::BindGroupLayout,
    pub render_pipeline: &'a wgpu::RenderPipeline,
    pub sampler: &'a wgpu::Sampler,
}

fn copy_region_to_location(
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    target_x: u32,
    target_y: u32,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) {
    encoder.copy_texture_to_texture(
        wgpu::TextureCopyView {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: origin_x,
                y: origin_y,
                z: 0,
            },
        },
        wgpu::TextureCopyView {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: target_x,
                y: target_y,
                z: 0,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth: 1,
        },
    );
}
