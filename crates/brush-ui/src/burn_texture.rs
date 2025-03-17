use std::sync::Arc;

use brush_render::{BBase, BFused};
use burn::tensor::{Float, Int, Tensor};
use burn_cubecl::BoolElement;
use burn_fusion::client::FusionClient;
use eframe::egui_wgpu::Renderer;
use egui::TextureId;
use egui::epaint::mutex::RwLock as EguiRwLock;
use wgpu::{CommandEncoderDescriptor, TexelCopyBufferLayout, TextureViewDescriptor};

struct TextureState {
    texture: wgpu::Texture,
    id: TextureId,
}

pub struct BurnTexture {
    state: Option<TextureState>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Arc<EguiRwLock<Renderer>>,
}

fn create_texture(size: glam::UVec2, device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Splat backbuffer"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    })
}

impl BurnTexture {
    pub const DEPTH_GRADIENTS: &'static [(f32, glam::Vec3)] = &[
        (0.0, glam::vec3(192.0, 192.0, 192.0)),
        (1.0, glam::vec3(0.0, 0.0, 192.0)),
        (2.0, glam::vec3(0.0, 192.0, 192.0)),
        (4.0, glam::vec3(0.0, 192.0, 0.0)),
        (8.0, glam::vec3(192.0, 192.0, 0.0)),
        (16.0, glam::vec3(192.0, 0.0, 0.0)),
        (32.0, glam::vec3(0.0, 0.0, 0.0)),
    ];

    pub fn new(
        renderer: Arc<EguiRwLock<Renderer>>,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self {
            state: None,
            device,
            queue,
            renderer,
        }
    }

    pub fn update_depth_texture<BT: BoolElement>(&mut self, depth: Tensor<BFused<BT>, 2>) -> TextureId {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("depth viewer encoder"),
            });

        let [h, w] = depth.shape().dims();
        let size = glam::uvec2(w as u32, h as u32);

        let dirty = if let Some(s) = self.state.as_ref() {
            s.texture.width() != size.x || s.texture.height() != size.y
        } else {
            true
        };

        if dirty {
            let texture = create_texture(glam::uvec2(w as u32, h as u32), &self.device);

            if let Some(s) = self.state.as_mut() {
                s.texture = texture;

                self.renderer.write().update_egui_texture_from_wgpu_texture(
                    &self.device,
                    &s.texture.create_view(&TextureViewDescriptor::default()),
                    wgpu::FilterMode::Linear,
                    s.id,
                );
            } else {
                let id = self.renderer.write().register_native_texture(
                    &self.device,
                    &texture.create_view(&TextureViewDescriptor::default()),
                    wgpu::FilterMode::Linear,
                );
                self.state = Some(TextureState { texture, id });
            }
        }

        let Some(s) = self.state.as_ref() else {
            unreachable!("Somehow failed to initialize")
        };
        let texture: &wgpu::Texture = &s.texture;

        let [height, width] = depth.dims();

        let channels = 1;
        let padded_shape = vec![height, width.div_ceil(64) * 64, channels];

        let depth = Tensor::<BBase<BT>, 3, Float>::from_data(
            depth
                .clone()
                .clamp(Self::DEPTH_GRADIENTS.first().unwrap().0, Self::DEPTH_GRADIENTS.last().unwrap().0)
                .reshape([0, 0, 1])
                .into_data(),
            &depth.device(),
        );

        let alpha = i32::from_ne_bytes(0xff000000u32.to_ne_bytes());
        let mut rgba = Tensor::<BBase<BT>, 3, Int>::zeros(depth.shape(), &depth.device());
        rgba = rgba + alpha;
        for i in 0..Self::DEPTH_GRADIENTS.len() - 1 {
            let (d0, c0) = Self::DEPTH_GRADIENTS[i];
            let (d1, c1) = Self::DEPTH_GRADIENTS[i + 1];
            let mask = depth.clone().greater_equal_elem(d0).bool_and(depth.clone().lower_elem(d1));
            let t = (depth.clone() - d0) / (d1 - d0);
            let r = Tensor::from_data((t.clone() * (c1.x - c0.x) + c0.x).into_data(), &depth.device());
            let g = Tensor::from_data((t.clone() * (c1.y - c0.y) + c0.y).into_data(), &depth.device());
            let b = Tensor::from_data((t.clone() * (c1.z - c0.z) + c0.z).into_data(), &depth.device());
            rgba = rgba.mask_where(
                mask,
                r + g * 0x0100 + b * 0x010000 + alpha,
            );
        }

        // Create padded tensor if needed. The bytes_per_row needs to be divisible
        // by 256 in WebGPU, so 4 bytes per pixel means width needs to be divisible by 64.
        let rgba = if width % 64 != 0 {
            let padded: Tensor<BBase<BT>, 3, Int> = Tensor::zeros(&padded_shape, &rgba.device());
            padded.slice_assign([0..height, 0..width], rgba)
        } else {
            rgba
        };

        let rgba = rgba.into_primitive();
        
        // Get a hold of the Burn resource.
        let client = &rgba.client;
        let img_res_handle = client.get_resource(rgba.handle.clone().binding());

        // Now flush commands to make sure the resource is fully ready.
        client.flush();

        // Put compute passes in encoder before copying the buffer.
        let bytes_per_row = Some(4 * padded_shape[1] as u32);

        // Now copy the buffer to the texture.
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &img_res_handle.resource().buffer,
                layout: TexelCopyBufferLayout {
                    offset: img_res_handle.resource().offset(),
                    bytes_per_row,
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);

        s.id
    }

    pub fn update_texture<BT: BoolElement>(&mut self, img: Tensor<BFused<BT>, 3>) -> TextureId {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("viewer encoder"),
            });

        let [h, w, _] = img.shape().dims();
        let size = glam::uvec2(w as u32, h as u32);

        let dirty = if let Some(s) = self.state.as_ref() {
            s.texture.width() != size.x || s.texture.height() != size.y
        } else {
            true
        };

        if dirty {
            let texture = create_texture(glam::uvec2(w as u32, h as u32), &self.device);

            if let Some(s) = self.state.as_mut() {
                s.texture = texture;

                self.renderer.write().update_egui_texture_from_wgpu_texture(
                    &self.device,
                    &s.texture.create_view(&TextureViewDescriptor::default()),
                    wgpu::FilterMode::Linear,
                    s.id,
                );
            } else {
                let id = self.renderer.write().register_native_texture(
                    &self.device,
                    &texture.create_view(&TextureViewDescriptor::default()),
                    wgpu::FilterMode::Linear,
                );
                self.state = Some(TextureState { texture, id });
            }
        }

        let Some(s) = self.state.as_ref() else {
            unreachable!("Somehow failed to initialize")
        };
        let texture: &wgpu::Texture = &s.texture;

        let [height, width, c] = img.dims();

        let padded_shape = vec![height, width.div_ceil(64) * 64, c];

        let img_prim = img.into_primitive().tensor();
        let fusion_client = img_prim.client.clone();
        let img = fusion_client.resolve_tensor_int::<BBase<BT>>(img_prim);
        let img: Tensor<BBase<BT>, 3, Int> = Tensor::from_primitive(img);

        // Create padded tensor if needed. The bytes_per_row needs to be divisible
        // by 256 in WebGPU, so 4 bytes per pixel means width needs to be divisible by 64.
        let img = if width % 64 != 0 {
            let padded: Tensor<BBase<BT>, 3, Int> = Tensor::zeros(&padded_shape, &img.device());
            padded.slice_assign([0..height, 0..width], img)
        } else {
            img
        };

        let img = img.into_primitive();

        // Get a hold of the Burn resource.
        let client = &img.client;
        let img_res_handle = client.get_resource(img.handle.clone().binding());

        // Now flush commands to make sure the resource is fully ready.
        client.flush();

        // Put compute passes in encoder before copying the buffer.
        let bytes_per_row = Some(4 * padded_shape[1] as u32);

        // Now copy the buffer to the texture.
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &img_res_handle.resource().buffer,
                layout: TexelCopyBufferLayout {
                    offset: img_res_handle.resource().offset(),
                    bytes_per_row,
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);

        s.id
    }

    pub fn id(&self) -> Option<TextureId> {
        self.state.as_ref().map(|s| s.id)
    }
}
