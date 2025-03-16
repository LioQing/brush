use brush_dataset::splat_export;
use brush_process::process_loop::{ControlMessage, ProcessMessage};
use brush_train::{scene::ViewImageType, train::{sobel, TrainBack, SOBEL_SCALE}};
use brush_ui::burn_texture::BurnTexture;
use burn::tensor::backend::AutodiffBackend;
use core::f32;
use egui::epaint::mutex::RwLock as EguiRwLock;
use std::sync::Arc;

use brush_render::{
    camera::{focal_to_fov, fov_to_focal},
    gaussian_splats::Splats,
};
use eframe::egui_wgpu::Renderer;
use egui::{Color32, Rect};
use glam::{Quat, UVec2, Vec3};
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;
use web_time::Instant;

use crate::app::{AppContext, AppPanel};

#[derive(Debug, Clone, Copy, PartialEq)]
struct RenderState {
    size: UVec2,
    cam_pos: Vec3,
    cam_rot: Quat,

    frame: f32,
}

struct ErrorDisplay {
    headline: String,
    context: Vec<String>,
}

pub(crate) struct ScenePanel {
    pub(crate) backbuffer: BurnTexture,
    pub(crate) depth_backbuffer: BurnTexture,
    pub(crate) sobel_backbuffer: BurnTexture,
    pub(crate) last_draw: Option<Instant>,

    view_splats: Vec<Splats<<TrainBack as AutodiffBackend>::InnerBackend>>,
    frame_count: u32,
    frame: f32,

    // Ui state.
    live_update: bool,
    paused: bool,
    err: Option<ErrorDisplay>,
    zen: bool,

    // Keep track of what was last rendered.
    last_state: Option<RenderState>,
}

impl ScenePanel {
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: Arc<EguiRwLock<Renderer>>,
        zen: bool,
    ) -> Self {
        Self {
            backbuffer: BurnTexture::new(renderer.clone(), device.clone(), queue.clone()),
            depth_backbuffer: BurnTexture::new(renderer.clone(), device.clone(), queue.clone()),
            sobel_backbuffer: BurnTexture::new(renderer, device, queue),
            last_draw: None,
            err: None,
            view_splats: vec![],
            live_update: true,
            paused: false,
            last_state: None,
            zen,
            frame_count: 0,
            frame: 0.0,
        }
    }

    pub(crate) fn draw_splats(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut AppContext,
        splats: &Splats<<TrainBack as AutodiffBackend>::InnerBackend>,
    ) {
        let size = brush_ui::size_for_splat_view(ui);

        if size.x < 8.0 || size.y < 8.0 {
            return;
        }

        let mut size = size.floor();

        if let Some(aspect_ratio) = context.view_aspect {
            if size.x / size.y > aspect_ratio {
                size.x = size.y * aspect_ratio;
            } else {
                size.y = size.x / aspect_ratio;
            }
        } else {
            let focal_y = fov_to_focal(context.camera.fov_y, size.y as u32) as f32;
            context.camera.fov_x = focal_to_fov(focal_y as f64, size.x as u32);
        }
        let size = glam::uvec2(size.x.round() as u32, size.y.round() as u32);

        let (rect, response) = ui.allocate_exact_size(
            egui::Vec2::new(size.x as f32, size.y as f32),
            egui::Sense::drag(),
        );

        context.controls.tick(&response, ui);

        let camera = &mut context.camera;

        // Create a camera that incorporates the model transform.
        let total_transform = context.model_local_to_world * context.controls.local_to_world();

        camera.position = total_transform.translation.into();
        camera.rotation = Quat::from_mat3a(&total_transform.matrix3);

        let state = RenderState {
            size,
            cam_pos: camera.position,
            cam_rot: camera.rotation,
            frame: self.frame,
        };

        let dirty = self.last_state != Some(state);

        if dirty {
            self.last_state = Some(state);

            // Check again next frame, as there might be more to animate.
            ui.ctx().request_repaint();
        }

        // If this viewport is re-rendering.
        if size.x > 0 && size.y > 0 && dirty {
            let _span = trace_span!("Render splats").entered();
            let (img, depth, _) = splats.render(&context.camera, size, false);
            self.backbuffer.update_texture(img);
            self.depth_backbuffer.update_depth_texture(depth.clone());
            self.sobel_backbuffer.update_sobel_texture(sobel(depth), SOBEL_SCALE);
        }

        if let Some(id) = self.backbuffer.id() {
            ui.scope(|ui| {
                let mut background = false;
                if let Some(view) = context.dataset.train.views.first() {
                    if view.image.color().has_alpha() && view.img_type == ViewImageType::Alpha {
                        background = true;
                        // if training views have alpha, show a background checker. Masked images
                        // should still use a black background.
                        brush_ui::draw_checkerboard(ui, rect, Color32::WHITE);
                    }
                }

                // If a scene is opaque, it assumes a black background.
                if !background {
                    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                }

                ui.painter().image(
                    id,
                    rect,
                    Rect {
                        min: egui::pos2(0.0, 0.0),
                        max: egui::pos2(1.0, 1.0),
                    },
                    Color32::WHITE,
                );
            });
        }

        let depth_or_sobel = self.sobel_backbuffer.id().is_some() || self.depth_backbuffer.id().is_some();
        let depth_and_sobel = self.sobel_backbuffer.id().is_some() && self.depth_backbuffer.id().is_some();
        let depth_xor_sobel = depth_or_sobel && !depth_and_sobel;

        if depth_or_sobel {
            ui.with_layout(
            if depth_xor_sobel {
                egui::Layout::right_to_left(egui::Align::TOP)
            } else {
                egui::Layout::left_to_right(egui::Align::TOP)
            },
            |ui| {
                let max_height = ui.available_rect_before_wrap().height() -
                    ui.text_style_height(&egui::TextStyle::Body) * 2.0;
                let size = match (size.y as f32) < max_height {
                    true => size.as_vec2(),
                    false => {
                        let scale = max_height / size.y as f32;
                        glam::vec2(size.x as f32 * scale, max_height)
                    }
                };

                let max_width = ui.available_rect_before_wrap().width();
                let size = if depth_and_sobel && size.x > max_width / 2.0 {
                    let scale = max_width / 2.0 / size.x;
                    glam::vec2(max_width / 2.0, size.y * scale)
                } else {
                    size
                };

                if let Some(id) = self.depth_backbuffer.id() {
                    let (depth_rect, _) = ui.allocate_exact_size(
                        egui::Vec2::new(size.x, size.y),
                        egui::Sense::empty(),
                    );

                    ui.painter().image(
                        id,
                        depth_rect,
                        Rect {
                            min: egui::pos2(0.0, 0.0),
                            max: egui::pos2(1.0, 1.0),
                        },
                        Color32::WHITE,
                    );
                }

                if let Some(id) = self.sobel_backbuffer.id() {
                    let (sobel_rect, _) = ui.allocate_exact_size(
                        egui::Vec2::new(size.x, size.y),
                        egui::Sense::empty(),
                    );

                    ui.painter().image(
                        id,
                        sobel_rect,
                        Rect {
                            min: egui::pos2(0.0, 0.0),
                            max: egui::pos2(1.0, 1.0),
                        },
                        Color32::WHITE,
                    );
                }
            });
        }
    }
}

impl AppPanel for ScenePanel {
    fn title(&self) -> String {
        "Scene".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, context: &mut AppContext) {
        match message {
            ProcessMessage::NewSource => {
                self.view_splats = vec![];
                self.frame_count = 0;
                self.live_update = true;
                self.paused = false;
                self.err = None;
                self.last_state = None;
                self.frame = 0.0;
            }
            ProcessMessage::ViewSplats {
                up_axis,
                splats,
                frame,
                total_frames,
            } => {
                if let Some(up_axis) = up_axis {
                    context.set_model_up(*up_axis);
                }

                if self.live_update {
                    self.view_splats.truncate(*frame as usize);
                    self.view_splats.push(*splats.clone());
                }
                self.frame_count = *total_frames;
                self.last_state = None;
            }
            ProcessMessage::TrainStep {
                splats,
                stats: _,
                iter: _,
                timestamp: _,
            } => {
                self.last_state = None;

                let splats = *splats.clone();

                if self.live_update {
                    self.view_splats = vec![splats];
                }
            }
            ProcessMessage::Error(e) => {
                let headline = e.to_string();
                let context = e.chain().skip(1).map(|cause| format!("{cause}")).collect();
                self.err = Some(ErrorDisplay { headline, context });
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        let cur_time = Instant::now();

        self.last_draw = Some(cur_time);

        // Empty scene, nothing to show.
        if !context.training() && self.view_splats.is_empty() && self.err.is_none() && !self.zen {
            ui.heading("Load a ply file or dataset to get started.");
            ui.add_space(5.0);
            ui.label(
                r#"
Load a pretrained .ply file to view it

Or load a dataset to train on. These are zip files with:
    - a transforms.json and images, like the nerfstudio dataset format.
    - COLMAP data, containing the `images` & `sparse` folder."#,
            );

            ui.add_space(10.0);

            if cfg!(debug_assertions) {
                ui.scope(|ui| {
                    ui.visuals_mut().override_text_color = Some(Color32::LIGHT_BLUE);
                    ui.heading(
                        "Note: running in debug mode, compile with --release for best performance",
                    );
                });

                ui.add_space(10.0);
            }

            #[cfg(target_family = "wasm")]
            ui.scope(|ui| {
                ui.visuals_mut().override_text_color = Some(Color32::YELLOW);
                ui.heading("Note: Running in browser is still experimental");

                ui.label(
                    r#"
In browser training is slower, and lower quality than the native app.

For bigger training runs consider using the native app."#,
                );
            });

            return;
        }

        if let Some(err) = self.err.as_ref() {
            ui.heading(format!("❌ {}", err.headline));

            ui.indent("err_context", |ui| {
                for c in &err.context {
                    ui.label(format!("• {c}"));
                    ui.add_space(2.0);
                }
            });
        } else if !self.view_splats.is_empty() {
            const FPS: f32 = 24.0;

            if !self.paused {
                self.frame += ui.input(|r| r.predicted_dt);
            }
            if self.view_splats.len() as u32 != self.frame_count {
                let max_t = (self.view_splats.len() - 1) as f32 / FPS;
                self.frame = self.frame.min(max_t);
            }

            let frame = (self.frame * FPS)
                .rem_euclid(self.frame_count as f32)
                .floor() as usize;
            let splats = self.view_splats[frame].clone();

            self.draw_splats(ui, context, &splats);

            if self.view_splats.len() > 1 && self.view_splats.len() as u32 == self.frame_count {
                let label = if self.paused {
                    "⏸ paused"
                } else {
                    "⏵ playing"
                };

                if ui.selectable_label(!self.paused, label).clicked() {
                    self.paused = !self.paused;
                }
            }

            ui.horizontal(|ui| {
                if context.loading() {
                    ui.horizontal(|ui| {
                        ui.label("Loading... Please wait.");
                        ui.spinner();
                    });
                }

                if context.training() {
                    ui.add_space(15.0);

                    let label = if self.paused {
                        "⏸ paused"
                    } else {
                        "⏵ training"
                    };

                    if ui.selectable_label(!self.paused, label).clicked() {
                        self.paused = !self.paused;
                        context.control_message(ControlMessage::Paused(self.paused));
                    }

                    ui.add_space(15.0);

                    ui.scope(|ui| {
                        ui.style_mut().visuals.selection.bg_fill = Color32::DARK_RED;
                        if ui
                            .selectable_label(self.live_update, "🔴 Live update splats")
                            .clicked()
                        {
                            self.live_update = !self.live_update;
                        }
                    });

                    ui.add_space(15.0);

                    if ui.button("⬆ Export").clicked() {
                        let splats = splats.clone();

                        let fut = async move {
                            let file = rrfd::save_file("export.ply").await;

                            // Not sure where/how to show this error if any.
                            match file {
                                Err(e) => {
                                    log::error!("Failed to save file: {e}");
                                }
                                Ok(file) => {
                                    let data = splat_export::splat_to_ply(splats).await;

                                    let data = match data {
                                        Ok(data) => data,
                                        Err(e) => {
                                            log::error!("Failed to serialize file: {e}");
                                            return;
                                        }
                                    };

                                    if let Err(e) = file.write(&data).await {
                                        log::error!("Failed to write file: {e}");
                                    }
                                }
                            }
                        };

                        tokio_wasm::task::spawn(fut);
                    }
                }

                ui.selectable_label(false, "Controls")
                    .on_hover_ui_at_pointer(|ui| {
                        ui.heading("Controls");

                        ui.label("• Left click and drag to orbit");
                        ui.label(
                            "• Right click, or left click + spacebar, and drag to look around.",
                        );
                        ui.label("• Middle click, or left click + control, and drag to pan");
                        ui.label("• Scroll to zoom");
                        ui.label("• WASD to fly, Q&E to move up & down.");
                        ui.label("• Z&C to roll, X to reset roll");
                        ui.label("• Shift to move faster");
                    });
            });
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
