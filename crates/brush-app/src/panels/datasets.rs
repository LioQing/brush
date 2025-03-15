use crate::app::{AppContext, AppPanel};
use brush_process::process_loop::ProcessMessage;
use brush_train::scene::{Scene, SceneView, ViewImageType, ViewType};
use brush_ui::burn_texture::BurnTexture;
use egui::{Slider, TextureHandle, TextureOptions, pos2};

struct SelectedView {
    index: usize,
    view_type: ViewType,
    texture_handle: TextureHandle,
}

fn selected_scene(t: ViewType, context: &AppContext) -> &Scene {
    if let Some(eval_scene) = context.dataset.eval.as_ref() {
        match t {
            ViewType::Train => &context.dataset.train,
            _ => eval_scene,
        }
    } else {
        &context.dataset.train
    }
}

impl SelectedView {
    fn get_view<'a>(&'a self, context: &'a AppContext) -> &'a SceneView {
        &selected_scene(self.view_type, context).views[self.index]
    }
}

pub(crate) struct DatasetPanel {
    view_type: ViewType,
    selected_view: Option<SelectedView>,
    selected_depth_view: Option<SelectedView>,
}

impl DatasetPanel {
    pub(crate) fn new() -> Self {
        Self {
            view_type: ViewType::Train,
            selected_view: None,
            selected_depth_view: None,
        }
    }
}

impl AppPanel for DatasetPanel {
    fn title(&self) -> String {
        "Dataset".to_owned()
    }

    fn on_message(&mut self, message: &ProcessMessage, context: &mut AppContext) {
        match message {
            ProcessMessage::NewSource => {
                *self = Self::new();
            }
            ProcessMessage::Dataset { data: d } => {
                // Set train view to last loaded camera.
                if let Some(view) = d.train.views.last() {
                    context.focus_view(view);
                }
                context.dataset = d.clone();
            }
            _ => {}
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut AppContext) {
        let pick_scene = selected_scene(self.view_type, context).clone();

        let mut nearest_view_ind = pick_scene.get_nearest_view(context.camera.local_to_world());

        if let Some(nearest) = nearest_view_ind.as_mut() {
            // Update image if dirty.
            // For some reason (bug in egui), this _has_ to be before drawing the image.
            // Otherwise egui releases the image too early and wgpu crashes.
            let mut dirty = self.selected_view.is_none();

            if let Some(view) = self.selected_view.as_ref() {
                dirty |= view.index != *nearest;
                dirty |= view.view_type != self.view_type;
            }

            if dirty {
                let view = &pick_scene.views[*nearest];
                let image = &view.image;
                let img_size = [image.width() as usize, image.height() as usize];
                let color_img = if image.color().has_alpha() {
                    let data = image.to_rgba8().into_vec();
                    egui::ColorImage::from_rgba_unmultiplied(img_size, &data)
                } else {
                    egui::ColorImage::from_rgb(img_size, &image.to_rgb8().into_vec())
                };
                let depth_img = if let Some(depth) = view.depth.as_ref() {
                    let depth = depth.to_luma16().into_vec();
                    
                    let mut data = vec![0xff000000u32; img_size[0] * img_size[1]];
                    for (d, rgba) in depth.iter().zip(data.iter_mut()) {
                        let d = *d as f32 / 1000.0; // Convert from mm to m.
                        let d = d.clamp(
                            BurnTexture::DEPTH_GRADIENTS.first().unwrap().0,
                            BurnTexture::DEPTH_GRADIENTS.last().unwrap().0,
                        );
                        for i in 0..BurnTexture::DEPTH_GRADIENTS.len() - 1 {
                            let (d0, c0) = BurnTexture::DEPTH_GRADIENTS[i];
                            let (d1, c1) = BurnTexture::DEPTH_GRADIENTS[i + 1];
                            if d >= d0 && d < d1 {
                                let t = (d - d0) / (d1 - d0);
                                let c = c0.lerp(c1, t).as_uvec3();
                                *rgba = c.x + c.y * 0x0100 + c.z * 0x010000 + 0xff000000;
                                break;
                            }
                        }
                    }

                    let data = data.iter().map(|&x| x.to_le_bytes()).flatten().collect::<Vec<_>>();

                    Some(egui::ColorImage::from_rgba_unmultiplied(img_size, &data))
                } else {
                    None
                };

                self.selected_view = Some(SelectedView {
                    index: *nearest,
                    view_type: self.view_type,
                    texture_handle: ui.ctx().load_texture(
                        "nearest_view_tex",
                        color_img,
                        TextureOptions::default(),
                    ),
                });
                self.selected_depth_view = depth_img.map(|depth_img| SelectedView {
                    index: *nearest,
                    view_type: self.view_type,
                    texture_handle: ui.ctx().load_texture(
                        "nearest_depth_view_tex",
                        depth_img,
                        TextureOptions::default(),
                    ),
                });
            }

            let view_count = pick_scene.views.len();

            if let Some(selected) = self.selected_view.as_ref() {
                let selected_view = selected.get_view(context).clone();
                let texture_handle = &selected.texture_handle;

                let img_size = texture_handle.size();
                let size = brush_ui::size_for_splat_view(ui);

                let size = egui::Image::new(texture_handle).shrink_to_fit().calc_size(
                    size,
                    Some(egui::vec2(img_size[0] as f32, img_size[1] as f32)),
                );
                let min = ui.cursor().min;
                let size = size.round();

                let rect = egui::Rect::from_min_size(min, size);

                match selected_view.img_type {
                    ViewImageType::Alpha => {
                        brush_ui::draw_checkerboard(ui, rect, egui::Color32::WHITE);
                    }
                    ViewImageType::Masked => {
                        brush_ui::draw_checkerboard(ui, rect, egui::Color32::DARK_RED);
                    }
                }

                ui.allocate_rect(rect, egui::Sense::click());

                ui.painter().image(
                    texture_handle.id(),
                    rect,
                    egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                if let Some(selected_depth) = self.selected_depth_view.as_ref() {
                    let max_height = ui.available_rect_before_wrap().height() -
                        ui.text_style_height(&egui::TextStyle::Body) * 2.0;
                    let size = match size.y < max_height {
                        true => size,
                        false => {
                            let scale = max_height / size.y;
                            egui::vec2(size.x * scale, max_height)
                        }
                    };
                    let (depth_rect, _) = ui.allocate_exact_size(
                        egui::vec2(size.x, size.y),
                        egui::Sense::empty(),
                    );

                    ui.painter().image(
                        selected_depth.texture_handle.id(),
                        depth_rect,
                        egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }

                ui.horizontal(|ui| {
                    let mut interacted = false;
                    if ui.button("⏪").clicked() {
                        *nearest = (*nearest + view_count - 1) % view_count;
                        interacted = true;
                    }
                    if ui
                        .add(
                            Slider::new(nearest, 0..=view_count - 1)
                                .suffix(format!("/ {view_count}"))
                                .custom_formatter(|num, _| format!("{}", num as usize + 1))
                                .custom_parser(|s| s.parse::<usize>().ok().map(|n| n as f64 - 1.0)),
                        )
                        .dragged()
                    {
                        interacted = true;
                    }
                    if ui.button("⏩").clicked() {
                        *nearest = (*nearest + 1) % view_count;
                        interacted = true;
                    }

                    ui.add_space(10.0);

                    if context.dataset.eval.is_some() {
                        for (t, l) in [ViewType::Train, ViewType::Eval]
                            .into_iter()
                            .zip(["train", "eval"])
                        {
                            if ui.selectable_label(self.view_type == t, l).clicked() {
                                self.view_type = t;
                                *nearest = 0;
                                interacted = true;
                            };
                        }
                    }

                    if interacted {
                        context.focus_view(&pick_scene.views[*nearest]);
                    }

                    ui.add_space(10.0);

                    let mask_info = if selected_view.image.color().has_alpha() {
                        if selected_view.img_type == ViewImageType::Alpha {
                            "rgb + alpha transparency"
                        } else {
                            "rgb, masked"
                        }
                    } else {
                        "rgb"
                    };

                    let info = format!(
                        "{} ({}x{} {})",
                        selected_view.path,
                        selected_view.image.width(),
                        selected_view.image.height(),
                        mask_info
                    );
                    ui.label(info);
                });
            }
        }

        if context.loading() && context.training() {
            ui.label("Loading...");
        }
    }

    fn inner_margin(&self) -> f32 {
        0.0
    }
}
