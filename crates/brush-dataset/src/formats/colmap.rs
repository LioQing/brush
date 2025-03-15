use std::{
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::DataStream;
use crate::{
    brush_vfs::BrushVfs, formats::{clamp_img_to_max_size, find_depth_path, find_mask_path, load_depth, load_image}, splat_import::SplatMessage, stream_fut_parallel, Dataset, LoadDataseConfig
};
use anyhow::{Context, Result};
use async_fn_stream::try_fn_stream;
use brush_render::{
    camera::{self, Camera},
    gaussian_splats::Splats,
    sh::rgb_to_sh,
};
use brush_train::scene::SceneView;
use burn::prelude::Backend;
use glam::Vec3;
use std::collections::HashMap;
use tokio_stream::StreamExt;

fn find_base_path(archive: &BrushVfs, search_path: &str) -> Option<PathBuf> {
    for path in archive.file_names() {
        if let Some(str) = path.to_str() {
            if str.to_lowercase().ends_with(search_path) {
                return path
                    .ancestors()
                    .nth(Path::new(search_path).components().count())
                    .map(|x| x.to_owned());
            }
        }
    }
    None
}

fn find_img_mask_depth(vfs: &BrushVfs, paths: &[PathBuf]) -> Result<(PathBuf, Option<PathBuf>, Option<PathBuf>)> {
    let mut path_masks = HashMap::new();
    let mut masks = vec![];
    let mut depths = vec![];

    // First pass: collect images & masks.
    for path in paths {
        let mask = find_mask_path(vfs, path);
        let depth = find_depth_path(vfs, path);
        path_masks.insert(path.clone(), (mask.clone(), depth.clone()));
        if let Some(mask_path) = mask {
            masks.push(mask_path);
        }
        if let Some(depth_path) = depth {
            depths.push(depth_path);
        }
    }

    // Remove masks from candidates - shouldn't count as an input image.
    for mask in masks {
        path_masks.remove(&mask);
    }

    // Remove depths from candidates - shouldn't count as an input image.
    for depth in depths {
        path_masks.remove(&depth);
    }

    // Sort and return the first candidate (alphabetically).
    path_masks
        .into_iter()
        .min_by_key(|kv| kv.0.clone())
        .map(|(img, (mask, depth))| (img, mask, depth))
        .context("No candidates found")
}

async fn read_views(
    vfs: BrushVfs,
    load_args: LoadDataseConfig,
) -> Result<Vec<impl Future<Output = Result<SceneView>>>> {
    log::info!("Loading colmap dataset");
    let mut vfs = vfs;

    let (is_binary, base_path) = if let Some(path) = find_base_path(&vfs, "cameras.bin") {
        (true, path)
    } else if let Some(path) = find_base_path(&vfs, "cameras.txt") {
        (false, path)
    } else {
        anyhow::bail!("No COLMAP data found (either text or binary.)")
    };

    let (cam_path, img_path) = if is_binary {
        (base_path.join("cameras.bin"), base_path.join("images.bin"))
    } else {
        (base_path.join("cameras.txt"), base_path.join("images.txt"))
    };

    let cam_model_data = {
        let mut cam_file = vfs.open_path(&cam_path).await?;
        colmap_reader::read_cameras(&mut cam_file, is_binary).await?
    };

    let img_infos = {
        let img_file = vfs.open_path(&img_path).await?;
        let mut buf_reader = tokio::io::BufReader::new(img_file);
        colmap_reader::read_images(&mut buf_reader, is_binary).await?
    };

    let mut img_info_list = img_infos.into_iter().collect::<Vec<_>>();

    log::info!("Loading colmap dataset with {} images", img_info_list.len());

    // Sort by image name. This is important to match the exact eval images mipnerf uses.
    img_info_list.sort_by_key(|key_img| key_img.1.name.clone());

    let handles = img_info_list
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .map(move |(_, img_info)| {
            let cam_data = cam_model_data[&img_info.camera_id].clone();
            let mut vfs = vfs.clone();

            // Create a future to handle loading the image.
            async move {
                let focal = cam_data.focal();

                let fovx = camera::focal_to_fov(focal.0, cam_data.width as u32);
                let fovy = camera::focal_to_fov(focal.1, cam_data.height as u32);

                let center = cam_data.principal_point();
                let center_uv = center / glam::vec2(cam_data.width as f32, cam_data.height as f32);

                // Colmap only specifies an image name, not a full path. We brute force
                // search for the image in the archive.
                let img_paths: Vec<_> = vfs
                    .file_names()
                    .filter(|p| p.ends_with(&img_info.name))
                    .collect();

                let (path, mask_path, depth_path) = find_img_mask_depth(&vfs, &img_paths)
                    .with_context(|| format!("Failed to find image {}", img_info.name))?;

                let (image, img_type) = load_image(&mut vfs, &path, mask_path.as_deref())
                    .await
                    .with_context(|| format!("Failed to load image {}", img_info.name))?;

                let depth = if let Some(depth_path) = depth_path.as_ref() {
                    Some(
                        load_depth(&mut vfs, depth_path)
                            .await
                            .with_context(|| format!(
                                "Failed to load depth for image {}", img_info.name
                            ))?
                    )
                } else { None };

                let image = clamp_img_to_max_size(Arc::new(image), load_args.max_resolution);
                let depth = depth.map(|d| clamp_img_to_max_size(Arc::new(d), load_args.max_resolution));

                // Convert w2c to c2w.
                let world_to_cam =
                    glam::Affine3A::from_rotation_translation(img_info.quat, img_info.tvec);
                let cam_to_world = world_to_cam.inverse();
                let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

                let camera = Camera::new(translation, quat, fovx, fovy, center_uv);

                let view = SceneView {
                    path: path.to_string_lossy().to_string(),
                    camera,
                    image,
                    img_type,
                    depth,
                };
                Ok(view)
            }
        })
        .collect();

    Ok(handles)
}

pub(crate) async fn load_dataset<B: Backend>(
    mut vfs: BrushVfs,
    load_args: &LoadDataseConfig,
    device: &B::Device,
) -> Result<(DataStream<SplatMessage<B>>, DataStream<Dataset>)> {
    let mut handles = read_views(vfs.clone(), load_args.clone()).await?;

    if let Some(subsample) = load_args.subsample_frames {
        handles = handles.into_iter().step_by(subsample as usize).collect();
    }

    let mut train_views = vec![];
    let mut eval_views = vec![];

    let load_args = load_args.clone();
    let device = device.clone();

    let mut i = 0;
    let stream = stream_fut_parallel(handles).map(move |view| {
        let view = view.context("Failed to load COLMAP view")?;

        if let Some(eval_period) = load_args.eval_split_every {
            if i % eval_period == 0 {
                eval_views.push(view);
            } else {
                train_views.push(view);
            }
        } else {
            train_views.push(view);
        }

        i += 1;
        Ok(Dataset::from_views(train_views.clone(), eval_views.clone()))
    });

    let init_stream = try_fn_stream(|emitter| async move {
        let points_path = vfs.file_names().find(|p| {
            if let Some(path) = p.to_str().map(|p| p.to_lowercase()) {
                path.ends_with("points3d.txt") || path.ends_with("points3d.bin")
            } else {
                false
            }
        });

        let Some(points_path) = points_path else {
            return Ok(());
        };

        let is_binary = matches!(
            points_path.extension().and_then(|p| p.to_str()),
            Some("bin")
        );

        // Extract COLMAP sfm points.
        let points_data = {
            let mut points_file = vfs
                .open_path(&points_path)
                .await
                .context("Failed to read COLMAP points file")?;
            colmap_reader::read_points3d(&mut points_file, is_binary).await
        };

        // Ignore empty points data.
        if let Ok(points_data) = points_data {
            if !points_data.is_empty() {
                log::info!("Starting from colmap points {}", points_data.len());

                let mut positions: Vec<Vec3> = points_data.values().map(|p| p.xyz).collect();
                let mut colors: Vec<f32> = points_data
                    .values()
                    .flat_map(|p| {
                        let sh = rgb_to_sh(glam::vec3(
                            p.rgb[0] as f32 / 255.0,
                            p.rgb[1] as f32 / 255.0,
                            p.rgb[2] as f32 / 255.0,
                        ));
                        [sh.x, sh.y, sh.z]
                    })
                    .collect();

                // Other dataloaders handle subsampling in the ply import. Here just
                // do it manually, maybe nice to unify at some point.
                if let Some(subsample) = load_args.subsample_points {
                    positions = positions.into_iter().step_by(subsample as usize).collect();
                    colors = colors.into_iter().step_by(subsample as usize * 3).collect();
                }

                let init_splat =
                    Splats::from_raw(&positions, None, None, Some(&colors), None, &device);
                emitter
                    .emit(SplatMessage {
                        meta: crate::splat_import::ParseMetadata {
                            up_axis: None,
                            total_splats: init_splat.num_splats(),
                            frame_count: 1,
                            current_frame: 0,
                        },
                        splats: init_splat,
                    })
                    .await;
            }
        }

        Ok(())
    });

    Ok((Box::pin(init_stream), Box::pin(stream)))
}
