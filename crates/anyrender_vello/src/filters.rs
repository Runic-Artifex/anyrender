//! CSS `filter` and `backdrop-filter` for the Vello renderer.
//!
//! Vello draws no filters, so a filtered layer is drawn in steps: its contents are
//! rendered by Vello into an offscreen texture, filtered there by the render passes in
//! `filters.wgsl`, and the result is registered with Vello as an image that the parent
//! scene draws through the layer's clip, blend mode and opacity. A backdrop filter
//! renders everything painted before the layer the same way and draws the filtered
//! backdrop behind the layer.
//!
//! The semantics are those of `anyrender_vello_cpu`'s filters and of browsers: the
//! colour functions run on the straight (unpremultiplied) sRGB colour and clamp after
//! each function, in float; blurs and drop shadows use the CSS length as the standard
//! deviation, scaled to device pixels by the layer transform; drop-shadow offsets are
//! rounded to whole device pixels, as `vello_cpu` does.

use anyrender::{
    Filter,
    filters::{EdgeMode, FilterEffect, FilterId, FilterInput, FilterSource},
};
use kurbo::{Affine, Rect};
use peniko::{
    ImageData,
    color::{AlphaColor, Srgb},
};
use vello::{AaConfig, RenderParams, Renderer as VelloRenderer, Scene};
use wgpu::util::DeviceExt;
use wgpu_context::DeviceHandle;

/// The most matrices one pass applies; longer runs take several passes (the
/// intermediate target is float, so splitting a run changes nothing).
const MAX_MATRICES: usize = 8;

/// One step of a filter chain, in device pixels.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Step {
    /// Consecutive colour matrices (row-major 4x5, on straight colour), each clamped.
    Matrices(Vec<[f32; 20]>),
    /// A Gaussian blur, transparent beyond the filter region.
    Blur { std_deviation: f32 },
    /// The input over its blurred, offset and coloured alpha.
    DropShadow {
        dx: i32,
        dy: i32,
        std_deviation: f32,
        color: AlphaColor<Srgb>,
    },
}

/// A filter graph the Vello renderer can draw: a linear chain of the primitives the CSS
/// filter functions lower to.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Chain {
    pub(crate) steps: Vec<Step>,
}

/// The blur kernel radius in texels for a standard deviation.
fn radius(std_deviation: f32) -> u32 {
    (3.0 * std_deviation).ceil().max(0.0) as u32
}

/// The uniform scale of a transform: the mean of its singular values (as
/// `vello_common::util::extract_scales`, which vello_cpu scales blurs with).
fn uniform_scale(transform: &Affine) -> f32 {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let s1 = a * a + b * b + c * c + d * d;
    let s2 = ((a * a - b * b + c * c - d * d).powi(2) + 4.0 * (a * b + c * d).powi(2)).sqrt();
    let scale_x = (0.5 * (s1 + s2)).sqrt().max(1e-6);
    let scale_y = (0.5 * (s1 - s2)).max(0.0).sqrt().max(1e-6);
    ((scale_x + scale_y) / 2.0) as f32
}

impl Chain {
    /// Convert a filter graph for a layer drawn with `transform`.
    ///
    /// Returns `None` for an empty graph, a graph whose primitives do not each read the
    /// previous result (as `Filter::linear_list` builds them for a CSS filter list), or one
    /// with a primitive this renderer does not implement; the layer is then drawn
    /// unfiltered, as `anyrender_vello_cpu` does.
    pub(crate) fn new(filter: &Filter, transform: &Affine) -> Option<Self> {
        let nodes = filter.nodes();
        if nodes.is_empty() || usize::from(filter.output().0) != nodes.len() - 1 {
            return None;
        }
        let scale = uniform_scale(transform);
        let [a, b, c, d, _, _] = transform.as_coeffs();
        let mut steps: Vec<Step> = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            let reads_previous = match &node.inputs.primary {
                None => true,
                Some(FilterInput::Result(FilterId(id))) => usize::from(*id) + 1 == index,
                Some(FilterInput::Source(source)) => {
                    index == 0 && *source == FilterSource::SourceGraphic
                }
            };
            if !reads_previous || node.inputs.secondary.is_some() {
                return None;
            }
            match &node.effect {
                FilterEffect::ColorMatrix(matrix) => {
                    if let Some(Step::Matrices(run)) = steps.last_mut() {
                        run.push(matrix.0);
                    } else {
                        steps.push(Step::Matrices(vec![matrix.0]));
                    }
                }
                FilterEffect::GaussianBlur(blur) if blur.edge_mode == EdgeMode::None => {
                    steps.push(Step::Blur {
                        std_deviation: blur.std_deviation.max(0.0) * scale,
                    });
                }
                FilterEffect::DropShadow(shadow) if shadow.edge_mode == EdgeMode::None => {
                    let (dx, dy) = (f64::from(shadow.dx), f64::from(shadow.dy));
                    steps.push(Step::DropShadow {
                        dx: (a * dx + c * dy).round() as i32,
                        dy: (b * dx + d * dy).round() as i32,
                        std_deviation: shadow.std_deviation.max(0.0) * scale,
                        color: shadow.color,
                    });
                }
                _ => return None,
            }
        }
        Some(Self { steps })
    }

    /// Whether the chain leaves every pixel as it is (`blur(0)`, `brightness(1)`, …).
    pub(crate) fn is_identity(&self) -> bool {
        const IDENTITY: [f32; 20] = [
            1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
        ];
        self.steps.iter().all(|step| match step {
            Step::Matrices(run) => run.iter().all(|m| *m == IDENTITY),
            Step::Blur { std_deviation } => radius(*std_deviation) == 0,
            Step::DropShadow { color, .. } => color.components[3] == 0.0,
        })
    }

    /// How far, in device pixels, the result of the chain at a pixel reads its input:
    /// the margin of input the filter region needs around the pixels it shows.
    pub(crate) fn reach(&self) -> (u32, u32) {
        self.steps.iter().fold((0, 0), |(x, y), step| match step {
            Step::Matrices(_) => (x, y),
            Step::Blur { std_deviation } => {
                (x + radius(*std_deviation), y + radius(*std_deviation))
            }
            Step::DropShadow {
                dx,
                dy,
                std_deviation,
                ..
            } => (
                x + radius(*std_deviation) + dx.unsigned_abs(),
                y + radius(*std_deviation) + dy.unsigned_abs(),
            ),
        })
    }
}

/// An integer device-space rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Region {
    /// The pixels of `bounds` inside the viewport, grown by `reach` (beyond the viewport
    /// too), and no larger than `max` per side. `None` if no pixel of `bounds` is visible.
    pub(crate) fn around(
        bounds: Rect,
        viewport: (u32, u32),
        reach: (u32, u32),
        max: u32,
    ) -> Option<Self> {
        let visible = bounds.intersect(Rect::new(
            0.0,
            0.0,
            f64::from(viewport.0),
            f64::from(viewport.1),
        ));
        if !(visible.width() > 0.0 && visible.height() > 0.0) {
            return None;
        }
        let visible = visible.expand();
        let (rx, ry) = (reach.0 as i32, reach.1 as i32);
        let x0 = visible.x0 as i32 - rx;
        let y0 = visible.y0 as i32 - ry;
        let x1 = visible.x1 as i32 + rx;
        let y1 = visible.y1 as i32 + ry;
        Some(Self {
            x: x0,
            y: y0,
            width: ((x1 - x0) as u32).min(max),
            height: ((y1 - y0) as u32).min(max),
        })
    }

    pub(crate) fn rect(&self) -> Rect {
        Rect::new(
            f64::from(self.x),
            f64::from(self.y),
            f64::from(self.x) + f64::from(self.width),
            f64::from(self.y) + f64::from(self.height),
        )
    }
}

/// Filter state a renderer keeps between frames: the filter pipelines of its device, the
/// images registered with Vello for the frame being drawn, and the frame's parameters.
#[derive(Default)]
pub(crate) struct FilterState {
    engine: Option<FilterEngine>,
    frame_images: Vec<ImageData>,
    /// The viewport of the frame being drawn.
    pub(crate) size: (u32, u32),
    /// The anti-aliasing of the frame, used for the layers rendered offscreen too.
    pub(crate) antialiasing: Option<AaConfig>,
}

impl FilterState {
    /// Unregister the frame's filtered images once the frame has been rendered.
    pub(crate) fn end_frame(&mut self, renderer: &mut VelloRenderer) {
        for image in self.frame_images.drain(..) {
            renderer.unregister_texture(image);
        }
    }

    /// Forget the pipelines (the device is going away).
    pub(crate) fn clear(&mut self) {
        self.engine = None;
        self.frame_images.clear();
    }
}

/// What a scene painter needs to draw filters: the Vello renderer and device of the
/// frame, and the renderer's filter state.
pub(crate) struct Gpu<'a> {
    pub(crate) renderer: &'a mut VelloRenderer,
    pub(crate) device: &'a DeviceHandle,
    pub(crate) state: &'a mut FilterState,
}

impl Gpu<'_> {
    fn engine(&mut self) -> &FilterEngine {
        let device = &self.device.device;
        if self
            .state
            .engine
            .as_ref()
            .is_none_or(|engine| engine.device != *device)
        {
            self.state.engine = Some(FilterEngine::new(device, &self.device.queue));
        }
        self.state.engine.as_ref().unwrap()
    }

    /// The largest texture side the device supports.
    pub(crate) fn max_size(&self) -> u32 {
        self.device.device.limits().max_texture_dimension_2d
    }

    /// Render `scene` (device coordinates) over `region` into a straight-alpha
    /// `Rgba8Unorm` texture of the region's size.
    pub(crate) fn render(&mut self, scene: &Scene, region: Region) -> wgpu::Texture {
        let texture = self.device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anyrender_vello filter source"),
            size: wgpu::Extent3d {
                width: region.width,
                height: region.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut shifted = Scene::new();
        shifted.append(
            scene,
            Some(Affine::translate((
                -f64::from(region.x),
                -f64::from(region.y),
            ))),
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render_to_texture(
                &self.device.device,
                &self.device.queue,
                &shifted,
                &view,
                &RenderParams {
                    base_color: peniko::Color::TRANSPARENT,
                    width: region.width,
                    height: region.height,
                    antialiasing_method: self.state.antialiasing.unwrap_or(AaConfig::Area),
                },
            )
            .expect("failed to render a filtered layer");
        texture
    }

    /// Filter the `size` texels of `source` starting at `offset` (transparent beyond
    /// the source, or mirrored if `mirror`) and register the result with Vello.
    pub(crate) fn filter(
        &mut self,
        source: &wgpu::Texture,
        offset: (i32, i32),
        mirror: bool,
        size: (u32, u32),
        chain: &Chain,
    ) -> ImageData {
        let texture = self.engine().apply(source, offset, mirror, size, chain);
        let image = self.renderer.register_texture(texture);
        self.state.frame_images.push(image.clone());
        image
    }
}

/// The filter pipelines of one device.
struct FilterEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    import: wgpu::RenderPipeline,
    matrix: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    export: wgpu::RenderPipeline,
    /// Bound as `original` by the passes that read one input.
    empty: wgpu::TextureView,
}

const FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The uniforms of a pass (`Params` in filters.wgsl).
#[derive(Default)]
struct Params {
    offset: (i32, i32),
    mode: u32,
    count: u32,
    sigma: f32,
    color: [f32; 4],
    matrices: Vec<[f32; 20]>,
}

impl Params {
    const SIZE: usize = 48 + 40 * 16;

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::SIZE);
        out.extend_from_slice(&self.offset.0.to_le_bytes());
        out.extend_from_slice(&self.offset.1.to_le_bytes());
        out.extend_from_slice(&self.mode.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        for value in [self.sigma, 0.0, 0.0, 0.0] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.color {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.matrices.iter().flatten() {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.resize(Self::SIZE, 0);
        out
    }
}

impl FilterEngine {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anyrender_vello filters"),
            source: wgpu::ShaderSource::Wgsl(include_str!("filters.wgsl").into()),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anyrender_vello filters"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anyrender_vello filters"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let empty = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("anyrender_vello filters empty"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FLOAT_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            device: device.clone(),
            queue: queue.clone(),
            import: pipeline("fs_import", FLOAT_FORMAT),
            matrix: pipeline("fs_matrix", FLOAT_FORMAT),
            blur: pipeline("fs_blur", FLOAT_FORMAT),
            shadow: pipeline("fs_shadow", FLOAT_FORMAT),
            export: pipeline("fs_export", wgpu::TextureFormat::Rgba8Unorm),
            layout,
            empty,
        }
    }

    fn target(&self, size: (u32, u32), format: wgpu::TextureFormat) -> wgpu::Texture {
        let mut usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        if format != FLOAT_FORMAT {
            // Vello copies registered images into its atlas.
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("anyrender_vello filter target"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::Texture,
        original: Option<&wgpu::Texture>,
        params: &Params,
        target: &wgpu::Texture,
    ) {
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let uniforms = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("anyrender_vello filter params"),
                contents: &params.bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let source_view = view(source);
        let original_view = original.map(view);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anyrender_vello filter pass"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        original_view.as_ref().unwrap_or(&self.empty),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        let target_view = view(target);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("anyrender_vello filter pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: wgpu::Texture,
        std_deviation: f32,
        size: (u32, u32),
    ) -> wgpu::Texture {
        let radius = radius(std_deviation);
        if radius == 0 {
            return source;
        }
        let mut current = source;
        for mode in [0, 1] {
            let next = self.target(size, FLOAT_FORMAT);
            let params = Params {
                mode,
                count: radius,
                sigma: std_deviation,
                ..Default::default()
            };
            self.pass(encoder, &self.blur, &current, None, &params, &next);
            current = next;
        }
        current
    }

    /// Filter `size` texels of the straight-alpha `source` from `offset` into a new
    /// straight-alpha `Rgba8Unorm` texture.
    fn apply(
        &self,
        source: &wgpu::Texture,
        offset: (i32, i32),
        mirror: bool,
        size: (u32, u32),
        chain: &Chain,
    ) -> wgpu::Texture {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anyrender_vello filters"),
            });
        let mut current = self.target(size, FLOAT_FORMAT);
        let params = Params {
            offset,
            mode: u32::from(mirror),
            ..Default::default()
        };
        self.pass(&mut encoder, &self.import, source, None, &params, &current);
        for step in &chain.steps {
            match step {
                Step::Matrices(run) => {
                    for matrices in run.chunks(MAX_MATRICES) {
                        let next = self.target(size, FLOAT_FORMAT);
                        let params = Params {
                            count: matrices.len() as u32,
                            matrices: matrices.to_vec(),
                            ..Default::default()
                        };
                        self.pass(&mut encoder, &self.matrix, &current, None, &params, &next);
                        current = next;
                    }
                }
                Step::Blur { std_deviation } => {
                    current = self.blur(&mut encoder, current, *std_deviation, size);
                }
                Step::DropShadow {
                    dx,
                    dy,
                    std_deviation,
                    color,
                } => {
                    let blurred = self.blur(&mut encoder, current.clone(), *std_deviation, size);
                    let next = self.target(size, FLOAT_FORMAT);
                    let params = Params {
                        offset: (*dx, *dy),
                        color: color.components,
                        ..Default::default()
                    };
                    self.pass(
                        &mut encoder,
                        &self.shadow,
                        &blurred,
                        Some(&current),
                        &params,
                        &next,
                    );
                    current = next;
                }
            }
        }
        let output = self.target(size, wgpu::TextureFormat::Rgba8Unorm);
        self.pass(
            &mut encoder,
            &self.export,
            &current,
            None,
            &Params::default(),
            &output,
        );
        self.queue.submit([encoder.finish()]);
        output
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    //! These tests render on a wgpu adapter (for example Mesa's lavapipe, selected with
    //! `VK_ICD_FILENAMES=…/lvp_icd.x86_64.json`). The expected values are Chromium's,
    //! as in `anyrender_vello_cpu`'s filter tests.
    use std::sync::Arc;

    use anyrender::{Filter, PaintScene, filters::FilterEffect, render_to_buffer};
    use kurbo::{Affine, Rect};
    use peniko::{Color, Fill, Mix};

    use crate::VelloImageRenderer;

    fn near(actual: [u8; 4], expected: [u8; 3], what: &str) {
        assert!(
            actual[..3]
                .iter()
                .zip(expected)
                .all(|(a, e)| a.abs_diff(e) <= 1),
            "{what}: {actual:?}, expected {expected:?}"
        );
    }

    /// Draw a 32x32 square of rgb(200, 100, 50) on white through `effects` and return
    /// the centre pixel and the pixel `dx` to the right of the square's right edge.
    fn render(effects: Vec<FilterEffect>, dx: usize) -> ([u8; 4], [u8; 4]) {
        let filter = Arc::new(Filter::linear_list(effects.into_iter()));
        let buffer = render_to_buffer::<VelloImageRenderer, _>(
            |scene| {
                let all = Rect::new(0.0, 0.0, 96.0, 64.0);
                scene.fill(Fill::NonZero, Affine::IDENTITY, Color::WHITE, None, &all);
                scene.push_layer(
                    Fill::NonZero,
                    Mix::Normal,
                    1.0,
                    Affine::IDENTITY,
                    &all,
                    Some(filter),
                    None,
                );
                let square = Rect::new(16.0, 16.0, 48.0, 48.0);
                let color = Color::from_rgb8(200, 100, 50);
                scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &square);
                scene.pop_layer();
            },
            96,
            64,
        );
        let pixel = |x: usize, y: usize| {
            let i = (y * 96 + x) * 4;
            [buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]]
        };
        (pixel(32, 32), pixel(47 + dx, 32))
    }

    #[test]
    fn css_colour_functions() {
        let cases = [
            (FilterEffect::brightness(0.5), [100, 50, 25]),
            (FilterEffect::grayscale(1.0), [118, 118, 118]),
            (FilterEffect::sepia(1.0), [165, 147, 114]),
            (FilterEffect::invert(1.0), [55, 155, 205]),
            (FilterEffect::contrast(0.5), [164, 114, 89]),
            (FilterEffect::saturate(0.5), [159, 109, 84]),
            (FilterEffect::hue_rotate(90f32.to_radians()), [50, 146, 35]),
            // Over white.
            (FilterEffect::opacity(0.5), [228, 178, 152]),
        ];
        for (effect, expected) in cases {
            let (centre, _) = render(vec![effect.clone()], 0);
            near(centre, expected, &format!("{effect:?}"));
        }
    }

    #[test]
    fn filter_lists_apply_every_function_in_order_in_float() {
        // Chromium: each result clamped, intermediate results not rounded (36, not 37).
        let (centre, _) = render(
            vec![FilterEffect::brightness(0.5), FilterEffect::contrast(2.0)],
            0,
        );
        assert_eq!(centre, [73, 0, 0, 255]);
        let (centre, _) = render(
            vec![FilterEffect::contrast(2.0), FilterEffect::brightness(0.5)],
            0,
        );
        assert_eq!(centre, [128, 36, 0, 255]);
    }

    #[test]
    fn drop_shadows_and_blurs_draw_outside_the_source() {
        let blue = Color::from_rgb8(0, 0, 255);
        let (_, outside) = render(vec![FilterEffect::drop_shadow(10.0, 0.0, 0.0, blue)], 5);
        assert_eq!(outside, [0, 0, 255, 255]);
        let (_, outside) = render(
            vec![
                FilterEffect::brightness(0.5),
                FilterEffect::drop_shadow(10.0, 0.0, 0.0, blue),
            ],
            5,
        );
        assert_eq!(outside, [0, 0, 255, 255]);
        let (_, outside) = render(vec![FilterEffect::blur(4.0)], 3);
        assert!(outside[1] > 100 && outside[1] < 250, "{outside:?}");
    }

    #[test]
    fn blurs_use_the_length_as_standard_deviation() {
        // A Gaussian with σ = 4 across the square's right edge (x = 48): at the centre
        // of pixel 51, 3.5 px outside, the square's colour has the weight
        // 1 - Φ(3.5 / 4) = 0.191 (the square is 32 px wide, so its far edge adds nothing).
        let (_, outside) = render(vec![FilterEffect::blur(4.0)], 4);
        let expected = 255.0 - 0.1908 * (255.0 - 100.0);
        assert!(
            (f32::from(outside[1]) - expected).abs() <= 2.0,
            "{outside:?}, expected green {expected}"
        );
    }

    #[test]
    fn unsupported_graphs_draw_unfiltered() {
        let (centre, _) = render(vec![FilterEffect::brightness(0.5), FilterEffect::Tile], 0);
        assert_eq!(centre, [200, 100, 50, 255]);
    }

    /// Red left of x = 50 and blue right of it, under a layer over (20, 20, 80, 80)
    /// with the given backdrop filter and a half-transparent white square in it. If
    /// `nested`, all of it is inside a clip layer and a layer with `saturate(2)`, which
    /// leaves the backdrop's colours as they are.
    fn render_backdrop(backdrop: Vec<FilterEffect>, nested: bool, alpha: f32) -> Vec<u8> {
        let backdrop = Arc::new(Filter::linear_list(backdrop.into_iter()));
        render_to_buffer::<VelloImageRenderer, _>(
            |scene| {
                let left = Rect::new(0.0, 0.0, 50.0, 100.0);
                let right = Rect::new(50.0, 0.0, 100.0, 100.0);
                let red = Color::from_rgb8(255, 0, 0);
                let blue = Color::from_rgb8(0, 0, 255);
                let all = Rect::new(0.0, 0.0, 100.0, 100.0);
                if nested {
                    // The backdrop is read through open layers, a filtered one too.
                    scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &all);
                    scene.push_layer(
                        Fill::NonZero,
                        Mix::Normal,
                        1.0,
                        Affine::IDENTITY,
                        &all,
                        Some(Arc::new(Filter::single(FilterEffect::saturate(2.0)))),
                        None,
                    );
                }
                scene.fill(Fill::NonZero, Affine::IDENTITY, red, None, &left);
                scene.fill(Fill::NonZero, Affine::IDENTITY, blue, None, &right);
                let panel = Rect::new(20.0, 20.0, 80.0, 80.0);
                scene.push_layer(
                    Fill::NonZero,
                    Mix::Normal,
                    alpha,
                    Affine::IDENTITY,
                    &panel,
                    None,
                    Some(backdrop),
                );
                let dot = Rect::new(60.0, 60.0, 70.0, 70.0);
                let white = Color::from_rgba8(255, 255, 255, 128);
                scene.fill(Fill::NonZero, Affine::IDENTITY, white, None, &dot);
                scene.pop_layer();
                if nested {
                    scene.pop_layer();
                    scene.pop_layer();
                }
            },
            100,
            100,
        )
    }

    fn pixel(buffer: &[u8], x: usize, y: usize) -> [u8; 4] {
        let i = (y * 100 + x) * 4;
        [buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]]
    }

    #[test]
    fn backdrop_filters_filter_what_is_behind_the_layer() {
        for nested in [false, true] {
            let buffer = render_backdrop(vec![FilterEffect::invert(1.0)], nested, 1.0);
            // Inside the layer: the inverted backdrop, then the layer's content over it.
            assert_eq!(pixel(&buffer, 30, 30), [0, 255, 255, 255]);
            assert_eq!(pixel(&buffer, 55, 30), [255, 255, 0, 255]);
            if !nested {
                near(pixel(&buffer, 65, 65), [255, 255, 128], "dot");
            }
            // Outside it, the backdrop is unchanged.
            assert_eq!(pixel(&buffer, 10, 30), [255, 0, 0, 255]);
            assert_eq!(pixel(&buffer, 90, 30), [0, 0, 255, 255]);
        }
    }

    #[test]
    fn backdrop_blurs_read_beyond_the_layer_but_stay_inside_it() {
        let buffer = render_backdrop(vec![FilterEffect::blur(4.0)], false, 1.0);
        // Across the red/blue edge both colours mix inside the layer only.
        let mixed = pixel(&buffer, 50, 30);
        assert!(mixed[0] > 60 && mixed[2] > 60, "{mixed:?}");
        assert_eq!(pixel(&buffer, 49, 10), [255, 0, 0, 255]);
        assert_eq!(pixel(&buffer, 50, 10), [0, 0, 255, 255]);
        // Near the layer's left edge the blur reads the red outside it: no fade.
        assert_eq!(pixel(&buffer, 21, 30), [255, 0, 0, 255]);
    }

    /// As in Chromium, the filtered backdrop is composited with the layer's opacity,
    /// then the layer's content with it over that.
    #[test]
    fn the_layer_opacity_applies_to_backdrop_and_content_separately() {
        let buffer = render_backdrop(vec![FilterEffect::invert(1.0)], false, 0.5);
        // Half the inverted red over red.
        near(pixel(&buffer, 30, 30), [128, 128, 128], "backdrop");
        // Half the white square (itself half transparent) over that, over blue.
        near(pixel(&buffer, 65, 65), [160, 160, 160], "dot");
    }

    #[test]
    fn backdrops_mirror_beyond_the_viewport() {
        // A blurred backdrop at the viewport's edge reads the mirrored backdrop, so a
        // uniform colour stays uniform there.
        let buffer = render_to_buffer::<VelloImageRenderer, _>(
            |scene| {
                let all = Rect::new(0.0, 0.0, 100.0, 100.0);
                let green = Color::from_rgb8(0, 128, 0);
                scene.fill(Fill::NonZero, Affine::IDENTITY, green, None, &all);
                let blur = Arc::new(Filter::single(FilterEffect::blur(6.0)));
                scene.push_layer(
                    Fill::NonZero,
                    Mix::Normal,
                    1.0,
                    Affine::IDENTITY,
                    &all,
                    None,
                    Some(blur),
                );
                scene.pop_layer();
            },
            100,
            100,
        );
        assert_eq!(pixel(&buffer, 0, 0), [0, 128, 0, 255]);
        assert_eq!(pixel(&buffer, 99, 50), [0, 128, 0, 255]);
    }

    #[test]
    fn layers_without_a_renderer_draw_unfiltered() {
        let mut scene = vello::Scene::new();
        let mut painter = crate::VelloScenePainter::new(&mut scene);
        let filter = Arc::new(Filter::single(FilterEffect::blur(4.0)));
        let all = Rect::new(0.0, 0.0, 10.0, 10.0);
        painter.push_layer(
            Fill::NonZero,
            Mix::Normal,
            1.0,
            Affine::IDENTITY,
            &all,
            Some(filter),
            None,
        );
        assert!(matches!(
            painter.layers.as_slice(),
            [crate::scene::OpenLayer::Plain]
        ));
        painter.pop_layer();
        assert!(painter.layers.is_empty());
    }
}
