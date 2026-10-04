use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId};
use kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use peniko::{
    BlendMode, BrushRef, Color, Extend, Fill, FontData, ImageBrush, ImageData, ImageQuality,
    ImageSampler, StyleRef,
};
use rustc_hash::FxHashMap;
use vello::Renderer as VelloRenderer;
use wgpu::Texture;
use wgpu_context::DeviceHandle;

use crate::filters::{Chain, FilterState, Gpu, Region};

/// Tolerance for flattening the clips of filtered layers, which are kept until the
/// layer is drawn.
const CLIP_TOLERANCE: f64 = 0.1;

pub struct VelloScenePainter<'r, 's> {
    pub(crate) renderer: Option<&'r mut VelloRenderer>,
    pub(crate) device_handle: Option<&'r DeviceHandle>,
    pub(crate) texture_handles: Option<&'r mut FxHashMap<ResourceId, ImageData>>,
    pub(crate) inner: &'s mut vello::Scene,
    /// The renderer's filter state. Without it (or without the renderer and device),
    /// filters and backdrop filters are not drawn.
    pub(crate) filters: Option<&'r mut FilterState>,
    /// The layers pushed and not yet popped, innermost last.
    pub(crate) layers: Vec<OpenLayer>,
}

/// A layer pushed on the painter.
pub(crate) enum OpenLayer {
    /// A layer of the Vello scene.
    Plain,
    /// A filtered layer: its contents go to a scene of their own, which is filtered and
    /// drawn into the parent scene when the layer is popped.
    Filter(Box<FilterLayer>),
}

/// How a layer is composited: its clip, blend mode and opacity.
pub(crate) struct LayerStyle {
    fill: Fill,
    blend: BlendMode,
    alpha: f32,
    transform: Affine,
    clip: BezPath,
}

impl LayerStyle {
    /// The device-space bounds of the clip.
    fn bounds(&self) -> Rect {
        self.transform.transform_rect_bbox(self.clip.bounding_box())
    }

    /// Draw `image`, covering `region`, into `scene` through this layer.
    fn draw(&self, scene: &mut vello::Scene, image: ImageData, region: Region) {
        scene.push_layer(
            self.fill,
            self.blend,
            self.alpha,
            self.transform,
            &self.clip,
        );
        let brush = ImageBrush {
            image,
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        };
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            &brush,
            Some(Affine::translate((
                f64::from(region.x),
                f64::from(region.y),
            ))),
            &region.rect(),
        );
        scene.pop_layer();
    }
}

pub(crate) struct FilterLayer {
    /// The scene the layer is drawn into (the painter's scene before the push).
    parent: vello::Scene,
    style: LayerStyle,
    chain: Chain,
}

impl Gpu<'_> {
    /// Filter `content` (a filtered layer's contents) and draw the result into `target`.
    /// Only what the layer's clip shows is filtered, with the margin the filter reads.
    fn draw_filtered(
        &mut self,
        content: &vello::Scene,
        style: &LayerStyle,
        chain: &Chain,
        target: &mut vello::Scene,
    ) {
        let Some(region) = Region::around(
            style.bounds(),
            self.state.size,
            chain.reach(),
            self.max_size(),
        ) else {
            return;
        };
        let source = self.render(content, region);
        let image = self.filter(&source, (0, 0), false, (region.width, region.height), chain);
        style.draw(target, image, region);
    }
}

/// The renderer, device and filter state of a painter, if it can draw filters.
fn gpu<'a>(
    renderer: &'a mut Option<&mut VelloRenderer>,
    device: Option<&'a DeviceHandle>,
    filters: &'a mut Option<&mut FilterState>,
) -> Option<Gpu<'a>> {
    Some(Gpu {
        renderer: renderer.as_deref_mut()?,
        device: device?,
        state: filters.as_deref_mut()?,
    })
}

impl VelloScenePainter<'_, '_> {
    /// Pop the filtered layers still open (and the layers inside them), so that their
    /// contents reach the scene. Renderers call this after the frame's drawing function.
    pub(crate) fn close_filter_layers(&mut self) {
        while self
            .layers
            .iter()
            .any(|layer| matches!(layer, OpenLayer::Filter(_)))
        {
            self.pop_layer();
        }
    }

    /// Everything painted so far, with the open layers closed (filtered layers filter
    /// what they hold so far): the backdrop of a layer pushed now.
    fn backdrop(&mut self) -> Option<vello::Scene> {
        let mut gpu = gpu(&mut self.renderer, self.device_handle, &mut self.filters)?;
        let mut scene = self.inner.clone();
        for layer in self.layers.iter().rev() {
            match layer {
                OpenLayer::Plain => scene.pop_layer(),
                OpenLayer::Filter(layer) => {
                    let mut parent = layer.parent.clone();
                    gpu.draw_filtered(&scene, &layer.style, &layer.chain, &mut parent);
                    scene = parent;
                }
            }
        }
        Some(scene)
    }

    /// Draw the backdrop filtered by `chain` behind a layer, as Chromium does (and
    /// `anyrender_vello_cpu`): the filtered backdrop is composited through the layer's
    /// clip with its opacity, and the layer is drawn over it. The filter reads the
    /// backdrop around the clip as far as it reaches (a blur), beyond the viewport
    /// mirrored.
    fn draw_filtered_backdrop(&mut self, chain: &Chain, style: &LayerStyle) {
        let Some(size) = self.filters.as_ref().map(|state| state.size) else {
            return;
        };
        let Some(max) = self
            .device_handle
            .map(|device| device.device.limits().max_texture_dimension_2d)
        else {
            return;
        };
        let Some(region) = Region::around(style.bounds(), size, chain.reach(), max) else {
            return;
        };
        let Some(backdrop) = self.backdrop() else {
            return;
        };
        let Some(mut gpu) = gpu(&mut self.renderer, self.device_handle, &mut self.filters) else {
            return;
        };
        let viewport = Region {
            x: 0,
            y: 0,
            width: size.0,
            height: size.1,
        };
        let source = gpu.render(&backdrop, viewport);
        let image = gpu.filter(
            &source,
            (region.x, region.y),
            true,
            (region.width, region.height),
            chain,
        );
        style.draw(self.inner, image, region);
    }
}

impl RenderContext for VelloScenePainter<'_, '_> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<ResourceId, anyrender::RegisterResourceError> {
        if let Some(renderer) = &mut self.renderer
            && let Some(texture_handles) = &mut self.texture_handles
        {
            if let Ok(texture) = resource.downcast::<Texture>() {
                let id = ResourceId::new();
                texture_handles.insert(id, renderer.register_texture(*texture));
                Ok(id)
            } else {
                Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
            }
        } else {
            Err(anyrender::RegisterResourceErrorKind::Unimplemented.into())
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        if let Some(renderer) = &mut self.renderer
            && let Some(texture_handles) = &mut self.texture_handles
            && let Some(handle) = texture_handles.remove(&resource_id)
        {
            renderer.unregister_texture(handle);
        }
    }

    fn renderer_specific_context(&self) -> Option<Box<dyn std::any::Any>> {
        self.device_handle
            .map(|device_handle| Box::new(device_handle.clone()) as _)
    }
}

impl VelloScenePainter<'_, '_> {
    pub fn new<'s>(scene: &'s mut vello::Scene) -> VelloScenePainter<'static, 's> {
        VelloScenePainter {
            renderer: None,
            device_handle: None,
            texture_handles: None,
            inner: scene,
            filters: None,
            layers: Vec::new(),
        }
    }
}

impl PaintScene for VelloScenePainter<'_, '_> {
    fn reset(&mut self) {
        self.inner.reset();
        self.layers.clear();
    }

    fn push_layer(
        &mut self,
        fill: Fill,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        let blend = blend.into();
        // Filters need the renderer to draw offscreen; a painter without one (a scene
        // painted for later use) draws the layer unfiltered.
        let can_filter =
            self.renderer.is_some() && self.device_handle.is_some() && self.filters.is_some();
        let convert = |filter: Option<Arc<Filter>>| {
            filter
                .filter(|_| can_filter)
                .and_then(|filter| Chain::new(&filter, &transform))
                .filter(|chain| !chain.is_identity())
        };

        let style = |blend| LayerStyle {
            fill,
            blend,
            alpha,
            transform,
            clip: clip.into_path(CLIP_TOLERANCE),
        };

        // A backdrop filter draws the filtered backdrop behind the layer, composited
        // normally whatever the layer's blend mode.
        if let Some(chain) = convert(backdrop_filter) {
            self.draw_filtered_backdrop(&chain, &style(BlendMode::default()));
        }

        if let Some(chain) = convert(filter) {
            let parent = std::mem::take(self.inner);
            self.layers.push(OpenLayer::Filter(Box::new(FilterLayer {
                parent,
                style: style(blend),
                chain,
            })));
        } else {
            self.inner.push_layer(fill, blend, alpha, transform, clip);
            self.layers.push(OpenLayer::Plain);
        }
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        self.inner.push_clip_layer(fill, transform, clip);
        self.layers.push(OpenLayer::Plain);
    }

    fn pop_layer(&mut self) {
        match self.layers.pop() {
            Some(OpenLayer::Filter(mut layer)) => {
                let parent = std::mem::take(&mut layer.parent);
                let content = std::mem::replace(self.inner, parent);
                if let Some(mut gpu) =
                    gpu(&mut self.renderer, self.device_handle, &mut self.filters)
                {
                    gpu.draw_filtered(&content, &layer.style, &layer.chain, self.inner);
                }
            }
            Some(OpenLayer::Plain) | None => self.inner.pop_layer(),
        }
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint_ref: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let paint_ref: PaintRef<'_> = paint_ref.into();
        let brush_ref: BrushRef<'_> = paint_ref.into();
        self.inner
            .stroke(style, transform, brush_ref, brush_transform, shape);
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let paint: PaintRef<'_> = paint.into();
        let brush_ref: BrushRef<'_> = match paint {
            Paint::Solid(color) => BrushRef::Solid(color),
            Paint::Gradient(gradient) => BrushRef::Gradient(gradient),
            Paint::Image(image) => BrushRef::Image(image),
            Paint::Resource(brush) => {
                let resource_id = brush.image;
                if let Some(texture_handle) = self
                    .texture_handles
                    .as_ref()
                    .and_then(|texture_handles| texture_handles.get(&resource_id))
                {
                    peniko::Brush::Image(ImageBrush {
                        image: texture_handle,
                        sampler: brush.sampler,
                    })
                } else {
                    BrushRef::Solid(Color::TRANSPARENT)
                }
            }
            Paint::Custom(_) => BrushRef::Solid(Color::TRANSPARENT),
        };

        self.inner
            .fill(style, transform, brush_ref, brush_transform, shape);
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        self.inner
            .draw_glyphs(font)
            .font_size(font_size)
            .hint(hint)
            .normalized_coords(normalized_coords)
            .font_embolden(vello::FontEmbolden::new(kurbo::Diagonal2::new(
                embolden.x, embolden.y,
            )))
            .brush(paint.into())
            .brush_alpha(brush_alpha)
            .transform(transform)
            .glyph_transform(glyph_transform)
            .draw(
                style,
                glyphs.map(|g: anyrender::Glyph| vello::Glyph {
                    id: g.id,
                    x: g.x,
                    y: g.y,
                }),
            );
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.inner
            .draw_blurred_rounded_rect(transform, rect, brush, radius, std_dev);
    }
}
