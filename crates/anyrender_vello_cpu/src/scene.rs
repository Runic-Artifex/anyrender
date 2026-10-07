use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, StyleRef};
use vello_cpu::{PaintType, Pixmap};

use crate::image_cache::{ImageCache, ImageCacheConfig};

const DEFAULT_TOLERANCE: f64 = 0.1;

pub struct VelloCpuScenePainter {
    pub(crate) render_ctx: vello_cpu::RenderContext,
    pub(crate) resources: vello_cpu::Resources,
    pub(crate) image_cache: ImageCache,
    /// The commands painted since the last reset, replayed to draw the backdrop of a
    /// layer with a backdrop filter, and the number of layers open.
    #[cfg(feature = "backdrop_filters")]
    recording: anyrender::recording::Scene,
    #[cfg(feature = "backdrop_filters")]
    open_layers: usize,
}

impl VelloCpuScenePainter {
    pub fn new(width: u16, height: u16) -> Self {
        Self::with_image_cache_config(width, height, ImageCacheConfig::default())
    }

    pub fn with_image_cache_config(width: u16, height: u16, config: ImageCacheConfig) -> Self {
        Self {
            render_ctx: vello_cpu::RenderContext::new(width, height),
            resources: vello_cpu::Resources::new(),
            image_cache: ImageCache::new(config),
            #[cfg(feature = "backdrop_filters")]
            recording: anyrender::recording::Scene::new(),
            #[cfg(feature = "backdrop_filters")]
            open_layers: 0,
        }
    }

    /// Forget the commands of the frame (called after each render and on reset).
    pub(crate) fn end_frame(&mut self) {
        #[cfg(feature = "backdrop_filters")]
        {
            self.recording.reset();
            self.open_layers = 0;
        }
    }

    /// Everything painted so far, with the open layers closed: the backdrop of a layer
    /// pushed now.
    #[cfg(feature = "backdrop_filters")]
    fn backdrop(&self) -> Pixmap {
        let mut painter =
            VelloCpuScenePainter::new(self.render_ctx.width(), self.render_ctx.height());
        painter.append_scene(self.recording.clone(), Affine::IDENTITY);
        for _ in 0..painter.open_layers {
            painter.pop_layer();
        }
        painter.render_ctx.flush();
        painter.finish()
    }

    /// Draw the backdrop filtered by `backdrop_filter` behind a layer with the given
    /// clip and opacity, as Chromium does: the filtered backdrop is composited with the
    /// layer's opacity, then the layer over it. The filter reads the backdrop around the
    /// clip as far as it reaches (a blur), beyond the viewport mirrored.
    #[cfg(feature = "backdrop_filters")]
    fn draw_filtered_backdrop(
        &mut self,
        backdrop_filter: vello_common::filter_effects::Filter,
        fill: Fill,
        alpha: f32,
        transform: Affine,
        clip: &kurbo::BezPath,
    ) {
        let backdrop = self.backdrop();
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(fill);
        self.render_ctx
            .push_layer(Some(clip), None, Some(alpha), None, None);
        let expansion = backdrop_filter.filter_expansion(&transform);
        let bounds = transform.transform_rect_bbox(clip.bounding_box());
        let area = Rect::new(
            bounds.x0 + expansion.x0,
            bounds.y0 + expansion.y0,
            bounds.x1 + expansion.x1,
            bounds.y1 + expansion.y1,
        );
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .push_layer(None, None, None, None, Some(backdrop_filter));
        self.render_ctx.set_transform(Affine::IDENTITY);
        self.render_ctx.set_paint(PaintType::Image(ImageBrush {
            image: vello_cpu::ImageSource::Pixmap(Arc::new(backdrop)),
            sampler: peniko::ImageSampler::default().with_extend(peniko::Extend::Reflect),
        }));
        self.render_ctx.reset_paint_transform();
        self.render_ctx.fill_rect(&area);
        self.render_ctx.pop_layer();
        self.render_ctx.pop_layer();
    }

    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image) => PaintType::Image(ImageBrush {
                image: self
                    .image_cache
                    .get_or_register(&mut self.resources, image.image),
                sampler: image.sampler,
            }),
            // TODO: custom paint
            Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
    }

    /// Advance the image cache's frame counter and evict stale entries.
    ///
    /// Should be called once per frame, after rendering.
    pub fn maintain(&mut self) {
        self.image_cache.maintain(&mut self.resources);
    }

    /// Drop all cached image conversions.
    pub fn clear_image_cache(&mut self) {
        self.image_cache.clear(&mut self.resources);
    }

    pub fn finish(mut self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
        self.render_ctx.render(pixmap.as_mut(), &mut self.resources);
        pixmap
    }
}

impl RenderContext for VelloCpuScenePainter {}
impl PaintScene for VelloCpuScenePainter {
    fn reset(&mut self) {
        self.render_ctx.reset();
        self.end_frame();
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
        let clip = clip.into_path(DEFAULT_TOLERANCE);

        #[cfg(feature = "filters")]
        let convert = |filter: &Option<Arc<Filter>>| {
            filter
                .clone()
                .and_then(crate::filters::convert_filter)
                .filter(|_| cfg!(not(feature = "multithreading")))
        };
        #[cfg(not(feature = "filters"))]
        let convert = |_: &Option<Arc<Filter>>| None;

        // A backdrop filter draws the filtered backdrop behind the layer.
        #[cfg(feature = "backdrop_filters")]
        if let Some(backdrop) = convert(&backdrop_filter) {
            self.draw_filtered_backdrop(backdrop, fill, alpha, transform, &clip);
        }
        #[cfg(not(feature = "backdrop_filters"))]
        let _ = backdrop_filter;
        #[cfg(feature = "backdrop_filters")]
        let filter_arg = filter.clone();

        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(fill);
        self.render_ctx.push_layer(
            Some(&clip),
            Some(blend),
            Some(alpha),
            None,
            convert(&filter),
        );

        #[cfg(feature = "backdrop_filters")]
        self.recording.push_layer(
            fill,
            blend,
            alpha,
            transform,
            &clip,
            filter_arg,
            backdrop_filter,
        );
        #[cfg(feature = "backdrop_filters")]
        {
            self.open_layers += 1;
        }
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        let clip = clip.into_path(DEFAULT_TOLERANCE);
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(fill);
        self.render_ctx.push_clip_layer(&clip);
        #[cfg(feature = "backdrop_filters")]
        {
            self.recording.push_clip_layer(fill, transform, &clip);
            self.open_layers += 1;
        }
    }

    fn pop_layer(&mut self) {
        self.render_ctx.pop_layer();
        #[cfg(feature = "backdrop_filters")]
        {
            self.recording.pop_layer();
            self.open_layers = self.open_layers.saturating_sub(1);
        }
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let paint = paint.into();
        #[cfg(feature = "backdrop_filters")]
        self.recording
            .stroke(style, transform, paint.clone(), brush_transform, shape);
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_stroke(style.clone());
        let paint = self.convert_paint(paint);
        self.render_ctx.set_paint(paint);
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let paint = paint.into();
        #[cfg(feature = "backdrop_filters")]
        self.recording
            .fill(style, transform, paint.clone(), brush_transform, shape);
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(style);
        let paint = self.convert_paint(paint);
        self.render_ctx.set_paint(paint);
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .fill_path(&shape.into_path(DEFAULT_TOLERANCE));
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
        glyphs: impl Iterator<Item = anyrender::Glyph> + Clone,
    ) {
        let paint = paint.into();
        let style: StyleRef<'a> = style.into();
        #[cfg(feature = "backdrop_filters")]
        self.recording.draw_glyphs(
            font,
            font_size,
            hint,
            normalized_coords,
            embolden,
            style,
            paint.clone(),
            brush_alpha,
            transform,
            glyph_transform,
            glyphs.clone(),
        );
        #[cfg(not(feature = "backdrop_filters"))]
        let _ = brush_alpha;
        self.render_ctx.set_transform(transform);
        let paint = self.convert_paint(paint);
        self.render_ctx.set_paint(paint);

        match style {
            StyleRef::Fill(fill) => {
                self.render_ctx.set_fill_rule(fill);
                let _ = self
                    .render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(layout_glyph_transform(glyph_transform))
                    .fill_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.render_ctx.set_stroke(stroke.clone());
                let _ = self
                    .render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(layout_glyph_transform(glyph_transform))
                    .stroke_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
        }
    }
    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        #[cfg(feature = "backdrop_filters")]
        self.recording
            .draw_box_shadow(transform, rect, color, radius, std_dev);
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_paint(PaintType::Solid(color));
        self.render_ctx.reset_paint_transform();
        self.render_ctx
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32, false);
    }
}

#[cfg(test)]
mod clip_rule_tests {
    use anyrender::{PaintScene, recording::Scene, render_to_buffer};
    use kurbo::{Affine, BezPath, Rect};
    use peniko::{Fill, Mix, color::palette::css::RED};

    use crate::VelloCpuImageRenderer;

    fn render_clip(fill: Fill, compositing_layer: bool, replay: bool) -> Vec<u8> {
        let path = BezPath::from_svg("M0 0H100V100H0Z M25 25H75V75H25Z").unwrap();
        let draw = |scene: &mut crate::VelloCpuScenePainter| {
            if replay {
                let mut recording = Scene::new();
                draw_clipped(&mut recording, fill, compositing_layer, &path);
                scene.append_scene(recording, Affine::IDENTITY);
            } else {
                draw_clipped(scene, fill, compositing_layer, &path);
            }
        };
        render_to_buffer::<VelloCpuImageRenderer, _>(draw, 100, 100)
    }

    fn draw_clipped(
        scene: &mut impl PaintScene,
        fill: Fill,
        compositing_layer: bool,
        path: &BezPath,
    ) {
        if compositing_layer {
            scene.push_layer(fill, Mix::Normal, 1.0, Affine::IDENTITY, path, None, None);
        } else {
            scene.push_clip_layer(fill, Affine::IDENTITY, path);
        }
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            RED,
            None,
            &Rect::new(0.0, 0.0, 100.0, 100.0),
        );
        scene.pop_layer();
    }

    fn assert_pixels(compositing_layer: bool, replay: bool) {
        for fill in [Fill::NonZero, Fill::EvenOdd] {
            let buffer = render_clip(fill, compositing_layer, replay);
            let pixel = |x: usize, y: usize| &buffer[(y * 100 + x) * 4..(y * 100 + x) * 4 + 4];
            assert_eq!(pixel(10, 10), &[255, 0, 0, 255]);
            assert_eq!(
                pixel(50, 50),
                if fill == Fill::EvenOdd {
                    &[0, 0, 0, 0]
                } else {
                    &[255, 0, 0, 255]
                }
            );
        }
    }

    #[test]
    fn clip_layers_respect_fill_rule() {
        assert_pixels(false, false);
    }

    #[test]
    fn compositing_layers_respect_fill_rule() {
        assert_pixels(true, false);
    }

    #[test]
    fn scene_replay_preserves_clip_rules() {
        assert_pixels(false, true);
        assert_pixels(true, true);
    }
}

#[cfg(all(test, feature = "backdrop_filters"))]
mod backdrop_filter_tests {
    use std::sync::Arc;

    use anyrender::{Filter, PaintScene, filters::FilterEffect, render_to_buffer};
    use kurbo::{Affine, Rect};
    use peniko::{Color, Fill, Mix};

    use crate::VelloCpuImageRenderer;

    /// Red left of x = 50 and blue right of it, under a layer over (20, 20, 80, 80)
    /// with the given backdrop filter and a half-transparent white square in it.
    pub(super) fn render(backdrop: Vec<FilterEffect>, nested: bool, alpha: f32) -> Vec<u8> {
        let backdrop = Arc::new(Filter::linear_list(backdrop.into_iter()));
        render_to_buffer::<VelloCpuImageRenderer, _>(
            |scene| {
                let left = Rect::new(0.0, 0.0, 50.0, 100.0);
                let right = Rect::new(50.0, 0.0, 100.0, 100.0);
                let red = Color::from_rgb8(255, 0, 0);
                let blue = Color::from_rgb8(0, 0, 255);
                if nested {
                    // The backdrop is read through open layers.
                    let all = Rect::new(0.0, 0.0, 100.0, 100.0);
                    scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &all);
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
                }
            },
            100,
            100,
        )
    }

    pub(super) fn pixel(buffer: &[u8], x: usize, y: usize) -> [u8; 4] {
        let i = (y * 100 + x) * 4;
        [buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]]
    }

    #[test]
    fn backdrop_filters_filter_what_is_behind_the_layer() {
        for nested in [false, true] {
            let buffer = render(vec![FilterEffect::invert(1.0)], nested, 1.0);
            // Inside the layer: the inverted backdrop, then the layer's content over it.
            assert_eq!(pixel(&buffer, 30, 30), [0, 255, 255, 255]);
            assert_eq!(pixel(&buffer, 55, 30), [255, 255, 0, 255]);
            assert_eq!(pixel(&buffer, 65, 65), [255, 255, 128, 255]);
            // Outside it, the backdrop is unchanged.
            assert_eq!(pixel(&buffer, 10, 30), [255, 0, 0, 255]);
            assert_eq!(pixel(&buffer, 90, 30), [0, 0, 255, 255]);
        }
    }

    #[test]
    fn backdrop_blurs_read_beyond_the_layer_but_stay_inside_it() {
        let buffer = render(vec![FilterEffect::blur(4.0)], false, 1.0);
        // Across the red/blue edge both colours mix inside the layer only.
        let mixed = pixel(&buffer, 50, 30);
        assert!(mixed[0] > 60 && mixed[2] > 60, "{mixed:?}");
        assert_eq!(pixel(&buffer, 49, 10), [255, 0, 0, 255]);
        assert_eq!(pixel(&buffer, 50, 10), [0, 0, 255, 255]);
        // Near the layer's left edge the blur reads the red outside it: no fade.
        assert_eq!(pixel(&buffer, 21, 30), [255, 0, 0, 255]);
    }
}

#[cfg(all(test, feature = "backdrop_filters"))]
mod backdrop_opacity_tests {
    use super::backdrop_filter_tests::{pixel, render};
    use anyrender::filters::FilterEffect;

    /// As in Chromium, the filtered backdrop is composited with the layer's opacity,
    /// then the layer's content with it over that.
    #[test]
    fn the_layer_opacity_applies_to_backdrop_and_content_separately() {
        let buffer = render(vec![FilterEffect::invert(1.0)], false, 0.5);
        let near = |actual: [u8; 4], expected: [u8; 3]| {
            assert!(
                actual[..3]
                    .iter()
                    .zip(expected)
                    .all(|(a, e)| a.abs_diff(e) <= 1),
                "{actual:?}, expected {expected:?}"
            )
        };
        // Half the inverted red over red.
        near(pixel(&buffer, 30, 30), [128, 128, 128]);
        // Half the white square (itself half transparent) over that, over blue.
        near(pixel(&buffer, 65, 65), [160, 160, 160]);
    }
}

/// The glyph transform in vello_cpu's convention. Callers give it in font
/// space (y up), as vello does: a synthetic italic is `Affine::skew(tan, 0)`.
/// vello_cpu applies it after flipping the outline into layout space (y down),
/// where that skew leans the glyphs left, so it is conjugated with the flip.
fn layout_glyph_transform(transform: Option<Affine>) -> Affine {
    transform.map_or(Affine::IDENTITY, |t| Affine::FLIP_Y * t * Affine::FLIP_Y)
}
