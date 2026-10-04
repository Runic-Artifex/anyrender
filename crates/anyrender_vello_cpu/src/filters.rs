use std::sync::Arc;

use anyrender::{
    Filter,
    filters::{EdgeMode, FilterEffect, FilterId, FilterInput},
};
use vello_common::filter_effects::FilterPrimitive;

/// Convert a filter graph that Vello CPU can draw: a linear chain (each primitive reads
/// the previous one's result, as `Filter::linear_list` builds for a CSS filter list) of
/// supported primitives.
///
/// Returns `None` for an empty graph, a graph with other inputs, or one with a primitive
/// Vello CPU does not implement; the layer is then drawn unfiltered.
pub(crate) fn convert_filter(filter: Arc<Filter>) -> Option<vello_common::filter_effects::Filter> {
    let nodes = filter.nodes();
    if nodes.is_empty() || usize::from(filter.output().0) != nodes.len() - 1 {
        return None;
    }

    let primitives = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let reads_previous = match &node.inputs.primary {
                None => true,
                Some(FilterInput::Result(FilterId(id))) => usize::from(*id) + 1 == index,
                Some(FilterInput::Source(source)) => {
                    index == 0 && *source == anyrender::filters::FilterSource::SourceGraphic
                }
            };
            if !reads_previous || node.inputs.secondary.is_some() {
                return None;
            }
            convert_filter_effect(&node.effect)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(vello_common::filter_effects::Filter::from_primitives(
        primitives,
    ))
}

pub(crate) fn convert_filter_effect(effect: &FilterEffect) -> Option<FilterPrimitive> {
    Some(match effect {
        FilterEffect::Flood(color) => FilterPrimitive::Flood { color: *color },
        FilterEffect::GaussianBlur(blur) => FilterPrimitive::GaussianBlur {
            std_deviation: blur.std_deviation,
            edge_mode: convert_edge_mode(blur.edge_mode),
        },
        FilterEffect::DropShadow(shadow) => FilterPrimitive::DropShadow {
            dx: shadow.dx,
            dy: shadow.dy,
            std_deviation: shadow.std_deviation,
            color: shadow.color,
            edge_mode: convert_edge_mode(shadow.edge_mode),
        },
        FilterEffect::Offset(offset) => FilterPrimitive::Offset {
            dx: offset.x as f32,
            dy: offset.y as f32,
        },
        FilterEffect::ColorMatrix(matrix) => FilterPrimitive::ColorMatrix { matrix: matrix.0 },
        FilterEffect::ComponentTransfer(_component_transfer_filter) => return None,
        FilterEffect::Blend(mode) => FilterPrimitive::Blend { mode: *mode },
        FilterEffect::Composite(_composite_operator) => return None,
        FilterEffect::Morphology(_morphology_filter) => return None,
        FilterEffect::ConvolveMatrix(_convolution_kernel) => return None,
        FilterEffect::Turbulence(_turbulence_filter) => return None,
        FilterEffect::DisplacementMap(_displacement_map_filter) => return None,
        FilterEffect::Image(_external_image_source) => return None,
        FilterEffect::Tile => return None,
        FilterEffect::DiffuseLighting(_diffuse_lighting_filter) => return None,
        FilterEffect::SpecularLighting(_specular_lighting_filter) => return None,
    })
}

fn convert_edge_mode(edge_mode: EdgeMode) -> vello_common::filter_effects::EdgeMode {
    match edge_mode {
        EdgeMode::Duplicate => vello_common::filter_effects::EdgeMode::Duplicate,
        EdgeMode::Wrap => vello_common::filter_effects::EdgeMode::Wrap,
        EdgeMode::Mirror => vello_common::filter_effects::EdgeMode::Mirror,
        EdgeMode::None => vello_common::filter_effects::EdgeMode::None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyrender::{Filter, PaintScene, filters::FilterEffect, render_to_buffer};
    use kurbo::{Affine, Rect};
    use peniko::{Color, Fill, Mix};

    use crate::VelloCpuImageRenderer;

    /// Draw a 32x32 square of rgb(200, 100, 50) on white through `effects` and return
    /// the centre pixel and the pixel `dx` to the right of the square's right edge.
    fn render(effects: Vec<FilterEffect>, dx: usize) -> ([u8; 4], [u8; 4]) {
        let filter = Arc::new(Filter::linear_list(effects.into_iter()));
        let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
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

    // The expected values are Chromium's: CSS filter functions work in sRGB.
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
            for (actual, expected) in centre.iter().zip(expected) {
                assert!(
                    actual.abs_diff(expected) <= 1,
                    "{effect:?}: {centre:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn filter_lists_apply_every_function_in_order() {
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
    fn unsupported_graphs_draw_unfiltered() {
        let (centre, _) = render(vec![FilterEffect::brightness(0.5), FilterEffect::Tile], 0);
        assert_eq!(centre, [200, 100, 50, 255]);
    }
}
