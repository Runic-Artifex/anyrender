// Copyright 2026 the AnyRender Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Filter passes for CSS `filter` and `backdrop-filter` (see filters.rs).
//
// Every pass draws a full-target triangle and reads its inputs with
// `textureLoad` at integer texel positions, so that no sampler filtering is
// involved. Intermediate targets are `Rgba32Float` with premultiplied colour;
// the passes that read Vello's output (`fs_import`) or write the image Vello
// composites (`fs_export`) convert from and to its straight-alpha `Rgba8Unorm`.

struct Params {
    // fs_import: source texel of target texel `p` is `p + offset`.
    // fs_shadow: the shadow of target texel `p` is read at `p - offset`.
    offset: vec2<i32>,
    // fs_import: 0 = transparent outside the source, 1 = mirrored.
    // fs_blur: 0 = horizontal, 1 = vertical.
    mode: u32,
    // fs_matrix: number of matrices; fs_blur: kernel radius in texels.
    count: u32,
    // fs_blur: standard deviation in texels.
    sigma: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    // fs_shadow: the shadow colour (straight alpha).
    color: vec4<f32>,
    // fs_matrix: `count` row-major 4x5 matrices of 20 floats, packed.
    matrices: array<vec4<f32>, 40>,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var original: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn texel(t: texture_2d<f32>, p: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(t));
    if p.x < 0 || p.y < 0 || p.x >= size.x || p.y >= size.y {
        return vec4<f32>(0.0);
    }
    return textureLoad(t, p, 0);
}

// Symmetric reflection: ... 2 1 0 | 0 1 2 ... n-1 | n-1 n-2 ...
fn mirror(x: i32, n: i32) -> i32 {
    let period = 2 * n;
    var m = x % period;
    if m < 0 {
        m += period;
    }
    if m >= n {
        m = period - 1 - m;
    }
    return m;
}

fn premultiply(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

fn unpremultiply(c: vec4<f32>) -> vec4<f32> {
    if c.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

@fragment
fn fs_import(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var p = vec2<i32>(floor(position.xy)) + params.offset;
    if params.mode == 1u {
        let size = vec2<i32>(textureDimensions(source));
        p = vec2<i32>(mirror(p.x, size.x), mirror(p.y, size.y));
    }
    return premultiply(texel(source, p));
}

fn matrix_entry(index: u32) -> f32 {
    return params.matrices[index / 4u][index % 4u];
}

// Colour matrices on the straight colour, clamped after each one (Filter
// Effects: every CSS filter function's result is clamped), in float.
@fragment
fn fs_matrix(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var c = unpremultiply(texel(source, vec2<i32>(floor(position.xy))));
    for (var k = 0u; k < params.count; k += 1u) {
        let base = k * 20u;
        var out: vec4<f32>;
        for (var row = 0u; row < 4u; row += 1u) {
            let r = base + row * 5u;
            out[row] = matrix_entry(r) * c.r + matrix_entry(r + 1u) * c.g
                + matrix_entry(r + 2u) * c.b + matrix_entry(r + 3u) * c.a + matrix_entry(r + 4u);
        }
        c = clamp(out, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    return premultiply(c);
}

// One direction of a separable Gaussian blur of the premultiplied colour,
// transparent beyond the target, truncated at `count` = ceil(3 sigma).
@fragment
fn fs_blur(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    let step = select(vec2<i32>(1, 0), vec2<i32>(0, 1), params.mode == 1u);
    let radius = i32(params.count);
    let scale = -0.5 / (params.sigma * params.sigma);
    var sum = vec4<f32>(0.0);
    var weights = 0.0;
    for (var i = -radius; i <= radius; i += 1) {
        let w = exp(f32(i * i) * scale);
        sum += w * texel(source, p + i * step);
        weights += w;
    }
    return sum / weights;
}

// `drop-shadow()`: the blurred alpha of the input (`source`), offset and
// coloured, with the input (`original`) composited over it.
@fragment
fn fs_shadow(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    let alpha = texel(source, p - params.offset).a;
    let shadow = premultiply(params.color) * alpha;
    let top = texel(original, p);
    return top + shadow * (1.0 - top.a);
}

@fragment
fn fs_export(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return unpremultiply(texel(source, vec2<i32>(floor(position.xy))));
}
