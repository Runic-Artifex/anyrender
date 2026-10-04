use anyrender::{ImageRenderer, RenderContext, ResourceId};
use peniko::ImageData;
use rustc_hash::FxHashMap;
use vello::{AaConfig, AaSupport, Renderer as VelloRenderer, RendererOptions, Scene as VelloScene};
use wgpu::TextureUsages;
use wgpu_context::{BufferRenderer, BufferRendererConfig, WGPUContext};

use crate::{DEFAULT_THREADS, VelloScenePainter, filters::FilterState};

pub struct VelloImageRenderer {
    buffer_renderer: BufferRenderer,
    vello_renderer: VelloRenderer,
    scene: VelloScene,
    texture_handles: FxHashMap<ResourceId, ImageData>,
    antialiasing_method: AaConfig,
    filters: FilterState,
}

impl VelloImageRenderer {
    /// Like [`ImageRenderer::new`], but reusing compiled pipelines from a cache
    /// wgpu built earlier. Creating the cache, persisting its data and deciding
    /// when it is stale are all the caller's responsibility.
    pub fn with_pipeline_cache(
        width: u32,
        height: u32,
        pipeline_cache: Option<wgpu::PipelineCache>,
    ) -> Self {
        Self::with_options(width, height, pipeline_cache, AaConfig::Area)
    }

    /// Like [`ImageRenderer::new`], with the anti-aliasing method of the renderer (area
    /// coverage by default; a window renderer uses `AaConfig::Msaa16` by default) and a
    /// pipeline cache as in [`with_pipeline_cache`](Self::with_pipeline_cache).
    pub fn with_options(
        width: u32,
        height: u32,
        pipeline_cache: Option<wgpu::PipelineCache>,
        antialiasing_method: AaConfig,
    ) -> Self {
        // Create WGPUContext
        let mut context = WGPUContext::new();

        // Create wgpu_context::BufferRenderer
        let buffer_renderer =
            pollster::block_on(context.create_buffer_renderer(BufferRendererConfig {
                width,
                height,
                usage: TextureUsages::STORAGE_BINDING,
            }))
            .expect("No compatible device found");

        // Create vello::Renderer
        let vello_renderer = VelloRenderer::new(
            buffer_renderer.device(),
            RendererOptions {
                use_cpu: false,
                num_init_threads: DEFAULT_THREADS,
                antialiasing_support: match antialiasing_method {
                    AaConfig::Area => AaSupport::area_only(),
                    AaConfig::Msaa8 => AaSupport {
                        area: false,
                        msaa8: true,
                        msaa16: false,
                    },
                    AaConfig::Msaa16 => AaSupport {
                        area: false,
                        msaa8: false,
                        msaa16: true,
                    },
                },
                pipeline_cache,
            },
        )
        .expect("Got non-Send/Sync error from creating renderer");

        Self {
            buffer_renderer,
            vello_renderer,
            scene: VelloScene::new(),
            texture_handles: FxHashMap::default(),
            antialiasing_method,
            filters: FilterState::default(),
        }
    }

    /// The anti-aliasing method this renderer draws with.
    pub fn antialiasing_method(&self) -> AaConfig {
        self.antialiasing_method
    }

    /// The adapter the renderer draws on.
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.buffer_renderer.device_handle.adapter.get_info()
    }
}

impl RenderContext for VelloImageRenderer {}
impl ImageRenderer for VelloImageRenderer {
    type ScenePainter<'a>
        = VelloScenePainter<'a, 'a>
    where
        Self: 'a;

    fn new(width: u32, height: u32) -> Self {
        Self::with_pipeline_cache(width, height, None)
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.buffer_renderer.resize(width, height);
    }

    fn reset(&mut self) {
        self.scene.reset();
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        cpu_buffer: &mut Vec<u8>,
    ) {
        let size = self.buffer_renderer.size();
        cpu_buffer.resize((size.width * size.height * 4) as usize, 0);
        self.render(draw_fn, cpu_buffer);
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        cpu_buffer: &mut [u8],
    ) {
        let size = self.buffer_renderer.size();
        self.filters.size = (size.width, size.height);
        self.filters.antialiasing = Some(self.antialiasing_method);
        let mut painter = VelloScenePainter {
            inner: &mut self.scene,
            renderer: Some(&mut self.vello_renderer),
            device_handle: Some(&self.buffer_renderer.device_handle),
            texture_handles: Some(&mut self.texture_handles),
            filters: Some(&mut self.filters),
            layers: Vec::new(),
        };
        draw_fn(&mut painter);
        painter.close_filter_layers();
        drop(painter);

        for handle in self.texture_handles.values() {
            self.vello_renderer.mark_override_image_dirty(handle);
        }

        self.vello_renderer
            .render_to_texture(
                self.buffer_renderer.device(),
                self.buffer_renderer.queue(),
                &self.scene,
                &self.buffer_renderer.target_texture_view(),
                &vello::RenderParams {
                    base_color: vello::peniko::Color::TRANSPARENT,
                    width: size.width,
                    height: size.height,
                    antialiasing_method: self.antialiasing_method,
                },
            )
            .expect("Got non-Send/Sync error from rendering");

        self.buffer_renderer.copy_texture_to_buffer(cpu_buffer);
        self.filters.end_frame(&mut self.vello_renderer);

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}
