use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, Window, Document};
use std::sync::Arc;
use log::{Level, LevelFilter};
use std::panic;
use crate::state::State;
use futures::executor::block_on;
use crate::render_device::RenderDevice;
use crate::system_info::SystemInfo;

fn init_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    log::info!("Panic hook initialized");
}

#[wasm_bindgen]
pub fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

#[wasm_bindgen]
pub fn init_logging() {
    // Initialize console_log
    console_log::init_with_level(Level::Info).expect("Failed to initialize logger");
    
    // Set up custom logger for wgpu
    log::set_max_level(LevelFilter::Info);
    log::set_logger(&WebLogger).expect("Failed to set logger");
    
    // Initialize panic hook
    init_panic_hook();
    
    log::info!("Logging initialized");
}

struct WebLogger;

impl log::Log for WebLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= Level::Info
    }
    
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let msg = format!("[{}] {}", record.level(), record.args());
            web_sys::console::log_1(&JsValue::from_str(&msg));
        }
    }
    
    fn flush(&self) {}
}

#[wasm_bindgen]
pub struct WebRenderer {
    state: Option<State>,
    animation_frame_id: Option<i32>,
    render_loop_closure: Option<Closure<dyn FnMut()>>,
}

#[wasm_bindgen]
impl WebRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log::info!("Creating WebRenderer");
        
        Self {
            state: None,
            animation_frame_id: None,
            render_loop_closure: None,
        }
    }
    
    #[wasm_bindgen]
    pub async fn initialize(&mut self, canvas_id: &str) -> Result<(), JsValue> {
        log::info!("Initializing WebRenderer with canvas ID: {}", canvas_id);
        
        // Get the canvas element
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window found"))?;
        let document = window.document().ok_or_else(|| JsValue::from_str("No document found"))?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str(&format!("Canvas with ID '{}' not found", canvas_id)))?
            .dyn_into::<HtmlCanvasElement>()?;
        
        // Create a simple system info for web
        let mut system_info = SystemInfo {
            cpu_info: "Web Browser".to_string(),
            cpu_cores: 1,
            cpu_threads: 1,
            memory_total: 0,
            memory_available: 0,
            gpu_info: vec!["WebGL".to_string()],
            vsync_enabled: true,
            selected_gpu_index: 0,
        };
        
        // Create a simple render device for web
        let render_device = RenderDevice {
            name: "WebGL".to_string(),
            vendor: "Browser".to_string(),
            device_type: wgpu::DeviceType::Other,
            backend: wgpu::Backend::WebGpu,
        };
        
        // Create the state
        let window_arc = Arc::new(canvas.clone());
        self.state = Some(
            State::new(
                window_arc,
                render_device,
                system_info,
                wgpu::PresentMode::Fifo, // Use VSync by default on web
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create state: {:?}", e)))?
        );
        
        log::info!("WebRenderer initialized successfully");
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn start_render_loop(&mut self) -> Result<(), JsValue> {
        log::info!("Starting render loop");
        
        // Get the window
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window found"))?;
        
        // Make sure we're not already running
        if self.animation_frame_id.is_some() {
            log::warn!("Render loop is already running");
            return Ok(());
        }
        
        // Get a reference to our state
        let state_ref = self.state.as_mut().ok_or_else(|| JsValue::from_str("State not initialized"))?;
        
        // Create a closure that will be called on each animation frame
        let f = Closure::new(move || {
            if let Some(state) = self.state.as_mut() {
                state.update();
                
                match state.render() {
                    Ok(_) => {},
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of memory error");
                        return;
                    },
                    Err(e) => {
                        log::error!("Render error: {:?}", e);
                    }
                }
                
                // Schedule the next frame
                let window = web_sys::window().expect("No window found");
                self.animation_frame_id = Some(window
                    .request_animation_frame(self.render_loop_closure.as_ref().unwrap().as_ref().unchecked_ref())
                    .expect("Failed to request animation frame"));
            }
        });
        
        // Store the closure
        self.render_loop_closure = Some(f);
        
        // Start the animation loop
        self.animation_frame_id = Some(window
            .request_animation_frame(self.render_loop_closure.as_ref().unwrap().as_ref().unchecked_ref())
            .expect("Failed to request animation frame"));
        
        log::info!("Render loop started");
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn stop_render_loop(&mut self) {
        log::info!("Stopping render loop");
        
        if let Some(id) = self.animation_frame_id {
            if let Some(window) = web_sys::window() {
                window.cancel_animation_frame(id).expect("Failed to cancel animation frame");
                self.animation_frame_id = None;
                log::info!("Render loop stopped");
            }
        }
        
        // Drop the closure to free memory
        self.render_loop_closure = None;
    }
}

#[wasm_bindgen]
pub fn init() {
    init_logging();
    log::info!("WebGL renderer initialized");
}

#[wasm_bindgen]
pub async fn start(canvas: HtmlCanvasElement) -> Result<(), JsValue> {
    log::info!("Starting WebGL renderer");
    
    let mut renderer = WebRenderer::new();
    renderer.initialize(canvas.id().as_str()).await?;
    renderer.start_render_loop()?;
    
    Ok(())
} 