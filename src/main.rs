use anyhow::Result;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent, DeviceEvent, DeviceId},
    event_loop::{ActiveEventLoop, EventLoop, ControlFlow},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use gpu_acceleration_test::{
    state::State,
    system_info::get_system_info,
    device_selector::prompt_device_selection,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
};

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    window_attributes: WindowAttributes,
    render_device: gpu_acceleration_test::render_device::RenderDevice,
    system_info: gpu_acceleration_test::system_info::SystemInfo,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Create window when the application is resumed
            let win = event_loop.create_window(self.window_attributes.clone()).unwrap();
            self.window = Some(Arc::new(win));
            
            // Create the state struct
            self.state = Some(pollster::block_on(State::new(
                self.window.as_ref().unwrap().clone(),
                self.render_device.clone(),
                self.system_info.clone(),
                wgpu::PresentMode::Fifo // Default to VSync enabled
            )).unwrap());
            
            // Log initial VSync status
            if let Some(state) = &self.state {
                log::info!("Application started with VSync {}: {} (Press 'V' to toggle)",
                    if state.system_info.vsync_enabled { "ON" } else { "OFF" },
                    if state.system_info.vsync_enabled {
                        "Frame rate limited to display refresh rate"
                    } else {
                        "Uncapped frame rate (may cause tearing)"
                    }
                );
            }
        }
    }
    
    fn window_event(&mut self, _event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        if let (Some(window), Some(state)) = (self.window.as_ref(), &mut self.state) {
            if window_id == window.id() {
                if !state.input(&event) {
                    match event {
                        WindowEvent::KeyboardInput { 
                            event: KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                            ..
                        } => {
                            // We'll handle exit in the about_to_wait method
                        },
                        WindowEvent::CloseRequested => {
                            // We'll handle exit in the about_to_wait method
                        },
                        WindowEvent::Resized(physical_size) => {
                            // Ensure resize is handled even if state.input doesn't handle it
                            state.resize(physical_size);
                        },
                        _ => {}
                    }
                }
            }
        }
    }
    
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.update();
            
            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    event_loop.exit();
                },
                Err(e) => eprintln!("{:?}", e),
            }
        }
    }
    
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, _event: DeviceEvent) {
        // Handle device events if needed
    }
}

fn main() -> Result<()> {
    // Initialize the logger
    env_logger::init();
    
    // Skip version logging since wgpu's API has changed
    log::info!("Starting application with wgpu and egui integration");
    
    // Create wgpu instance for system info
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    // Get system info for device selection
    let mut system_info = get_system_info(&instance);
    
    // Prompt user to select a render device if multiple are available
    let render_device = prompt_device_selection(&mut system_info);
    
    // Add logging for debugging render device configuration
    log::debug!("Render device selected: {:?}", render_device);
    
    // Create event loop
    let event_loop = EventLoop::new().unwrap();
    
    // Set control flow to Poll for continuous rendering
    event_loop.set_control_flow(ControlFlow::Poll);
    
    // Create window attributes
    let window_attributes = WindowAttributes::default()
        .with_title(WINDOW_TITLE)
        .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .with_resizable(true);
    
    // Create our application handler
    let mut app = App {
        window: None,
        state: None,
        window_attributes,
        render_device,
        system_info,
    };
    
    // Start the event loop
    event_loop.run_app(&mut app)?;

    Ok(())
}
