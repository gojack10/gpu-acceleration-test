use anyhow::Result;
use std::sync::Arc;
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};
// No longer using WindowAttributes, we'll use Window directly

use gpu_acceleration_test::{
    render_device::RenderDevice,
    state::State,
    system_info::get_system_info,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
};

fn main() -> Result<()> {
    // Initialize the logger
    env_logger::init();
    
    // Skip version logging since wgpu's API has changed
    log::info!("Starting application with wgpu and egui integration");
    
    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    
    // Use WindowBuilder in winit 0.30
    let window = WindowBuilder::default()
        .with_title(WINDOW_TITLE)
        .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(&event_loop)
        .unwrap();
    
    let window = Arc::new(window);
    
    // Create wgpu instance for system info
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    // Get system info for device selection
    let system_info = get_system_info(&instance);
    
    // Select render device (default to first GPU)
    let render_device = if !system_info.gpus.is_empty() && system_info.gpus.len() > 1 {
        RenderDevice::GPU(system_info.gpus[1].clone())
    } else {
        RenderDevice::CPU
    };
    
    // Run the benchmark for the selected device
    run_benchmark(
        window.clone(),
        render_device,
        system_info,
        &event_loop,
    )?;
    
    Ok(())
}

fn run_benchmark(
    window: Arc<Window>, 
    render_device: RenderDevice, 
    system_info: gpu_acceleration_test::system_info::SystemInfo,
    event_loop: &EventLoop<()>
) -> Result<()> {
    // Create the state
    let mut state = pollster::block_on(State::new(window.clone(), render_device, system_info))?;
    
    // Start the event loop
    event_loop.run_app(move |event, elwt| {
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        WindowEvent::KeyboardInput { 
                            event: KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                            ..
                        } => {
                            elwt.exit();
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;
    
    Ok(())
}
