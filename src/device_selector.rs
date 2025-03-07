use std::io::{self, Write};
use crate::render_device::RenderDevice;
use crate::system_info::SystemInfo;
use log::info;

/// Prompts the user to select a rendering device from the available options
pub fn prompt_device_selection(system_info: &mut SystemInfo) -> RenderDevice {
    // If there's only one option (CPU), just return it without prompting
    if system_info.gpus.len() <= 1 {
        info!("No GPU devices detected, using CPU rendering");
        
        // Set vsync to enabled by default
        system_info.vsync_enabled = true;
        info!("VSync enabled by default (press V to toggle during runtime)");
        
        return RenderDevice::CPU;
    }

    println!("\nAvailable rendering devices:");
    println!("---------------------------");
    
    // Display all available devices
    for (i, gpu) in system_info.gpus.iter().enumerate() {
        println!("{}: {}", i, gpu);
    }
    
    println!("\nPlease select a rendering device (0-{}):", system_info.gpus.len() - 1);
    
    // Read user input
    let mut selection = String::new();
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut selection).unwrap();
    
    // Parse the selection
    let render_device = match selection.trim().parse::<usize>() {
        Ok(index) if index < system_info.gpus.len() => {
            // Update the selected GPU in the system info
            system_info.selected_gpu = index;
            
            // Return the appropriate render device
            if index == 0 {
                info!("Selected CPU rendering");
                RenderDevice::CPU
            } else {
                info!("Selected GPU: {}", system_info.gpus[index]);
                RenderDevice::GPU(system_info.gpus[index].clone())
            }
        },
        _ => {
            // Invalid selection, default to CPU
            println!("Invalid selection, defaulting to CPU rendering");
            system_info.selected_gpu = 0;
            RenderDevice::CPU
        }
    };
    
    // Set vsync to enabled by default
    system_info.vsync_enabled = true;
    info!("VSync enabled by default (press V to toggle during runtime)");
    
    render_device
} 