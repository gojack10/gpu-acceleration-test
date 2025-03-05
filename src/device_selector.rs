use std::io::{self, Write};
use crate::render_device::RenderDevice;
use crate::system_info::SystemInfo;
use log::info;

/// Prompts the user to select a rendering device from the available options
pub fn prompt_device_selection(system_info: &mut SystemInfo) -> RenderDevice {
    // If there's only one option (CPU), just return it without prompting
    if system_info.gpus.len() <= 1 {
        info!("No GPU devices detected, using CPU rendering");
        
        // Even with CPU rendering, prompt for vsync setting
        prompt_vsync_setting(system_info);
        
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
    
    // After device selection, prompt for vsync setting
    prompt_vsync_setting(system_info);
    
    render_device
}

/// Prompts the user to select vsync setting (on/off)
fn prompt_vsync_setting(system_info: &mut SystemInfo) {
    println!("\nVSync settings:");
    println!("---------------------------");
    println!("0: VSync On (limits framerate to monitor refresh rate)");
    println!("1: VSync Off (uncapped framerate, may cause tearing)");
    println!("\nPlease select VSync setting (0-1):");
    
    // Read user input
    let mut selection = String::new();
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut selection).unwrap();
    
    // Parse the selection
    match selection.trim().parse::<usize>() {
        Ok(0) => {
            system_info.vsync_enabled = true;
            info!("VSync enabled");
        },
        Ok(1) => {
            system_info.vsync_enabled = false;
            info!("VSync disabled");
        },
        _ => {
            // Invalid selection, default to vsync on
            println!("Invalid selection, defaulting to VSync On");
            system_info.vsync_enabled = true;
        }
    }
} 