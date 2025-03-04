use sysinfo::{Cpu, CpuRefreshKind, RefreshKind, System};
use crate::render_device::RenderDevice;

#[derive(Clone)]
pub struct SystemInfo {
    pub cpus: Vec<String>,
    pub gpus: Vec<String>,
    pub selected_cpu: usize,
    pub selected_gpu: usize,
}

pub fn get_system_info(instance: &wgpu::Instance) -> SystemInfo {
    // Get CPU info
    let sys = System::new_with_specifics(
        RefreshKind::new().with_cpu(CpuRefreshKind::everything()),
    );
    let cpus = sys
        .cpus()
        .iter()
        .map(|cpu| format!("{} ({})", cpu.brand(), cpu.name()))
        .collect::<Vec<String>>();

    // Get GPU info
    let adapters = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .collect::<Vec<_>>();
    
    let mut gpus = Vec::new();
    
    // Add CPU (software rendering) as first option
    gpus.push("CPU (Software Rendering)".to_string());
    
    // Add all GPUs with their info
    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::Cpu {
            gpus.push(format!("{} - {}", info.name, info.backend.to_str()));
        }
    }

    SystemInfo {
        cpus,
        gpus,
        selected_cpu: 0,
        selected_gpu: 0,
    }
}

pub fn select_render_device(system_info: &SystemInfo) -> RenderDevice {
    if system_info.selected_gpu == 0 {
        RenderDevice::CPU
    } else {
        RenderDevice::GPU(system_info.gpus[system_info.selected_gpu].clone())
    }
} 