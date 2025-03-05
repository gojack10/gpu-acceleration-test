use sysinfo::{CpuRefreshKind, RefreshKind, System};
use crate::render_device::RenderDevice;

#[derive(Clone)]
pub struct SystemInfo {
    pub cpus: Vec<String>,
    pub gpus: Vec<String>,
    pub selected_cpu: usize,
    pub selected_gpu: usize,
    // Additional CPU information
    pub cpu_model: String,
    pub cpu_architecture: String,
    pub cpu_usage: f32,
    pub rendering_threads: usize,
    // Additional GPU information
    pub gpu_model: String,
    pub gpu_architecture: String,
    pub gpu_utilization: f32,
    pub vram_used: u64,
    pub vram_total: u64,
    pub api_backend: String,
    pub vsync_enabled: bool,
}

impl SystemInfo {
    pub fn get_render_device(&self) -> RenderDevice {
        if self.selected_gpu == 0 || self.gpus.is_empty() {
            RenderDevice::CPU
        } else {
            RenderDevice::GPU(self.gpus[self.selected_gpu].clone())
        }
    }
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

    // Get CPU architecture
    let cpu_architecture = if cfg!(target_arch = "x86_64") {
        "x86_64".to_string()
    } else if cfg!(target_arch = "x86") {
        "x86".to_string()
    } else if cfg!(target_arch = "aarch64") {
        "ARM64".to_string()
    } else if cfg!(target_arch = "arm") {
        "ARM".to_string()
    } else {
        "Unknown".to_string()
    };

    // Get CPU model
    let cpu_model = if !cpus.is_empty() {
        cpus[0].clone()
    } else {
        "Unknown CPU".to_string()
    };

    // Calculate CPU usage (average across all cores)
    let cpu_usage = if !sys.cpus().is_empty() {
        sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32
    } else {
        0.0
    };

    // Determine rendering threads (use number of logical cores)
    let rendering_threads = num_cpus::get();

    // Get GPU info
    let adapters = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .collect::<Vec<_>>();
    
    let mut gpus = Vec::new();
    
    // Add CPU (software rendering) as first option
    gpus.push("CPU (Software Rendering)".to_string());
    
    // Default GPU information
    let mut gpu_model = "N/A".to_string();
    let mut gpu_architecture = "N/A".to_string();
    let mut api_backend = "Software".to_string();
    
    // Add all GPUs with their info
    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::Cpu {
            let gpu_name = format!("{} - {}", info.name, info.backend.to_str());
            gpus.push(gpu_name.clone());
            
            // Store first GPU info as default
            if gpu_model == "N/A" {
                gpu_model = info.name.clone();
                // Attempt to determine architecture from name (simplified)
                if info.name.contains("NVIDIA") {
                    if info.name.contains("RTX") {
                        if info.name.contains("40") {
                            gpu_architecture = "Ada Lovelace".to_string();
                        } else if info.name.contains("30") {
                            gpu_architecture = "Ampere".to_string();
                        } else if info.name.contains("20") {
                            gpu_architecture = "Turing".to_string();
                        } else {
                            gpu_architecture = "NVIDIA RTX".to_string();
                        }
                    } else if info.name.contains("GTX") {
                        gpu_architecture = "NVIDIA GTX".to_string();
                    } else {
                        gpu_architecture = "NVIDIA".to_string();
                    }
                } else if info.name.contains("AMD") || info.name.contains("Radeon") {
                    if info.name.contains("RX 7") {
                        gpu_architecture = "RDNA 3".to_string();
                    } else if info.name.contains("RX 6") {
                        gpu_architecture = "RDNA 2".to_string();
                    } else if info.name.contains("RX 5") {
                        gpu_architecture = "RDNA".to_string();
                    } else {
                        gpu_architecture = "AMD Radeon".to_string();
                    }
                } else if info.name.contains("Intel") {
                    if info.name.contains("Arc") {
                        gpu_architecture = "Intel Xe HPG".to_string();
                    } else if info.name.contains("Iris") {
                        gpu_architecture = "Intel Xe".to_string();
                    } else {
                        gpu_architecture = "Intel".to_string();
                    }
                } else if info.name.contains("Apple") {
                    gpu_architecture = "Apple Silicon".to_string();
                }
                
                api_backend = info.backend.to_str().to_string();
            }
        }
    }

    SystemInfo {
        cpus,
        gpus,
        selected_cpu: 0,
        selected_gpu: 0,
        cpu_model,
        cpu_architecture,
        cpu_usage,
        rendering_threads,
        gpu_model,
        gpu_architecture,
        gpu_utilization: 0.0, // Will need to be updated during runtime
        vram_used: 0,         // Will need to be updated during runtime
        vram_total: 0,        // Will need to be updated during runtime
        api_backend,
        vsync_enabled: false,
    }
}

pub fn select_render_device(system_info: &SystemInfo) -> RenderDevice {
    if system_info.selected_gpu == 0 {
        RenderDevice::CPU
    } else {
        RenderDevice::GPU(system_info.gpus[system_info.selected_gpu].clone())
    }
} 