#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
        "aarch64".to_string()
    } else if cfg!(target_arch = "arm") {
        "arm".to_string()
    } else {
        "unknown".to_string()
    };

    // Get CPU usage
    let cpu_usage = sys.global_cpu_info().cpu_usage();

    // Get number of CPU cores and threads
    let num_cores = num_cpus::get_physical();
    let num_threads = num_cpus::get();

    // Get memory info
    let total_memory = sys.total_memory();
    let available_memory = sys.available_memory();

    // Get GPU info
    let adapters = instance
        .enumerate_adapters(wgpu::Backends::all())
        .map(|adapter| adapter.get_info().name.clone())
        .collect::<Vec<String>>();

    // Create the system info
    SystemInfo {
        cpus,
        gpus: if adapters.is_empty() { vec!["Unknown GPU".to_string()] } else { adapters },
        selected_cpu: 0,
        selected_gpu: 0,
        cpu_model: sys.global_cpu_info().brand().to_string(),
        cpu_architecture,
        cpu_usage,
        rendering_threads: num_threads,
        gpu_model: "Unknown".to_string(),
        gpu_architecture: "Unknown".to_string(),
        gpu_utilization: 0.0,
        vram_used: 0,
        vram_total: 0,
        api_backend: "wgpu".to_string(),
        vsync_enabled: true,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn get_system_info(_instance: &wgpu::Instance) -> SystemInfo {
    // Create a simple system info for web
    SystemInfo {
        cpus: vec!["Web Browser CPU".to_string()],
        gpus: vec!["WebGL".to_string()],
        selected_cpu: 0,
        selected_gpu: 0,
        cpu_model: "Web Browser CPU".to_string(),
        cpu_architecture: "wasm32".to_string(),
        cpu_usage: 0.0,
        rendering_threads: 1,
        gpu_model: "WebGL".to_string(),
        gpu_architecture: "WebGL".to_string(),
        gpu_utilization: 0.0,
        vram_used: 0,
        vram_total: 0,
        api_backend: "WebGL".to_string(),
        vsync_enabled: true,
    }
}

pub fn select_render_device(system_info: &SystemInfo) -> RenderDevice {
    if system_info.selected_gpu == 0 {
        RenderDevice::CPU
    } else {
        RenderDevice::GPU(system_info.gpus[system_info.selected_gpu].clone())
    }
} 