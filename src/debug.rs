#[cfg(not(target_arch = "wasm32"))]
use egui::{Color32, Align2, FontFamily, FontDefinitions, FontData};
use winit::{
    event::WindowEvent,
    keyboard::{KeyCode, PhysicalKey},
    event::ElementState,
};
use crate::render_device::RenderDevice;
use crate::system_info::SystemInfo;
#[cfg(not(target_arch = "wasm32"))]
use sysinfo::{System, RefreshKind, CpuRefreshKind};
use log::{info, debug};
use std::time::Instant;
use std::collections::BTreeMap;
use glam::Vec3;


// FONT SIZE CONFIGURATION
const DEBUG_FONT_SIZE: f32 = 22.0;

pub struct DebugState {
    pub enabled: bool,
    pub last_cpu_usage_check: Instant,
    pub cpu_usage: f32,
    #[cfg(not(target_arch = "wasm32"))]
    pub system: System,
    pub font_loaded: bool,
    pub font_load_attempted: bool,
    pub show_debug: bool,
    pub fps: f32,
    pub frame_time: f32,
    pub last_scale_factor: f32,
}

impl Default for DebugState {
    fn default() -> Self {
        Self {
            enabled: false,
            last_cpu_usage_check: Instant::now(),
            cpu_usage: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            system: System::new_with_specifics(
                RefreshKind::new().with_cpu(CpuRefreshKind::everything()),
            ),
            font_loaded: false,
            font_load_attempted: false,
            show_debug: true,
            fps: 0.0,
            frame_time: 0.0,
            last_scale_factor: 0.0,
        }
    }
}

impl DebugState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn toggle(&mut self) {
        // Toggle debug mode for 3D axes only
        // Text display will always be visible regardless of this setting
        self.enabled = !self.enabled;
        
        // Keep show_debug always true to ensure text is always displayed
        self.show_debug = true;
        
        info!("Debug axes toggled: {}", self.enabled);
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { 
                event: winit::event::KeyEvent {
                    physical_key: PhysicalKey::Code(key_code),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                if *key_code == KeyCode::KeyD {
                    self.toggle();
                    return true;
                }
            },
            _ => {},
        }
        
        false
    }

    pub fn update(&mut self, system_info: Option<&mut SystemInfo>) {
        // CPU usage check - update every second
        let now = Instant::now();
        if now.duration_since(self.last_cpu_usage_check).as_secs_f32() >= 1.0 {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.system.refresh_cpu();
                self.cpu_usage = self.system.global_cpu_info().cpu_usage();
            }
            
            #[cfg(target_arch = "wasm32")]
            {
                // For WASM, just use a placeholder value
                self.cpu_usage = 5.0; // Placeholder value
            }
            
            self.last_cpu_usage_check = now;
            
            // Update system info if provided
            if let Some(sys_info) = system_info {
                // Update CPU usage
                sys_info.cpu_usage = self.cpu_usage;
                
                // In a real application, you would update GPU metrics here
                // For demonstration, we'll simulate some values
                if sys_info.selected_gpu > 0 {
                    // Simulate GPU utilization (random value between 30-90%)
                    sys_info.gpu_utilization = 30.0 + (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() % 60) as f32;
                    
                    // Simulate VRAM usage (increasing over time, then resetting)
                    let cycle = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() % 30) as u64;
                    
                    // Total VRAM (simulated as 8GB)
                    sys_info.vram_total = 8 * 1024 * 1024 * 1024;
                    
                    // Used VRAM (cycles from 1GB to 6GB)
                    sys_info.vram_used = (1 + cycle / 5) * 1024 * 1024 * 1024;
                    if sys_info.vram_used > sys_info.vram_total {
                        sys_info.vram_used = sys_info.vram_total;
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_fonts(&mut self, ctx: &egui::Context) {
        if self.font_load_attempted && self.font_loaded {
            return;
        }
        
        self.font_load_attempted = true;
        
        let font_definitions = self.create_font_definitions();
        ctx.set_fonts(font_definitions);
        
        self.configure_text_styles(ctx);
        
        self.font_loaded = true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn configure_text_styles(&self, ctx: &egui::Context) {
        let pixels_per_point = ctx.pixels_per_point();
        
        // Calculate font sizes based on scale factor to ensure readability
        // Base sizes are for 1.0 scale factor
        let base_heading_size = 24.0;
        let base_body_size = 16.0; 
        let base_mono_size = 14.0;
        let base_button_size = 16.0;
        let base_small_size = 12.0;
        
        // Apply scale factor with a minimum to avoid "No font size matching" errors
        // Use a higher minimum to completely avoid the error
        let scale_multiplier = pixels_per_point.max(1.01); // Ensure we never go below 1.01 for scale
        
        // Use a minimum absolute font size to avoid the "No font size matching" error
        // Fonts must be at least 1.0 * pixels_per_point in actual rendered size
        let min_font_size = pixels_per_point.max(1.0);
        
        let heading_size = (base_heading_size * scale_multiplier).max(min_font_size);
        let body_size = (base_body_size * scale_multiplier).max(min_font_size);
        let mono_size = (base_mono_size * scale_multiplier).max(min_font_size);
        let button_size = (base_button_size * scale_multiplier).max(min_font_size);
        let small_size = (base_small_size * scale_multiplier).max(min_font_size);
        
        // Log the available text styles
        let mut style = (*ctx.style()).clone();
        
        // Create a map of text styles to font ids
        style.text_styles = [
            (egui::TextStyle::Heading, egui::FontId::new(heading_size, egui::FontFamily::Proportional)),
            (egui::TextStyle::Body, egui::FontId::new(body_size, egui::FontFamily::Proportional)),
            (egui::TextStyle::Monospace, egui::FontId::new(mono_size, egui::FontFamily::Monospace)),
            (egui::TextStyle::Button, egui::FontId::new(button_size, egui::FontFamily::Proportional)),
            (egui::TextStyle::Small, egui::FontId::new(small_size, egui::FontFamily::Proportional)),
        ].into();
        
        // Note: style.interaction.font_size doesn't exist in this egui version
        // Skip this step or adjust for your egui version
        
        // Set the style to the context
        ctx.set_style(style);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(
        &mut self,
        ctx: &egui::Context,
        fps: f32,
        render_device: &RenderDevice,
        system_info: &SystemInfo,
        window_size: (u32, u32),
        rotation: f32,
        position: Vec3,
        velocity: Vec3,
    ) {
        self.fps = fps;
        self.frame_time = 1000.0 / fps;
        
        let mut current_scale_factor = ctx.pixels_per_point();
        
        // Ensure scale factor is never below 1.0, which causes the "No font matching" error
        if current_scale_factor < 1.0 {
            ctx.set_pixels_per_point(1.0);
            current_scale_factor = 1.0;
            ctx.request_repaint();
        }
        
        // Check if scale factor has changed
        if (current_scale_factor - self.last_scale_factor).abs() > 0.001 {
            // Reset font flags to force reload
            self.font_loaded = false;
            self.font_load_attempted = false;
            
            // Force a repaint to ensure changes are applied
            ctx.request_repaint();
            
            self.last_scale_factor = current_scale_factor;
        }
        
        // Always ensure fonts are loaded before rendering
        if !self.font_loaded {
            self.load_fonts(ctx);
        }
        
        // Create a consistent font using the global font size variable
        let font_id = egui::FontId::new(DEBUG_FONT_SIZE, egui::FontFamily::Proportional);
        
        // Use transparent background for all windows
        let ctx_visuals = ctx.style().visuals.clone();
        let mut transparent_visuals = egui::style::Visuals::dark();
        transparent_visuals.window_fill = Color32::from_rgba_premultiplied(0, 0, 0, 0); // Fully transparent
        
        // Set shadow to none by setting offset to zero and color to transparent
        transparent_visuals.window_shadow.offset = egui::Vec2::ZERO;
        transparent_visuals.window_shadow.blur = 0.0;
        transparent_visuals.window_shadow.spread = 0.0;
        transparent_visuals.window_shadow.color = Color32::TRANSPARENT;
        
        transparent_visuals.widgets.noninteractive.bg_fill = Color32::TRANSPARENT; // Transparent widget backgrounds
        transparent_visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
        transparent_visuals.widgets.hovered.bg_fill = Color32::from_rgba_premultiplied(255, 255, 255, 10); // Very slight highlight
        transparent_visuals.widgets.active.bg_fill = Color32::from_rgba_premultiplied(255, 255, 255, 20);
        ctx.set_visuals(transparent_visuals);
        
        // Calculate a reasonable max width based on window size
        // Use at most 40% of window width, but not less than 200px and not more than 400px
        let max_width = ((window_size.0 as f32) * 0.4).max(200.0).min(400.0);
        
        // Single left-aligned window with all information
        egui::Window::new("Stats")
            .title_bar(false)
            .resizable(true)  // Allow resizing
            .default_width(max_width) // Set default width
            .min_width(150.0) // Set minimum width to prevent too narrow window
            .anchor(Align2::LEFT_TOP, egui::Vec2::new(10.0, 10.0))
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                // Apply consistent styling
                ui.style_mut().override_font_id = Some(font_id.clone());
                ui.style_mut().visuals.override_text_color = Some(Color32::WHITE);
                
                // Set text wrapping
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                
                // Use vertical layout with auto-width
                ui.vertical(|ui| {
                    ui.set_max_width(max_width);
                    
                    // Performance stats
                    ui.colored_label(Color32::WHITE, format!("FPS: {:.1}", self.fps));
                    ui.colored_label(Color32::WHITE, format!("Frame Time: {:.2} ms", self.frame_time));
                    
                    // Add render device information with much more detail
                    ui.add_space(10.0);
                    
                    match render_device {
                        RenderDevice::CPU => {
                            ui.colored_label(Color32::YELLOW, "CPU RENDERING (SOFTWARE)");
                            
                            // CPU Model and Manufacturer
                            let cpu_model_text = format!("CPU: {}", system_info.cpu_model);
                            ui.add(egui::Label::new(egui::RichText::new(cpu_model_text).color(Color32::WHITE)).wrap());
                            
                            // CPU Architecture
                            ui.colored_label(Color32::LIGHT_GRAY, format!("Architecture: {}", system_info.cpu_architecture));
                            
                            // Software Renderer
                            ui.colored_label(Color32::LIGHT_GRAY, "Renderer: Software (CPU-based)");
                            
                            // Rendering Thread Count
                            ui.colored_label(Color32::LIGHT_GRAY, format!("Rendering Threads: {}", system_info.rendering_threads));
                            
                            // CPU Usage
                            ui.colored_label(Color32::LIGHT_GRAY, format!("CPU Usage: {:.1}%", system_info.cpu_usage));
                            
                            // VSync Status and Toggle Key
                            ui.add_space(5.0);
                            let vsync_status = if system_info.vsync_enabled {
                                egui::RichText::new("VSync: ON").color(Color32::GREEN)
                            } else {
                                egui::RichText::new("VSync: OFF").color(Color32::YELLOW)
                            };
                            ui.label(vsync_status);
                            ui.colored_label(Color32::LIGHT_GRAY, "Press 'V' to toggle VSync");
                        },
                        RenderDevice::GPU(_name) => {
                            ui.colored_label(Color32::GREEN, "HARDWARE ACCELERATED RENDERING");
                            
                            // GPU Model and Manufacturer
                            let gpu_text = format!("GPU: {}", system_info.gpu_model);
                            ui.add(egui::Label::new(egui::RichText::new(gpu_text).color(Color32::WHITE)).wrap());
                            
                            // GPU Architecture
                            ui.colored_label(Color32::LIGHT_GRAY, format!("Architecture: {}", system_info.gpu_architecture));
                            
                            // GPU Utilization
                            ui.colored_label(Color32::LIGHT_GRAY, format!("GPU Utilization: {:.1}%", system_info.gpu_utilization));
                            
                            // VRAM Usage
                            let vram_mb_used = system_info.vram_used / (1024 * 1024);
                            let vram_mb_total = system_info.vram_total / (1024 * 1024);
                            
                            if system_info.vram_total > 0 {
                                ui.colored_label(Color32::LIGHT_GRAY, format!("VRAM: {}MB / {}MB", vram_mb_used, vram_mb_total));
                            } else {
                                ui.colored_label(Color32::LIGHT_GRAY, "VRAM: Unknown");
                            }
                            
                            // API Backend
                            ui.colored_label(Color32::LIGHT_GRAY, format!("API Backend: {}", system_info.api_backend));
                            
                            // VSync Status and Toggle Key
                            ui.add_space(5.0);
                            let vsync_status = if system_info.vsync_enabled {
                                egui::RichText::new("VSync: ON").color(Color32::GREEN)
                            } else {
                                egui::RichText::new("VSync: OFF").color(Color32::YELLOW)
                            };
                            ui.label(vsync_status);
                            ui.colored_label(Color32::LIGHT_GRAY, "Press 'V' to toggle VSync");
                        }
                    }
                    
                    // Add spacing between render device info and cube info
                    ui.add_space(10.0);
                    
                    // Cube information
                    ui.colored_label(Color32::WHITE, format!("Window Size: {}x{}", window_size.0, window_size.1));
                    ui.colored_label(Color32::WHITE, format!("Cube Rotation: {:.2} rad", rotation));
                    
                    // Use labels with wrapping for position and velocity
                    let pos_text = format!("Cube Position: ({:.2}, {:.2}, {:.2})", 
                        position.x, position.y, position.z);
                    ui.add(egui::Label::new(egui::RichText::new(pos_text).color(Color32::WHITE)).wrap());
                    
                    let vel_text = format!("Cube Velocity: ({:.2}, {:.2}, {:.2})", 
                        velocity.x, velocity.y, velocity.z);
                    ui.add(egui::Label::new(egui::RichText::new(vel_text).color(Color32::WHITE)).wrap());
                });
            });
        
        // Restore original visuals
        ctx.set_visuals(ctx_visuals);
    }

    #[cfg(target_arch = "wasm32")]
    pub fn render(
        &mut self,
        _ctx: &(),
        _fps: f32,
        _render_device: &RenderDevice,
        _system_info: &SystemInfo,
        _window_size: (u32, u32),
        _rotation: f32,
        _position: Vec3,
        _velocity: Vec3,
    ) {
        // No-op implementation for WebAssembly
    }
} 