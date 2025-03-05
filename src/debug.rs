use egui::{Color32, Align2, FontFamily, FontDefinitions, FontData};
use winit::{
    event::WindowEvent,
    keyboard::{KeyCode, PhysicalKey},
    event::ElementState,
};
use crate::render_device::RenderDevice;
use crate::system_info::SystemInfo;
use sysinfo::{System, RefreshKind, CpuRefreshKind};
use log::{info, debug};
use std::time::Instant;
use std::collections::BTreeMap;


// FONT SIZE CONFIGURATION
const DEBUG_FONT_SIZE: f32 = 22.0;

pub struct DebugState {
    pub enabled: bool,
    pub last_cpu_usage_check: Instant,
    pub cpu_usage: f32,
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

    pub fn update(&mut self) {
        // CPU usage check - update every second
        let now = Instant::now();
        if now.duration_since(self.last_cpu_usage_check).as_secs_f32() >= 1.0 {
            self.system.refresh_cpu();
            self.cpu_usage = self.system.global_cpu_info().cpu_usage();
            self.last_cpu_usage_check = now;
        }
    }

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

    fn create_font_definitions(&self) -> FontDefinitions {
        let mut font_data = BTreeMap::new();
        let mut families = BTreeMap::new();
        
        // Try to load the font from the assets directory
        match std::fs::read("assets/fonts/JetBrainsMono-Regular.ttf") {
            Ok(font_data_bytes) => {
                // Add the font to the font data map
                font_data.insert("jetbrains_mono".to_owned(), FontData::from_owned(font_data_bytes));
                
                // Add the font to the proportional and monospace families
                families.insert(
                    FontFamily::Proportional,
                    vec!["jetbrains_mono".to_owned()],
                );
                families.insert(
                    FontFamily::Monospace,
                    vec!["jetbrains_mono".to_owned()],
                );
            }
            Err(err) => {
                debug!("Failed to load font: {:?}", err);
                
                // Fall back to the default font
                families.insert(
                    FontFamily::Proportional,
                    vec![egui::FontFamily::Name("sans-serif".into()).to_owned().to_string()],
                );
                families.insert(
                    FontFamily::Monospace,
                    vec![egui::FontFamily::Name("monospace".into()).to_owned().to_string()],
                );
            }
        }
        
        let mut font_defs = FontDefinitions {
            font_data,
            families,
            ..Default::default()
        };
        
        // Make sure we have font sizes that work with different DPI scales
        // This is crucial for the "No font size matching 1 pixels per point" error
        for (_font_name, font_info) in font_defs.font_data.iter_mut() {
            // Set tweak values to accommodate different DPI scales
            font_info.tweak.scale = 1.0;
            font_info.tweak.y_offset_factor = 0.0;
            
            // Note: oversample_height_in_pixels is not available in this version of egui
            // Adjust this part based on your egui version
        }
        
        font_defs
    }

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

    pub fn render(
        &mut self, 
        ctx: &egui::Context, 
        fps: f32, 
        _render_device: &RenderDevice,
        _system_info: &SystemInfo,
        window_size: (u32, u32),
        cube_rotation: f32,
        cube_position: glam::Vec3,
        cube_velocity: glam::Vec3,
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
                    
                    match _render_device {
                        RenderDevice::CPU => {
                            ui.colored_label(Color32::YELLOW, "CPU RENDERING (SOFTWARE)");
                            if !_system_info.cpus.is_empty() {
                                // Use label with wrapping for potentially long CPU names
                                let cpu_text = format!("CPU: {}", _system_info.cpus[_system_info.selected_cpu]);
                                ui.add(egui::Label::new(egui::RichText::new(cpu_text).color(Color32::WHITE)).wrap());
                            }
                        },
                        RenderDevice::GPU(name) => {
                            ui.colored_label(Color32::GREEN, "HARDWARE ACCELERATED RENDERING");
                            
                            // Use label with wrapping for potentially long GPU names
                            let gpu_text = format!("GPU: {}", name);
                            ui.add(egui::Label::new(egui::RichText::new(gpu_text).color(Color32::WHITE)).wrap());
                            
                            // Extract backend info if available
                            if name.contains(" - ") {
                                let parts: Vec<&str> = name.split(" - ").collect();
                                if parts.len() > 1 {
                                    let backend_text = format!("API Backend: {}", parts[1]);
                                    ui.add(egui::Label::new(egui::RichText::new(backend_text).color(Color32::LIGHT_GRAY)).wrap());
                                }
                            }
                        }
                    }
                    
                    // Add spacing between render device info and cube info
                    ui.add_space(10.0);
                    
                    // Cube information
                    ui.colored_label(Color32::WHITE, format!("Window Size: {}x{}", window_size.0, window_size.1));
                    ui.colored_label(Color32::WHITE, format!("Cube Rotation: {:.2} rad", cube_rotation));
                    
                    // Use labels with wrapping for position and velocity
                    let pos_text = format!("Cube Position: ({:.2}, {:.2}, {:.2})", 
                        cube_position.x, cube_position.y, cube_position.z);
                    ui.add(egui::Label::new(egui::RichText::new(pos_text).color(Color32::WHITE)).wrap());
                    
                    let vel_text = format!("Cube Velocity: ({:.2}, {:.2}, {:.2})", 
                        cube_velocity.x, cube_velocity.y, cube_velocity.z);
                    ui.add(egui::Label::new(egui::RichText::new(vel_text).color(Color32::WHITE)).wrap());
                });
            });
        
        // Restore original visuals
        ctx.set_visuals(ctx_visuals);
    }
} 