use anyhow::Result;
use glam::{Mat4, Quat, Vec3};
use std::{sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    window::Window,
    keyboard::{KeyCode, PhysicalKey},
    event::ElementState,
};
use egui;
use log::debug;

// For egui_wgpu_backend
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
// For egui_winit
use egui_winit::State as EguiState;

use crate::{
    texture::{Texture, create_depth_texture, create_texture_from_bytes},
    vertex::{Vertex, Uniforms, create_cube_vertices},
    render_device::RenderDevice,
    system_info::SystemInfo,
    debug::DebugState,
};

pub struct State {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    pub window: Arc<Window>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub textures: Vec<Texture>,
    pub texture_bind_group: wgpu::BindGroup,
    pub depth_texture: Texture,
    pub egui_state: EguiState,
    pub egui_renderer: RenderPass,
    pub frame_times: Vec<f32>,
    pub fps: f32,
    pub rotation: f32,
    pub last_update: Instant,
    pub render_device: RenderDevice,
    pub system_info: SystemInfo,
    pub debug_state: DebugState,
    pub debug_axis_buffer: Option<wgpu::Buffer>,
    pub debug_axis_pipeline: Option<wgpu::RenderPipeline>,
    pub world_axis_buffer: Option<wgpu::Buffer>,
    pub world_axis_bind_group: Option<wgpu::BindGroup>,
    pub world_axis_uniform_buffer: Option<wgpu::Buffer>,
    pub debug_axis_positions: Vec<Vec3>,
    pub world_axis_positions: Vec<Vec3>,
    pub cube_position: Vec3,
    pub cube_velocity: Vec3,
    pub vsync_enabled: bool, // Track VSync state for manual frame limiting
}

impl State {
    pub async fn new(
        window: Arc<Window>, 
        render_device: RenderDevice, 
        mut system_info: SystemInfo,
        present_mode: wgpu::PresentMode,
    ) -> Result<Self> {
        let size = window.inner_size();

        // Create the instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        // Create the surface
        let surface = instance.create_surface(window.clone()).unwrap();
        
        // Select the appropriate adapter based on rendering device
        let adapter = match &render_device {
            RenderDevice::CPU => {
                instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .into_iter()
                    .filter(|adapter| {
                        // Find CPU adapter (software rendering)
                        adapter.get_info().device_type == wgpu::DeviceType::Cpu
                    })
                    .next()
                    .or_else(|| {
                        // Fallback to any adapter if no CPU found
                        instance.enumerate_adapters(wgpu::Backends::all()).into_iter().next()
                    })
                    .unwrap()
            },
            RenderDevice::GPU(name) => {
                // Try to find the specific GPU by name
                let selected_gpu = instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .into_iter()
                    .find(|adapter| {
                        let info = adapter.get_info();
                        format!("{} - {}", info.name, info.backend.to_str()) == *name
                    });
                
                selected_gpu.unwrap_or_else(|| {
                    // Fallback to any GPU adapter
                    instance
                        .enumerate_adapters(wgpu::Backends::all())
                        .into_iter()
                        .filter(|adapter| {
                            adapter.get_info().device_type != wgpu::DeviceType::Cpu
                        })
                        .next()
                        .or_else(|| {
                            // Final fallback to any adapter
                            instance.enumerate_adapters(wgpu::Backends::all()).into_iter().next()
                        })
                        .unwrap()
                })
            }
        };
        
        let adapter_info = adapter.get_info();
        debug!("Using adapter: {} ({})", adapter_info.name, adapter_info.backend.to_str());
        
        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
            
        // Configure the surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
            
        // Log supported present modes
        log::info!("Supported present modes: {:?}", surface_caps.present_modes);
        log::info!("Requested present mode: {:?}", present_mode);
        
        // Check if the requested present mode is supported
        let actual_present_mode = if surface_caps.present_modes.contains(&present_mode) {
            log::info!("Requested present mode is supported");
            present_mode
        } else {
            log::warn!("Requested present mode {:?} is not supported, falling back to Fifo", present_mode);
            wgpu::PresentMode::Fifo
        };
        
        // Update the vsync_enabled flag in system_info based on the actual present mode
        system_info.vsync_enabled = match actual_present_mode {
            wgpu::PresentMode::Immediate => false,
            _ => true, // Fifo and Mailbox both use VSync
        };
        
        // Store the vsync_enabled value for later use
        let vsync_enabled = system_info.vsync_enabled;
        
        log::info!("Using present mode: {:?}, VSync: {}", actual_present_mode, vsync_enabled);
        log::info!("Initializing State with vsync_enabled: {}", vsync_enabled);
            
        // Create the surface configuration
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: actual_present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        // Configure the surface with our configuration
        surface.configure(&device, &config);
        
        // Load the shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        
        // Create depth texture
        let depth_texture = create_depth_texture(&device, &config, "depth_texture");
        
        // Load the texture
        let block_bytes = include_bytes!("../assets/textures/block.png");
        let block_texture = create_texture_from_bytes(&device, &queue, block_bytes, "block_texture").unwrap();
        
        debug!("Loaded texture with dimensions: {} x {}", 
               block_texture.texture.size().width, 
               block_texture.texture.size().height);
        
        // Create texture bind group layout
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&block_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&block_texture.sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

        // Create uniform bind group
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms {
                model_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        // Create the render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Log the depth texture format for debugging
        debug!("Depth texture format: {:?}", depth_texture.texture.format());
        
        // Create the cube vertices and indices
        let (vertices, indices) = create_cube_vertices();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;

        // Initialize egui
        let egui_context = egui::Context::default();
        let viewport_id = egui::ViewportId::default();
        let egui_state = EguiState::new(
            egui_context.clone(),
            viewport_id,
            window.as_ref(),
            None,
            None,
            None,
        );
        
        // Initialize egui_wgpu_backend
        let egui_renderer = egui_wgpu_backend::RenderPass::new(
            &device,
            surface_format,
            1,
        );

        // Create debug axis buffer for object axes
        let debug_axis_buffer = create_debug_axis_buffer(&device);
        
        // Create debug axis buffer for world space axes (bottom left)
        let world_axis_buffer = create_debug_axis_buffer(&device);
        
        // Create world axis uniform buffer for positioning the axes in bottom left
        let world_axis_uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("World Axis Uniform Buffer"),
                contents: bytemuck::cast_slice(&[Uniforms {
                    model_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        
        // Store axis endpoint positions for label rendering
        let arrow_size = 0.15;
        let debug_axis_positions = vec![
            Vec3::new(1.0 + arrow_size, 0.0, 0.0), // X axis endpoint with arrow
            Vec3::new(0.0, 1.0 + arrow_size, 0.0), // Y axis endpoint with arrow
            Vec3::new(0.0, 0.0, 1.0 + arrow_size), // Z axis endpoint with arrow
        ];
        
        // Same for world axis positions
        let world_axis_positions = debug_axis_positions.clone();
        
        // Create world axis bind group
        let world_axis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: world_axis_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("world_axis_bind_group"),
        });
        
        // Create the debug axis pipeline
        let debug_axis_pipeline = create_debug_axis_pipeline(&device, &render_pipeline_layout, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            uniform_buffer,
            uniform_bind_group,
            textures: vec![block_texture],
            texture_bind_group,
            depth_texture,
            egui_state,
            egui_renderer,
            frame_times: vec![16.7; 30],
            fps: 60.0,
            rotation: 0.0,
            last_update: Instant::now(),
            render_device,
            system_info,
            debug_state: DebugState::default(),
            debug_axis_buffer: Some(debug_axis_buffer),
            debug_axis_pipeline: Some(debug_axis_pipeline),
            world_axis_buffer: Some(world_axis_buffer),
            world_axis_bind_group: Some(world_axis_bind_group),
            world_axis_uniform_buffer: Some(world_axis_uniform_buffer),
            debug_axis_positions,
            world_axis_positions,
            cube_position: Vec3::ZERO,
            cube_velocity: Vec3::ZERO,
            vsync_enabled,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            log::info!("Resizing window to {}x{}, preserving present mode: {:?}", 
                new_size.width, new_size.height, self.config.present_mode);
            
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            // Preserve the current present mode instead of forcing Immediate
            // This allows vsync toggle to work properly
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        // First check if debug state wants to handle this input
        debug!("State received window event: {:?}", event);
        if self.debug_state.handle_input(event) {
            debug!("Debug state handled the event");
            return true;
        }

        match event {
            WindowEvent::KeyboardInput { 
                event: winit::event::KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Space),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                // Handle space key
                true
            },
            WindowEvent::KeyboardInput { 
                event: winit::event::KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyV),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                // Log before toggle
                log::info!("V key pressed - toggling VSync");
                log::info!("Before toggle - Present mode: {:?}, VSync enabled: {}", 
                    self.config.present_mode, self.system_info.vsync_enabled);
                
                // Toggle VSync mode
                self.toggle_vsync();
                
                // Log after toggle
                log::info!("After toggle - Present mode: {:?}, VSync enabled: {}", 
                    self.config.present_mode, self.system_info.vsync_enabled);
                
                true
            },
            WindowEvent::Resized(physical_size) => {
                self.resize(*physical_size);
                true
            },
            _ => false,
        }
    }

    pub fn update(&mut self) {
        // Calculate FPS
        let now = Instant::now();
        let dt = now - self.last_update;
        self.last_update = now;

        // Update FPS counter
        let frame_time = dt.as_secs_f32() * 1000.0; // in ms
        self.frame_times.remove(0);
        self.frame_times.push(frame_time);
        
        let avg_frame_time = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        self.fps = 1000.0 / avg_frame_time;
        
        // If vsync is disabled and we're in Immediate mode, force high FPS by requesting redraws
        if !self.vsync_enabled && self.config.present_mode == wgpu::PresentMode::Immediate {
            // Request a redraw to force higher frame rate
            self.window.request_redraw();
            
            // On macOS, we need to be more aggressive to overcome Metal's VSync enforcement
            #[cfg(target_os = "macos")]
            {
                // Check if FPS is still capped around 60
                if self.fps < 70.0 {
                    // We're still capped, try more aggressive approach
                    static mut TOGGLE_COUNT: u32 = 0;
                    
                    unsafe {
                        TOGGLE_COUNT += 1;
                        
                        // Every 60 frames, log that we're still capped
                        if TOGGLE_COUNT % 60 == 0 {
                            log::warn!("FPS still appears to be capped at {:.1} despite Immediate mode", self.fps);
                            log::warn!("macOS Metal may be enforcing VSync at the driver level");
                            log::warn!("Try running with: METAL_DEVICE_WRAPPER_TYPE=1 cargo run");
                        }
                    }
                }
            }
        }
        
        // Check if FPS is consistent with the current present mode
        // This helps detect if vsync changes are actually taking effect
        static mut LAST_FPS_CHECK: Option<Instant> = None;
        static mut LAST_PRESENT_MODE_CHECK: Option<wgpu::PresentMode> = None;
        
        unsafe {
            let should_check = match LAST_FPS_CHECK {
                Some(last_time) if now.duration_since(last_time).as_secs_f32() >= 2.0 => true,
                None => true,
                _ => false,
            };
            
            if should_check {
                LAST_FPS_CHECK = Some(now);
                
                // Check if FPS is consistent with present mode
                let expected_fps_range = match self.config.present_mode {
                    wgpu::PresentMode::Fifo => {
                        // With VSync, FPS should be close to refresh rate (typically 60)
                        (55.0, 65.0)
                    },
                    wgpu::PresentMode::Immediate => {
                        // Without VSync, FPS should be higher than refresh rate
                        // or at least not capped at exactly refresh rate
                        (70.0, 10000.0)
                    },
                    _ => (0.0, 10000.0),
                };
                
                let fps_matches_mode = self.fps >= expected_fps_range.0 && self.fps <= expected_fps_range.1;
                
                // Only log if present mode has changed since last check
                if let Some(last_mode) = LAST_PRESENT_MODE_CHECK {
                    if last_mode != self.config.present_mode || !fps_matches_mode {
                        if !fps_matches_mode {
                            log::warn!("FPS ({:.1}) doesn't match expected range for {:?} ({:.1}-{:.1})", 
                                self.fps, self.config.present_mode, expected_fps_range.0, expected_fps_range.1);
                            
                            // If we're in Immediate mode but FPS is still capped, something is wrong
                            if self.config.present_mode == wgpu::PresentMode::Immediate && self.fps < 70.0 {
                                log::warn!("VSync appears to still be active despite being in Immediate mode!");
                                log::warn!("This might be due to OS-level VSync enforcement or another frame limiting mechanism");
                            }
                        } else {
                            log::info!("FPS ({:.1}) matches expected range for {:?} ({:.1}-{:.1})", 
                                self.fps, self.config.present_mode, expected_fps_range.0, expected_fps_range.1);
                        }
                    }
                }
                
                LAST_PRESENT_MODE_CHECK = Some(self.config.present_mode);
            }
        }
        
        // Log FPS and present mode every second (approximately)
        static mut LAST_FPS_LOG: Option<Instant> = None;
        unsafe {
            let should_log = match LAST_FPS_LOG {
                Some(last_time) if now.duration_since(last_time).as_secs_f32() >= 1.0 => true,
                None => true,
                _ => false,
            };
            
            if should_log {
                LAST_FPS_LOG = Some(now);
                
                // Log FPS and present mode
                log::debug!("FPS: {:.1}, Frame Time: {:.2}ms, Present Mode: {:?}, VSync: {}", 
                    self.fps, 
                    avg_frame_time,
                    self.config.present_mode,
                    self.system_info.vsync_enabled
                );
                
                // Track present mode changes over time
                static mut LAST_PRESENT_MODE: Option<wgpu::PresentMode> = None;
                let current_mode = self.config.present_mode;
                
                if let Some(last_mode) = LAST_PRESENT_MODE {
                    if last_mode != current_mode {
                        log::info!("Present mode changed from {:?} to {:?} during update", 
                            last_mode, current_mode);
                        
                        // Run diagnostic check when present mode changes
                        self.check_present_mode_status("Present mode changed during update");
                    }
                }
                
                LAST_PRESENT_MODE = Some(current_mode);
            }
        }

        // Update rotation
        self.rotation += dt.as_secs_f32();

        // Update debug state with system info
        self.debug_state.update(Some(&mut self.system_info));

        // Update cube position and velocity (for debug display)
        // In this example, we'll just use a simple orbit
        let orbit_radius = 1.0;
        self.cube_position = Vec3::new(
            orbit_radius * self.rotation.sin(),
            0.0,
            orbit_radius * self.rotation.cos()
        );
        
        self.cube_velocity = Vec3::new(
            orbit_radius * self.rotation.cos(),
            0.0,
            -orbit_radius * self.rotation.sin()
        );

        // Calculate aspect ratio
        let aspect = self.config.width as f32 / self.config.height as f32;
        
        // Create projection matrix
        // Use a fixed field of view to maintain consistent perspective
        let fov = 45.0_f32.to_radians();
        let proj = Mat4::perspective_rh(fov, aspect, 0.1, 100.0);
        
        // Create view matrix with fixed camera position
        // This ensures the cube stays at a constant distance from the camera
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 1.5, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        
        // Apply texture to cube, then rotate the cube, then app starts
        // This follows the correct order of operations for proper texture orientation
        
        // 1. Create the base model matrix (identity) - represents the cube with texture applied
        let base_model = Mat4::IDENTITY;
        
        // 2. Apply fixed rotation to align the texture correctly
        // This is applied during the initial setup, before continuous animation
        let x_rotation = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
        
        // 3. Apply continuous Y-rotation for animation (happens during app runtime)
        let y_rotation_matrix = Mat4::from_rotation_y(self.rotation);
        
        // 4. Combine: First apply texture (implicit in base_model), 
        // then fixed rotation for alignment, then continuous animation
        let model = y_rotation_matrix * x_rotation * base_model;
        
        // Combine into model-view-projection matrix
        let mvp = proj * view * model;
        
        // Update uniform buffer
        let uniform_data = Uniforms {
            model_view_proj: mvp.to_cols_array_2d(),
        };
        
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );

        // Update debug world axis uniform if in debug mode
        if self.debug_state.enabled && self.world_axis_uniform_buffer.is_some() {
            // Create a special transform for the world axes at bottom left
            let aspect = self.config.width as f32 / self.config.height as f32;
            
            // Calculate scale factor based on window size for consistent UI scaling
            let min_dimension = self.config.width.min(self.config.height) as f32;
            let scale_factor = min_dimension / 768.0; // Base scale on reference size of 768
            
            // Scale and position the axes in the bottom left corner
            let scale = 0.15 * scale_factor; // Scale of the axes, also affected by window size
            let x_pos = -0.85; // Position in NDC (-1 to 1)
            let y_pos = -0.85;
            
            // Create model-view-projection matrix for fixed screen space position
            let world_axis_model = Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                Quat::IDENTITY,
                Vec3::new(x_pos, y_pos, 0.0)
            );
            
            // Create orthographic projection for screen space
            let ortho = Mat4::orthographic_rh(-aspect, aspect, -1.0, 1.0, -10.0, 10.0);
            
            // Combine matrices
            let world_mvp = ortho * world_axis_model;
            
            // Update uniform buffer
            let world_uniform_data = Uniforms {
                model_view_proj: world_mvp.to_cols_array_2d(),
            };
            
            self.queue.write_buffer(
                self.world_axis_uniform_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&[world_uniform_data]),
            );
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Get the current frame
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Log the render pipeline and depth attachment configuration
        debug!("Render pipeline depth stencil configuration: checking");
        debug!("Using depth texture with format: {:?}", self.depth_texture.texture.format());
        
        // Start a render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            
            // Draw debug axes if in debug mode
            if self.debug_state.enabled && 
                self.debug_axis_buffer.is_some() && 
                self.debug_axis_pipeline.is_some() {
                // 1. Draw object-attached axes
                render_pass.set_pipeline(self.debug_axis_pipeline.as_ref().unwrap());
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.debug_axis_buffer.as_ref().unwrap().slice(..));
                render_pass.draw(0..72, 0..1); // 3 axes, 24 triangles total (8 per axis), 3 vertices per triangle
                
                // 2. Draw world space axes in bottom left corner
                if self.world_axis_bind_group.is_some() && self.world_axis_buffer.is_some() {
                    render_pass.set_bind_group(0, self.world_axis_bind_group.as_ref().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, self.world_axis_buffer.as_ref().unwrap().slice(..));
                    render_pass.draw(0..72, 0..1); // 3 axes, 24 triangles total (8 per axis), 3 vertices per triangle
                }
            }
        }
        
        // Get matrices for label positioning
        let aspect = self.config.width as f32 / self.config.height as f32;
        let _proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let _camera_view = Mat4::look_at_rh(
            Vec3::new(0.0, 1.5, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        
        // Create rotation for model (for object-attached axes)
        let rotation = Quat::from_rotation_y(self.rotation);
        let model = Mat4::from_quat(rotation);
        
        // Calculate world positions of object-attached axes
        let _rotated_axis_positions: Vec<Vec3> = self.debug_axis_positions.iter()
            .map(|pos| model.transform_point3(*pos))
            .collect();
        
        // Calculate scale factor based on window size for consistent UI scaling
        let min_dimension = self.config.width.min(self.config.height) as f32;
        let scale_factor = min_dimension / 768.0; // Base scale on reference size of 768
        
        // World axis positions in screen space (for bottom left corner)
        let scale = 0.15 * scale_factor;
        let x_pos = -0.85;
        let y_pos = -0.85;
        
        let world_axis_model = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::IDENTITY,
            Vec3::new(x_pos, y_pos, 0.0)
        );
        
        let ortho = Mat4::orthographic_rh(-aspect, aspect, -1.0, 1.0, -10.0, 10.0);
        let _world_mvp = ortho * world_axis_model; // Prefix with underscore to indicate intentionally unused
        
        // Create scaled world axis positions
        let _scaled_world_positions: Vec<Vec3> = self.world_axis_positions.iter()
            .map(|pos| {
                let scaled_pos = *pos * scale;
                Vec3::new(scaled_pos.x + x_pos, scaled_pos.y + y_pos, scaled_pos.z)
            })
            .collect();
        
        // Render egui UI
        let screen_descriptor = ScreenDescriptor {
            physical_width: self.config.width,
            physical_height: self.config.height,
            scale_factor: 1.0,
        };
        
        let egui_context = self.egui_state.egui_ctx().clone();
        
        let raw_input = self.egui_state.take_egui_input(self.window.as_ref());
        let full_output = egui_context.run(raw_input, |ctx| {
            // Always render debug information using our debug state
            // This will now show both text panels regardless of debug state
            self.debug_state.render(
                ctx,
                self.fps,
                &self.render_device,
                &self.system_info,
                (self.config.width, self.config.height),
                self.rotation,
                self.cube_position,
                self.cube_velocity,
            );
        });
        
        // Handle egui output and render
        self.egui_state.handle_platform_output(self.window.as_ref(), full_output.platform_output);
        
        // Check if shapes were actually generated
        if full_output.shapes.is_empty() {
            debug!("WARNING: No shapes generated by egui!");
        }
        
        // Now tessellate the shapes into paint jobs
        let paint_jobs = egui_context.tessellate(full_output.shapes, 1.0);
        
        // Update egui
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            paint_jobs.as_slice(),
            &screen_descriptor,
        );
        
        // Add textures from egui to the renderer
        if let Err(e) = self.egui_renderer.add_textures(&self.device, &self.queue, &full_output.textures_delta) {
            debug!("Failed to add textures to renderer: {:?}", e);
        }

        // Render egui
        let render_result = self.egui_renderer.execute(
            &mut encoder,
            &view,
            paint_jobs.as_slice(),
            &screen_descriptor,
            None,
        );
        
        // Just log errors, not success
        if let Err(e) = &render_result {
            debug!("Renderer execution failed: {:?}", e);
        }
        
        // Free textures that are no longer needed
        if let Err(e) = self.egui_renderer.remove_textures(full_output.textures_delta) {
            debug!("Failed to remove textures from renderer: {:?}", e);
        }
        
        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }

    pub fn toggle_vsync(&mut self) {
        // Log the current present mode and FPS before changing
        log::info!("Current present mode before toggle: {:?}, FPS: {:.1}", self.config.present_mode, self.fps);
        
        // Store initial FPS for comparison
        let initial_fps = self.fps;
        
        // Diagnostic check before toggle
        self.check_present_mode_status("Before toggle");
        
        // We need to get the supported present modes
        // Since we can't access the adapter directly from the device in newer wgpu versions,
        // we'll create a temporary instance and adapter to check capabilities
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(self.window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).unwrap();
        
        let surface_caps = surface.get_capabilities(&adapter);
        log::info!("Supported present modes: {:?}", surface_caps.present_modes);
        
        // Check if we're on macOS
        #[cfg(target_os = "macos")]
        let is_macos = true;
        #[cfg(not(target_os = "macos"))]
        let is_macos = false;
        
        // Cycle through available present modes: Fifo (VSync) -> Immediate (Uncapped) -> Mailbox (if available)
        let new_mode = match self.config.present_mode {
            wgpu::PresentMode::Fifo => {
                // Switch to Immediate (no VSync) if supported
                self.system_info.vsync_enabled = false;
                self.vsync_enabled = false; // Also update the State struct's flag
                if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
                    wgpu::PresentMode::Immediate
                } else {
                    // If Immediate is not supported, stay with Fifo
                    self.system_info.vsync_enabled = true;
                    self.vsync_enabled = true; // Also update the State struct's flag
                    wgpu::PresentMode::Fifo
                }
            },
            wgpu::PresentMode::Immediate => {
                // Try Mailbox if supported, otherwise back to Fifo
                self.system_info.vsync_enabled = true;
                self.vsync_enabled = true; // Also update the State struct's flag
                if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                    wgpu::PresentMode::Mailbox
                } else {
                    wgpu::PresentMode::Fifo
                }
            },
            wgpu::PresentMode::Mailbox => {
                // Back to Fifo
                self.system_info.vsync_enabled = true;
                self.vsync_enabled = true; // Also update the State struct's flag
                wgpu::PresentMode::Fifo
            },
            // For any other modes, default to Fifo
            _ => {
                self.system_info.vsync_enabled = true;
                self.vsync_enabled = true; // Also update the State struct's flag
                wgpu::PresentMode::Fifo
            }
        };
        
        // Update the config with the new present mode
        let old_mode = self.config.present_mode;
        self.config.present_mode = new_mode;
        
        // Verify if the mode is actually changing
        if old_mode == new_mode {
            log::warn!("Present mode didn't change! Still at {:?}", new_mode);
        } else {
            log::info!("Present mode changed from {:?} to {:?}", old_mode, new_mode);
        }
        
        // Force a complete reconfiguration of the surface to ensure changes take effect
        // This is more aggressive than just calling configure
        log::info!("Forcing surface reconfiguration to apply present mode change");
        
        // First, reconfigure the surface with the new settings
        self.surface.configure(&self.device, &self.config);
        
        // Then, force a complete refresh by recreating the depth texture
        // This ensures the rendering pipeline is fully reset
        self.depth_texture = create_depth_texture(&self.device, &self.config, "depth_texture");
        
        // Force a frame to be rendered immediately to apply changes
        // This helps ensure the new present mode takes effect right away
        log::info!("Forcing immediate frame render to apply present mode");
        
        // Use our specialized method to force the present mode update
        self.force_present_mode_update();
        
        // Diagnostic check after toggle
        self.check_present_mode_status("After toggle");
        
        // Add a direct check for Metal's VSync enforcement
        if self.config.present_mode == wgpu::PresentMode::Immediate {
            log::warn!("On macOS, Metal may enforce VSync at the driver level regardless of the present mode.");
            log::warn!("Try setting the environment variable: METAL_DEVICE_WRAPPER_TYPE=1");
            log::warn!("Example: METAL_DEVICE_WRAPPER_TYPE=1 cargo run");
            
            // Check if we're on macOS and try to force uncapped FPS
            #[cfg(target_os = "macos")]
            {
                log::info!("Attempting to force uncapped FPS on macOS...");
                
                // Try to set the CAMetalLayer's displaySyncEnabled property to false
                // This is a more direct approach to disable VSync on macOS
                
                // First, check if the METAL_DEVICE_WRAPPER_TYPE environment variable is set
                if std::env::var("METAL_DEVICE_WRAPPER_TYPE").is_err() {
                    log::warn!("METAL_DEVICE_WRAPPER_TYPE environment variable is not set.");
                    log::warn!("This may prevent disabling VSync on macOS.");
                }
                
                // Force a more aggressive approach to disable VSync
                // This involves directly manipulating the frame timing
                log::info!("Using alternative approach to force uncapped FPS");
                
                // Set a flag to use a more aggressive frame timing approach
                self.vsync_enabled = false;
                
                // Request high performance mode from the window system
                self.window.request_inner_size(self.window.inner_size());
            }
        }
        
        // Monitor FPS for a few seconds to see if it changes
        log::info!("Starting FPS monitoring to verify vsync change...");
        
        // Create a separate thread to monitor FPS changes
        let window_clone = self.window.clone();
        let present_mode = self.config.present_mode;
        let vsync_enabled = self.vsync_enabled;
        let initial_fps = initial_fps;
        
        std::thread::spawn(move || {
            // Wait a moment for the change to take effect
            std::thread::sleep(std::time::Duration::from_millis(500));
            
            // Request a redraw to ensure we get fresh frames
            window_clone.request_redraw();
            
            // Wait a bit longer to allow FPS to stabilize
            std::thread::sleep(std::time::Duration::from_millis(1000));
            
            // Log expected FPS behavior
            match present_mode {
                wgpu::PresentMode::Fifo => {
                    log::info!("With VSync ON (Fifo), FPS should be limited to refresh rate (typically 60 FPS)");
                },
                wgpu::PresentMode::Immediate => {
                    log::info!("With VSync OFF (Immediate), FPS should be uncapped and potentially higher than refresh rate");
                },
                wgpu::PresentMode::Mailbox => {
                    log::info!("With VSync ON (Mailbox), FPS should be high but synchronized (no tearing)");
                },
                _ => {
                    log::info!("Unknown present mode behavior");
                }
            }
            
            // Request another redraw
            window_clone.request_redraw();
        });
        
        // Log the new present mode with a user-friendly message
        log::info!("VSync is now {}: {} (Present Mode: {:?})", 
            if self.system_info.vsync_enabled { "ON" } else { "OFF" },
            if self.system_info.vsync_enabled { 
                "Frame rate limited to display refresh rate" 
            } else { 
                "Uncapped frame rate (may cause tearing)" 
            },
            self.config.present_mode
        );
    }
    
    // Diagnostic function to check present mode status
    fn check_present_mode_status(&self, context: &str) {
        // Create a temporary instance and adapter to check the actual capabilities
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(self.window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).unwrap();
        
        let adapter_info = adapter.get_info();
        let surface_caps = surface.get_capabilities(&adapter);
        
        log::info!("=== PRESENT MODE DIAGNOSTIC: {} ===", context);
        log::info!("Current adapter: {} ({})", adapter_info.name, adapter_info.backend.to_str());
        log::info!("Current config present mode: {:?}", self.config.present_mode);
        log::info!("Supported present modes: {:?}", surface_caps.present_modes);
        log::info!("Is current mode supported: {}", surface_caps.present_modes.contains(&self.config.present_mode));
        log::info!("SystemInfo vsync_enabled flag: {}", self.system_info.vsync_enabled);
        log::info!("State vsync_enabled flag: {}", self.vsync_enabled);
        log::info!("=== END DIAGNOSTIC ===");
    }

    // Helper method to force flush the graphics pipeline and ensure present mode changes take effect
    fn force_present_mode_update(&mut self) {
        log::info!("Forcing present mode update to {:?}", self.config.present_mode);
        
        // Create a simple command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Force Present Mode Update Encoder"),
        });
        
        // Submit the command encoder to flush the pipeline
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Force a wait for the device to idle
        // This ensures all previous commands are processed before continuing
        log::info!("Waiting for device to complete all operations...");
        
        // We can't directly wait for the device to idle in wgpu,
        // but we can create and map a buffer as a synchronization point
        let buffer_size = 4; // Just need a small buffer
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sync Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create a staging buffer with some data
        let staging_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Staging Buffer"),
            contents: &[0, 0, 0, 0],
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        
        // Copy from staging buffer to the map-read buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sync Encoder"),
        });
        encoder.copy_buffer_to_buffer(&staging_buffer, 0, &buffer, 0, buffer_size);
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Map the buffer - this will block until the copy is complete
        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        // Wait for the mapping to complete
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(_) = receiver.recv() {
            // Buffer is now mapped, which means all previous operations are complete
            log::info!("Device operations completed, present mode should now be active");
            
            // Unmap the buffer
            drop(buffer_slice.get_mapped_range());
            buffer.unmap();
        }
        
        // Reconfigure the surface one more time to ensure the present mode is applied
        self.surface.configure(&self.device, &self.config);
    }
}

// Helper function to create the debug axis rendering pipeline
pub fn create_debug_axis_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Debug Axis Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Debug Axis Pipeline"),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[Vertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_debug_main", // Use specialized fragment shader
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList, // Use triangles for thicker axes
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // Don't cull for debug axes
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

// Helper function to create debug axis buffer
pub fn create_debug_axis_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    // Create vertices for XYZ axes (red = X, green = Y, blue = Z)
    // Using thicker lines by creating triangular prisms for each axis
    let axis_thickness = 0.03; // Thickness of the axes
    let arrow_size = 0.15; // Size of the arrow tips
    
    let axis_vertices = vec![
        // X-axis (red) - thicker line using two triangles
        // First triangle
        Vertex { position: [0.0, -axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, -axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Second triangle
        Vertex { position: [0.0, -axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [0.0, axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Third triangle
        Vertex { position: [0.0, -axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, -axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Fourth triangle
        Vertex { position: [0.0, -axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [0.0, axis_thickness, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // X-axis arrow tip (pyramid at the end)
        // First triangle (bottom face)
        Vertex { position: [1.0, -axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, -axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0 + arrow_size, 0.0, 0.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Second triangle (top face)
        Vertex { position: [1.0, axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0 + arrow_size, 0.0, 0.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Third triangle (left face)
        Vertex { position: [1.0, -axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, -axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0 + arrow_size, 0.0, 0.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Fourth triangle (right face)
        Vertex { position: [1.0, -axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0, axis_thickness, axis_thickness], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        Vertex { position: [1.0 + arrow_size, 0.0, 0.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
        
        // Y-axis (green) - thicker line using two triangles
        // First triangle
        Vertex { position: [-axis_thickness, 0.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [-axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Second triangle
        Vertex { position: [-axis_thickness, 0.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 0.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Third triangle
        Vertex { position: [-axis_thickness, 0.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [-axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Fourth triangle
        Vertex { position: [-axis_thickness, 0.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 0.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Y-axis arrow tip (pyramid at the end)
        // First triangle (bottom face)
        Vertex { position: [-axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [-axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.0, 1.0 + arrow_size, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Second triangle (top face)
        Vertex { position: [axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.0, 1.0 + arrow_size, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Third triangle (left face)
        Vertex { position: [-axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, -axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.0, 1.0 + arrow_size, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Fourth triangle (right face)
        Vertex { position: [-axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [axis_thickness, 1.0, axis_thickness], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.0, 1.0 + arrow_size, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },
        
        // Z-axis (blue) - thicker line using two triangles
        // First triangle
        Vertex { position: [-axis_thickness, -axis_thickness, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [-axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Second triangle
        Vertex { position: [-axis_thickness, -axis_thickness, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, -axis_thickness, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Third triangle
        Vertex { position: [-axis_thickness, axis_thickness, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [-axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Fourth triangle
        Vertex { position: [-axis_thickness, axis_thickness, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, axis_thickness, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Z-axis arrow tip (pyramid at the end)
        // First triangle (bottom face)
        Vertex { position: [-axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [-axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [0.0, 0.0, 1.0 + arrow_size], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Second triangle (top face)
        Vertex { position: [axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [0.0, 0.0, 1.0 + arrow_size], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Third triangle (left face)
        Vertex { position: [-axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, -axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [0.0, 0.0, 1.0 + arrow_size], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        
        // Fourth triangle (right face)
        Vertex { position: [-axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [axis_thickness, axis_thickness, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
        Vertex { position: [0.0, 0.0, 1.0 + arrow_size], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },
    ];
    
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Debug Axis Buffer"),
        contents: bytemuck::cast_slice(&axis_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
} 