use anyhow::Result;
use glam::{Mat4, Quat, Vec3, Vec4};
use std::{sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    window::Window,
    keyboard::{KeyCode, PhysicalKey},
    event::ElementState,
};
use egui::{pos2, Color32, Pos2, Align2};
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
    pub debug_mode: bool,
    pub debug_axis_buffer: Option<wgpu::Buffer>,
    pub debug_axis_pipeline: Option<wgpu::RenderPipeline>,
    pub world_axis_buffer: Option<wgpu::Buffer>,
    pub world_axis_bind_group: Option<wgpu::BindGroup>,
    pub world_axis_uniform_buffer: Option<wgpu::Buffer>,
    pub debug_axis_positions: Vec<Vec3>,
    pub world_axis_positions: Vec<Vec3>,
}

impl State {
    pub async fn new(window: Arc<Window>, render_device: RenderDevice, system_info: SystemInfo) -> Result<Self> {
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
        println!("Using adapter: {} ({})", adapter_info.name, adapter_info.backend.to_str());
        
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
            
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
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
        let debug_axis_positions = vec![
            Vec3::new(1.0, 0.0, 0.0), // X axis endpoint
            Vec3::new(0.0, 1.0, 0.0), // Y axis endpoint
            Vec3::new(0.0, 0.0, 1.0), // Z axis endpoint
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
            debug_mode: false,
            debug_axis_buffer: Some(debug_axis_buffer),
            debug_axis_pipeline: Some(debug_axis_pipeline),
            world_axis_buffer: Some(world_axis_buffer),
            world_axis_bind_group: Some(world_axis_bind_group),
            world_axis_uniform_buffer: Some(world_axis_uniform_buffer),
            debug_axis_positions,
            world_axis_positions,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { 
                event: winit::event::KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyD),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                self.debug_mode = !self.debug_mode;
                debug!("Debug mode: {}", self.debug_mode);
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

        // Update rotation
        self.rotation += dt.as_secs_f32();

        // Create projection matrix
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        
        // Create view matrix
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 1.5, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        
        // Create model matrix with rotation
        let rotation = Quat::from_rotation_y(self.rotation);
        let model = Mat4::from_quat(rotation);
        
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
        if self.debug_mode && self.world_axis_uniform_buffer.is_some() {
            // Create a special transform for the world axes at bottom left
            let aspect = self.config.width as f32 / self.config.height as f32;
            
            // Scale and position the axes in the bottom left corner
            let scale = 0.15; // Scale of the axes
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
            if self.debug_mode && 
               self.debug_axis_buffer.is_some() && 
               self.debug_axis_pipeline.is_some() {
                // 1. Draw object-attached axes
                render_pass.set_pipeline(self.debug_axis_pipeline.as_ref().unwrap());
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.debug_axis_buffer.as_ref().unwrap().slice(..));
                render_pass.draw(0..6, 0..1); // 3 lines, 2 vertices each
                
                // 2. Draw world space axes in bottom left corner
                if self.world_axis_bind_group.is_some() && self.world_axis_buffer.is_some() {
                    render_pass.set_bind_group(0, self.world_axis_bind_group.as_ref().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, self.world_axis_buffer.as_ref().unwrap().slice(..));
                    render_pass.draw(0..6, 0..1); // 3 lines, 2 vertices each
                }
            }
        }
        
        // Get matrices for label positioning
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 1.5, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        
        // Create rotation for model (for object-attached axes)
        let rotation = Quat::from_rotation_y(self.rotation);
        let model = Mat4::from_quat(rotation);
        
        // Calculate world positions of object-attached axes
        let rotated_axis_positions: Vec<Vec3> = self.debug_axis_positions.iter()
            .map(|pos| model.transform_point3(*pos))
            .collect();
        
        // World axis positions in screen space (for bottom left corner)
        let scale = 0.15;
        let x_pos = -0.85;
        let y_pos = -0.85;
        
        let world_axis_model = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::IDENTITY,
            Vec3::new(x_pos, y_pos, 0.0)
        );
        
        let ortho = Mat4::orthographic_rh(-aspect, aspect, -1.0, 1.0, -10.0, 10.0);
        let world_mvp = ortho * world_axis_model;
        
        // Create scaled world axis positions
        let scaled_world_positions: Vec<Vec3> = self.world_axis_positions.iter()
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
            egui::Window::new("Stats").show(ctx, |ui| {
                ui.label(format!("FPS: {:.1}", self.fps));
                ui.label(format!("Frame Time: {:.2} ms", 1000.0 / self.fps));
                ui.label(format!("Rendering Device: {}", self.render_device));
                
                ui.separator();
                
                ui.heading("CPU Information");
                if !self.system_info.cpus.is_empty() {
                    ui.label(&self.system_info.cpus[self.system_info.selected_cpu]);
                }
                
                ui.separator();
                
                ui.heading("GPU Information");
                if self.system_info.selected_gpu < self.system_info.gpus.len() {
                    ui.label(&self.system_info.gpus[self.system_info.selected_gpu]);
                }
            });
            
            // Render axis labels if in debug mode
            if self.debug_mode {
                // Helper function to project 3D point to 2D screen space
                let project_point = |point: Vec3, mvp: Mat4| -> Option<Pos2> {
                    // Transform point to clip space
                    let clip_pos = mvp * Vec4::new(point.x, point.y, point.z, 1.0);
                    
                    // Check if point is behind camera
                    if clip_pos.w <= 0.0 {
                        return None;
                    }
                    
                    // Convert to normalized device coordinates
                    let ndc = Vec3::new(
                        clip_pos.x / clip_pos.w,
                        clip_pos.y / clip_pos.w,
                        clip_pos.z / clip_pos.w
                    );
                    
                    // Convert to screen coordinates
                    let screen_x = (ndc.x + 1.0) * 0.5 * self.config.width as f32;
                    let screen_y = (1.0 - ndc.y) * 0.5 * self.config.height as f32; // Y is flipped
                    
                    Some(pos2(screen_x, screen_y))
                };
                
                // Project object-attached axis positions to screen space
                let mvp = proj * view * model;
                
                let axis_labels = ["X", "Y", "Z"];
                let axis_colors = [Color32::RED, Color32::GREEN, Color32::BLUE];
                
                // Draw object axis labels
                for (i, pos) in rotated_axis_positions.iter().enumerate() {
                    if let Some(screen_pos) = project_point(*pos, mvp) {
                        // Draw axis label
                        egui::Area::new(format!("axis_label_{}", i))
                            .movable(false)
                            .anchor(Align2::CENTER_CENTER, screen_pos)
                            .show(ctx, |ui| {
                                ui.colored_label(axis_colors[i], axis_labels[i]);
                            });
                    }
                }
                
                // Project world axis positions to screen space
                for (i, pos) in scaled_world_positions.iter().enumerate() {
                    // For world axes in the corner, we need orthographic projection
                    if let Some(screen_pos) = project_point(*pos, ortho) {
                        // Draw world axis label
                        egui::Area::new(format!("world_axis_label_{}", i))
                            .movable(false)
                            .anchor(Align2::CENTER_CENTER, screen_pos)
                            .show(ctx, |ui| {
                                ui.colored_label(axis_colors[i], axis_labels[i]);
                            });
                    }
                }
            }
        });
        
        // Handle egui output and render
        self.egui_state.handle_platform_output(self.window.as_ref(), full_output.platform_output);
        let paint_jobs = egui_context.tessellate(full_output.shapes, 1.0);
        
        // Add debug logging
        debug!("Paint jobs type: {:?}", std::any::type_name_of_val(&paint_jobs));
        debug!("Egui renderer expected type: egui_wgpu_backend::renderer::ClippedPrimitive");
        debug!("First paint job element type (if any): {:?}", 
               paint_jobs.first().map(|_| std::any::type_name_of_val(paint_jobs.first().unwrap())));
        
        // Update egui
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            paint_jobs.as_slice(),
            &screen_descriptor,
        );

        // Render egui
        let _ = self.egui_renderer.execute(
            &mut encoder,
            &view,
            paint_jobs.as_slice(),
            &screen_descriptor,
            None,
        );
        
        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
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
            topology: wgpu::PrimitiveTopology::LineList, // Use lines for axes
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
    let axis_vertices = vec![
        // X-axis (red)
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] }, // origin
        Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] }, // x+
        
        // Y-axis (green)
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] }, // origin
        Vertex { position: [0.0, 1.0, 0.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] }, // y+
        
        // Z-axis (blue)
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] }, // origin
        Vertex { position: [0.0, 0.0, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] }, // z+
    ];
    
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Debug Axis Buffer"),
        contents: bytemuck::cast_slice(&axis_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
} 