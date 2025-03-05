use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext as WebGL, WebGlProgram, WebGlShader, WebGlBuffer, WebGlTexture, WebGlUniformLocation};
use log::{Level, LevelFilter};
use std::panic;
use crate::system_info::SystemInfo;
use std::cell::RefCell;
use std::rc::Rc;
use std::f32::consts::PI;
use crate::vertex::create_cube_vertices;
use glam::{Mat4, Quat, Vec3};

// Store WebGL context and resources
thread_local! {
    static RENDER_CONTEXT: RefCell<Option<WebGL>> = RefCell::new(None);
    static SHADER_PROGRAM: RefCell<Option<WebGlProgram>> = RefCell::new(None);
    static VERTEX_BUFFER: RefCell<Option<WebGlBuffer>> = RefCell::new(None);
    static INDEX_BUFFER: RefCell<Option<WebGlBuffer>> = RefCell::new(None);
    static TEXTURE: RefCell<Option<WebGlTexture>> = RefCell::new(None);
    static MVP_UNIFORM: RefCell<Option<WebGlUniformLocation>> = RefCell::new(None);
    static NUM_INDICES: RefCell<u32> = RefCell::new(0);
    static ROTATION: RefCell<f32> = RefCell::new(0.0);
    static LAST_RENDER_TIME: RefCell<f64> = RefCell::new(0.0);
    static FPS_COUNTER: RefCell<String> = RefCell::new("FPS: 0".to_string());
    static FRAME_TIMES: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(100));
}

fn init_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    log::info!("Panic hook initialized");
}

#[wasm_bindgen]
pub fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

#[wasm_bindgen]
pub fn init_logging() {
    // Initialize console_log
    console_log::init_with_level(Level::Info).expect("Failed to initialize logger");
    
    // Set up custom logger for wgpu
    log::set_max_level(LevelFilter::Info);
    log::set_logger(&WebLogger).expect("Failed to set logger");
    
    // Initialize panic hook
    init_panic_hook();
    
    log::info!("Logging initialized");
}

struct WebLogger;

impl log::Log for WebLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= Level::Info
    }
    
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let msg = format!("[{}] {}", record.level(), record.args());
            web_sys::console::log_1(&JsValue::from_str(&msg));
        }
    }
    
    fn flush(&self) {}
}

#[wasm_bindgen]
pub fn init() {
    init_logging();
    log::info!("WebGL renderer initialized");
}

fn compile_shader(
    context: &WebGL,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    
    context.shader_source(&shader, source);
    context.compile_shader(&shader);
    
    if context
        .get_shader_parameter(&shader, WebGL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

fn link_program(
    context: &WebGL,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader program"))?;
    
    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);
    
    if context
        .get_program_parameter(&program, WebGL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program")))
    }
}

async fn load_texture_image(url: &str) -> Result<web_sys::HtmlImageElement, JsValue> {
    let image = web_sys::HtmlImageElement::new()?;
    
    // Create a promise for image loading
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        // Create a clone of reject for the error callback
        let reject_clone = reject.clone();
        
        let success_callback = Closure::once(Box::new(move || {
            resolve.call0(&JsValue::NULL).unwrap();
        }) as Box<dyn FnMut()>);
        
        let error_callback = Closure::once(Box::new(move |_| {
            reject_clone.call0(&JsValue::NULL).unwrap();
        }) as Box<dyn FnMut(JsValue)>);
        
        // Set up image properties
        image.set_cross_origin(Some("anonymous"));
        image.set_src(url);
        
        // Add event listeners
        let success_ref = success_callback.as_ref().unchecked_ref();
        let error_ref = error_callback.as_ref().unchecked_ref();
        
        // We can't use ? operator in this closure, so we'll handle errors manually
        match image.add_event_listener_with_callback("load", success_ref) {
            Ok(_) => {},
            Err(e) => {
                web_sys::console::error_1(&e);
                reject.call1(&JsValue::NULL, &e).unwrap();
                return;
            }
        }
        
        match image.add_event_listener_with_callback("error", error_ref) {
            Ok(_) => {},
            Err(e) => {
                web_sys::console::error_1(&e);
                reject.call1(&JsValue::NULL, &e).unwrap();
                return;
            }
        }
        
        // Prevent callbacks from being dropped
        success_callback.forget();
        error_callback.forget();
    });
    
    // Wait for the image to load
    wasm_bindgen_futures::JsFuture::from(promise).await?;
    
    Ok(image)
}

#[wasm_bindgen]
pub async fn initialize(canvas: HtmlCanvasElement) -> Result<bool, JsValue> {
    log::info!("Initializing WebGL renderer with canvas");
    
    // Create a simple system info for web
    let _system_info = SystemInfo {
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
    };
    
    // Get WebGL2 context
    let context = canvas
        .get_context("webgl2")
        .map_err(|_| JsValue::from_str("Failed to get WebGL2 context"))?
        .ok_or_else(|| JsValue::from_str("No WebGL2 context found"))?
        .dyn_into::<WebGL>()?;
    
    // Set up viewport
    context.viewport(0, 0, canvas.width() as i32, canvas.height() as i32);
    
    // Enable depth testing
    context.enable(WebGL::DEPTH_TEST);
    
    // Clear to black
    context.clear_color(0.0, 0.0, 0.0, 1.0);
    context.clear(WebGL::COLOR_BUFFER_BIT | WebGL::DEPTH_BUFFER_BIT);
    
    // Vertex shader source
    let vertex_shader_source = r#"#version 300 es
        precision highp float;
        
        in vec3 position;
        in vec2 texCoord;
        in vec3 normal;
        
        uniform mat4 modelViewProjection;
        
        out vec2 vTexCoord;
        out vec3 vNormal;
        
        void main() {
            vTexCoord = texCoord;
            vNormal = normal;
            gl_Position = modelViewProjection * vec4(position, 1.0);
        }
    "#;
    
    // Fragment shader source
    let fragment_shader_source = r#"#version 300 es
        precision highp float;
        
        in vec2 vTexCoord;
        in vec3 vNormal;
        
        uniform sampler2D textureSampler;
        
        out vec4 fragColor;
        
        void main() {
            vec4 texColor = texture(textureSampler, vTexCoord);
            
            // Simple lighting
            vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
            float diffuse = max(dot(vNormal, lightDir), 0.3);
            
            fragColor = texColor * diffuse;
        }
    "#;
    
    // Compile shaders
    let vert_shader = compile_shader(
        &context,
        WebGL::VERTEX_SHADER,
        vertex_shader_source,
    ).map_err(|e| JsValue::from_str(&format!("Vertex shader error: {}", e)))?;
    
    let frag_shader = compile_shader(
        &context,
        WebGL::FRAGMENT_SHADER,
        fragment_shader_source,
    ).map_err(|e| JsValue::from_str(&format!("Fragment shader error: {}", e)))?;
    
    // Link program
    let program = link_program(&context, &vert_shader, &frag_shader)
        .map_err(|e| JsValue::from_str(&format!("Program linking error: {}", e)))?;
    
    // Use the program
    context.use_program(Some(&program));
    
    // Get attribute locations
    let position_attrib = context.get_attrib_location(&program, "position");
    let tex_coord_attrib = context.get_attrib_location(&program, "texCoord");
    let normal_attrib = context.get_attrib_location(&program, "normal");
    
    // Get uniform location
    let mvp_uniform = context.get_uniform_location(&program, "modelViewProjection")
        .ok_or_else(|| JsValue::from_str("Could not get modelViewProjection uniform location"))?;
    
    // Create vertex and index buffers
    let (vertices, indices) = create_cube_vertices();
    
    // Create and bind vertex buffer
    let vertex_buffer = context.create_buffer()
        .ok_or_else(|| JsValue::from_str("Failed to create vertex buffer"))?;
    context.bind_buffer(WebGL::ARRAY_BUFFER, Some(&vertex_buffer));
    
    // Convert vertices to Float32Array and upload to GPU
    let vertices_array = js_sys::Float32Array::new_with_length((vertices.len() * 8) as u32);
    let mut offset = 0;
    
    for vertex in &vertices {
        // Position (3 floats)
        vertices_array.set_index(offset, vertex.position[0]);
        vertices_array.set_index(offset + 1, vertex.position[1]);
        vertices_array.set_index(offset + 2, vertex.position[2]);
        
        // Texture coordinates (2 floats)
        vertices_array.set_index(offset + 3, vertex.tex_coords[0]);
        vertices_array.set_index(offset + 4, vertex.tex_coords[1]);
        
        // Normal (3 floats)
        vertices_array.set_index(offset + 5, vertex.normal[0]);
        vertices_array.set_index(offset + 6, vertex.normal[1]);
        vertices_array.set_index(offset + 7, vertex.normal[2]);
        
        offset += 8;
    }
    
    context.buffer_data_with_array_buffer_view(
        WebGL::ARRAY_BUFFER,
        &vertices_array,
        WebGL::STATIC_DRAW,
    );
    
    // Create and bind index buffer
    let index_buffer = context.create_buffer()
        .ok_or_else(|| JsValue::from_str("Failed to create index buffer"))?;
    context.bind_buffer(WebGL::ELEMENT_ARRAY_BUFFER, Some(&index_buffer));
    
    // Convert indices to Uint16Array and upload to GPU
    let indices_array = js_sys::Uint16Array::new_with_length(indices.len() as u32);
    for (i, &index) in indices.iter().enumerate() {
        indices_array.set_index(i as u32, index);
    }
    
    context.buffer_data_with_array_buffer_view(
        WebGL::ELEMENT_ARRAY_BUFFER,
        &indices_array,
        WebGL::STATIC_DRAW,
    );
    
    // Create vertex array object
    let vao = context.create_vertex_array()
        .ok_or_else(|| JsValue::from_str("Failed to create vertex array object"))?;
    context.bind_vertex_array(Some(&vao));
    
    // Set up vertex attributes
    context.bind_buffer(WebGL::ARRAY_BUFFER, Some(&vertex_buffer));
    
    // Position attribute
    context.enable_vertex_attrib_array(position_attrib as u32);
    context.vertex_attrib_pointer_with_i32(
        position_attrib as u32,
        3,
        WebGL::FLOAT,
        false,
        8 * 4, // 8 floats per vertex (3 position + 2 texcoord + 3 normal) * 4 bytes per float
        0,
    );
    
    // Texture coordinate attribute
    context.enable_vertex_attrib_array(tex_coord_attrib as u32);
    context.vertex_attrib_pointer_with_i32(
        tex_coord_attrib as u32,
        2,
        WebGL::FLOAT,
        false,
        8 * 4,
        3 * 4, // Offset by 3 floats (position)
    );
    
    // Normal attribute
    context.enable_vertex_attrib_array(normal_attrib as u32);
    context.vertex_attrib_pointer_with_i32(
        normal_attrib as u32,
        3,
        WebGL::FLOAT,
        false,
        8 * 4,
        5 * 4, // Offset by 5 floats (position + texcoord)
    );
    
    // Bind index buffer
    context.bind_buffer(WebGL::ELEMENT_ARRAY_BUFFER, Some(&index_buffer));
    
    // Load texture
    let image = load_texture_image("assets/textures/block.png").await?;
    
    // Create and bind texture
    let texture = context.create_texture()
        .ok_or_else(|| JsValue::from_str("Failed to create texture"))?;
    context.active_texture(WebGL::TEXTURE0);
    context.bind_texture(WebGL::TEXTURE_2D, Some(&texture));
    
    // Set texture parameters
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_WRAP_S, WebGL::CLAMP_TO_EDGE as i32);
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_WRAP_T, WebGL::CLAMP_TO_EDGE as i32);
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_MIN_FILTER, WebGL::LINEAR as i32);
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_MAG_FILTER, WebGL::LINEAR as i32);
    
    // Upload image to texture
    context.tex_image_2d_with_u32_and_u32_and_html_image_element(
        WebGL::TEXTURE_2D,
        0,
        WebGL::RGBA as i32,
        WebGL::RGBA,
        WebGL::UNSIGNED_BYTE,
        &image,
    )?;
    
    // Set texture uniform
    let sampler_uniform = context.get_uniform_location(&program, "textureSampler")
        .ok_or_else(|| JsValue::from_str("Could not get textureSampler uniform location"))?;
    context.uniform1i(Some(&sampler_uniform), 0); // Use texture unit 0
    
    // Store resources in thread local storage
    RENDER_CONTEXT.with(|rc| {
        *rc.borrow_mut() = Some(context);
    });
    
    SHADER_PROGRAM.with(|sp| {
        *sp.borrow_mut() = Some(program);
    });
    
    VERTEX_BUFFER.with(|vb| {
        *vb.borrow_mut() = Some(vertex_buffer);
    });
    
    INDEX_BUFFER.with(|ib| {
        *ib.borrow_mut() = Some(index_buffer);
    });
    
    TEXTURE.with(|t| {
        *t.borrow_mut() = Some(texture);
    });
    
    MVP_UNIFORM.with(|mvp| {
        *mvp.borrow_mut() = Some(mvp_uniform);
    });
    
    NUM_INDICES.with(|ni| {
        *ni.borrow_mut() = indices.len() as u32;
    });
    
    // Log success
    log::info!("WebGL renderer initialized successfully");
    
    // Return success
    Ok(true)
}

#[wasm_bindgen]
pub fn start_render_loop() -> Result<(), JsValue> {
    log::info!("Starting render loop");
    
    // Create a simple animation frame request
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    
    // Create the render function
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // Request the next animation frame
        request_animation_frame(&f);
        
        // Render a frame
        render_frame();
    }) as Box<dyn FnMut()>));
    
    // Request the first animation frame
    request_animation_frame(&g);
    
    Ok(())
}

fn render_frame() {
    RENDER_CONTEXT.with(|rc| {
        if let Some(context) = &*rc.borrow() {
            // Get current time for animation
            let now = js_sys::Date::now();
            let last_time = LAST_RENDER_TIME.with(|lt| *lt.borrow());
            
            // Calculate delta time in seconds
            let delta_time = if last_time > 0.0 {
                (now - last_time) / 1000.0
            } else {
                0.0
            } as f32;
            
            // Update last render time
            LAST_RENDER_TIME.with(|lt| {
                *lt.borrow_mut() = now;
            });
            
            // Update rotation
            ROTATION.with(|r| {
                let mut rotation = r.borrow_mut();
                *rotation += delta_time * 0.5; // Rotate at 0.5 radians per second
                if *rotation > 2.0 * PI {
                    *rotation -= 2.0 * PI;
                }
            });
            
            // Update FPS counter
            FRAME_TIMES.with(|ft| {
                let mut frame_times = ft.borrow_mut();
                frame_times.push(delta_time);
                
                // Keep only the last 100 frames
                if frame_times.len() > 100 {
                    frame_times.remove(0);
                }
                
                // Calculate average FPS
                if !frame_times.is_empty() {
                    let avg_frame_time = frame_times.iter().sum::<f32>() / frame_times.len() as f32;
                    let fps = if avg_frame_time > 0.0 { 1.0 / avg_frame_time } else { 0.0 };
                    
                    FPS_COUNTER.with(|fc| {
                        *fc.borrow_mut() = format!("FPS: {:.1}", fps);
                    });
                }
            });
            
            // Clear the canvas
            context.clear_color(0.0, 0.0, 0.0, 1.0);
            context.clear(WebGL::COLOR_BUFFER_BIT | WebGL::DEPTH_BUFFER_BIT);
            
            // Get the current rotation
            let rotation = ROTATION.with(|r| *r.borrow());
            
            // Create model-view-projection matrix
            let aspect = context.drawing_buffer_width() as f32 / context.drawing_buffer_height() as f32;
            
            // Create projection matrix (perspective)
            let projection = Mat4::perspective_rh(45.0 * (PI / 180.0), aspect, 0.1, 100.0);
            
            // Create view matrix (camera)
            let view = Mat4::look_at_rh(
                Vec3::new(0.0, 0.0, 3.0), // Camera position
                Vec3::new(0.0, 0.0, 0.0), // Look at origin
                Vec3::new(0.0, 1.0, 0.0), // Up vector
            );
            
            // Create model matrix (object transformation)
            let rotation_quat = Quat::from_rotation_y(rotation);
            let model = Mat4::from_quat(rotation_quat);
            
            // Combine matrices: projection * view * model
            let mvp = projection * view * model;
            
            // Convert to column-major array for WebGL
            let mvp_array = mvp.to_cols_array();
            
            // Set the uniform
            MVP_UNIFORM.with(|mvp_uniform| {
                if let Some(uniform) = &*mvp_uniform.borrow() {
                    context.uniform_matrix4fv_with_f32_array(
                        Some(uniform),
                        false,
                        &mvp_array,
                    );
                }
            });
            
            // Draw the cube
            NUM_INDICES.with(|ni| {
                let num_indices = *ni.borrow();
                context.draw_elements_with_i32(
                    WebGL::TRIANGLES,
                    num_indices as i32,
                    WebGL::UNSIGNED_SHORT,
                    0,
                );
            });
            
            // Render debug text
            render_debug_text(context);
        }
    });
}

fn render_debug_text(context: &WebGL) {
    // Get the FPS counter text
    let fps_text = FPS_COUNTER.with(|fc| fc.borrow().clone());
    
    // Get system info
    let system_info = format!(
        "WebGL Renderer\nCPU: Web Browser\nGPU: WebGL"
    );
    
    // Create a 2D canvas for text rendering
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.create_element("canvas").unwrap().dyn_into::<web_sys::HtmlCanvasElement>().unwrap();
    
    // Set canvas size to match WebGL context
    canvas.set_width(context.drawing_buffer_width() as u32);
    canvas.set_height(context.drawing_buffer_height() as u32);
    
    // Get 2D context
    let ctx = canvas.get_context("2d").unwrap().unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>().unwrap();
    
    // Set text properties
    ctx.set_font("16px monospace");
    ctx.set_fill_style(&JsValue::from_str("white").into());
    
    // Draw FPS counter at top-right
    ctx.fill_text(&fps_text, context.drawing_buffer_width() as f64 - 100.0, 20.0).unwrap();
    
    // Draw system info at top-left
    let lines = system_info.split('\n');
    let mut y = 20.0;
    for line in lines {
        ctx.fill_text(line, 10.0, y).unwrap();
        y += 20.0;
    }
    
    // Create a texture from the canvas
    let texture = context.create_texture().unwrap();
    context.active_texture(WebGL::TEXTURE1);
    context.bind_texture(WebGL::TEXTURE_2D, Some(&texture));
    
    // Set texture parameters
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_WRAP_S, WebGL::CLAMP_TO_EDGE as i32);
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_WRAP_T, WebGL::CLAMP_TO_EDGE as i32);
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_MIN_FILTER, WebGL::LINEAR as i32);
    context.tex_parameteri(WebGL::TEXTURE_2D, WebGL::TEXTURE_MAG_FILTER, WebGL::LINEAR as i32);
    
    // Upload canvas to texture
    context.tex_image_2d_with_u32_and_u32_and_html_canvas_element(
        WebGL::TEXTURE_2D,
        0,
        WebGL::RGBA as i32,
        WebGL::RGBA,
        WebGL::UNSIGNED_BYTE,
        &canvas,
    ).unwrap();
    
    // Restore active texture
    context.active_texture(WebGL::TEXTURE0);
}

fn request_animation_frame(f: &Rc<RefCell<Option<Closure<dyn FnMut()>>>>) {
    let window = web_sys::window().expect("no global `window` exists");
    window
        .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
        .expect("failed to request animation frame");
} 